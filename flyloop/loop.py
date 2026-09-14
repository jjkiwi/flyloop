"""The closed sensorimotor loop.

    body.observe() -> compound eye -> temporal filter -> Poisson drive
      -> LIF brain -> descending readout -> locomotor command -> body.step()

The brain runs at its own fine timestep (0.1 ms) and the body at a coarse
control step (typically 10 ms), because that is the real relationship: neurons
are fast and actuators are not.  Mixing the two up is the most common source of
a loop that oscillates for no visible reason.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .brain.lif import LIFBrain, LIFParams
from .connectome.schema import Connectome
from .motor.descending import DescendingReadout, LocomotorCommand
from .vision.mapping import ColumnMap
from .vision.ommatidia import CompoundEye, TemporalFilter, to_rates


@dataclass
class LoopResult:
    """Per-control-step log of a closed-loop run."""

    log: pd.DataFrame
    spike_counts: np.ndarray
    duration: float

    def summary(self) -> str:
        log = self.log
        return (
            f"{self.duration:.2f} s, {len(log)} control steps\n"
            f"  mean forward {log['forward'].mean():+.3f}   "
            f"mean |turn| {log['turn'].abs().mean():.3f}\n"
            f"  mean stop    {log['stop'].mean():.3f}   "
            f"escaped: {bool(log['escape'].max())}\n"
            f"  total spikes {int(self.spike_counts.sum()):,}"
        )


class ClosedLoop:
    """Wires a connectome, an eye, a readout and a body into one running system."""

    def __init__(
        self,
        connectome: Connectome,
        body,
        *,
        eye: CompoundEye | None = None,
        params: LIFParams | None = None,
        input_cell_type: str = "PR",
        max_input_rate: float = 200.0,
        input_gain: float = 400.0,
        seed: int = 0,
    ):
        self.c = connectome
        self.body = body
        self.eye = eye or CompoundEye(n_columns=128)
        self.brain = LIFBrain(connectome, params, seed=seed)
        self.readout = DescendingReadout(connectome)
        self.filter = TemporalFilter()
        self.map = ColumnMap.from_connectome(connectome, self.eye, cell_type=input_cell_type)
        self.max_input_rate = max_input_rate
        self.input_gain = input_gain
        self.command = LocomotorCommand()
        self._steps_per_control = max(1, int(round(body.dt / self.brain.p.dt)))

    def reset(self) -> None:
        self.brain.reset()
        self.readout.reset()
        self.filter.reset()
        self.body.reset()
        self.command = LocomotorCommand()

    def control_step(self, counts: np.ndarray | None = None) -> dict[str, float]:
        """One control step: look, run the brain, decide, move.

        ``counts``, if given, accumulates per-neuron spike counts across *every*
        brain step of this control step, not just the last one.
        """
        pano = self.body.observe()
        sample = self.eye.sample(pano)
        # A dark object on a bright background is a *decrement*; the escape
        # system cares about contrast change, so the negated high-pass output
        # is what should excite the input neurons.
        change = {s: -v for s, v in self.filter(sample, self.body.dt).items()}

        self.brain.drive.clear()
        for idx, vals in self.map.drive_rates(change):
            self.brain.drive.set(
                idx, to_rates(vals, max_rate=self.max_input_rate, gain=self.input_gain)
            )

        spikes_this_step = 0
        for _ in range(self._steps_per_control):
            spk = self.brain.step()
            spikes_this_step += len(spk)
            if counts is not None and len(spk):
                np.add.at(counts, spk, 1)
            self.command = self.readout.update(spk, self.brain.p.dt)

        self.body.step(self.command)
        row = dict(self.command.as_dict())
        row["escape_source"] = self.command.escape_source
        row.update(self.body.state())
        row["spikes"] = spikes_this_step
        row.update({f"dn_{k}": v for k, v in self.readout.rates.items()})
        return row

    def run(self, duration: float, *, progress: bool = False) -> LoopResult:
        """Run the loop for ``duration`` seconds of simulated time."""
        n = int(round(duration / self.body.dt))
        counts = np.zeros(self.c.n, dtype=np.int64)
        rows = []
        for i in range(n):
            rows.append(self.control_step(counts))
            if progress and i % max(1, n // 10) == 0:  # pragma: no cover
                print(f"  {100 * i / n:3.0f}%", flush=True)
        return LoopResult(pd.DataFrame(rows), counts, n * self.body.dt)
