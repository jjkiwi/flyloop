"""Non-connectome controllers to compare against.

Every claim of the form "the fly brain does X" needs an answer to "and what
does ten lines of arithmetic on the same input do?".  Usually the answer is
"the same thing, better", and that is a finding worth knowing before spending a
month on it, not after.

These baselines consume exactly the same visual input as the brain does, so the
comparison is about the controller and not about the sensor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..body.kinematic import KinematicBody
from ..motor.descending import LocomotorCommand
from ..vision.ommatidia import CompoundEye, TemporalFilter


class ReactiveBaseline:
    """Escape when the dark fraction of the visual field expands fast enough.

    This is a crude angular-expansion detector, roughly what an engineer would
    write in an afternoon.  If the connectome model cannot beat it on escape
    latency, false-alarm rate, or robustness to clutter, say so plainly.
    """

    def __init__(
        self,
        eye: CompoundEye | None = None,
        *,
        expansion_threshold: float = 0.6,
        turn_gain: float = 3.0,
    ):
        self.eye = eye or CompoundEye(n_columns=128)
        self.filter = TemporalFilter()
        self.expansion_threshold = expansion_threshold
        self.turn_gain = turn_gain
        self.reset()

    def reset(self) -> None:
        self.filter.reset()
        self._prev_dark = None
        self.escaped = False

    def __call__(self, panorama: np.ndarray, dt: float) -> LocomotorCommand:
        sample = self.eye.sample(panorama)
        dark = {s: float((v < 0.5).mean()) for s, v in sample.items()}
        total = dark["L"] + dark["R"]
        rate = 0.0 if self._prev_dark is None else (total - self._prev_dark) / dt
        self._prev_dark = total
        if rate > self.expansion_threshold:
            self.escaped = True
        turn = float(np.clip(self.turn_gain * (dark["L"] - dark["R"]), -1.0, 1.0))
        return LocomotorCommand(
            forward=0.0, turn=turn, stop=1.0 if self.escaped else 0.0, escape=self.escaped
        )


def run_baseline(arena, *, duration: float = 2.6, dt: float = 0.01, **kw) -> pd.DataFrame:
    """Drive a :class:`KinematicBody` with :class:`ReactiveBaseline`."""
    body = KinematicBody(arena, dt=dt)
    ctrl = ReactiveBaseline(**kw)
    rows = []
    for _ in range(int(round(duration / dt))):
        cmd = ctrl(body.observe(), dt)
        body.step(cmd)
        row = dict(cmd.as_dict())
        row.update(body.state())
        rows.append(row)
    return pd.DataFrame(rows)
