"""Descending neurons as the interface between brain and body.

The descending neurons are the only route from the brain to the leg motor
circuits, and a handful of them are well characterised behaviourally.  That
makes them the natural API of the animal:

    DNa01   forward walking / speed
    DNa02   steering; the left-right difference sets turn direction
    MDN     backward walking
    DNp09   stopping and freezing, recruited by looming
    GF      the giant fibre; one spike triggers an escape jump

Reading these populations instead of inventing a bespoke output layer is what
keeps the project honest.  The moment you start fitting an arbitrary linear
readout on top of 160k neurons until the robot walks, it is the readout that is
walking, not the fly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..connectome.schema import Connectome

#: Descending populations this readout knows how to use.
DN_NAMES = ("DNa01", "DNa02", "MDN", "DNp09", "GF")


@dataclass
class LocomotorCommand:
    """What the brain is currently asking the body to do."""

    forward: float = 0.0  # -1 (backward) .. +1 (forward)
    turn: float = 0.0  # -1 (left) .. +1 (right)
    stop: float = 0.0  # 0 .. 1
    escape: bool = False

    def as_dict(self) -> dict[str, float]:
        return {
            "forward": self.forward,
            "turn": self.turn,
            "stop": self.stop,
            "escape": float(self.escape),
        }


class DescendingReadout:
    """Turns descending-neuron spike trains into a locomotor command.

    Rates are estimated with a leaky spike counter per population, which is the
    same exponential filter a motor neuron pool applies anyway.

    ``reference_rate`` is the firing rate at which a population is treated as
    fully driving its behaviour.  It is a calibration constant, not a fitted
    parameter: set it once from the recorded firing rates of the population and
    leave it alone.
    """

    def __init__(
        self,
        connectome: Connectome,
        *,
        tau: float = 50e-3,
        reference_rate: float = 100.0,
        escape_threshold: float = 1.0,
        missing: str = "warn",
    ):
        self.tau = tau
        self.reference_rate = reference_rate
        self.escape_threshold = escape_threshold
        self.pops: dict[str, np.ndarray] = {}
        self.absent: list[str] = []
        for name in DN_NAMES:
            for side in ("L", "R"):
                idx = self._side_population(connectome, name, side)
                self.pops[f"{name}_{side}"] = idx
            if len(self.pops[f"{name}_L"]) + len(self.pops[f"{name}_R"]) == 0:
                self.absent.append(name)
        if self.absent:
            msg = (
                f"connectome {connectome.name!r} contains none of: {self.absent}. "
                "The corresponding behaviours cannot be produced."
            )
            if missing == "raise":
                raise KeyError(msg)
            if missing == "warn":
                import warnings

                warnings.warn(msg, stacklevel=2)
        self.rates = {k: 0.0 for k in self.pops}
        self._gf_fired = False

    @staticmethod
    def _side_population(c: Connectome, name: str, side: str) -> np.ndarray:
        m = (c.neurons["type"].astype(str) == name)
        if "side" in c.neurons.columns:
            m &= c.neurons["side"].astype(str).str.upper().str.startswith(side)
        return np.flatnonzero(m.to_numpy())

    def reset(self) -> None:
        self.rates = {k: 0.0 for k in self.pops}
        self._gf_fired = False

    def update(self, spikes: np.ndarray, dt: float) -> LocomotorCommand:
        """Feed one timestep of spikes; returns the current command."""
        decay = float(np.exp(-dt / self.tau))
        hit = np.zeros(0, dtype=bool)
        for key, idx in self.pops.items():
            self.rates[key] *= decay
            if len(idx) and len(spikes):
                hit = np.isin(spikes, idx)
                n = int(hit.sum())
                if n:
                    self.rates[key] += n / (len(idx) * self.tau)
        return self.command()

    def _norm(self, key: str) -> float:
        return float(np.clip(self.rates[key] / self.reference_rate, 0.0, 1.0))

    def command(self) -> LocomotorCommand:
        """Current command, without advancing time."""
        fwd = 0.5 * (self._norm("DNa01_L") + self._norm("DNa01_R"))
        bwd = 0.5 * (self._norm("MDN_L") + self._norm("MDN_R"))
        stop = 0.5 * (self._norm("DNp09_L") + self._norm("DNp09_R"))
        # DNa02 steers by left-right imbalance.  Activity on the left DNa02
        # turns the animal left, so the signed difference is R minus L.
        turn = self._norm("DNa02_R") - self._norm("DNa02_L")
        gf = max(self.rates["GF_L"], self.rates["GF_R"])
        if gf >= self.escape_threshold:
            self._gf_fired = True
        return LocomotorCommand(
            forward=float(np.clip(fwd - bwd, -1.0, 1.0)) * (1.0 - stop),
            turn=float(np.clip(turn, -1.0, 1.0)),
            stop=stop,
            escape=self._gf_fired,
        )
