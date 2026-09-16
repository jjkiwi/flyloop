"""Descending neurons as the interface between brain and body.

The descending neurons are the only route from the brain to the leg motor
circuits, and a handful of them are well characterised behaviourally. That
makes them the natural API of the animal:

    DNa01           forward walking / speed
    DNa02           steering; the left-right difference sets turn direction

**The DNa02 readout is not this project's guess.** Rayshubskiy, Holtz & Wilson
(*eLife* 102230; first posted as bioRxiv 2020.04.04.024703) record from both
copies of DNa02 at once and report that the fly's rotational velocity is
*linearly proportional to the right-left difference in DNa02 activity* -- a
"see-saw", where excitation of one copy comes with inhibition of its
contralateral twin. ``turn = DNa02_R - DNa02_L`` is that relationship, with the
same functional form, so the readout is taken from the literature rather than
fitted here.

Two limits of leaning on it. DNa02 sits below DNa03 and LAL013 in the steering
hierarchy (Westeinde et al.), so reading DNa02 alone reads one level of a stack;
and Yang et al. (*Cell* 2024) describe finer structure -- DNa02 shortens strides
on the inside of a turn -- that a scalar turn command cannot express.
    MDN             backward walking
    DNp09           stopping and freezing
    DNp01 ("GF")    the giant fibre; classically, one spike triggers escape
    DNp02, DNp04    direct targets of the LC4 looming population

Reading these populations instead of inventing a bespoke output layer is what
keeps the project honest. The moment you start fitting an arbitrary linear
readout on top of 160k neurons until the robot walks, it is the readout that is
walking, not the fly.

**Why the escape channel is a set, not the giant fibre.** The obvious design is
to read escape off DNp01/DNp09, which is what the textbook account of looming
suggests and what this project originally did. The ommatid project (MIT) ran
that design against the real MaleCNS v1.0 wiring on a hexapod in September 2026
and measured the opposite: across 210 trials *DNp01, DNp09 and MDN stayed at
0 Hz in every condition*, and the looming-evoked rise was carried by DNp04
(+12.8 Hz) and DNp02 (+5.6 Hz), the two direct LC4 targets. So the default
escape channel here is the union, and which member actually fires is a result
to report rather than an assumption to build in.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from ..connectome.schema import Connectome

#: Populations backing each behavioural role.
FORWARD_NAMES = ("DNa01",)
STEERING_NAMES = ("DNa02",)
BACKWARD_NAMES = ("MDN",)
STOP_NAMES = ("DNp09",)
#: "GF" is the classical name for the cell the Janelia datasets call DNp01;
#: both are listed so a readout works whichever naming a dataset uses.
ESCAPE_NAMES = ("DNp01", "GF", "DNp02", "DNp04", "DNp09")

#: Every population this readout tracks.
DN_NAMES = tuple(
    dict.fromkeys(
        FORWARD_NAMES + STEERING_NAMES + BACKWARD_NAMES + STOP_NAMES + ESCAPE_NAMES
    )
)

ROLES = {
    "forward": FORWARD_NAMES,
    "steering": STEERING_NAMES,
    "backward": BACKWARD_NAMES,
    "stop": STOP_NAMES,
    "escape": ESCAPE_NAMES,
}


@dataclass
class LocomotorCommand:
    """What the brain is currently asking the body to do."""

    forward: float = 0.0  # -1 (backward) .. +1 (forward)
    turn: float = 0.0  # -1 (left) .. +1 (right)
    stop: float = 0.0  # 0 .. 1
    escape: bool = False
    #: Which escape population crossed threshold, if any. A result, not a knob.
    escape_source: str | None = None

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
    fully driving its behaviour. It is a calibration constant, not a fitted
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
        escape_populations: tuple[str, ...] = ESCAPE_NAMES,
        missing: str = "warn",
    ):
        self.tau = tau
        self.reference_rate = reference_rate
        self.escape_threshold = escape_threshold
        self.escape_populations = escape_populations
        self.pops: dict[str, np.ndarray] = {}
        self.present: list[str] = []
        for name in dict.fromkeys(DN_NAMES + tuple(escape_populations)):
            total = 0
            for side in ("L", "R"):
                idx = self._side_population(connectome, name, side)
                self.pops[f"{name}_{side}"] = idx
                total += len(idx)
            if total:
                self.present.append(name)

        # Warn per behavioural role, not per name: a dataset that calls the
        # giant fibre DNp01 is not missing anything by lacking "GF".
        self.absent_roles = [
            role
            for role, names in ROLES.items()
            if not any(n in self.present for n in names)
        ]
        if self.absent_roles and missing != "ignore":
            msg = (
                f"connectome {connectome.name!r} has no neurons for these roles: "
                f"{self.absent_roles}. Those behaviours cannot be produced. "
                f"Populations found: {self.present or 'none'}."
            )
            if missing == "raise":
                raise KeyError(msg)
            warnings.warn(msg, stacklevel=2)

        self.rates = dict.fromkeys(self.pops, 0.0)
        self._escape_source: str | None = None

    @staticmethod
    def _side_population(c: Connectome, name: str, side: str) -> np.ndarray:
        m = c.neurons["type"].astype(str) == name
        if "side" in c.neurons.columns:
            m &= c.neurons["side"].astype(str).str.upper().str.startswith(side)
        return np.flatnonzero(m.to_numpy())

    def reset(self) -> None:
        self.rates = dict.fromkeys(self.pops, 0.0)
        self._escape_source = None

    def update(self, spikes: np.ndarray, dt: float) -> LocomotorCommand:
        """Feed one timestep of spikes; returns the current command."""
        decay = float(np.exp(-dt / self.tau))
        for key, idx in self.pops.items():
            self.rates[key] *= decay
            if len(idx) and len(spikes):
                n = int(np.isin(spikes, idx).sum())
                if n:
                    self.rates[key] += n / (len(idx) * self.tau)
        return self.command()

    def _norm(self, key: str) -> float:
        return float(np.clip(self.rates[key] / self.reference_rate, 0.0, 1.0))

    def _role_rate(self, names: tuple[str, ...]) -> float:
        """Mean normalised rate across both sides of every population in a role."""
        vals = [
            self._norm(f"{n}_{s}")
            for n in names
            for s in ("L", "R")
            if len(self.pops.get(f"{n}_{s}", ()))
        ]
        return float(np.mean(vals)) if vals else 0.0

    def command(self) -> LocomotorCommand:
        """Current command, without advancing time."""
        fwd = self._role_rate(FORWARD_NAMES)
        bwd = self._role_rate(BACKWARD_NAMES)
        stop = self._role_rate(STOP_NAMES)
        # DNa02 steers by left-right imbalance. Activity on the left DNa02
        # turns the animal left, so the signed difference is R minus L.
        turn = sum(self._norm(f"{n}_R") - self._norm(f"{n}_L") for n in STEERING_NAMES)

        if self._escape_source is None:
            for name in self.escape_populations:
                rate = max(
                    self.rates.get(f"{name}_L", 0.0), self.rates.get(f"{name}_R", 0.0)
                )
                if rate >= self.escape_threshold:
                    self._escape_source = name
                    break

        return LocomotorCommand(
            forward=float(np.clip(fwd - bwd, -1.0, 1.0)) * (1.0 - stop),
            turn=float(np.clip(turn, -1.0, 1.0)),
            stop=stop,
            escape=self._escape_source is not None,
            escape_source=self._escape_source,
        )
