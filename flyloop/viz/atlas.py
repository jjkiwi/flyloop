"""Where the active neurons actually are, in the animal's own coordinates.

**What this can and cannot draw.** The prepared MaleCNS files carry exactly one
spatial number per neuron: ``somaLocation``, the position of its cell body.
There are no skeletons in them, and the services that serve skeletons --
neuPrint, FlyWire's Codex -- are unreachable from here. So this draws a *cloud
of cell bodies*, not morphology. The well-known rainbow renders of the fly brain
show neurites filling the neuropil; the cell bodies sit on the rind around it,
so the shape here is a shell, not a solid. Saying otherwise would be a lie about
the data.

That limitation matters less than it looks for this project's question. We are
not asking what a neuron looks like; we are asking *which* neurons carry a
decision, and where in the animal they sit.

**The highlight is a difference, not an activation.** Lighting up everything the
stimulus excites answers the wrong question -- most of it responds to "there is
an object" rather than "the object is on my right". The same mirror-pair logic
that made Run 10 honest applies here: the neurons responsible for the *action*
are the ones whose activity flips when the target moves from one side to the
other. :func:`lateralised_activation` returns exactly that difference.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..connectome.schema import Connectome

#: Which metadata column is which anatomical axis, established from the data:
#: somata labelled L sit at x~74k and R at x~23k, and the VNC sits at z~105k
#: against the brain's z~30k.
AXES = {"x": "left-right", "y": "dorsal-ventral", "z": "anterior-posterior"}

#: Everything in front of this z is brain; behind it is nerve cord.
BRAIN_Z_MAX = 60_000.0

#: Superclass -> colour, for the context cloud.
PALETTE = {
    "ol_intrinsic": "#3b6ea5",
    "cb_intrinsic": "#8a5fb0",
    "vnc_intrinsic": "#2f8f6f",
    "visual_projection": "#4aa3c7",
    "descending_neuron": "#e0a020",
    "vnc_motor": "#d9534f",
    "cb_sensory": "#b06a8a",
    "vnc_sensory": "#5f8f4a",
    "ol_sensory": "#6f9fd8",
}
OTHER_COLOUR = "#4a4a58"


@dataclass
class Atlas:
    """Soma positions for one connectome, with the unlocated cells dropped."""

    xyz: np.ndarray  # (n_located, 3)
    rows: np.ndarray  # indices back into the connectome
    super_class: np.ndarray
    type: np.ndarray
    name: str

    @classmethod
    def from_connectome(cls, c: Connectome) -> "Atlas":
        need = ("soma_x", "soma_y", "soma_z")
        missing = [col for col in need if col not in c.neurons.columns]
        if missing:
            raise KeyError(
                f"connectome {c.name!r} has no soma coordinates ({missing}). "
                "Load it with flyloop.connectome.data_prep, which parses "
                "somaLocation; the synthetic fixture has none."
            )
        xyz = c.neurons[list(need)].to_numpy(dtype=float)
        rows = np.flatnonzero(np.isfinite(xyz).all(axis=1))
        return cls(
            xyz=xyz[rows],
            rows=rows,
            super_class=c.neurons["super_class"].to_numpy()[rows],
            type=c.neurons["type"].to_numpy()[rows],
            name=c.name,
        )

    def __len__(self) -> int:
        return len(self.rows)

    @property
    def brain(self) -> np.ndarray:
        """Mask of the cells in the brain rather than the nerve cord."""
        return self.xyz[:, 2] < BRAIN_Z_MAX

    def colours(self) -> list[str]:
        return [PALETTE.get(s, OTHER_COLOUR) for s in self.super_class]

    def values(self, per_neuron: np.ndarray) -> np.ndarray:
        """Restrict a full-length per-neuron vector to the located cells."""
        v = np.asarray(per_neuron)
        if len(v) < self.rows.max() + 1:
            raise ValueError(
                f"vector has {len(v)} entries, too short for neuron row {self.rows.max()}"
            )
        return v[self.rows]

    def report(self) -> str:
        n_brain = int(self.brain.sum())
        return (
            f"atlas of {self.name}: {len(self):,} located somata "
            f"({n_brain:,} brain, {len(self) - n_brain:,} nerve cord)"
        )


# ---------------------------------------------------------------- activation


def situation_activation(
    c: Connectome,
    *,
    bearing_deg: float = 35.0,
    half_width_deg: float = 5.7,
    hops: int = 5,
    bias: float = 0.0,
    view=None,
) -> np.ndarray:
    """Per-neuron activation with an object at ``bearing_deg``.

    Bearing is positive to the fly's right, the convention fixed in
    :meth:`flyloop.loop_rate.Target.seen_from`. The value returned is each
    neuron's peak activation across the synaptic hops, which is what the
    descending readout takes elsewhere in this project.
    """
    from ..brain.rate import RateBrain, steady_state
    from ..vision.hexproject import HexWorldView

    view = view or HexWorldView(c)
    brain = RateBrain(c, view.sensory, num_layers=hops, default_bias=bias)
    pattern = view.pattern([(bearing_deg, half_width_deg)])
    res = brain.run(steady_state(pattern, hops))
    return res.activations.max(axis=1)


def lateralised_activation(c: Connectome, *, bearing_deg: float = 35.0, **kw) -> pd.DataFrame:
    """What changes when the same object moves from one side to the other.

    Returns a frame with the two activations and their difference. Positive
    ``lateralised`` means the neuron prefers the target on the fly's right.

    This is the per-neuron version of Run 10's mirror pair, and it is here for
    the same reason: a neuron that responds equally to both is reporting that an
    object exists, not deciding which way to turn.
    """
    from ..vision.hexproject import HexWorldView

    view = kw.pop("view", None) or HexWorldView(c)
    right = situation_activation(c, bearing_deg=+bearing_deg, view=view, **kw)
    left = situation_activation(c, bearing_deg=-bearing_deg, view=view, **kw)
    return pd.DataFrame(
        {
            "right": right,
            "left": left,
            "lateralised": right - left,
            "both": 0.5 * (right + left),
        }
    )
