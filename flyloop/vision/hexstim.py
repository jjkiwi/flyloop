"""Stimuli in the fly's own hexagonal eye coordinates.

Runs 1 to 3 injected current straight into LC4. That answers "what does this
connectome connect to LC4", but it cannot answer anything about vision: LC4
activity was imposed rather than earned, so the optic lobe between the eye and
LC4 was never exercised, and hypotheses about optomotor responses or phototaxis
were untestable by construction.

This module closes that gap. The prepared MaleCNS metadata assigns 23,720
optic-lobe neurons to hexagonal columns -- 892 on the right, 875 on the left --
so a pattern can be painted onto those columns and delivered to the lamina
monopolar cells that actually receive it.

**Which cells to drive, and the sign.** Photoreceptors are histaminergic and
*inhibit* their targets, so the lamina monopolar cells L1 and L2 depolarise when
light *decreases*. A dark object on a bright background is a light decrement, so
the columns it covers are the columns where L1 and L2 are driven. L1 heads the
ON pathway and L2 the OFF pathway; both are driven here, which is a
simplification stated rather than hidden -- this model has no photoreceptor
stage, so the sign inversion is applied by hand at the input.

The stimulus geometry comes from ``connectome-interpreter`` so that it matches
the hex coordinate convention of the tables the columns came from.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..connectome.schema import Connectome


def hex_coords(
    c: Connectome, *, cell_type: str = "L1", side: str = "R"
) -> list[str]:
    """Hex column labels (``"x,y"``) covered by one cell type in one eye."""
    n = c.neurons
    m = (n["type"].astype(str) == cell_type) & n["hex1"].notna()
    if "side" in n.columns:
        m &= n["side"].astype(str) == side
    rows = n[m]
    if rows.empty:
        raise KeyError(
            f"connectome {c.name!r} has no hex-assigned {cell_type!r} on side {side!r}. "
            "This needs the prepared metadata; see docs/DATA.md."
        )
    return [f"{int(a)},{int(b)}" for a, b in zip(rows["hex1"], rows["hex2"], strict=True)]


def column_neurons(
    c: Connectome, *, cell_types: tuple[str, ...] = ("L1", "L2"), side: str = "R"
) -> dict[str, np.ndarray]:
    """Map each hex column label to the neuron rows of ``cell_types`` in it."""
    n = c.neurons
    m = n["type"].astype(str).isin(cell_types) & n["hex1"].notna()
    if "side" in n.columns:
        m &= n["side"].astype(str) == side
    rows = np.flatnonzero(m.to_numpy())
    labels = [
        f"{int(a)},{int(b)}"
        for a, b in zip(
            n["hex1"].to_numpy()[rows], n["hex2"].to_numpy()[rows], strict=True
        )
    ]
    out: dict[str, list[int]] = {}
    for label, row in zip(labels, rows, strict=True):
        out.setdefault(label, []).append(int(row))
    return {k: np.asarray(v, dtype=np.int64) for k, v in out.items()}


def centre_column(coords: list[str]) -> str:
    """The column closest to the centre of the field."""
    xy = np.array([[float(v) for v in c.split(",")] for c in coords])
    d = ((xy - xy.mean(axis=0)) ** 2).sum(axis=1)
    return coords[int(np.argmin(d))]


@dataclass
class HexStimulus:
    """A sequence of frames, each naming the hex columns that are stimulated."""

    frames: list[list[str]]
    name: str
    meta: dict = field(default_factory=dict)

    @property
    def n_frames(self) -> int:
        return len(self.frames)

    def sizes(self) -> list[int]:
        return [len(f) for f in self.frames]

    def describe(self) -> str:
        return f"{self.name}: {self.n_frames} frames, columns lit {self.sizes()}"


def looming(
    coords: list[str], *, n_time: int = 8, centre: str | None = None
) -> HexStimulus:
    """A dark disc expanding from one column outwards, one hex ring per frame."""
    from connectome_interpreter.external_map import looming_stimulus

    start = centre or centre_column(coords)
    frames = [list(f) for f in looming_stimulus([start], coords, n_time=n_time)]
    return HexStimulus(frames, "looming", {"centre": start, "n_time": n_time})


def receding(coords: list[str], *, n_time: int = 8, centre: str | None = None) -> HexStimulus:
    """The same disc, shrinking. The control for an approaching object."""
    s = looming(coords, n_time=n_time, centre=centre)
    return HexStimulus(list(reversed(s.frames)), "receding", dict(s.meta))


def static_disc(
    coords: list[str], *, n_time: int = 8, centre: str | None = None, at: int | None = None
) -> HexStimulus:
    """A disc of constant size, held for the whole trial.

    The size is taken from the middle of the equivalent looming sequence, so it
    covers a comparable area without ever changing. An object that appears
    abruptly at full size is *not* this control -- ommatid's first protocol used
    one and found it produced a looming-like response by itself, because an
    abrupt onset is a strong transient. Here the disc is present from frame one.
    """
    s = looming(coords, n_time=n_time, centre=centre)
    idx = n_time // 2 if at is None else at
    held = s.frames[idx]
    return HexStimulus(
        [list(held) for _ in range(n_time)], "static", dict(s.meta, held_frame=idx)
    )


def hemifield(coords: list[str], *, bright: str = "L", n_time: int = 8) -> HexStimulus:
    """One half of the field dark, the other untouched. Tests phototaxis.

    ``bright`` names the half that is *not* driven, since the drive represents a
    light decrement.
    """
    xy = np.array([[float(v) for v in c.split(",")] for c in coords])
    mid = xy[:, 0].mean()
    dark = [c for c, x in zip(coords, xy[:, 0], strict=True) if (x > mid) == (bright == "L")]
    return HexStimulus(
        [list(dark) for _ in range(n_time)], f"hemifield_dark_{'R' if bright == 'L' else 'L'}",
        {"bright": bright},
    )


def grating(
    coords: list[str], *, direction: int = +1, period: int = 6, n_time: int = 8
) -> HexStimulus:
    """A square-wave grating drifting across the field. Tests the optomotor response.

    ``direction`` is +1 or -1 along the first hex axis. A wide-field drifting
    pattern should, in an intact animal, turn it the same way -- the hypothesis
    ommatid pre-registered and did not confirm.
    """
    xy = np.array([[float(v) for v in c.split(",")] for c in coords])
    x = xy[:, 0]
    frames = []
    for t in range(n_time):
        phase = direction * t * (period / n_time)
        dark = [
            c
            for c, xi in zip(coords, x, strict=True)
            if ((xi + phase) % period) < period / 2
        ]
        frames.append(dark)
    return HexStimulus(
        frames, f"grating_{'R' if direction > 0 else 'L'}",
        {"direction": direction, "period": period},
    )


#: The stimulus set, matching the protocol ommatid pre-registered.
def stimulus_set(coords: list[str], *, n_time: int = 8) -> dict[str, HexStimulus]:
    """Looming and its controls, plus the optomotor and phototaxis stimuli."""
    return {
        "looming": looming(coords, n_time=n_time),
        "receding": receding(coords, n_time=n_time),
        "static": static_disc(coords, n_time=n_time),
        "grating_R": grating(coords, direction=+1, n_time=n_time),
        "grating_L": grating(coords, direction=-1, n_time=n_time),
        "dark_L": hemifield(coords, bright="R", n_time=n_time),
        "dark_R": hemifield(coords, bright="L", n_time=n_time),
    }


def stimulus_table(stimuli: dict[str, HexStimulus]) -> pd.DataFrame:
    """Columns lit per frame for each stimulus, for checking before running."""
    return pd.DataFrame({k: v.sizes() for k, v in stimuli.items()})
