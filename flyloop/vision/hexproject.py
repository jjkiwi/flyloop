"""Putting a world onto the fly's hex columns.

The metadata gives every optic-lobe neuron a hexagonal column (``hex1``,
``hex2``) but not a viewing direction, so something has to bridge lattice
coordinates and the world. This module fits an affine map from the hex lattice
onto each eye's field of view.

**That map is an approximation and the approximation is the point of this
docstring.** The real retinotopic map is curved and the interommatidial angle
varies across the eye; an affine fit gets the topology right -- neighbouring
columns look in neighbouring directions, left is left -- and the metric wrong.
It is good enough to ask whether the animal turns toward an object, and not
good enough for anything quantitative about angles.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..connectome.schema import Connectome


@dataclass
class EyeProjection:
    """Viewing direction of every hex column of one eye."""

    labels: list[str]
    azimuth: np.ndarray  # degrees, 0 straight ahead, positive toward that eye
    elevation: np.ndarray  # degrees, positive up
    side: str

    @property
    def n(self) -> int:
        return len(self.labels)

    def visible(
        self, bearing: float, half_width: float, half_height: float = 90.0
    ) -> list[str]:
        """Columns covered by an object at ``bearing`` of angular radius ``half_width``."""
        d = (self.azimuth - bearing + 180.0) % 360.0 - 180.0
        m = (np.abs(d) <= half_width) & (np.abs(self.elevation) <= half_height)
        return [self.labels[i] for i in np.flatnonzero(m)]


def project_eye(
    c: Connectome,
    *,
    side: str = "R",
    cell_type: str = "L1",
    fov_azimuth: float = 150.0,
    fov_elevation: float = 120.0,
    gaze_offset: float = 45.0,
) -> EyeProjection:
    """Fit hex columns of one eye onto a field of view.

    ``gaze_offset`` is how far to the side each eye looks. The two eyes overlap
    in front, which is what lets a frontal object drive both and a lateral one
    drive only its own side.
    """
    n = c.neurons
    m = (n["type"].astype(str) == cell_type) & n["hex1"].notna()
    if "side" in n.columns:
        m &= n["side"].astype(str) == side
    rows = n[m]
    if rows.empty:
        raise KeyError(f"no hex-assigned {cell_type!r} on side {side!r}")

    h1 = rows["hex1"].to_numpy(dtype=float)
    h2 = rows["hex2"].to_numpy(dtype=float)
    labels = [f"{int(a)},{int(b)}" for a, b in zip(h1, h2, strict=True)]

    def scale(v: np.ndarray, span: float) -> np.ndarray:
        lo, hi = v.min(), v.max()
        if hi == lo:
            return np.zeros_like(v)
        return (v - lo) / (hi - lo) * span - span / 2.0

    # The hex axes are skewed relative to azimuth/elevation; using their
    # difference for elevation undoes most of that shear.
    az = scale(h1, fov_azimuth)
    el = scale(h2 - h1, fov_elevation)
    sign = +1.0 if side == "R" else -1.0
    return EyeProjection(
        labels=labels,
        azimuth=az * sign + gaze_offset * sign,
        elevation=el,
        side=side,
    )


class HexWorldView:
    """Renders objects in an arena onto both eyes' hex columns."""

    def __init__(
        self,
        c: Connectome,
        *,
        cell_types: tuple[str, ...] = ("L1", "L2"),
        **projection_kwargs,
    ):
        from .hexstim import column_neurons

        self.eyes = {s: project_eye(c, side=s, **projection_kwargs) for s in ("L", "R")}
        self.columns = {
            s: column_neurons(c, cell_types=cell_types, side=s) for s in ("L", "R")
        }
        order: list[int] = []
        for s in ("L", "R"):
            for rows in self.columns[s].values():
                order.extend(int(r) for r in rows)
        self.sensory = np.unique(np.asarray(order, dtype=np.int64))
        self._pos = {int(v): i for i, v in enumerate(self.sensory)}

    def pattern(self, objects: list[tuple[float, float]]) -> np.ndarray:
        """Sensory input vector for objects given as ``(bearing, half_width)``.

        Bearings are in degrees relative to the animal's heading. A dark object
        drives L1 and L2 in the columns it covers -- the histaminergic sign
        inversion, applied by hand because this model has no photoreceptors.
        """
        v = np.zeros(len(self.sensory), dtype=np.float32)
        for side, eye in self.eyes.items():
            cols = self.columns[side]
            for bearing, half_width in objects:
                for label in eye.visible(bearing, half_width):
                    for r in cols.get(label, ()):
                        v[self._pos[int(r)]] = 1.0
        return v

    def coverage(self, objects: list[tuple[float, float]]) -> dict[str, int]:
        """How many columns of each eye an object covers. Useful for sanity checks."""
        out = {}
        for side, eye in self.eyes.items():
            seen = set()
            for bearing, half_width in objects:
                seen.update(eye.visible(bearing, half_width))
            out[side] = len(seen)
        return out
