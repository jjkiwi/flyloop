"""Compound-eye sampling: a camera frame becomes one luminance per ommatidium.

A fly does not see a rectangular image.  Each eye is a roughly hexagonal
lattice of ommatidia, each looking in its own direction, with an interommatidial
angle of about 5 degrees and a total field of view that is enormous compared
with any camera: the two eyes together cover nearly the whole sphere.

Getting this mapping right is the least glamorous and most load-bearing part of
the visual front end.  If you feed a connectome-derived optic lobe a naively
downsampled camera image, the retinotopy is wrong and every motion computation
downstream is wrong with it -- but nothing crashes, so the error is invisible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def hex_directions(
    n: int, fov_azimuth: float = 150.0, fov_elevation: float = 120.0
) -> tuple[np.ndarray, np.ndarray]:
    """``n`` viewing directions on a hexagonal lattice inside an elliptical field.

    Returns azimuth and elevation in degrees, ordered from the centre outwards so
    that truncating the arrays keeps the field centred.
    """
    if n <= 0:
        return np.empty(0), np.empty(0)
    a, b = fov_azimuth / 2.0, fov_elevation / 2.0
    # Start from a spacing that would place ~n points in the ellipse, then grow
    # the lattice until there are enough.
    spacing = float(np.sqrt(np.pi * a * b / max(n, 1)))
    for _ in range(64):
        cols = int(np.ceil(2 * a / spacing)) + 2
        rows = int(np.ceil(2 * b / (spacing * np.sqrt(3) / 2))) + 2
        i = np.arange(-cols, cols + 1)
        j = np.arange(-rows, rows + 1)
        jj, ii = np.meshgrid(j, i, indexing="ij")
        az = (ii + 0.5 * (jj % 2)) * spacing
        el = jj * spacing * np.sqrt(3) / 2
        inside = (az / a) ** 2 + (el / b) ** 2 <= 1.0
        if inside.sum() >= n:
            az, el = az[inside], el[inside]
            order = np.argsort(az**2 + el**2)
            return az[order][:n], el[order][:n]
        spacing *= 0.95
    raise RuntimeError("could not fit the requested number of ommatidia")  # pragma: no cover


@dataclass
class Eye:
    """One eye: a set of viewing directions in the head's frame of reference."""

    azimuth: np.ndarray  # degrees, positive = toward that eye's side
    elevation: np.ndarray  # degrees, positive = up
    side: str

    @property
    def n(self) -> int:
        return len(self.azimuth)


class CompoundEye:
    """A pair of eyes that samples an equirectangular panorama.

    ``n_columns`` is per eye.  NeuroMechFly v2 uses 721 ommatidia per eye; the
    ommatid robot project resamples its camera onto 892 retinotopic columns.
    Either is a reasonable target; the default here is small so tests are fast.
    """

    def __init__(
        self,
        n_columns: int = 128,
        *,
        fov_azimuth: float = 150.0,
        fov_elevation: float = 120.0,
        gaze_offset: float = 45.0,
    ):
        az, el = hex_directions(n_columns, fov_azimuth, fov_elevation)
        self.eyes = {
            "L": Eye(az - gaze_offset, el, "L"),
            "R": Eye(az + gaze_offset, el, "R"),
        }
        self.n_columns = n_columns

    def sample(self, panorama: np.ndarray) -> dict[str, np.ndarray]:
        """Luminance per ommatidium for each eye.

        ``panorama`` is ``(H, W)`` in [0, 1], covering azimuth -180..180 degrees
        left to right and elevation +90..-90 degrees top to bottom.
        """
        if panorama.ndim == 3:
            panorama = panorama.mean(axis=2)
        h, w = panorama.shape
        out = {}
        for side, eye in self.eyes.items():
            az = ((eye.azimuth + 180.0) % 360.0) - 180.0
            col = np.clip(((az + 180.0) / 360.0 * w).astype(int), 0, w - 1)
            row = np.clip(((90.0 - eye.elevation) / 180.0 * h).astype(int), 0, h - 1)
            out[side] = panorama[row, col].astype(np.float32)
        return out


class TemporalFilter:
    """Per-column high-pass filter: the fly's visual system reports change.

    Photoreceptors adapt, so a static scene produces little sustained drive.
    Without this stage a bright wall looks like a permanent looming object.
    """

    def __init__(self, tau: float = 50e-3):
        self.tau = tau
        self._mean: dict[str, np.ndarray] = {}

    def __call__(self, sample: dict[str, np.ndarray], dt: float) -> dict[str, np.ndarray]:
        alpha = float(np.exp(-dt / self.tau))
        out = {}
        for side, v in sample.items():
            m = self._mean.get(side)
            if m is None or m.shape != v.shape:
                m = v.copy()
            m = alpha * m + (1 - alpha) * v
            self._mean[side] = m
            out[side] = v - m
        return out

    def reset(self) -> None:
        self._mean.clear()


def to_rates(
    values: np.ndarray, *, max_rate: float = 200.0, gain: float = 400.0
) -> np.ndarray:
    """Map a signed per-column signal to non-negative Poisson rates in Hz."""
    return np.clip(values * gain, 0.0, max_rate).astype(np.float32)
