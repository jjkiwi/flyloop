"""The contract every body must satisfy.

Keeping this interface narrow is what lets the same brain drive a stub, a
MuJoCo fly and -- in stage 5 -- a real hexapod over a socket.  The body owes the
brain exactly two things: a panorama to look at, and the consequences of the
last command.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from ..motor.descending import LocomotorCommand


class Body(Protocol):
    """Anything that can be driven by a :class:`LocomotorCommand`."""

    dt: float

    def reset(self) -> None:
        """Return to the start of an episode."""

    def step(self, command: LocomotorCommand) -> None:
        """Apply one control step."""

    def observe(self) -> np.ndarray:
        """Equirectangular panorama in [0, 1], shape ``(H, W)``.

        Azimuth runs -180..180 degrees left to right, elevation +90..-90 top to
        bottom, matching :meth:`flyloop.vision.CompoundEye.sample`.
        """

    def state(self) -> dict[str, float]:
        """Scalar telemetry for logging and for experiment scoring."""
