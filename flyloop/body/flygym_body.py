"""Stage 4: the NeuroMechFly v2 biomechanical body, via FlyGym and MuJoCo.

Kept behind an optional import so the rest of the project installs and tests
without MuJoCo.  ``pip install 'flyloop[body]'`` enables it.

The adapter's job is only translation: locomotor command in, joint targets out,
camera frame back.  All of the connectome work happens upstream, which means
switching from this to a physical hexapod in stage 5 touches this file and
nothing else.
"""

from __future__ import annotations

import numpy as np

from ..motor.descending import LocomotorCommand
from ..motor.gait import TripodGait


class FlyGymBody:
    """Wraps a FlyGym ``NeuroMechFly`` simulation behind the :class:`Body` protocol."""

    def __init__(self, dt: float = 1e-4, render: bool = False, **sim_kwargs):
        try:
            from flygym import Fly, SingleFlySimulation
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "FlyGym is not installed. Install the biomechanical body with:\n"
                "    pip install 'flyloop[body]'\n"
                "See docs/DATA.md for what stage 4 needs."
            ) from exc
        from flygym import Fly, SingleFlySimulation

        self.dt = dt
        self.gait = TripodGait()
        self._fly = Fly(enable_vision=True, **sim_kwargs)
        self._sim = SingleFlySimulation(fly=self._fly, timestep=dt)
        self._render = render
        self._obs: dict | None = None
        self.reset()

    def reset(self) -> None:  # pragma: no cover - optional dependency
        self.gait.reset()
        self._obs, _ = self._sim.reset()
        self.t = 0.0

    def step(self, command: LocomotorCommand) -> None:  # pragma: no cover
        joints = self.gait.step(command.forward, command.turn, self.dt)
        action = {"joints": TripodGait.flatten(joints)}
        self._obs, _, _, _, _ = self._sim.step(action)
        self.t += self.dt

    def observe(self) -> np.ndarray:  # pragma: no cover
        """FlyGym returns per-ommatidium intensities directly.

        Those are already in the fly's own retinotopic coordinates, so they are
        returned as a two-row 'panorama' that :class:`CompoundEye` must not be
        applied to a second time.  Use :meth:`ommatidia` instead when driving
        the brain from this body.
        """
        vision = self._obs.get("vision")
        if vision is None:
            raise RuntimeError("this Fly was built without enable_vision=True")
        return np.asarray(vision, dtype=np.float32)

    def ommatidia(self) -> dict[str, np.ndarray]:  # pragma: no cover
        """Per-eye ommatidial intensities, ready to drive photoreceptor neurons."""
        v = self.observe()
        return {"L": v[0].mean(axis=-1), "R": v[1].mean(axis=-1)}

    def state(self) -> dict[str, float]:  # pragma: no cover
        pos = np.asarray(self._obs["fly"][0], dtype=float)
        return {"t": self.t, "x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2])}
