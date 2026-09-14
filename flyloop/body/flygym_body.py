"""Stage 4: the NeuroMechFly v2 biomechanical body, via FlyGym and MuJoCo.

**Status: written against the FlyGym 2.x API but not executed in this
repository's CI.** FlyGym 2.1 requires Python >= 3.12 and MuJoCo, neither of
which the default development environment here has. Every class and method name
below was read out of the ``flygym-2.1.0`` wheel rather than guessed, but
"compiles against the right names" is not "works". Treat the first successful
run as the acceptance test, and see ``docs/PLAN.md``.

A note on versions, because it cost time to find: FlyGym was rewritten in March
2026 and 2.x is not backward compatible. The old ``Fly`` + ``SingleFlySimulation``
interface now lives in the separate ``flygym-gymnasium`` package. The 2.x entry
points are ``flygym.Simulation`` and the ``flygym.compose`` builders.

What FlyGym gives us that we should not rewrite: the real ommatidial lattice.
``flygym/assets/model/neuromechfly/vision/ommatidia_id_map.npy`` maps a
512x450 rendered eye image onto **721 ommatidia per eye**, with
``pale_mask.npy`` marking the pale/yellow spectral subtypes. That is a measured
lattice; :mod:`flyloop.vision.ommatidia` only approximates one.
"""

from __future__ import annotations

import numpy as np

from ..motor.descending import LocomotorCommand
from ..motor.gait import LEGS, TripodGait

_INSTALL_HINT = (
    "FlyGym 2.x is not installed. Install the biomechanical body with:\n"
    "    pip install 'flyloop[body]'\n"
    "It needs Python >= 3.12; flyvis caps at < 3.13, so use Python 3.12."
)


class FlyGymBody:
    """Drives a NeuroMechFly v2 fly from a :class:`LocomotorCommand`."""

    def __init__(
        self,
        dt: float = 1e-4,
        *,
        world: str = "flat",
        warmup: float = 0.05,
        fly_name: str = "fly",
    ):
        try:
            from flygym import Simulation
            from flygym.compose import ActuatorType, FlatGroundWorld, NeuroMechFly
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(_INSTALL_HINT) from exc

        if world != "flat":  # pragma: no cover - only one world wired up so far
            raise ValueError(f"unsupported world {world!r}; only 'flat' is wired up")

        self.dt = dt
        self.gait = TripodGait()
        self._actuator_type = ActuatorType
        self._fly = NeuroMechFly(name=fly_name)
        self._fly.add_vision()
        self._world = FlatGroundWorld()
        self._world.add_fly(self._fly)
        self._sim = Simulation(self._world, timestep=dt)
        self._name = fly_name
        self._warmup = warmup
        self._joint_order = self._fly.get_actuated_jointdofs_order(
            ActuatorType.POSITION
        )
        self._gait_map = self._build_gait_map()
        self.reset()

    # ------------------------------------------------------------- actuation

    def _build_gait_map(self) -> list[tuple[int, str, int]]:
        """Match this project's 18 gait outputs onto the fly's own joint DOFs.

        NeuroMechFly actuates far more degrees of freedom than a hexapod's three
        per leg, so the mapping is by name: each actuated DOF whose name mentions
        a leg and one of coxa/femur/tibia gets the corresponding gait output;
        every other DOF is left at its neutral pose.

        Raising here rather than sending a wrong-length or wrong-order vector is
        deliberate. A silently mismatched actuator order produces a fly that
        twitches convincingly and means nothing.
        """
        joints = [str(j) for j in self._joint_order]
        mapping: list[tuple[int, str, int]] = []
        for i, name in enumerate(joints):
            low = name.lower()
            leg = next((leg for leg in LEGS if leg.lower() in low), None)
            if leg is None:
                continue
            for k, segment in enumerate(("coxa", "femur", "tibia")):
                if segment in low:
                    mapping.append((i, leg, k))
                    break
        if not mapping:
            raise RuntimeError(
                "could not match any gait output to a NeuroMechFly joint. "
                f"The fly's actuated DOFs are named: {joints[:12]}... "
                "Update FlyGymBody._build_gait_map for this FlyGym version "
                "rather than guessing an actuator order."
            )
        return mapping

    def reset(self) -> None:  # pragma: no cover - optional dependency
        self.gait.reset()
        self._sim.reset()
        if self._warmup:
            self._sim.warmup(self._warmup)
        self.t = 0.0
        self._neutral = self._sim.get_joint_angles(self._name).copy()

    def step(self, command: LocomotorCommand) -> None:  # pragma: no cover
        joints = self.gait.step(command.forward, command.turn, self.dt)
        targets = self._neutral.copy()
        for idx, leg, k in self._gait_map:
            targets[idx] = self._neutral[idx] + joints[leg][k]
        self._sim.set_actuator_inputs(
            self._name, self._actuator_type.POSITION, targets
        )
        self._sim.step()
        self.t += self.dt

    # ------------------------------------------------------------ perception

    def ommatidia(self) -> dict[str, np.ndarray]:  # pragma: no cover
        """Per-ommatidium intensity for each eye, from FlyGym's own retina.

        ``get_ommatidia_readouts`` returns ``(2, n_ommatidia, 2)``: left and
        right eye, then the yellow and pale channels, with a zero in whichever
        channel the ommatidium is not. Summing the last axis therefore recovers
        one intensity per ommatidium without inventing a spectral model.
        """
        readouts = self._sim.get_ommatidia_readouts(self._name)
        merged = readouts.sum(axis=-1)
        return {"L": merged[0].astype(np.float32), "R": merged[1].astype(np.float32)}

    def observe(self) -> np.ndarray:  # pragma: no cover
        """Not available: this body reports ommatidia, not a panorama.

        FlyGym renders directly into the fly's retinotopic coordinates. Passing
        that through :class:`~flyloop.vision.CompoundEye` would resample an
        already-correct lattice onto an approximate one and scramble the
        retinotopy. Use :meth:`ommatidia`.
        """
        raise NotImplementedError(
            "FlyGymBody exposes ommatidia() directly; do not resample it through "
            "CompoundEye. See docs/DATA.md."
        )

    def state(self) -> dict[str, float]:  # pragma: no cover
        pos = np.asarray(self._sim.get_body_positions(self._name)[0], dtype=float)
        return {"t": self.t, "x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2])}
