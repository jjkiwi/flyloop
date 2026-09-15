"""A physically simulated fly, driven by the same descending readout.

Every result in this project so far moved a point on a plane. This puts the
identical command into a body with legs, mass and contact physics:
NeuroMechFly v2 in MuJoCo, 42 actuated degrees of freedom, walking on a
tripod gait with stumbling and retraction correction.

The interface is what makes it a drop-in. FlyGym's ``HybridTurningController``
takes a **two-element descending signal**, one drive per side, which is exactly
the shape :class:`~flyloop.motor.descending.DescendingReadout` already produces
from the left-right difference of DNa02. Nothing about the brain, the readout or
the stimulus changes; only what the command moves.

**Which FlyGym.** This targets the 1.x API, published as ``flygym-gymnasium``,
because 2.x requires Python >= 3.12 while ``flyvis`` caps at < 3.13 -- and 1.x
runs on everything from 3.10. If a machine has 3.12 exactly, 2.x is the better
simulator; on anything else this is the one that works.

**Vision stays analytic.** The fly's bearing to a target is computed from the
pose rather than rendered through its eyes, which is also what the
connectome-in-a-body projects do (Fly.exe calls it "an analytic encoder, not
optical simulation"). Rendering the real optics needs a GPU and would not change
what the brain receives here, since the retinotopic pattern is built from
bearing and angular size either way.
"""

from __future__ import annotations

import numpy as np

from ..motor.descending import LocomotorCommand

_HINT = (
    "The physics body needs FlyGym 1.x and MuJoCo:\n"
    "    pip install flygym-gymnasium mujoco\n"
    "Headless machines without EGL or OSMesa can still run the physics; set\n"
    "MUJOCO_GL=disable and log the trajectory instead of rendering."
)

#: Leg segments carrying contact sensors, as the turning controller expects.
CONTACT_SEGMENTS = ("Tibia", "Tarsus1", "Tarsus2", "Tarsus3", "Tarsus4", "Tarsus5")
LEGS = ("LF", "LM", "LH", "RF", "RM", "RH")


class NeuroMechFlyBody:
    """NeuroMechFly v2 under a :class:`LocomotorCommand`.

    Positions are millimetres and headings radians, matching the simulator.
    ``theta`` decreases when the fly turns right, which is the same convention
    :class:`~flyloop.body.KinematicBody` uses, so a target bearing computed as
    positive-to-the-right works unchanged against either body.
    """

    def __init__(
        self,
        *,
        dt: float = 1e-4,
        control_dt: float = 0.05,
        base_drive: float = 1.0,
        turn_gain: float = 1.4,
        seed: int = 0,
    ):
        try:
            from flygym_gymnasium import Fly
            from flygym_gymnasium.examples.locomotion import HybridTurningController
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(_HINT) from exc

        self.dt = control_dt
        self.physics_dt = dt
        self.base_drive = base_drive
        self.turn_gain = turn_gain
        self._seed = seed
        self._steps = max(1, int(round(control_dt / dt)))

        fly = Fly(
            enable_adhesion=True,
            contact_sensor_placements=[
                f"{leg}{seg}" for leg in LEGS for seg in CONTACT_SEGMENTS
            ],
        )
        self._sim = HybridTurningController(fly=fly, cameras=[], timestep=dt)
        self.reset()

    # ---------------------------------------------------------------- driving

    def action(self, command: LocomotorCommand) -> np.ndarray:
        """Translate a locomotor command into the controller's two drives.

        A positive ``turn`` means right, and the controller turns right when the
        *left* drive is the larger of the two -- the inner legs take shorter
        steps. Getting this pair the wrong way round produces a fly that walks
        smoothly away from whatever it is looking at, which is a failure mode
        this project has already paid for once.
        """
        fwd = float(np.clip(command.forward, 0.0, 1.0)) * self.base_drive
        turn = float(np.clip(command.turn, -1.0, 1.0)) * self.turn_gain
        return np.clip([fwd + turn, fwd - turn], -1.0, 1.5)

    def reset(self) -> None:
        obs, _ = self._sim.reset(seed=self._seed)
        self._obs = obs
        self.t = 0.0
        self._origin = np.asarray(obs["fly"][0][:2], dtype=float).copy()
        self._yaw0 = float(obs["fly"][2][0])

    def step(self, command: LocomotorCommand) -> None:
        a = self.action(command)
        for _ in range(self._steps):
            self._obs, _, _, _, _ = self._sim.step(a)
        self.t += self.dt

    # ------------------------------------------------------------------ state

    def state(self) -> dict[str, float]:
        pos = np.asarray(self._obs["fly"][0][:2], dtype=float) - self._origin
        yaw = float(self._obs["fly"][2][0]) - self._yaw0
        return {
            "t": self.t,
            "x": float(pos[0]),
            "y": float(pos[1]),
            "theta": float((yaw + np.pi) % (2 * np.pi) - np.pi),
            "z": float(self._obs["fly"][0][2]),
        }

    def joint_angles(self) -> np.ndarray:  # pragma: no cover - telemetry
        """Current joint angles, for anyone who wants to draw the legs."""
        return np.asarray(self._obs["joints"][0], dtype=float)
