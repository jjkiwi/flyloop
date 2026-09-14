"""A minimal body: a point that moves on a plane among cylindrical obstacles.

There is no physics here, and that is on purpose.  Stage 4 swaps this for
NeuroMechFly, but every part of the loop -- eye geometry, spike-rate readout,
experiment scoring -- can be debugged against this in milliseconds instead of
waiting on MuJoCo.  When a behaviour appears only after the real body is
attached, you want to already know the rest of the pipeline is sound.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..motor.descending import LocomotorCommand


@dataclass
class Pillar:
    """A dark cylinder standing on the plane."""

    x: float
    y: float
    radius: float = 0.5
    height: float = 1.0
    #: Constant velocity, used to make an object loom toward a stationary animal.
    vx: float = 0.0
    vy: float = 0.0


@dataclass
class Arena:
    pillars: list[Pillar] = field(default_factory=list)

    def step(self, dt: float) -> None:
        for p in self.pillars:
            p.x += p.vx * dt
            p.y += p.vy * dt


class KinematicBody:
    """Pose integration plus a ray-cast panorama."""

    def __init__(
        self,
        arena: Arena | None = None,
        *,
        dt: float = 0.01,
        speed: float = 0.03,  # metres per second at full forward drive
        turn_rate: float = 2.0,  # radians per second at full turn
        panorama_size: tuple[int, int] = (64, 180),
    ):
        self.arena = arena or Arena()
        self.dt = dt
        self.speed = speed
        self.turn_rate = turn_rate
        self.panorama_size = panorama_size
        self._initial = [
            (p.x, p.y, p.vx, p.vy) for p in self.arena.pillars
        ]
        self.reset()

    def reset(self) -> None:
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.t = 0.0
        self.jumped = False
        for p, (x, y, vx, vy) in zip(self.arena.pillars, self._initial, strict=True):
            p.x, p.y, p.vx, p.vy = x, y, vx, vy

    def step(self, command: LocomotorCommand) -> None:
        dt = self.dt
        if command.escape and not self.jumped:
            # An escape jump: a single large displacement away from the heading.
            self.x -= 0.05 * np.cos(self.theta)
            self.y -= 0.05 * np.sin(self.theta)
            self.jumped = True
        self.theta -= command.turn * self.turn_rate * dt
        v = command.forward * self.speed
        self.x += v * np.cos(self.theta) * dt
        self.y += v * np.sin(self.theta) * dt
        self.arena.step(dt)
        self.t += dt

    # ------------------------------------------------------------- rendering

    def observe(self) -> np.ndarray:
        """Render the arena as seen from the current pose."""
        h, w = self.panorama_size
        pano = np.ones((h, w), dtype=np.float32)
        az = np.linspace(-np.pi, np.pi, w, endpoint=False)
        el = np.linspace(np.pi / 2, -np.pi / 2, h)
        for p in self.arena.pillars:
            dx, dy = p.x - self.x, p.y - self.y
            dist = float(np.hypot(dx, dy))
            if dist <= p.radius:
                return np.zeros((h, w), dtype=np.float32)  # inside the object
            bearing = np.arctan2(dy, dx) - self.theta
            bearing = (bearing + np.pi) % (2 * np.pi) - np.pi
            half_width = float(np.arcsin(np.clip(p.radius / dist, -1.0, 1.0)))
            half_height = float(np.arctan2(p.height / 2.0, dist))
            d_az = (az - bearing + np.pi) % (2 * np.pi) - np.pi
            mask_az = np.abs(d_az) <= half_width
            mask_el = np.abs(el) <= half_height
            if mask_az.any() and mask_el.any():
                pano[np.ix_(mask_el, mask_az)] = 0.0
        return pano

    def state(self) -> dict[str, float]:
        nearest = min(
            (float(np.hypot(p.x - self.x, p.y - self.y)) for p in self.arena.pillars),
            default=float("inf"),
        )
        return {
            "t": self.t,
            "x": self.x,
            "y": self.y,
            "theta": float((self.theta + np.pi) % (2 * np.pi) - np.pi),
            "nearest_obstacle": nearest,
            "jumped": float(self.jumped),
        }


def looming_arena(
    approach_speed: float = 0.5,
    start_distance: float = 1.5,
    radius: float = 0.3,
    height: float = 0.8,
) -> Arena:
    """An object on a collision course from straight ahead.

    The defaults give an object that subtends a few degrees at the start and
    fills much of the frontal field just before contact, which is the regime
    the escape system is tuned for.  A distant small object is a valid negative
    control, not a looming stimulus.
    """
    return Arena(
        [Pillar(x=start_distance, y=0.0, radius=radius, height=height, vx=-approach_speed)]
    )


def cluttered_arena(n: int = 8, radius: float = 1.5, seed: int = 0) -> Arena:
    """Static pillars scattered around the animal, for navigation experiments."""
    rng = np.random.default_rng(seed)
    angles = rng.uniform(0, 2 * np.pi, n)
    dists = rng.uniform(0.5, radius, n)
    return Arena(
        [
            Pillar(x=float(d * np.cos(a)), y=float(d * np.sin(a)), radius=0.1, height=0.6)
            for a, d in zip(angles, dists, strict=True)
        ]
    )
