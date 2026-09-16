"""Flying at a target, with dopamine that grows as it gets closer.

The loop is quasi-static: at each control step the world is rendered onto the
fly's hex columns, the rate network is run to steady state over a fixed number
of synaptic hops, the descending neurons are read, and the body moves. The
dynamics live in the body, not in the neurons -- which is exactly the
approximation :mod:`flyloop.brain.rate` forces, since its time axis is the
synaptic-hop axis and it has no memory (see docs/RESULTS.md, Run 5).

For approaching an object that is a defensible approximation: the animal's
decision at each moment depends on where the object is now. For anything about
motion direction it is not, and this loop cannot be used for that.

Reward reaches the brain through the fly's own PAM cluster and acts on its own
KC->MBON synapses. The steering readout is unchanged and unfitted: turn is the
left-right difference of DNa02, as in every run before this one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .body.kinematic import Arena, KinematicBody, Pillar
from .brain.dopamine import dopaminergic, proximity_reward
from .brain.rate import population_index, rate_brain, steady_state
from .connectome.schema import Connectome
from .motor.descending import LocomotorCommand
from .vision.hexproject import HexWorldView


@dataclass
class Target:
    """What the fly is flying at."""

    x: float = 1.5
    y: float = 0.0
    radius: float = 0.15

    def as_pillar(self) -> Pillar:
        return Pillar(x=self.x, y=self.y, radius=self.radius, height=0.8)

    def seen_from(self, x: float, y: float, theta: float) -> tuple[float, float, float]:
        """Bearing in degrees, angular half-width in degrees, and distance.

        **Bearing is positive to the fly's right.** Three conventions have to
        agree or the animal turns away from what it is looking at, which is
        exactly the bug this sign fixes:

        * :class:`~flyloop.vision.hexproject.EyeProjection` gives the right eye
          positive azimuth,
        * the readout takes turn as ``DNa02_R - DNa02_L``, positive for right,
        * :class:`~flyloop.body.KinematicBody` turns right when ``turn`` is
          positive (it decrements ``theta``).

        The world's own angle ``atan2(dy, dx) - theta`` is positive
        counter-clockwise, which is to the *left*, so it is negated here once
        rather than being patched at three call sites.
        """
        dx, dy = self.x - x, self.y - y
        dist = float(np.hypot(dx, dy))
        bearing = np.degrees(theta - np.arctan2(dy, dx))
        bearing = (bearing + 180.0) % 360.0 - 180.0
        half_width = float(
            np.degrees(np.arcsin(np.clip(self.radius / max(dist, self.radius), -1, 1)))
        )
        return float(bearing), half_width, dist


class RateLoop:
    """One episode of approaching a target, with optional dopamine."""

    def __init__(
        self,
        connectome: Connectome,
        *,
        target: Target | None = None,
        hops: int = 5,
        bias: float = 0.0,
        dt: float = 0.05,
        speed: float = 0.08,
        turn_rate: float = 3.0,
        turn_gain: float = 3.0,
        dopamine: bool = True,
        reward_scale: float = 0.8,
        learning_rate: float = 0.05,
        view: HexWorldView | None = None,
        body=None,
        backend: str = "fast",
    ):
        self.c = connectome
        self.target = target or Target()
        self.hops = hops
        self.dt = dt
        self.speed = speed
        self.turn_rate = turn_rate
        self.turn_gain = turn_gain
        self.use_dopamine = dopamine
        self.reward_scale = reward_scale

        self.view = view or HexWorldView(connectome)
        self.pam = dopaminergic(connectome, cluster="PAM")
        # Reward is an external input to the dopaminergic neurons, which is
        # where it enters the animal too.
        self.sensory = np.unique(np.concatenate([self.view.sensory, self.pam]))
        self._pos = {int(v): i for i, v in enumerate(self.sensory)}
        self._visual_slot = np.array(
            [self._pos[int(v)] for v in self.view.sensory], dtype=np.int64
        )
        self._pam_slot = np.array([self._pos[int(v)] for v in self.pam], dtype=np.int64)

        self.brain = rate_brain(
            connectome,
            self.sensory,
            backend=backend,
            num_layers=hops,
            default_bias=bias,
        )
        self.plastic = self.brain.plasticity(learning_rate=learning_rate)

        self.record = {}
        for side in ("L", "R"):
            self.record.update(
                population_index(connectome, ("DNa02", "DNa01", "LC4"), side=side)
            )
        # Any body exposing reset(), step(command) and state() with x, y and
        # theta will do. The default is the kinematic stub; pass a
        # NeuroMechFlyBody to put the same readout in a physics simulation.
        self.body = body or KinematicBody(
            Arena([self.target.as_pillar()]), dt=dt, speed=speed, turn_rate=turn_rate
        )
        self.reset()

    def reset(self) -> None:
        self.body.reset()
        self.t = 0.0
        self.reward = 0.0

    # ------------------------------------------------------------------ step

    def step(self) -> dict[str, float]:
        st = self.body.state()
        bearing, half_width, dist = self.target.seen_from(st["x"], st["y"], st["theta"])

        v = np.zeros(len(self.sensory), dtype=np.float32)
        v[self._visual_slot] = self.view.pattern([(bearing, half_width)])
        reward = proximity_reward(dist, scale=self.reward_scale) if self.use_dopamine else 0.0
        v[self._pam_slot] = reward

        res = self.brain.run(steady_state(v, self.hops), record=self.record)
        peak = res.peak()
        turn = float(peak.get("DNa02_R", 0.0) - peak.get("DNa02_L", 0.0))
        forward = 0.5 * (peak.get("DNa01_L", 0.0) + peak.get("DNa01_R", 0.0))

        kc_activity = res.activations[self.plastic.kc].max(axis=1)
        depression = self.plastic.step(kc_activity, reward)

        cmd = LocomotorCommand(
            forward=float(np.clip(0.4 + forward, 0.0, 1.0)),
            turn=float(np.clip(self.turn_gain * turn, -1.0, 1.0)),
        )
        self.body.step(cmd)
        self.t += self.dt
        self.reward = reward

        return {
            "t": self.t,
            "x": st["x"],
            "y": st["y"],
            "theta": st["theta"],
            "distance": dist,
            "bearing": bearing,
            "half_width": half_width,
            "turn": cmd.turn,
            "forward": cmd.forward,
            "reward": reward,
            "dopamine_depression": depression,
            "DNa02_L": float(peak.get("DNa02_L", 0.0)),
            "DNa02_R": float(peak.get("DNa02_R", 0.0)),
            "LC4_L": float(peak.get("LC4_L", 0.0)),
            "LC4_R": float(peak.get("LC4_R", 0.0)),
        }

    def run(self, n_steps: int = 40, *, progress: bool = False) -> pd.DataFrame:
        rows = []
        for i in range(n_steps):
            row = self.step()
            rows.append(row)
            if progress:  # pragma: no cover
                print(
                    f"    {i:3d}  d={row['distance']:.3f}  bearing={row['bearing']:+6.1f}"
                    f"  turn={row['turn']:+.2f}  reward={row['reward']:.3f}",
                    flush=True,
                )
            if row["distance"] <= self.target.radius:
                break
        return pd.DataFrame(rows)


def approach_score(log: pd.DataFrame, *, radius: float | None = None) -> dict[str, float]:
    """Did it get closer, and how fast?

    ``closed_fraction`` is how much of the starting distance was removed; it is
    the number to compare across conditions, since it is scale free. Pass
    ``radius`` to have ``reached`` mean anything.
    """
    if log.empty:
        return {"start": np.nan, "final": np.nan, "closed_fraction": np.nan}
    start = float(log["distance"].iloc[0])
    final = float(log["distance"].iloc[-1])
    return {
        "start": start,
        "final": final,
        "closest": float(log["distance"].min()),
        "closed_fraction": (start - final) / start if start else np.nan,
        "mean_abs_bearing": float(log["bearing"].abs().mean()),
        "steps": len(log),
        "reached": bool(radius is not None and final <= radius),
    }
