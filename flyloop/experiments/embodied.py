"""The same readout, in a body with legs.

Every result before this one moved a point on a plane. Here the identical
descending command -- the left-right difference of DNa02, unfitted, straight out
of the signed connectome -- drives NeuroMechFly v2 in MuJoCo: 42 actuated joints,
a tripod gait, contact physics, and a fly that falls over if you ask it to.

**Why the measurement is a mirror pair.** A walking NeuroMechFly does not go
straight. Told to walk straight for 2 s it drifts by +15, -0 and -41 degrees on
three different seeds -- a spread of 55 degrees, several times any steering signal
we expect. A single episode therefore says nothing at all.

So each measurement is two episodes from the *same body seed*: one with the
target at +bearing, one at -bearing. The gait noise is common to both; the
stimulus is what differs. The statistic is the difference of the two heading
changes, which cancels the drift:

    fixation = (dtheta_left - dtheta_right) / 2

positive when the fly turned toward the target in both mirror images. Reporting
one episode's heading change instead would be reporting gait noise.

The commanded turn is reported next to it. That one is pure brain -- it never
touches the body -- so it separates "the readout has the right sign" from "the
legs delivered it".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..connectome.schema import Connectome
from ..loop_rate import RateLoop, Target

#: Millimetres. NeuroMechFly walks about 14 mm/s, so this is a few seconds away.
TARGET_DISTANCE = 25.0
#: Millimetres. A pillar this wide subtends about 6 degrees at 25 mm.
TARGET_RADIUS = 2.5
#: Millimetres, the e-folding distance of the proximity reward at this scale.
REWARD_SCALE = 8.0


def target_at(bearing_deg: float, *, distance: float = TARGET_DISTANCE) -> Target:
    """A target seen at ``bearing_deg`` by a fly at the origin facing +x.

    Bearing is positive to the fly's right, so the world angle is its negative
    (see :meth:`flyloop.loop_rate.Target.seen_from`).
    """
    a = np.radians(-bearing_deg)
    return Target(
        x=float(distance * np.cos(a)),
        y=float(distance * np.sin(a)),
        radius=TARGET_RADIUS,
    )


def embodied_episode(
    c: Connectome,
    *,
    bearing_deg: float,
    seed: int = 0,
    steps: int = 30,
    control_dt: float = 0.05,
    dopamine: bool = True,
    progress: bool = False,
) -> pd.DataFrame:
    """One episode of the rate readout driving the physics body."""
    from ..body.nmf_body import NeuroMechFlyBody

    body = NeuroMechFlyBody(control_dt=control_dt, seed=seed)
    loop = RateLoop(
        c,
        target=target_at(bearing_deg),
        dt=control_dt,
        reward_scale=REWARD_SCALE,
        dopamine=dopamine,
        body=body,
    )
    log = loop.run(steps, progress=progress)
    log["bearing0"] = bearing_deg
    log["seed"] = seed
    return log


def mirror_pair(
    c: Connectome, *, bearing_deg: float = 35.0, seed: int = 0, **kw
) -> dict[str, float]:
    """Two mirrored episodes from one body seed, and the drift-free statistic."""
    right = embodied_episode(c, bearing_deg=+bearing_deg, seed=seed, **kw)
    left = embodied_episode(c, bearing_deg=-bearing_deg, seed=seed, **kw)
    return {
        "seed": seed,
        "fixation_deg": float(np.degrees(_heading_change(left) - _heading_change(right)) / 2.0),
        "commanded_turn": float(right["turn"].mean() - left["turn"].mean()) / 2.0,
        "dtheta_right_deg": float(np.degrees(_heading_change(right))),
        "dtheta_left_deg": float(np.degrees(_heading_change(left))),
        "closed_right_mm": float(right["distance"].iloc[0] - right["distance"].iloc[-1]),
        "closed_left_mm": float(left["distance"].iloc[0] - left["distance"].iloc[-1]),
        "_logs": (right, left),
    }


def _heading_change(log: pd.DataFrame) -> float:
    """Net heading change over an episode, unwrapped so a big turn is not aliased."""
    return float(np.unwrap(log["theta"].to_numpy())[-1] - log["theta"].to_numpy()[0])


def embodied_experiment(
    c: Connectome, *, seeds: tuple[int, ...] = (0, 1, 2), **kw
) -> pd.DataFrame:
    """Mirror pairs over several body seeds."""
    rows = []
    for s in seeds:
        r = mirror_pair(c, seed=s, **kw)
        r.pop("_logs")
        rows.append(r)
    return pd.DataFrame(rows)
