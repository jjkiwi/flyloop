"""The physics body, and the sign convention that has to hold across three modules."""

import os

import numpy as np
import pytest

from flyloop.experiments.embodied import _heading_change, target_at
from flyloop.motor.descending import LocomotorCommand

pytest.importorskip("flygym_gymnasium", reason="physics body needs FlyGym 1.x")
os.environ.setdefault("MUJOCO_GL", "disable")


@pytest.fixture(scope="module")
def body():
    from flyloop.body.nmf_body import NeuroMechFlyBody

    return NeuroMechFlyBody(control_dt=0.05, seed=0)


# ------------------------------------------------------------------ geometry


def test_positive_bearing_puts_the_target_on_the_right():
    """Bearing is positive to the fly's right, so a right target has y < 0."""
    assert target_at(+35.0).y < 0
    assert target_at(-35.0).y > 0
    assert target_at(0.0).y == pytest.approx(0.0, abs=1e-9)


def test_target_at_round_trips_through_seen_from():
    for bearing in (-80.0, -35.0, 0.0, 12.5, 35.0, 80.0):
        seen, _, dist = target_at(bearing).seen_from(0.0, 0.0, 0.0)
        assert seen == pytest.approx(bearing, abs=1e-6)
        assert dist == pytest.approx(25.0, rel=1e-9)


def test_heading_change_unwraps_past_pi():
    """A turn through half a circle must read as a big turn, not a tiny one."""
    import pandas as pd

    theta = np.linspace(0.0, 1.2 * np.pi, 40)
    log = pd.DataFrame({"theta": (theta + np.pi) % (2 * np.pi) - np.pi})
    assert _heading_change(log) == pytest.approx(1.2 * np.pi, abs=1e-6)


# -------------------------------------------------------------- the adapter


def test_turning_right_drives_the_left_side_harder(body):
    """Positive turn means right, and the controller turns right on left drive.

    Swapping this pair produces a fly that walks smoothly away from whatever it
    is looking at. That bug has already cost this project one full run, so the
    mapping is pinned here rather than left to the docstring.
    """
    right = body.action(LocomotorCommand(forward=1.0, turn=+0.5))
    left = body.action(LocomotorCommand(forward=1.0, turn=-0.5))
    assert right[0] > right[1]
    assert left[1] > left[0]
    assert right == pytest.approx(left[::-1])


def test_straight_command_is_symmetric(body):
    a = body.action(LocomotorCommand(forward=1.0, turn=0.0))
    assert a[0] == pytest.approx(a[1])


def test_drives_are_clipped_to_the_controller_range(body):
    a = body.action(LocomotorCommand(forward=1.0, turn=+1.0))
    assert a.max() <= 1.5
    assert a.min() >= -1.0


def test_commands_outside_the_unit_range_saturate(body):
    assert body.action(LocomotorCommand(forward=9.0, turn=9.0)) == pytest.approx(
        body.action(LocomotorCommand(forward=1.0, turn=1.0))
    )
    assert body.action(LocomotorCommand(forward=-9.0, turn=0.0)) == pytest.approx(
        body.action(LocomotorCommand(forward=0.0, turn=0.0))
    )


# ------------------------------------------------------------- the simulator


def test_reset_puts_the_fly_back_at_its_own_origin(body):
    body.reset()
    st = body.state()
    assert (st["x"], st["y"], st["theta"], st["t"]) == pytest.approx((0, 0, 0, 0))
    assert st["z"] > 0  # standing on the floor, not through it


@pytest.mark.slow
def test_a_right_turn_command_actually_turns_the_body_right(body):
    """The whole point: positive turn has to decrease theta in the physics too.

    This is the only test here that runs MuJoCo. It is slow (tens of seconds)
    because the tripod gait needs a few hundred milliseconds of walking before a
    heading difference exceeds the gait's own wobble.
    """
    headings = {}
    for label, turn in (("right", +1.0), ("left", -1.0)):
        body.reset()
        for _ in range(8):
            body.step(LocomotorCommand(forward=1.0, turn=turn))
        headings[label] = body.state()["theta"]
    assert headings["right"] < headings["left"]
    assert np.degrees(headings["left"] - headings["right"]) > 45.0
