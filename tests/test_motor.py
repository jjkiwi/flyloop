import numpy as np
import pytest

from flyloop.brain import LIFBrain
from flyloop.connectome.synthetic import eye_columns, synthetic_connectome
from flyloop.motor import DescendingReadout, TripodGait
from flyloop.motor.gait import LEGS, TRIPOD_A, TRIPOD_B


def test_readout_warns_when_descending_neurons_are_absent():
    import pandas as pd
    import scipy.sparse as sp

    from flyloop.connectome.schema import Connectome

    c = Connectome(
        pd.DataFrame({"id": [0], "type": ["x"], "nt": ["gaba"], "side": ["L"]}),
        sp.csr_matrix((1, 1)),
        name="empty",
    )
    with pytest.warns(UserWarning, match="DNa01"):
        DescendingReadout(c)


def test_looming_on_one_side_turns_away_from_it():
    c = synthetic_connectome()
    cols = eye_columns(c)
    turns = {}
    for side in ("L", "R"):
        b = LIFBrain(c, seed=4)
        ro = DescendingReadout(c)
        b.drive.set(cols[side], 200.0)
        for _ in range(3000):
            ro.update(b.step(), b.p.dt)
        turns[side] = ro.command().turn
    assert turns["L"] > 0.5, "threat on the left should turn right"
    assert turns["R"] < -0.5, "threat on the right should turn left"


def test_giant_fibre_latches_escape():
    c = synthetic_connectome()
    b = LIFBrain(c, seed=4)
    ro = DescendingReadout(c)
    b.drive.set(eye_columns(c)["L"], 200.0)
    for _ in range(3000):
        ro.update(b.step(), b.p.dt)
    assert ro.command().escape
    b.drive.clear()
    for _ in range(2000):
        ro.update(b.step(), b.p.dt)
    assert ro.command().escape, "an escape decision should not silently un-fire"


def test_stopping_suppresses_forward_drive():
    ro = DescendingReadout(synthetic_connectome())
    ro.rates["DNa01_L"] = ro.rates["DNa01_R"] = 100.0
    assert ro.command().forward == pytest.approx(1.0)
    ro.rates["DNp09_L"] = ro.rates["DNp09_R"] = 100.0
    assert ro.command().forward == pytest.approx(0.0)


def test_backward_drive_opposes_forward():
    ro = DescendingReadout(synthetic_connectome())
    ro.rates["MDN_L"] = ro.rates["MDN_R"] = 100.0
    assert ro.command().forward < 0


def test_tripod_groups_are_in_antiphase():
    g = TripodGait()
    j = g.step(1.0, 0.0, 0.02)
    a = np.mean([j[leg][1] for leg in TRIPOD_A])
    b = np.mean([j[leg][1] for leg in TRIPOD_B])
    assert (a > 0) != (b > 0), "the two tripods must not lift together"


def test_gait_output_is_eighteen_joints():
    g = TripodGait()
    assert TripodGait.flatten(g.step(0.5, 0.0, 0.01)).shape == (18,)
    assert len(LEGS) == 6


def test_zero_command_produces_no_motion():
    g = TripodGait()
    out = TripodGait.flatten(g.step(0.0, 0.0, 0.01))
    assert np.allclose(out, 0.0)


def test_turning_shortens_the_inner_stride():
    g = TripodGait()
    for _ in range(5):
        j = g.step(1.0, 0.0, 0.01)
    straight = abs(j["LF"][0])
    g.reset()
    for _ in range(5):
        j = g.step(1.0, -1.0, 0.01)  # turn left: left legs are on the inside
    assert abs(j["LF"][0]) < straight
