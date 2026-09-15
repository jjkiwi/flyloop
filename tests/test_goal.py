"""Flying at a target: the sign conventions, the reward, and the plasticity."""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from flyloop.brain.dopamine import (
    dopaminergic,
    mushroom_body_plasticity,
    proximity_reward,
)
from flyloop.connectome.schema import Connectome
from flyloop.loop_rate import Target, approach_score
from flyloop.vision.hexproject import project_eye

pytest.importorskip("torch", reason="optional 'data' extra")


def _eye_connectome(radius: int = 3) -> Connectome:
    rows, nid = [], 0
    for side in ("L", "R"):
        for x in range(-radius, radius + 1):
            for y in range(-radius, radius + 1):
                rows.append(
                    {
                        "id": nid,
                        "type": "L1",
                        "nt": "glutamate",
                        "side": side,
                        "hex1": x,
                        "hex2": y,
                    }
                )
                nid += 1
    return Connectome(pd.DataFrame(rows), sp.csr_matrix((nid, nid)), name="eyes")


# --- the sign convention that turned the fly away from the target ----------


def test_bearing_is_positive_to_the_right():
    """A target off the fly's right shoulder must read positive.

    The eye projection gives the right eye positive azimuth, the readout takes
    turn as DNa02_R - DNa02_L, and the body turns right on a positive turn. If
    bearing disagrees with those, the animal steers away from what it sees.
    """
    t = Target(x=0.0, y=-1.0, radius=0.1)  # to the right when facing +x
    bearing, _, dist = t.seen_from(0.0, 0.0, 0.0)
    assert bearing > 0
    assert dist == pytest.approx(1.0)


def test_bearing_is_negative_to_the_left():
    t = Target(x=0.0, y=1.0, radius=0.1)
    bearing, _, _ = t.seen_from(0.0, 0.0, 0.0)
    assert bearing < 0


def test_bearing_is_zero_straight_ahead():
    t = Target(x=2.0, y=0.0, radius=0.1)
    bearing, _, _ = t.seen_from(0.0, 0.0, 0.0)
    assert bearing == pytest.approx(0.0, abs=1e-9)


def test_right_eye_looks_right():
    c = _eye_connectome()
    right = project_eye(c, side="R")
    left = project_eye(c, side="L")
    assert right.azimuth.mean() > 0 > left.azimuth.mean()


def test_an_object_on_the_right_is_seen_by_the_right_eye():
    c = _eye_connectome()
    right = project_eye(c, side="R")
    left = project_eye(c, side="L")
    assert len(right.visible(60.0, 15.0)) > 0
    assert len(left.visible(60.0, 15.0)) == 0


def test_a_closer_target_subtends_a_wider_angle():
    t = Target(x=2.0, y=0.0, radius=0.2)
    _, far, _ = t.seen_from(0.0, 0.0, 0.0)
    _, near, _ = t.seen_from(1.5, 0.0, 0.0)
    assert near > far


# --- reward ---------------------------------------------------------------


def test_reward_grows_as_the_target_gets_closer():
    assert proximity_reward(0.1) > proximity_reward(1.0) > proximity_reward(5.0)


def test_reward_is_bounded_and_has_no_singularity():
    """Bounded in [0, 1], maximal at contact, and finite everywhere.

    An inverse-distance reward would blow up at contact; the exponential does
    not, and it underflows cleanly to zero far away rather than going negative.
    """
    assert proximity_reward(0.0) == pytest.approx(1.0)
    assert proximity_reward(1e6) == 0.0
    for d in (0.0, 0.5, 2.0, 50.0, 1e6):
        assert 0.0 <= proximity_reward(d) <= 1.0


def test_reward_floor_is_respected():
    assert proximity_reward(1e6, floor=0.2) == pytest.approx(0.2)


def test_missing_dopaminergic_population_fails_loudly():
    c = _eye_connectome()
    with pytest.raises(KeyError, match="dopaminergic"):
        dopaminergic(c)


# --- plasticity -----------------------------------------------------------


def _mb_connectome() -> Connectome:
    """Two Kenyon cells onto one MBON, plus a PAM neuron."""
    neurons = pd.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "type": ["KCg", "KCab", "MBON01", "PAM01"],
            "nt": ["acetylcholine", "acetylcholine", "glutamate", "dopamine"],
            "side": ["R"] * 4,
        }
    )
    dense = np.zeros((4, 4), dtype=np.float32)
    dense[0, 2] = 0.4
    dense[1, 2] = 0.4
    return Connectome(
        neurons, sp.csr_matrix(dense), name="mb", meta={"matrix_kind": "inprop"}
    )


def _plasticity(c, lr=0.5):
    """Build the sparse tensor the way RateBrain does, coalescing included."""
    import torch

    coo = c.W.T.tocoo()
    idx = torch.from_numpy(np.vstack([coo.row, coo.col]).astype(np.int64))
    val = torch.from_numpy(coo.data.astype(np.float32))
    sparse = torch.sparse_coo_tensor(idx, val, (c.n, c.n)).coalesce()
    values = sparse.values()
    return (
        mushroom_body_plasticity(c, values, sparse.indices(), learning_rate=lr),
        values,
    )


def test_positions_survive_coalescing():
    """coalesce() sorts entries; positions taken before it point at the wrong ones."""
    c = _mb_connectome()
    p, values = _plasticity(c)
    found = sorted(abs(float(v)) for v in values[p.entry_index])
    assert found == pytest.approx([0.4, 0.4]), "entry_index must find the KC->MBON weights"
    assert p.depression() == pytest.approx(0.0), "no learning yet, so no depression"


def test_plasticity_finds_the_kc_to_mbon_synapses():
    c = _mb_connectome()
    p, _ = _plasticity(c)
    assert p.n_synapses == 2
    assert len(p.kc) == 2 and len(p.mbon) == 1


def test_dopamine_depresses_only_the_active_kenyon_cell():
    """The rule is a coincidence detector, not a global gain change."""
    c = _mb_connectome()
    p, values = _plasticity(c)
    before = values[p.entry_index].clone()
    p.step(np.array([1.0, 0.0]), dopamine=1.0)
    after = values[p.entry_index]
    active = p.entry_pre == 0
    assert abs(after[active].item()) < abs(before[active].item())
    assert after[~active].item() == pytest.approx(before[~active].item())


def test_no_dopamine_means_no_learning():
    c = _mb_connectome()
    p, values = _plasticity(c)
    before = values[p.entry_index].clone()
    p.step(np.array([1.0, 1.0]), dopamine=0.0)
    assert (values[p.entry_index] == before).all()


def test_depression_never_passes_the_floor():
    c = _mb_connectome()
    p, values = _plasticity(c, lr=1.0)
    for _ in range(50):
        p.step(np.array([1.0, 1.0]), dopamine=1.0)
    floor = np.abs(p.initial) * p.min_fraction
    assert (np.abs(values[p.entry_index].numpy()) >= floor - 1e-6).all()


def test_wrong_activity_length_is_rejected():
    c = _mb_connectome()
    p, _ = _plasticity(c)
    with pytest.raises(ValueError, match="Kenyon cells"):
        p.step(np.zeros(7), dopamine=1.0)


# --- scoring --------------------------------------------------------------


def test_approach_score_measures_the_fraction_closed():
    log = pd.DataFrame({"distance": [2.0, 1.5, 1.0], "bearing": [10.0, 5.0, 0.0],
                        "half_width": [5.0] * 3})
    s = approach_score(log, radius=0.2)
    assert s["closed_fraction"] == pytest.approx(0.5)
    assert s["reached"] is False


def test_reached_needs_a_radius_to_mean_anything():
    log = pd.DataFrame(
        {"distance": [2.0, 0.1], "bearing": [0.0, 0.0], "half_width": [5.0, 5.0]}
    )
    assert approach_score(log)["reached"] is False
    assert approach_score(log, radius=0.2)["reached"] is True
