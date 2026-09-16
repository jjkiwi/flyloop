"""Reflex with a learned base: the invariants that keep the result readable."""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from flyloop.connectome.schema import Connectome
from flyloop.hybrid import TURN_RAD_PER_S, WALK_MM_PER_S, HybridLoop
from flyloop.loop_rate import Target

pytest.importorskip("connectome_interpreter", reason="optional 'data' extra")

#: Everything the loop needs: L1 photoreceptor targets carrying a hex
#: retinotopy so HexWorldView can paint an object, ORNs for two odours, PAM for
#: dopamine, Kenyon cells and MBONs for the plasticity, and the descending
#: neurons that the steering readout and the MBON valence both name.
OTHER_TYPES = (
    ["ORN_DM1"] * 2
    + ["ORN_DM4"] * 2
    + ["ORN_DA1"] * 2
    + ["ORN_VA2"] * 2
    + ["PAM01"] * 4
    + ["KCg-m"] * 6
    + ["MBON01"] * 4
    + ["DNa02"] * 2
    + ["DNa01"] * 2
    + ["MDN"] * 2
    + ["DNp09"] * 2
    + ["LC4"] * 2
    + ["LPLC2"] * 2
)


@pytest.fixture(scope="module")
def toy() -> Connectome:
    rng = np.random.default_rng(0)
    rows, nid = [], 0
    for side in ("L", "R"):
        for x in range(-2, 3):
            for y in range(-2, 3):
                rows.append(
                    {
                        "id": nid,
                        "type": "L1",
                        "nt": "acetylcholine",
                        "side": side,
                        "hex1": x,
                        "hex2": y,
                    }
                )
                nid += 1
    for i, kind in enumerate(OTHER_TYPES):
        nt = (
            "dopamine"
            if kind.startswith("PAM")
            else ("gaba" if kind.startswith("MBON") else "acetylcholine")
        )
        rows.append(
            {
                "id": nid,
                "type": kind,
                "nt": nt,
                "side": "L" if i % 2 else "R",
                "hex1": np.nan,
                "hex2": np.nan,
            }
        )
        nid += 1
    neurons = pd.DataFrame(rows)
    m = len(neurons)
    # Sparse and weak. At 25% density with weights near 1.0 every neuron in a
    # graph this small saturates its tanh at 1.0, whatever the deliberate
    # wiring below says -- which is how the first version of this fixture
    # measured a learned effect of exactly zero and looked fine doing it.
    dense = rng.random((m, m)).astype(np.float32)
    dense[dense < 0.90] = 0.0
    dense *= 0.02
    np.fill_diagonal(dense, 0.0)
    kinds = neurons["type"].astype(str).to_numpy()
    dense *= np.where(np.char.startswith(kinds.astype(str), "MBON"), -1.0, 1.0)[:, None]

    # The olfactory chain is wired deliberately rather than left to the random
    # draw: ORN -> KC -> MBON -> DNa02 is the path the learned bias travels, and
    # a fixture that happens not to contain it tests nothing.
    #
    # The weights are small on purpose. Wired strongly, the MBONs saturate the
    # tanh at 1.0 and stay there however hard the KC->MBON synapses are
    # depressed -- 52% depression moved the readout by exactly zero. Saturation
    # hides learning, and a fixture that saturates would have passed every test
    # here while measuring nothing. Real MBONs sit far from saturation.
    def rows_of(prefix):
        return np.flatnonzero(np.char.startswith(kinds.astype(str), prefix))

    orn, kc = rows_of("ORN"), rows_of("KC")
    mbon, dna02 = rows_of("MBON"), np.flatnonzero(kinds == "DNa02")
    dense[np.ix_(orn, kc)] = 0.08
    dense[np.ix_(kc, mbon)] = 0.08
    dense[np.ix_(mbon, dna02)] = -0.08
    return Connectome(neurons, sp.csr_matrix(dense), name="toy", meta={"matrix_kind": "inprop"})


@pytest.fixture
def loop(toy):
    return HybridLoop(toy, target=Target(x=25.0, y=0.0, radius=2.5), hops=3, gain=1.0)


# ------------------------------------------------------- the paired readout


def test_without_training_the_learned_component_is_exactly_zero(loop):
    """The paired design makes this true by construction, not by luck.

    Trained weights and naive weights are the same object before any pairing,
    so their difference has to vanish. If it does not, the two passes are not
    seeing the same input and every learned number downstream is noise.
    """
    row = loop.step(smell=loop.odours["trained"], odour_name="trained")
    assert row["learned_component"] == pytest.approx(0.0, abs=1e-12)


def test_the_paired_readout_puts_the_weights_back(loop):
    """It swaps naive weights in mid-step; failing to restore corrupts everything."""
    loop.train(3, odour_name="trained")
    before = np.asarray(loop.brain.values[loop.plastic.entry_index]).copy()
    loop.step(smell=loop.odours["trained"], odour_name="trained")
    after = np.asarray(loop.brain.values[loop.plastic.entry_index])
    assert after == pytest.approx(before, abs=1e-12)


def test_training_moves_the_learned_component_off_zero(loop):
    loop.train(6, odour_name="trained")
    row = loop.step(smell=loop.odours["trained"], odour_name="trained")
    assert row["learned_component"] != 0.0
    assert loop.plastic.depression() > 0


# ------------------------------------------------------------ the modulation


def test_no_odour_leaves_the_reflex_exactly_alone(loop):
    """With nothing in the air this must be Run 10, bit for bit."""
    loop.learned.gain = 1e6
    loop.train(4, odour_name="trained")
    row = loop.step(smell=None, odour_name=None)
    assert row["modulation"] == 1.0


def test_modulation_never_inverts_the_turn(loop):
    """However hard the gain is cranked, which way to turn stays the eye's call."""
    loop.train(6, odour_name="trained")
    loop.learned.gain = 1e12
    row = loop.step(smell=loop.odours["trained"], odour_name="trained")
    assert row["modulation"] >= 0.0
    assert np.sign(row["turn"]) in (0.0, np.sign(row["DNa02_R"] - row["DNa02_L"]))


# ------------------------------------------------------------------ plumbing


def test_the_kinematic_stub_is_calibrated_to_the_physics_body(loop):
    """Both bodies must speak millimetres, or Target means different things."""
    assert loop.body.speed == WALK_MM_PER_S
    assert loop.body.turn_rate == TURN_RAD_PER_S


def test_training_and_behaviour_share_one_timeline(loop):
    loop.train(4, odour_name="trained")
    loop.run(3, odour_name="trained")
    log = loop.log()
    assert list(log["phase"].unique()) == ["train", "behave"]
    assert log["t"].is_monotonic_increasing
    assert [p.name for p in loop.phases] == ["train", "behave"]


def test_an_unknown_odour_is_refused(loop):
    with pytest.raises(KeyError):
        loop.train(1, odour_name="lavender")


def test_reset_clears_the_log_but_not_the_learning(loop):
    loop.train(4, odour_name="trained")
    depressed = loop.plastic.depression()
    loop.reset()
    assert loop.log().empty
    assert loop.plastic.depression() == pytest.approx(depressed)


def test_a_saturated_mushroom_body_cannot_report_learning():
    """Why the fixture is wired weakly, kept as an executable warning.

    Drive the network hard enough and every tanh pins at 1.0. The KC->MBON
    synapses still depress -- the plasticity is working, and reports 52% -- but
    the MBONs cannot move, so the learned component reads exactly zero. A suite
    built on a saturated fixture goes green while measuring nothing, which is
    what the first draft of this file did.
    """
    rng = np.random.default_rng(0)
    rows, nid = [], 0
    for side in ("L", "R"):
        for x in range(-2, 3):
            for y in range(-2, 3):
                rows.append(
                    {
                        "id": nid,
                        "type": "L1",
                        "nt": "acetylcholine",
                        "side": side,
                        "hex1": x,
                        "hex2": y,
                    }
                )
                nid += 1
    for i, kind in enumerate(OTHER_TYPES):
        nt = (
            "dopamine"
            if kind.startswith("PAM")
            else ("gaba" if kind.startswith("MBON") else "acetylcholine")
        )
        rows.append(
            {
                "id": nid,
                "type": kind,
                "nt": nt,
                "side": "L" if i % 2 else "R",
                "hex1": np.nan,
                "hex2": np.nan,
            }
        )
        nid += 1
    neurons = pd.DataFrame(rows)
    m = len(neurons)
    dense = rng.random((m, m)).astype(np.float32)  # dense and strong: saturating
    dense[dense < 0.75] = 0.0
    np.fill_diagonal(dense, 0.0)
    kinds = neurons["type"].astype(str).to_numpy()
    dense *= np.where(np.char.startswith(kinds.astype(str), "MBON"), -1.0, 1.0)[:, None]
    hot = Connectome(
        neurons, sp.csr_matrix(dense), name="saturated", meta={"matrix_kind": "inprop"}
    )

    loop = HybridLoop(hot, target=Target(x=25.0, y=0.0, radius=2.5), hops=3)
    loop.train(6, odour_name="trained")
    row = loop.step(smell=loop.odours["trained"], odour_name="trained")
    assert loop.plastic.depression() > 0.1, "the synapses really did change"
    assert row["mbon_readout"] == pytest.approx(1.0), "and the MBONs are pinned"
    assert row["learned_component"] == 0.0, "so the readout cannot see it"
