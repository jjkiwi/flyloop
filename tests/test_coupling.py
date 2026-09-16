"""The junction between the learned system and the reflex.

The whole hybrid rests on one measured connection -- MBONs supply about half a
percent of DNa02's input -- so these tests pin what that coupling does, and
especially what it is not allowed to do.
"""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from flyloop.brain.coupling import LearnedBias, learned_bias, measured_share
from flyloop.connectome.schema import Connectome

TYPES = (
    ["MBON01"] * 4
    + ["KCg-m"] * 4
    + ["DNa02"] * 2
    + ["DNa01"] * 2
    + ["MDN"] * 2
    + ["DNp09"] * 2
    + ["LC4"] * 2
)


def _connectome(mbon_to_dna02: float = 0.5, other_to_dna02: float = 0.5) -> Connectome:
    n = len(TYPES)
    neurons = pd.DataFrame(
        {
            "id": np.arange(n),
            "type": TYPES,
            "nt": ["acetylcholine"] * n,
            "side": ["L", "R"] * (n // 2),
        }
    )
    t = np.asarray(TYPES)
    mbon = np.flatnonzero(t == "MBON01")
    dna02 = np.flatnonzero(t == "DNa02")
    lc4 = np.flatnonzero(t == "LC4")
    dense = np.zeros((n, n), dtype=np.float32)
    for m in mbon:
        dense[m, dna02] = mbon_to_dna02 / len(mbon)
    for cell in lc4:
        dense[cell, dna02] = other_to_dna02 / len(lc4)
    # MBONs also reach the approach and withdraw outputs, so a valence exists.
    dense[mbon[:2][:, None], np.flatnonzero(t == "DNa01")] = 0.4
    dense[mbon[2:][:, None], np.flatnonzero(t == "MDN")] = 0.4
    dense[mbon[2:][:, None], np.flatnonzero(t == "DNp09")] = 0.2
    return Connectome(neurons, sp.csr_matrix(dense), name="toy", meta={"matrix_kind": "inprop"})


# ------------------------------------------------------------ the share


def test_share_is_measured_from_the_wiring():
    c = _connectome(mbon_to_dna02=0.3, other_to_dna02=0.7)
    assert measured_share(c) == pytest.approx(0.3, rel=1e-5)


def test_share_is_zero_when_there_is_no_junction():
    c = _connectome(mbon_to_dna02=0.0, other_to_dna02=1.0)
    assert measured_share(c) == pytest.approx(0.0)


def test_share_is_zero_for_a_target_that_is_absent():
    assert measured_share(_connectome(), target="DNwhatever") == 0.0


# ----------------------------------------------------------- the readout


def test_gain_scales_the_strength_linearly():
    """gain=1 means the anatomy as measured; everything else is a choice."""
    c = _connectome(mbon_to_dna02=0.4)
    base = learned_bias(c, gain=1.0)
    assert base.strength == pytest.approx(base.share)
    assert learned_bias(c, gain=10.0).strength == pytest.approx(10 * base.share)


def test_readout_takes_the_peak_over_hops():
    lb = LearnedBias(rows=np.array([0, 1]), valence=np.array([1.0, 1.0]), share=0.5, _norm=2.0)
    flat = np.array([1.0, 0.0, 0.0])
    over_hops = np.array([[0.2, 1.0], [0.0, 0.0], [0.0, 0.0]])
    assert lb.readout(over_hops) == pytest.approx(lb.readout(flat))


def test_approach_weighted_cells_push_the_modulation_up():
    c = _connectome()
    lb = learned_bias(c, gain=20.0)
    approach = lb.rows[lb.valence > 0]
    withdraw = lb.rows[lb.valence < 0]
    assert len(approach) and len(withdraw), "fixture must split the ensemble"

    a = np.zeros(len(c.neurons))
    a[approach] = 1.0
    w = np.zeros(len(c.neurons))
    w[withdraw] = 1.0
    assert lb.modulation(a) > 1.0
    assert lb.modulation(w) < 1.0


def test_silence_leaves_the_reflex_untouched():
    """No odour, no learned signal: the fly must behave exactly as Run 10."""
    c = _connectome()
    lb = learned_bias(c, gain=500.0)
    assert lb.modulation(np.zeros(len(c.neurons))) == pytest.approx(1.0)


def test_modulation_never_goes_negative():
    """A learned aversion may cancel approach; it may not reverse the steering.

    Which way the fly turns is the visual pathway's answer. Letting the
    mushroom body flip its sign would make the learned signal command the
    reflex rather than bias it, which is not what 0.5% of DNa02's input does.
    """
    c = _connectome()
    lb = learned_bias(c, gain=100000.0)
    withdraw = lb.rows[lb.valence < 0]
    a = np.zeros(len(c.neurons))
    a[withdraw] = 1.0
    assert lb.modulation(a) == 0.0


def test_report_names_the_gain_and_the_share():
    """Anything that shows this number has to show what it was scaled by."""
    c = _connectome(mbon_to_dna02=0.25, other_to_dna02=0.75)  # a quarter of the input
    text = learned_bias(c, gain=8.0).report()
    assert "gain 8" in text
    assert "25.000%" in text or "25.0%" in text
