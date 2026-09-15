"""The MBON ensemble readout: a balance, not an average."""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from flyloop.connectome.schema import Connectome
from flyloop.experiments.ensemble import (
    APPROACH_OUTPUT,
    WITHDRAW_OUTPUT,
    effective_influence,
    mbon_ensemble,
)


def _valence_connectome() -> Connectome:
    """Two MBONs of opposite valence, plus the motor output they act on."""
    rows = [
        {"id": 0, "type": "MBON_app", "nt": "acetylcholine", "side": "R"},
        {"id": 1, "type": "MBON_wdr", "nt": "acetylcholine", "side": "R"},
        {"id": 2, "type": "DNa01", "nt": "acetylcholine", "side": "R"},
        {"id": 3, "type": "MDN", "nt": "acetylcholine", "side": "R"},
        {"id": 4, "type": "DNp09", "nt": "acetylcholine", "side": "R"},
    ]
    dense = np.zeros((5, 5), dtype=np.float32)
    dense[0, 2] = 1.0   # MBON_app -> DNa01 (forward): approach
    dense[1, 3] = 1.0   # MBON_wdr -> MDN (backward): withdrawal
    return Connectome(
        pd.DataFrame(rows), sp.csr_matrix(dense), name="valence",
        meta={"matrix_kind": "inprop"},
    )


def _prefix(c: Connectome) -> Connectome:
    """Same graph but with types the MBON prefix search will find."""
    n = c.neurons.copy()
    n["type"] = n["type"].str.replace("MBON_app", "MBON30").str.replace(
        "MBON_wdr", "MBON27"
    )
    return Connectome(n, c.W, name=c.name, meta=c.meta)


def test_influence_is_signed_so_inhibition_subtracts():
    rows = [
        {"id": 0, "type": "A", "nt": "acetylcholine", "side": "R"},
        {"id": 1, "type": "B", "nt": "gaba", "side": "R"},
        {"id": 2, "type": "T", "nt": "acetylcholine", "side": "R"},
    ]
    dense = np.zeros((3, 3), dtype=np.float32)
    dense[0, 2] = 1.0
    dense[1, 2] = -1.0
    c = Connectome(pd.DataFrame(rows), sp.csr_matrix(dense), name="t",
                   meta={"matrix_kind": "inprop"})
    inf = effective_influence(c, np.array([0, 1]), np.array([2]), hops=1, decay=1.0)
    assert inf[0] > 0 > inf[1]


def test_two_routes_of_opposite_sign_cancel():
    """A source reaching a target both ways nets out -- the point of a balance."""
    rows = [
        {"id": 0, "type": "S", "nt": "acetylcholine", "side": "R"},
        {"id": 1, "type": "I", "nt": "gaba", "side": "R"},
        {"id": 2, "type": "T", "nt": "acetylcholine", "side": "R"},
    ]
    dense = np.zeros((3, 3), dtype=np.float32)
    dense[0, 2] = 1.0   # direct excitation, 1 hop
    dense[0, 1] = 1.0   # via an inhibitory cell, 2 hops
    dense[1, 2] = -1.0
    c = Connectome(pd.DataFrame(rows), sp.csr_matrix(dense), name="t",
                   meta={"matrix_kind": "inprop"})
    direct = effective_influence(c, np.array([0]), np.array([2]), hops=1, decay=1.0)
    both = effective_influence(c, np.array([0]), np.array([2]), hops=2, decay=1.0)
    assert direct[0] > 0
    assert both[0] < direct[0], "the inhibitory route must subtract"


def test_decay_discounts_longer_paths():
    c = _valence_connectome()
    near = effective_influence(c, np.array([0]), np.array([2]), hops=4, decay=1.0)
    far = effective_influence(c, np.array([0]), np.array([2]), hops=4, decay=0.25)
    assert abs(far[0]) < abs(near[0])


def test_hops_must_be_positive():
    c = _valence_connectome()
    with pytest.raises(ValueError, match="hops"):
        effective_influence(c, np.array([0]), np.array([2]), hops=0)


def test_valence_is_approach_minus_withdrawal():
    ens = mbon_ensemble(_prefix(_valence_connectome()), hops=2, decay=1.0)
    t = ens.table()
    assert t["MBON30"] > 0, "driving forward walking is approach"
    assert t["MBON27"] < 0, "driving backward walking is withdrawal"


def test_missing_output_neurons_fail_loudly():
    c = _prefix(_valence_connectome())
    with pytest.raises(KeyError, match="DNxx"):
        mbon_ensemble(c, approach=("DNxx",))


def test_connectome_without_mbons_is_rejected():
    c = Connectome(
        pd.DataFrame({"id": [0], "type": ["DNa01"], "nt": ["acetylcholine"],
                      "side": ["R"]}),
        sp.csr_matrix((1, 1)), name="t", meta={"matrix_kind": "inprop"},
    )
    with pytest.raises(KeyError, match="MBONs"):
        mbon_ensemble(c)


def test_the_balance_sees_what_the_mean_cannot():
    """Two MBONs of opposite valence moving together cancel in a mean.

    This is why Run 8 saw nothing: averaging 97 cells of opposite sign hides a
    change that a signed, weighted sum reports.
    """
    ens = mbon_ensemble(_prefix(_valence_connectome()), hops=2, decay=1.0)
    n = 5
    before = np.zeros(n, dtype=np.float32)
    before[ens.rows] = [0.5, 0.5]
    after = np.zeros(n, dtype=np.float32)
    # the approach cell falls, the withdrawal cell rises by the same amount
    after[ens.rows] = [0.4, 0.6]

    mean_change = after[ens.rows].mean() - before[ens.rows].mean()
    balance_change = ens.readout(after) - ens.readout(before)
    assert mean_change == pytest.approx(0.0), "the mean is blind to this"
    assert abs(balance_change) > 1e-6, "the balance is not"


def test_readout_is_a_weighted_sum():
    ens = mbon_ensemble(_prefix(_valence_connectome()), hops=2, decay=1.0)
    acts = np.zeros(5, dtype=np.float32)
    acts[ens.rows] = [1.0, 0.0]
    assert ens.readout(acts) == pytest.approx(float(ens.valence[0]))


def test_default_outputs_are_the_ones_used_all_project():
    assert APPROACH_OUTPUT == ("DNa01",)
    assert WITHDRAW_OUTPUT == ("MDN", "DNp09")
