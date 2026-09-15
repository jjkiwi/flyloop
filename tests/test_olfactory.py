"""Odour conditioning: the sparse code it needs, and the controls it carries."""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from flyloop.connectome.schema import Connectome
from flyloop.experiments.olfactory import (
    RESPONSE_THRESHOLD,
    SPARSE_KC_SLOPE,
    ConditioningResult,
    kc_slopes,
    mushroom_body,
    odour,
    olfactory_receptors,
)

pytest.importorskip("torch", reason="optional 'data' extra")


def _mb_connectome() -> Connectome:
    """ORNs of two glomeruli onto separate Kenyon cells onto one MBON."""
    rows = []
    nid = 0
    for g in ("DM1", "DA1"):
        for _ in range(2):
            rows.append({"id": nid, "type": f"ORN_{g}", "nt": "acetylcholine", "side": "R"})
            nid += 1
    for k in range(2):
        rows.append({"id": nid, "type": f"KCg{k}", "nt": "acetylcholine", "side": "R"})
        nid += 1
    rows.append({"id": nid, "type": "MBON01", "nt": "glutamate", "side": "R"})
    rows.append({"id": nid + 1, "type": "PAM01", "nt": "dopamine", "side": "R"})
    neurons = pd.DataFrame(rows)
    n = len(neurons)
    dense = np.zeros((n, n), dtype=np.float32)
    dense[0, 4] = dense[1, 4] = 0.4  # DM1 ORNs -> first KC
    dense[2, 5] = dense[3, 5] = 0.4  # DA1 ORNs -> second KC
    dense[4, 6] = dense[5, 6] = 0.4  # both KCs -> MBON
    return Connectome(
        neurons, sp.csr_matrix(dense), name="mb", meta={"matrix_kind": "inprop"}
    )


def test_receptors_are_found():
    c = _mb_connectome()
    assert len(olfactory_receptors(c)) == 4


def test_missing_receptors_fail_loudly():
    c = Connectome(
        pd.DataFrame({"id": [0], "type": ["KCg"], "nt": ["acetylcholine"], "side": ["R"]}),
        sp.csr_matrix((1, 1)),
        name="nose-less",
        meta={"matrix_kind": "inprop"},
    )
    with pytest.raises(KeyError, match="ORN"):
        olfactory_receptors(c)


def test_mushroom_body_populations():
    mb = mushroom_body(_mb_connectome())
    assert len(mb["kc"]) == 2
    assert len(mb["mbon"]) == 1


def test_odour_activates_only_its_glomerulus():
    c = _mb_connectome()
    r = olfactory_receptors(c)
    v = odour(c, ("DM1",), r)
    assert v.sum() == 2
    types = c.neurons["type"].astype(str).to_numpy()[r]
    assert set(types[v > 0]) == {"ORN_DM1"}


def test_unknown_glomerulus_is_rejected():
    c = _mb_connectome()
    with pytest.raises(KeyError, match="ORN_ZZ9"):
        odour(c, ("ZZ9",), olfactory_receptors(c))


def test_kc_slopes_cover_every_kenyon_type():
    c = _mb_connectome()
    s = kc_slopes(c, 0.35)
    assert set(s) == {"KCg0", "KCg1"}
    assert all(v == 0.35 for v in s.values())


def test_sparse_slope_is_the_calibrated_one():
    """The default reproduces the ~5% sparse odour code, measured on MaleCNS.

    At the model's own default of 5.0 a single glomerulus activates 88% of the
    Kenyon cells and two odours overlap by 99.8%, which makes odour-specific
    learning impossible by construction.
    """
    assert SPARSE_KC_SLOPE == 0.35
    assert 0.0 < RESPONSE_THRESHOLD < 1.0


# --- the learning index and its controls ----------------------------------


def _result(paired_change: float, unpaired_change: float, **kw) -> ConditioningResult:
    before = pd.DataFrame({"MBON": [1.0, 1.0]}, index=["paired", "unpaired"])
    after = pd.DataFrame(
        {"MBON": [1.0 + paired_change, 1.0 + unpaired_change]},
        index=["paired", "unpaired"],
    )
    return ConditioningResult(
        before=before,
        after=after,
        sparseness=kw.get("sparseness", {"paired": 0.045, "unpaired": 0.015}),
        overlap=kw.get("overlap", 0.06),
        depression=kw.get("depression", 0.03),
        rewards=kw.get("rewards", [0.1, 0.5]),
        graph="test",
        kc_slope=SPARSE_KC_SLOPE,
    )


def test_learning_index_is_positive_when_the_rewarded_odour_falls_more():
    assert _result(-0.010, -0.002).learning_index() > 0


def test_learning_index_is_zero_without_odour_specificity():
    """If both odours change equally, nothing odour specific was learned."""
    assert _result(-0.005, -0.005).learning_index() == pytest.approx(0.0)


def test_learning_index_is_negative_if_the_wrong_odour_falls():
    assert _result(-0.002, -0.010).learning_index() < 0


def test_collapsed_sparse_code_is_flagged():
    r = _result(-0.01, -0.01, overlap=0.95)
    assert any("odour specific" in n for n in
               ConditioningResult(r.before, r.after, r.sparseness, 0.95, r.depression,
                                  r.rewards, "test", SPARSE_KC_SLOPE,
                                  notes=["the two odours share 95% of their Kenyon "
                                         "cells, so nothing odour specific can be "
                                         "learned; lower kc_slope"]).notes)


def test_report_mentions_both_odours():
    text = _result(-0.01, -0.002).report()
    assert "paired" in text and "unpaired" in text and "learning index" in text


def test_within_odour_effect_cancels_the_odour_identity_term():
    """The same odour, rewarded versus not -- the comparison that is valid.

    The learning index does not flip when the reward is swapped on real data,
    so it mixes a reward term with an odour term. Comparing one odour against
    itself removes the second.
    """
    from flyloop.experiments.olfactory import within_odour_effect

    rewarded = _result(-0.0076, -0.0042)  # odour A paired
    unrewarded = _result(-0.0042, -0.0058)  # odour A now the unpaired one
    effect = within_odour_effect(rewarded, unrewarded)
    assert effect < 0, "A must fall further when it is the rewarded odour"
    assert effect == pytest.approx(-0.0076 - -0.0058)


def test_within_odour_effect_is_zero_without_learning():
    from flyloop.experiments.olfactory import within_odour_effect

    flat = _result(-0.004, -0.004)
    assert within_odour_effect(flat, flat) == pytest.approx(0.0)


def test_mbon_targets_are_descending_neurons_only():
    """The readout has to find where MBON output goes, not where vision goes."""
    from flyloop.experiments.olfactory import mbon_targets

    rows = [
        {"id": 0, "type": "MBON01", "nt": "glutamate", "side": "R",
         "super_class": "central"},
        {"id": 1, "type": "DNp52", "nt": "acetylcholine", "side": "R",
         "super_class": "descending_neuron"},
        {"id": 2, "type": "DNa02", "nt": "acetylcholine", "side": "R",
         "super_class": "descending_neuron"},
        {"id": 3, "type": "SomeInterneuron", "nt": "gaba", "side": "R",
         "super_class": "cb_intrinsic"},
    ]
    dense = np.zeros((4, 4), dtype=np.float32)
    dense[0, 1] = 0.5   # MBON -> DNp52, strong
    dense[0, 2] = 0.01  # MBON -> DNa02, weak
    dense[0, 3] = 0.9   # MBON -> interneuron, strongest but not descending
    c = Connectome(
        pd.DataFrame(rows), sp.csr_matrix(dense), name="t",
        meta={"matrix_kind": "inprop"},
    )
    targets = mbon_targets(c, n=5)
    assert targets[0] == "DNp52", "strongest descending target must come first"
    assert "SomeInterneuron" not in targets
    assert "DNa02" in targets and targets.index("DNa02") > targets.index("DNp52")


def test_readout_accepts_arbitrary_cell_types():
    from flyloop.experiments.olfactory import _populations

    c = _mb_connectome()
    pops = _populations(c, ("KC", "MBON", "MBON01", "NoSuchType"))
    assert set(pops) == {"KC", "MBON", "MBON01"}
    assert len(pops["MBON01"]) == 1
    assert "NoSuchType" not in pops, "absent types are dropped, not faked"


def test_mbon_learning_and_output_are_reported_separately():
    """Plasticity acts where KC input lands; behaviour needs descending output.

    On MaleCNS those are different MBONs (r = -0.42), which is why olfactory
    conditioning stops at the mushroom body. The helper has to keep the two
    quantities apart rather than summarising them into one score.
    """
    from flyloop.experiments.olfactory import mbon_learning_vs_output

    rows = [
        {"id": 0, "type": "KCg", "nt": "acetylcholine", "side": "R",
         "super_class": "cb_intrinsic"},
        {"id": 1, "type": "MBON14", "nt": "glutamate", "side": "R",
         "super_class": "central"},
        {"id": 2, "type": "MBON33", "nt": "acetylcholine", "side": "R",
         "super_class": "central"},
        {"id": 3, "type": "DNp52", "nt": "acetylcholine", "side": "R",
         "super_class": "descending_neuron"},
    ]
    dense = np.zeros((4, 4), dtype=np.float32)
    dense[0, 1] = 0.9   # KC -> MBON14: learns a lot
    dense[0, 2] = 0.1   # KC -> MBON33: learns little
    dense[2, 3] = 0.5   # MBON33 -> DN: drives behaviour
    c = Connectome(
        pd.DataFrame(rows), sp.csr_matrix(dense), name="t",
        meta={"matrix_kind": "inprop"},
    )
    t = mbon_learning_vs_output(c)
    assert t.loc["MBON14", "kc_input"] > t.loc["MBON33", "kc_input"]
    assert t.loc["MBON14", "dn_output"] == 0.0
    assert t.loc["MBON33", "dn_output"] > 0.0


def test_mbon_analysis_is_empty_without_a_mushroom_body():
    from flyloop.experiments.olfactory import mbon_learning_vs_output

    c = Connectome(
        pd.DataFrame({"id": [0], "type": ["ORN_DM1"], "nt": ["acetylcholine"],
                      "side": ["R"]}),
        sp.csr_matrix((1, 1)), name="t", meta={"matrix_kind": "inprop"},
    )
    assert mbon_learning_vs_output(c).empty
