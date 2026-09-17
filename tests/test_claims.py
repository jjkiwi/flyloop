"""The numbers this project states in prose, checked against the live data.

Run 16 found a claim that had survived fifteen runs because it was never a
number: "reward reaches the brain through the fly's own PAM cluster". Every
quantitative check here compares a value some function returns, and that
sentence returns nothing, so nothing compared it against anything. It was
plausible, correct about the biology, and sat next to code that worked.

This file is the answer to that. Each test is a claim made in `README.md` or
`docs/RESULTS.md`, asserted against the real connectome. They are not unit
tests -- the code under test is the *documentation* -- and they fail when a
finding drifts, which is the point.

They need the prepared datasets, so they skip without them, and they are marked
slow because loading MaleCNS costs about ten seconds.
"""

import os
from pathlib import Path

import numpy as np
import pytest

DATA_ROOT = Path(
    os.environ.get("FLYLOOP_DATA_ROOT", "/home/user/yijieyin/connectome_data_prep")
)

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not (DATA_ROOT / "data").is_dir(),
        reason=f"needs a connectome_data_prep clone at {DATA_ROOT}",
    ),
]


@pytest.fixture(scope="module")
def malecns():
    from flyloop.connectome.data_prep import load_dataset

    return load_dataset(DATA_ROOT, "malecns", matrix="inprop")


@pytest.fixture(scope="module")
def flywire():
    from flyloop.connectome.data_prep import load_dataset

    return load_dataset(DATA_ROOT, "fafb", matrix="inprop")


def _share(c, source_prefix: str, target: str) -> float:
    types = c.neurons["type"].astype(str)
    src = np.flatnonzero(types.str.startswith(source_prefix).to_numpy())
    tgt = np.flatnonzero((types == target).to_numpy())
    A = abs(c.W.tocsr())
    total = np.asarray(A.sum(axis=0)).ravel()[tgt]
    frm = np.asarray(A[src][:, tgt].sum(axis=0)).ravel()
    return float(np.divide(frm, total, out=np.zeros_like(frm), where=total > 0).mean())


# ------------------------------------------------- Run 12: the junction


def test_mbon_supplies_about_half_a_percent_of_dna02(malecns):
    """README and Run 12: the one door a learned valence has into steering."""
    assert _share(malecns, "MBON", "DNa02") == pytest.approx(0.00526, abs=2e-4)


def test_mbon_reaches_neither_dna01_nor_dnp09(malecns):
    assert _share(malecns, "MBON", "DNa01") == pytest.approx(0.0, abs=1e-9)
    assert _share(malecns, "MBON", "DNp09") == pytest.approx(0.0, abs=1e-9)


def test_vision_cannot_be_learned_through_the_mushroom_body(malecns):
    """Run 12's load-bearing zero: LC4 and LPLC2 supply nothing to Kenyon cells.

    This is why the learned modality has to be an odour, and the whole hybrid
    architecture follows from it.
    """
    types = malecns.neurons["type"].astype(str).to_numpy()
    A = abs(malecns.W.tocsr())
    lc = np.flatnonzero(np.isin(types, ["LC4", "LPLC2"]))
    kc = np.flatnonzero(np.char.startswith(types.astype(str), "KC"))
    assert float(A[lc][:, kc].sum() / A[:, kc].sum()) == pytest.approx(0.0, abs=1e-6)


def test_the_indirect_routes_are_about_four_times_the_direct_one(malecns):
    from flyloop.brain.coupling import measured_share

    direct = measured_share(malecns, hops=1)
    within4 = measured_share(malecns, hops=4)
    assert direct == pytest.approx(0.00521, abs=2e-4)
    assert within4 == pytest.approx(0.02142, abs=5e-4)
    assert within4 / direct == pytest.approx(4.11, abs=0.15)


# --------------------------------------- Run 14: it replicates on FlyWire


def test_the_junction_replicates_in_a_second_animal(flywire):
    assert _share(flywire, "MBON", "DNa02") == pytest.approx(0.00515, abs=3e-4)


def test_vision_into_the_mushroom_body_is_zero_in_both(flywire):
    types = flywire.neurons["type"].astype(str).to_numpy()
    A = abs(flywire.W.tocsr())
    lc = np.flatnonzero(np.isin(types, ["LC4", "LPLC2"]))
    kc = np.flatnonzero(np.char.startswith(types.astype(str), "KC"))
    assert float(A[lc][:, kc].sum() / A[:, kc].sum()) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("dataset", ["malecns", "flywire"])
def test_mbon31_and_32_carry_nearly_all_of_it(dataset, request):
    """Run 14, and Li et al. 2020 from the hemibrain: the same two cells twice."""
    c = request.getfixturevalue(dataset)
    total = _share(c, "MBON", "DNa02")
    pair = _share(c, "MBON31", "DNa02") + _share(c, "MBON32", "DNa02")
    assert pair / total > 0.9


# ------------------------------------------- Run 15: the sign table


@pytest.mark.parametrize("dataset", ["malecns", "flywire"])
def test_every_glutamatergic_neuron_is_signed_negative(dataset, request):
    """53,975 neurons across two reconstructions, no exceptions."""
    c = request.getfixturevalue(dataset)
    glut = c.neurons[c.neurons["nt"].astype(str) == "glutamate"]
    assert len(glut) > 20_000
    assert set(glut["dataset_sign"].unique()) == {-1}


def test_flywire_photoreceptors_are_mislabelled_and_we_override_them(flywire):
    """FlyWire's classifier has no histamine class; identity wins for R1-8."""
    from flyloop.connectome.signs import photoreceptor_check

    check = photoreceptor_check(flywire.neurons)
    assert not check["ok"], "if FlyWire ever ships histamine, revisit the override"
    types = flywire.neurons["type"].astype(str)
    pr = np.flatnonzero(types.str.match(r"^R[1-8]").to_numpy())
    l1 = np.flatnonzero((types == "L1").to_numpy())
    edges = flywire.W.tocsr()[pr][:, l1]
    assert edges.nnz > 1000
    assert (edges.data > 0).sum() == 0, "photoreceptor output must be inhibitory"


def test_malecns_photoreceptors_need_no_override(malecns):
    from flyloop.connectome.signs import photoreceptor_check

    assert photoreceptor_check(malecns.neurons)["ok"]


# ------------------------------- Run 16: what the model deliberately deletes


def test_the_pam_cluster_carries_no_signal(malecns):
    """Run 16. Dopamine is signed 0, so reward is addressed, not propagated.

    If this ever fails, the prose in dopamine.py and loop_rate.py is wrong
    again -- in the other direction.
    """
    from flyloop.brain.dopamine import dopaminergic

    pam = dopaminergic(malecns, cluster="PAM")
    assert len(pam) == 316
    assert malecns.W.tocsr()[pam].nnz == 0


def test_the_readout_barely_notices_the_silenced_neurons(malecns):
    """2.3% of all weight is deleted, but under 1% of what DNa02 reads."""
    from flyloop.connectome.signs import sign_vector

    signs = sign_vector(malecns.neurons["nt"], malecns.neurons.get("type")).to_numpy()
    mute = np.flatnonzero(signs == 0)
    assert 2_500 < len(mute) < 3_500
    types = malecns.neurons["type"].astype(str).to_numpy()
    # Measured on magnitudes before signing; the signed matrix has already
    # removed these rows and would answer 0.00% to any question about them.
    raw = abs(malecns.W.tocsr())
    tgt = np.flatnonzero(types == "DNa02")
    assert float(raw[mute][:, tgt].sum()) == pytest.approx(0.0, abs=1e-9)


def test_the_eye_drives_lamina_cells_not_photoreceptors(malecns):
    """Run 16: L1 is cholinergic and L2 glutamatergic, so the front end is ON/OFF."""
    from flyloop.vision.hexproject import HexWorldView

    view = HexWorldView(malecns)
    driven = set(malecns.neurons["type"].astype(str).to_numpy()[view.sensory])
    assert driven == {"L1", "L2"}


# ----------------------------------------- Run 13: the open-loop controller


def test_the_turn_command_is_roughly_linear_in_bearing(malecns):
    """And the control says that linearity belongs to the wiring, not the shape."""
    from flyloop.experiments.bias import lateral_bias, tuning_curve

    out = lateral_bias(tuning_curve(malecns))
    assert out["linear_r2"] > 0.75
    assert out["peak_turn"] == pytest.approx(0.0824, abs=0.005)


def test_a_centred_object_still_leans_right(malecns):
    """Run 13's lean. Mirror pairs cancel it; a single-sided measurement does not."""
    from flyloop.experiments.bias import lateral_bias, tuning_curve

    out = lateral_bias(tuning_curve(malecns))
    assert out["centre_bias"] > 0
    assert out["centre_bias_fraction"] == pytest.approx(0.14, abs=0.04)
