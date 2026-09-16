"""Loading prepared connectome matrices.

The real datasets are gigabytes and live outside the repository, so these tests
build a miniature one on disk with the same layout and column names.
"""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from flyloop.connectome.data_prep import (
    find_files,
    load_dataset,
    load_prepared,
    sign_report,
)


def _write_dataset(folder, *, prefix="mini", ad_only=False, with_sign=True):
    folder.mkdir(parents=True, exist_ok=True)
    meta = pd.DataFrame(
        {
            "bodyid": [10, 20, 30, 40],
            "type": ["LC4", "DNp04", "INi", "PR"],
            "top_nt": ["acetylcholine", "acetylcholine", "gaba", "histamine"],
            "somaSide": ["L", "L", "R", "R"],
            "superclass": ["visual_projection", "descending_neuron", "cb", "ol_sensory"],
            "assignedOlHex1": [np.nan, np.nan, np.nan, 3.0],
            "assignedOlHex2": [np.nan, np.nan, np.nan, 4.0],
            # Bracketed voxel coordinates, exactly as the real files store them.
            # The third is deliberately absent: 14% of MaleCNS has no located
            # soma and those rows must survive as NaN, not vanish.
            "somaLocation": [
                "[10 20 30]",
                "[40 50 60]",
                np.nan,
                "[70 80 90]",
            ],
            # The datasets' own rule: glutamate and GABA negative, rest positive.
            "sign": [1, 1, -1, 1],
            "idx": [0, 1, 2, 3],
        }
    )
    if not with_sign:
        meta = meta.drop(columns=["sign"])
    meta.to_csv(folder / f"{prefix}_all_neuron_meta.csv", index=False)
    # (pre, post) synapse counts, deliberately including a sub-threshold edge.
    dense = np.array(
        [
            [0, 40, 0, 0],
            [0, 0, 7, 0],
            [0, 3, 0, 0],  # 3 synapses: below a threshold of 5
            [20, 0, 0, 0],
        ],
        dtype=np.float32,
    )
    kind = "_ad_syncount" if ad_only else "_syncount"
    name = f"{prefix}{kind}_all_neuron.npz"
    sp.save_npz(folder / name, sp.csr_matrix(dense))
    return folder


@pytest.fixture
def mini(tmp_path):
    return _write_dataset(tmp_path / "mini")


def test_loads_neurons_and_signed_matrix(mini):
    c = load_prepared(mini, min_synapses=5)
    assert c.n == 4
    assert list(c.neurons["type"]) == ["LC4", "DNp04", "INi", "PR"]
    # LC4 (acetylcholine) -> DNp04 stays positive.
    assert c.W[0, 1] == pytest.approx(+40)
    # The GABAergic interneuron's output is negative.
    assert c.W[2, 1] == 0.0, "3 synapses is below the threshold and must be dropped"


def test_threshold_keeps_weak_edges_when_asked(mini):
    c = load_prepared(mini, min_synapses=1)
    assert c.W[2, 1] == pytest.approx(-3)


def test_orientation_is_pre_by_post(mini):
    """W[pre, post]: the LC4 row carries its output onto DNp04."""
    c = load_prepared(mini, min_synapses=5)
    assert c.W[0, 1] != 0
    assert c.W[1, 0] == 0


def test_histamine_sign_differs_from_the_dataset(mini):
    ours = load_prepared(mini, min_synapses=5, sign_source="flyloop")
    theirs = load_prepared(mini, min_synapses=5, sign_source="dataset")
    # The photoreceptor drives neuron 0; we call histamine inhibitory, they do not.
    assert ours.W[3, 0] < 0
    assert theirs.W[3, 0] > 0


def test_sign_report_names_the_disagreement(mini):
    c = load_prepared(mini, min_synapses=5)
    rep = sign_report(c.neurons)
    assert "histamine" in set(rep["nt"])
    row = rep[rep["nt"] == "histamine"].iloc[0]
    assert row["ours"] == -1 and row["dataset"] == 1


def test_sign_report_is_empty_without_a_dataset_column(tmp_path):
    folder = _write_dataset(tmp_path / "nosign", prefix="ns", with_sign=False)
    c = load_prepared(folder, min_synapses=5)
    assert sign_report(c.neurons).empty


def test_metadata_carries_hex_columns(mini):
    c = load_prepared(mini, min_synapses=5)
    assert {"hex1", "hex2"}.issubset(c.neurons.columns)
    assert c.neurons["hex1"].notna().sum() == 1


def test_axon_dendrite_variant_is_recorded(tmp_path):
    folder = _write_dataset(tmp_path / "adonly", prefix="ad", ad_only=True)
    files = find_files(folder)
    assert files.axon_dendrite
    c = load_prepared(folder, min_synapses=5)
    assert c.meta["axon_dendrite_only"] is True


def test_plain_matrix_wins_over_the_split_one(tmp_path):
    folder = _write_dataset(tmp_path / "both", prefix="b")
    sp.save_npz(folder / "b_ad_syncount_all_neuron.npz", sp.csr_matrix(np.zeros((4, 4))))
    assert not find_files(folder).axon_dendrite
    assert find_files(folder, prefer_axon_dendrite=True).axon_dendrite


def test_provenance_is_recorded(mini):
    c = load_prepared(mini, min_synapses=5)
    assert c.meta["provenance"] == "YijieYin/connectome_data_prep"
    assert c.meta["sign_source"] == "flyloop"
    assert c.meta["min_synapses"] == 5


def test_bad_sign_source_is_rejected(mini):
    with pytest.raises(ValueError, match="sign_source"):
        load_prepared(mini, sign_source="vibes")


def test_shape_mismatch_is_caught(tmp_path):
    folder = _write_dataset(tmp_path / "bad", prefix="x")
    sp.save_npz(folder / "x_syncount_all_neuron.npz", sp.csr_matrix(np.zeros((3, 3))))
    with pytest.raises(ValueError, match="metadata must be in matrix order"):
        load_prepared(folder)


def test_missing_matrix_lists_what_was_found(tmp_path):
    folder = tmp_path / "empty"
    folder.mkdir()
    pd.DataFrame({"bodyid": [1], "type": ["a"], "top_nt": ["gaba"]}).to_csv(
        folder / "e_meta.csv", index=False
    )
    with pytest.raises(FileNotFoundError, match="no 'syncount' matrix"):
        load_prepared(folder)


def test_unknown_dataset_name_is_rejected(tmp_path):
    with pytest.raises(KeyError, match="unknown dataset"):
        load_dataset(tmp_path, "not_a_connectome")


def test_unknown_matrix_kind_is_rejected(mini):
    with pytest.raises(ValueError, match="matrix must be one of"):
        load_prepared(mini, matrix="vibes")


def test_inprop_matrix_is_selected_when_asked(tmp_path):
    """The rate model needs input proportions, not counts."""
    import numpy as np
    import scipy.sparse as sp

    folder = _write_dataset(tmp_path / "both_kinds", prefix="k")
    dense = np.zeros((4, 4), dtype=np.float32)
    dense[0, 1] = 0.75  # a proportion, not a count
    sp.save_npz(folder / "k_inprop_all_neuron.npz", sp.csr_matrix(dense))

    counts = load_prepared(folder, matrix="syncount", min_synapses=5)
    props = load_prepared(folder, matrix="inprop")
    assert counts.meta["matrix_kind"] == "syncount"
    assert props.meta["matrix_kind"] == "inprop"
    assert props.W[0, 1] == pytest.approx(0.75)
    # min_synapses must not silently delete proportional weights below 5
    assert props.W.nnz == 1
    assert props.meta["min_synapses"] is None


# ---------------------------------------------- metadata in matrix order


def _shuffled_dataset(folder, *, prefix="mini"):
    """The same miniature dataset with its CSV rows permuted.

    FlyWire ships like this: the first row of its metadata is matrix row 90,908.
    MaleCNS happens to ship sorted, which is why a loader that trusts CSV order
    reads one dataset correctly and silently mislabels every neuron in the other.
    """
    _write_dataset(folder, prefix=prefix)
    meta = pd.read_csv(folder / f"{prefix}_all_neuron_meta.csv")
    meta = meta.iloc[[2, 0, 3, 1]].reset_index(drop=True)
    meta.to_csv(folder / f"{prefix}_all_neuron_meta.csv", index=False)
    return folder


def test_rows_are_reordered_by_idx_not_trusted_from_the_csv(tmp_path):
    folder = _shuffled_dataset(tmp_path / "shuffled")
    c = load_prepared(folder, min_synapses=5)
    # _write_dataset lays the types out in idx order; the shuffle must be undone.
    assert c.neurons["type"].tolist() == ["LC4", "DNp04", "INi", "PR"]
    assert c.neurons["id"].tolist() == [10, 20, 30, 40]


def test_a_shuffled_load_matches_an_unshuffled_one(tmp_path):
    """The whole point: the same data in either row order must give one graph."""
    straight = load_prepared(_write_dataset(tmp_path / "a"), min_synapses=5)
    shuffled = load_prepared(_shuffled_dataset(tmp_path / "b"), min_synapses=5)
    assert shuffled.neurons["type"].tolist() == straight.neurons["type"].tolist()
    assert (shuffled.W != straight.W).nnz == 0


def test_a_duplicated_idx_is_refused(tmp_path):
    folder = _write_dataset(tmp_path / "dup")
    meta = pd.read_csv(folder / "mini_all_neuron_meta.csv")
    meta.loc[1, "idx"] = 0
    meta.to_csv(folder / "mini_all_neuron_meta.csv", index=False)
    with pytest.raises(ValueError, match="not unique"):
        load_prepared(folder, min_synapses=5)


def test_an_idx_that_is_not_a_permutation_is_refused(tmp_path):
    """Gaps mean the CSV is a subset of the matrix, and nothing lines up."""
    folder = _write_dataset(tmp_path / "gappy")
    meta = pd.read_csv(folder / "mini_all_neuron_meta.csv")
    meta["idx"] = [0, 1, 2, 99]
    meta.to_csv(folder / "mini_all_neuron_meta.csv", index=False)
    with pytest.raises(ValueError, match="permutation"):
        load_prepared(folder, min_synapses=5)


def test_metadata_without_an_idx_column_is_left_alone(tmp_path):
    folder = _write_dataset(tmp_path / "noidx")
    meta = pd.read_csv(folder / "mini_all_neuron_meta.csv").drop(columns=["idx"])
    meta.to_csv(folder / "mini_all_neuron_meta.csv", index=False)
    assert load_prepared(folder, min_synapses=5).neurons["type"].iloc[0] == "LC4"


def test_flywire_column_names_and_side_words_are_understood(tmp_path):
    """FlyWire writes root_id/cell_type and spells the sides out."""
    folder = _write_dataset(tmp_path / "fw")
    meta = pd.read_csv(folder / "mini_all_neuron_meta.csv").rename(
        columns={"bodyid": "root_id", "type": "cell_type", "somaSide": "side"}
    )
    meta["side"] = meta["side"].map({"L": "left", "R": "right"})
    meta.to_csv(folder / "mini_all_neuron_meta.csv", index=False)
    c = load_prepared(folder, min_synapses=5)
    assert c.neurons["type"].tolist() == ["LC4", "DNp04", "INi", "PR"]
    assert c.neurons["side"].tolist() == ["L", "L", "R", "R"]


def test_two_spellings_of_one_field_do_not_become_two_columns(tmp_path):
    """MaleCNS carries both `type` and `cell_type`; keeping both breaks lookups."""
    folder = _write_dataset(tmp_path / "both")
    meta = pd.read_csv(folder / "mini_all_neuron_meta.csv")
    meta["cell_type"] = "SOMETHING ELSE"
    meta.to_csv(folder / "mini_all_neuron_meta.csv", index=False)
    c = load_prepared(folder, min_synapses=5)
    assert isinstance(c.neurons["type"], pd.Series)
    assert c.neurons["type"].iloc[0] == "LC4"
