"""Soma coordinates, and the map they let us draw."""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from flyloop.connectome.data_prep import parse_soma
from flyloop.connectome.schema import Connectome
from flyloop.viz.atlas import BRAIN_Z_MAX, Atlas

# -------------------------------------------------------------- parsing


def test_parses_bracketed_voxel_coordinates():
    out = parse_soma(pd.Series(["[10 20 30]", "[40 50 60]"]))
    assert out == pytest.approx(np.array([[10.0, 20, 30], [40, 50, 60]]))


def test_missing_somata_become_nan_and_keep_their_row():
    """Alignment with the connectivity matrix matters more than completeness."""
    out = parse_soma(pd.Series(["[1 2 3]", np.nan, "[7 8 9]"]))
    assert len(out) == 3
    assert np.isnan(out[1]).all()
    assert out[2] == pytest.approx([7.0, 8.0, 9.0])


def test_malformed_entries_do_not_raise():
    out = parse_soma(pd.Series(["[1 2]", "nonsense", "[1 2 3 4]", "[5 6 7]"]))
    assert np.isnan(out[:3]).all()
    assert out[3] == pytest.approx([5.0, 6.0, 7.0])


def test_all_missing_is_handled():
    out = parse_soma(pd.Series([np.nan, np.nan]))
    assert out.shape == (2, 3)
    assert np.isnan(out).all()


# ---------------------------------------------------------------- atlas


def _connectome(coords) -> Connectome:
    n = len(coords)
    neurons = pd.DataFrame(
        {
            "id": np.arange(n),
            "type": [f"T{i}" for i in range(n)],
            "nt": ["acetylcholine"] * n,
            "side": ["L"] * n,
            "super_class": ["cb_intrinsic"] * n,
        }
    )
    xyz = np.asarray(coords, dtype=float)
    for i, col in enumerate(("soma_x", "soma_y", "soma_z")):
        neurons[col] = xyz[:, i]
    return Connectome(neurons, sp.csr_matrix((n, n)), name="toy")


def test_atlas_drops_unlocated_cells_but_remembers_their_rows():
    c = _connectome([[1, 2, 3], [np.nan] * 3, [4, 5, 6]])
    a = Atlas.from_connectome(c)
    assert len(a) == 2
    assert a.rows.tolist() == [0, 2]
    assert a.type.tolist() == ["T0", "T2"]


def test_values_realigns_a_full_length_vector():
    """A per-neuron result is indexed by connectome row, not by atlas row."""
    c = _connectome([[1, 2, 3], [np.nan] * 3, [4, 5, 6]])
    a = Atlas.from_connectome(c)
    assert a.values(np.array([10.0, 99.0, 30.0])).tolist() == [10.0, 30.0]


def test_values_rejects_a_vector_that_is_too_short():
    c = _connectome([[1, 2, 3], [4, 5, 6]])
    a = Atlas.from_connectome(c)
    with pytest.raises(ValueError, match="too short"):
        a.values(np.array([1.0]))


def test_brain_mask_splits_at_the_neck():
    c = _connectome([[0, 0, BRAIN_Z_MAX - 1], [0, 0, BRAIN_Z_MAX + 1]])
    assert Atlas.from_connectome(c).brain.tolist() == [True, False]


def test_a_connectome_without_coordinates_says_so():
    from flyloop.connectome.synthetic import synthetic_connectome

    with pytest.raises(KeyError, match="soma coordinates"):
        Atlas.from_connectome(synthetic_connectome())
