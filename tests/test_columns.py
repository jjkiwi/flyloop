"""Retinotopy from published columnar tables.

The logic tests run anywhere; the ones that touch the real tables skip when
connectome-interpreter is absent, since that is an optional extra.
"""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from flyloop.connectome.schema import Connectome
from flyloop.vision import ColumnMap, CompoundEye
from flyloop.vision.columns import column_order, columnar_indices, load_columnar_table

ci = pytest.importorskip("connectome_interpreter", reason="optional 'data' extra")


@pytest.fixture(scope="module")
def table():
    return load_columnar_table("mcns_right")


def _toy_table():
    return pd.DataFrame(
        {
            "hex1": [1, 1, 2, 2],
            "hex2": [7, 8, 7, 8],
            "L1": [100, 101, 102, 103],
            "x": [0, 1, 2, 3],
            "y": [0, 1, 2, 3],
        }
    )


def test_column_order_runs_outward_from_the_centre():
    t = _toy_table()
    o = column_order(t)
    d = np.hypot(t["x"] - t["x"].mean(), t["y"] - t["y"].mean()).to_numpy()[o]
    assert np.all(np.diff(d) >= -1e-9)


def test_missing_neurons_are_dropped_not_filled():
    t = _toy_table()
    idx = columnar_indices(t, pd.Series([103, 100, 999, 101]), cell_type="L1")
    assert len(idx) == 3  # 102 is not in the connectome
    assert 2 not in idx  # and row 2 (id 999) is never claimed


def test_unknown_cell_type_names_what_is_available():
    with pytest.raises(KeyError, match="Tm9"):
        columnar_indices(_toy_table(), pd.Series([100]), cell_type="Tm9")


def test_duplicate_ids_do_not_crash():
    idx = columnar_indices(_toy_table(), pd.Series([100, 100, 101]), cell_type="L1")
    assert len(idx) == 2


def test_the_published_table_has_892_columns(table):
    """920 rows in the file, 28 of them exact duplicates."""
    assert len(table) == 892
    assert table.duplicated(subset=["hex1", "hex2"]).sum() == 0


def test_fafb_table_pivots_to_one_row_per_column():
    t = load_columnar_table("fafb_right")
    assert {"hex1", "hex2", "x", "y"}.issubset(t.columns)
    assert "L1" in t.columns
    assert t.duplicated(subset=["hex1", "hex2"]).sum() == 0


def test_unknown_dataset_is_rejected():
    with pytest.raises(KeyError):
        load_columnar_table("no_such_eye")


def _l1_connectome(table, side="R"):
    ids = table["L1"].dropna().astype("int64").to_numpy()
    n = len(ids)
    neurons = pd.DataFrame(
        {"id": ids, "type": ["L1"] * n, "nt": ["histamine"] * n, "side": [side] * n}
    )
    return Connectome(neurons, sp.csr_matrix((n, n)), name="l1-only")


def test_column_map_from_the_real_table_is_monocular(table):
    m = ColumnMap.from_columnar_table(_l1_connectome(table), CompoundEye(256))
    assert len(m.indices["R"]) == 256
    assert m.monocular, "the published tables cover the right optic lobe only"
    assert m.assumptions == []


def test_mirroring_is_recorded_as_an_assumption(table):
    ids = table["L1"].dropna().astype("int64").to_numpy()
    n = len(ids)
    both = pd.DataFrame(
        {
            "id": np.concatenate([ids, ids + 10_000_000]),
            "type": ["L1"] * (2 * n),
            "nt": ["histamine"] * (2 * n),
            "side": ["R"] * n + ["L"] * n,
        }
    )
    c = Connectome(both, sp.csr_matrix((2 * n, 2 * n)), name="both-eyes")
    m = ColumnMap.from_columnar_table(c, CompoundEye(128), mirror=True)
    assert not m.monocular
    assert any("mirrored" in a for a in m.assumptions)


def test_asking_for_more_ommatidia_than_columns_fails_loudly(table):
    with pytest.raises(ValueError, match="matched only"):
        ColumnMap.from_columnar_table(_l1_connectome(table), CompoundEye(900))
