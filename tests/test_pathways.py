"""Asking the graph before asking the simulation."""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from flyloop.connectome.pathways import direct_drive, drive_balance, input_balance
from flyloop.connectome.schema import Connectome
from flyloop.connectome.synthetic import synthetic_connectome


@pytest.fixture(scope="module")
def cx():
    return synthetic_connectome()


def _toy():
    neurons = pd.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "type": ["LC4", "DNp04", "DNp09", "INi"],
            "nt": ["acetylcholine", "acetylcholine", "acetylcholine", "gaba"],
            "side": ["L", "L", "L", "L"],
        }
    )
    dense = np.zeros((4, 4), dtype=np.float32)
    dense[0, 1] = 100.0  # LC4 -> DNp04
    dense[3, 1] = -40.0  # inhibition onto DNp04
    return Connectome(neurons, sp.csr_matrix(dense), name="toy")


def test_absent_pathway_reads_as_exactly_zero():
    d = direct_drive(_toy(), "LC4", ("DNp04", "DNp09"), sides=("L",))
    assert d.loc["DNp04", "L"] == pytest.approx(100.0)
    assert d.loc["DNp09", "L"] == 0.0


def test_missing_source_is_rejected():
    with pytest.raises(KeyError, match="NoSuchCell"):
        direct_drive(_toy(), "NoSuchCell", ("DNp04",))


def test_missing_target_side_reads_as_nan():
    d = direct_drive(_toy(), "LC4", ("DNp04",), sides=("R",))
    assert np.isnan(d.loc["DNp04", "R"])


def test_input_balance_separates_excitation_from_inhibition():
    b = input_balance(_toy(), "DNp04", side="L")
    assert b["excitation"] == pytest.approx(100.0)
    assert b["inhibition"] == pytest.approx(40.0)
    assert b["ratio"] == pytest.approx(2.5)


def test_input_balance_lists_the_strongest_inputs():
    b = input_balance(_toy(), "DNp04", side="L", top=4)
    assert set(b["top_inputs"]["type"]) == {"LC4", "INi"}


def test_drive_balance_reports_the_share_of_excitation():
    t = drive_balance(_toy(), "LC4", ("DNp04", "DNp09"), side="L")
    assert t.loc["DNp04", "direct_share_of_E"] == pytest.approx(1.0)


def test_share_is_nan_when_nothing_excites_the_target_at_all():
    """Undefined is not the same as zero, and must not read as zero.

    DNp09 in the toy receives nothing from anyone. On the real connectome it
    receives 4,393 excitatory synapses but none from LC4, which is a share of
    0.0 -- a different and much more interesting fact.
    """
    t = drive_balance(_toy(), "LC4", ("DNp09",), side="L")
    assert np.isnan(t.loc["DNp09", "direct_share_of_E"])


def test_share_is_zero_when_the_source_specifically_does_not_connect():
    neurons = pd.DataFrame(
        {
            "id": [0, 1, 2],
            "type": ["LC4", "DNp09", "Other"],
            "nt": ["acetylcholine"] * 3,
            "side": ["L", "L", "L"],
        }
    )
    dense = np.zeros((3, 3), dtype=np.float32)
    dense[2, 1] = 50.0  # something else drives DNp09; LC4 does not
    c = Connectome(neurons, sp.csr_matrix(dense), name="toy2")
    t = drive_balance(c, "LC4", ("DNp09",), side="L")
    assert t.loc["DNp09", "direct_share_of_E"] == 0.0


def test_works_on_the_synthetic_fixture(cx):
    d = direct_drive(cx, "LC4", ("DNp04", "DNp02", "DNp01"))
    assert (d["L"] > 0).all(), "the fixture wires LC4 onto all three in parallel"
    t = drive_balance(cx, "LC4", ("DNp04",), side="L")
    assert 0.0 <= t.loc["DNp04", "direct_share_of_E"] <= 1.0
