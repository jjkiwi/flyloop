"""Control graphs are what turn a demo into a measurement."""

import numpy as np
import pytest

from flyloop.connectome.controls import (
    CONTROLS,
    control_graphs,
    relabel_neurons,
    rewire_degree_preserving,
    scramble_signs,
)
from flyloop.connectome.signs import sign_vector
from flyloop.connectome.synthetic import synthetic_connectome


@pytest.fixture(scope="module")
def cx():
    return synthetic_connectome()


def test_every_control_keeps_the_neuron_count(cx):
    for name, g in control_graphs(cx, seed=1).items():
        assert g.n == cx.n, name


def test_rewiring_preserves_out_degree_up_to_collapsed_self_loops(cx):
    g = rewire_degree_preserving(cx, seed=1)
    # The shuffle preserves degrees exactly; dropping self-loops and merging
    # duplicate pairs then costs a small fraction of the edges.
    assert g.n_connections <= cx.n_connections
    assert g.n_connections > 0.98 * cx.n_connections


def test_rewiring_actually_changes_the_wiring(cx):
    g = rewire_degree_preserving(cx, seed=1)
    a = set(zip(*cx.W.tocoo().nonzero(), strict=True))
    b = set(zip(*g.W.tocoo().nonzero(), strict=True))
    assert len(a & b) < 0.2 * len(a), "the shuffle barely moved anything"


def test_relabelling_preserves_topology_exactly(cx):
    g = relabel_neurons(cx, seed=1)
    assert g.n_connections == cx.n_connections
    a = np.sort(np.diff(cx.W.tocsr().indptr))
    b = np.sort(np.diff(g.W.tocsr().indptr))
    assert np.array_equal(a, b)


def test_relabelling_moves_the_cell_types(cx):
    g = relabel_neurons(cx, seed=1)
    assert not np.array_equal(
        cx.neurons["type"].to_numpy(), g.neurons["type"].to_numpy()
    )
    # The same cells still exist, just somewhere else.
    assert sorted(cx.neurons["type"]) == sorted(g.neurons["type"])


def test_sign_scrambling_preserves_the_transmitter_counts(cx):
    g = scramble_signs(cx, seed=1)
    assert sorted(cx.neurons["nt"]) == sorted(g.neurons["nt"])
    assert g.n_connections == cx.n_connections


def test_sign_scrambling_changes_which_neurons_are_inhibitory(cx):
    g = scramble_signs(cx, seed=1)
    assert not np.array_equal(
        sign_vector(cx.neurons["nt"]).to_numpy(), sign_vector(g.neurons["nt"]).to_numpy()
    )


def test_signs_stay_consistent_with_the_presynaptic_neuron(cx):
    """Every outgoing synapse of a neuron must carry that neuron's sign."""
    for g in (rewire_degree_preserving(cx, seed=2), scramble_signs(cx, seed=2)):
        coo = g.W.tocoo()
        expected = sign_vector(g.neurons["nt"]).to_numpy()[coo.row]
        assert np.all(np.sign(coo.data) == expected[: len(coo.data)])


def test_controls_are_reproducible(cx):
    for fn in CONTROLS.values():
        a, b = fn(cx, seed=7), fn(cx, seed=7)
        assert (a.W != b.W).nnz == 0


def test_different_seeds_give_different_graphs(cx):
    a = rewire_degree_preserving(cx, seed=1)
    b = rewire_degree_preserving(cx, seed=2)
    assert (a.W != b.W).nnz > 0


def test_control_metadata_records_what_was_done(cx):
    g = rewire_degree_preserving(cx, seed=3)
    assert g.meta["control"] == "rewire_degree_preserving"
    assert g.meta["control_seed"] == 3
    assert "rewired" in g.name
