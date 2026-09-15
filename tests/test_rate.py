"""The rate model, and the API trap in its time axis."""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from flyloop.brain.rate import RateBrain, population_index, steady_state
from flyloop.connectome.schema import Connectome

pytest.importorskip("connectome_interpreter", reason="optional 'data' extra")
pytest.importorskip("torch", reason="optional 'data' extra")


def _chain(kind: str = "inprop") -> Connectome:
    """A -> B -> C, each carrying most of its target's input."""
    neurons = pd.DataFrame(
        {
            "id": [0, 1, 2],
            "type": ["A", "B", "C"],
            "nt": ["acetylcholine"] * 3,
            "side": ["R"] * 3,
        }
    )
    dense = np.zeros((3, 3), dtype=np.float32)
    dense[0, 1] = 0.9
    dense[1, 2] = 0.9
    return Connectome(
        neurons, sp.csr_matrix(dense), name="chain", meta={"matrix_kind": kind}
    )


def test_steady_state_repeats_the_pattern():
    x = steady_state(np.array([1.0, 0.0, 1.0]), 4)
    assert x.shape == (3, 4)
    assert (x[:, 0] == x[:, -1]).all()


def test_synapse_counts_are_rejected():
    with pytest.raises(ValueError, match="input-proportion"):
        RateBrain(_chain("syncount"), np.array([0]))


def test_signal_propagates_along_the_chain():
    c = _chain()
    b = RateBrain(c, np.array([0]), num_layers=4, default_bias=0.0)
    rec = population_index(c, ("A", "B", "C"), side="R")
    r = b.run(steady_state(np.array([1.0]), 4), record=rec)
    peak = r.peak()
    assert peak["A_R"] > 0
    assert peak["B_R"] > 0, "one hop must carry"
    assert peak["C_R"] > 0, "two hops must carry"


def test_orientation_is_not_transposed():
    """A drives C through B, not the other way round."""
    c = _chain()
    forward = RateBrain(c, np.array([0]), num_layers=4).run(
        steady_state(np.array([1.0]), 4), record=population_index(c, ("C",), side="R")
    )
    backward = RateBrain(c, np.array([2]), num_layers=4).run(
        steady_state(np.array([1.0]), 4), record=population_index(c, ("A",), side="R")
    )
    assert forward.peak()["C_R"] > 0
    assert backward.peak()["A_R"] == 0


def test_input_shape_is_checked():
    c = _chain()
    b = RateBrain(c, np.array([0]), num_layers=3)
    with pytest.raises(ValueError, match="n_sensory"):
        b.run(np.zeros((2, 3)))


def test_the_time_axis_is_the_hop_axis():
    """Frame t is injected at hop t, so early input gets more amplification.

    This is the trap that invalidated a looming-versus-receding comparison:
    the same pattern placed early and late gives different answers, measuring
    hop count rather than the stimulus.
    """
    c = _chain()
    b = RateBrain(c, np.array([0]), num_layers=4, default_bias=0.0)
    rec = population_index(c, ("C",), side="R")
    early = np.zeros((1, 4), dtype=np.float32)
    early[0, 0] = 1.0
    late = np.zeros((1, 4), dtype=np.float32)
    late[0, 3] = 1.0
    assert b.run(early, record=rec).peak()["C_R"] > b.run(late, record=rec).peak()["C_R"]


def test_bias_gives_neurons_baseline_activity():
    c = _chain()
    rec = population_index(c, ("B",), side="R")
    silent = RateBrain(c, np.array([0]), num_layers=3, default_bias=0.0)
    biased = RateBrain(c, np.array([0]), num_layers=3, default_bias=0.3)
    quiet = silent.run(steady_state(np.array([0.0]), 3), record=rec).peak()["B_R"]
    warm = biased.run(steady_state(np.array([0.0]), 3), record=rec).peak()["B_R"]
    assert quiet == 0.0
    assert warm > 0.0


def test_population_index_skips_absent_types():
    c = _chain()
    assert "Z_R" not in population_index(c, ("A", "Z"), side="R")
