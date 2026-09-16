"""The scipy rate model against the PyTorch one it replaces.

A reimplementation is only as trustworthy as the test that pins it. These run
both models over the same graphs and stimuli and require agreement to float
tolerance; if one of them fails, ``flyloop.brain.fast`` is the one that is wrong.
"""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from flyloop.brain.fast import FastRateBrain
from flyloop.connectome.schema import Connectome

torch_available = pytest.importorskip  # re-exported for readability below


def _graph(kind: str = "inprop", *, seed: int = 0, n: int = 60) -> Connectome:
    """A small signed graph with a mix of transmitters and cell types."""
    rng = np.random.default_rng(seed)
    nt = rng.choice(
        ["acetylcholine", "gaba", "glutamate", "histamine"], size=n, p=[0.6, 0.2, 0.15, 0.05]
    )
    neurons = pd.DataFrame(
        {
            "id": np.arange(n),
            "type": [f"T{i % 11}" for i in range(n)],
            "nt": nt,
            "side": rng.choice(["L", "R"], size=n),
        }
    )
    dense = rng.random((n, n)).astype(np.float32)
    dense[dense < 0.85] = 0.0  # keep it sparse
    dense *= np.where(np.isin(nt, ["gaba", "glutamate", "histamine"]), -1.0, 1.0)[:, None]
    np.fill_diagonal(dense, 0.0)
    return Connectome(neurons, sp.csr_matrix(dense), name="toy", meta={"matrix_kind": kind})


def _reference(c, sensory, **kw):
    pytest.importorskip("connectome_interpreter", reason="optional 'data' extra")
    pytest.importorskip("torch", reason="optional 'data' extra")
    from flyloop.brain.rate import RateBrain

    return RateBrain(c, sensory, **kw)


# ------------------------------------------------------- equivalence


@pytest.mark.parametrize("hops", [1, 2, 5])
def test_matches_the_reference_model(hops):
    c = _graph()
    sensory = np.arange(6)
    stim = np.tile(np.linspace(0.2, 1.0, len(sensory), dtype=np.float32)[:, None], hops)

    slow = _reference(c, sensory, num_layers=hops).run(stim)
    fast = FastRateBrain(c, sensory, num_layers=hops).run(stim)
    assert fast.activations.shape == slow.activations.shape
    assert fast.activations == pytest.approx(slow.activations, abs=1e-5)


def test_matches_with_per_type_bias_and_slope():
    """Baseline activity and excitability are where a reimplementation drifts."""
    c = _graph(seed=3)
    sensory = np.arange(5)
    stim = np.full((len(sensory), 4), 0.7, dtype=np.float32)
    kw = dict(
        num_layers=4,
        default_bias=0.05,
        tanh_steepness=3.0,
        bias_by_type={"T1": 0.3, "T4": 0.9},
        slope_by_type={"T2": 8.0, "T7": 0.5},
    )
    slow = _reference(c, sensory, **kw).run(stim)
    fast = FastRateBrain(c, sensory, **kw).run(stim)
    assert fast.activations == pytest.approx(slow.activations, abs=1e-5)


def test_negative_bias_is_made_positive_by_both():
    """The reference stores raw biases and uses abs(); that is replicated."""
    c = _graph(seed=5)
    sensory = np.arange(4)
    stim = np.full((len(sensory), 3), 0.5, dtype=np.float32)
    kw = dict(num_layers=3, bias_by_type={"T0": -0.4})
    slow = _reference(c, sensory, **kw).run(stim)
    fast = FastRateBrain(c, sensory, **kw).run(stim)
    assert fast.activations == pytest.approx(slow.activations, abs=1e-5)
    # and it really is the absolute value, not a clamp to zero
    plain = FastRateBrain(c, sensory, num_layers=3, bias_by_type={"T0": 0.4})
    assert fast.activations == pytest.approx(plain.run(stim).activations, abs=1e-6)


def test_populations_are_recorded_the_same_way():
    c = _graph(seed=7)
    sensory = np.arange(5)
    stim = np.full((len(sensory), 3), 0.8, dtype=np.float32)
    rec = {"first": np.arange(10), "second": np.arange(10, 25)}
    slow = _reference(c, sensory, num_layers=3).run(stim, record=rec)
    fast = FastRateBrain(c, sensory, num_layers=3).run(stim, record=rec)
    assert list(fast.populations.columns) == list(slow.populations.columns)
    assert fast.populations.to_numpy() == pytest.approx(slow.populations.to_numpy(), abs=1e-5)


def test_plastic_entries_point_at_the_same_synapses():
    """The KC->MBON positions must agree, or learning edits the wrong weights."""
    c = _graph(seed=11)
    c.neurons.loc[:9, "type"] = "KCg-m"
    c.neurons.loc[10:19, "type"] = "MBON01"
    sensory = np.arange(4)

    fast = FastRateBrain(c, sensory, num_layers=3)
    slow = _reference(c, sensory, num_layers=3)
    pf, ps = fast.plasticity(), slow.plasticity()
    assert pf.n_synapses == ps.n_synapses
    assert np.asarray(fast.values)[pf.entry_index] == pytest.approx(
        np.asarray(slow.values.detach().cpu().numpy())[ps.entry_index], abs=1e-6
    )


# ------------------------------------------------------------ on its own


def test_synapse_counts_are_rejected():
    with pytest.raises(ValueError, match="input-proportion"):
        FastRateBrain(_graph("syncount"), np.arange(3))


def test_input_shape_is_checked():
    brain = FastRateBrain(_graph(), np.arange(5), num_layers=2)
    with pytest.raises(ValueError, match="n_sensory=5"):
        brain.run(np.zeros((4, 2), dtype=np.float32))


def test_indices_line_up_with_values():
    brain = FastRateBrain(_graph(seed=13), np.arange(3), num_layers=2)
    post, pre = brain.indices
    assert len(post) == len(pre) == len(brain.values)
    dense = brain.W.toarray()
    assert dense[post, pre] == pytest.approx(brain.values)


def test_the_chooser_returns_the_backend_it_was_asked_for():
    from flyloop.brain.fast import FastRateBrain as Fast
    from flyloop.brain.rate import rate_brain

    c, sensory = _graph(seed=17), np.arange(3)
    assert isinstance(rate_brain(c, sensory, num_layers=2), Fast)
    with pytest.raises(ValueError, match="unknown backend"):
        rate_brain(c, sensory, backend="quantum", num_layers=2)


def test_the_loop_defaults_to_the_fast_backend():
    """The closed loop is where the 23x actually buys something."""
    import inspect

    from flyloop.loop_rate import RateLoop

    assert inspect.signature(RateLoop).parameters["backend"].default == "fast"
