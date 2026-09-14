import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from flyloop.brain.lif import (
    LIFBrain,
    LIFParams,
    peak_psp_factor,
    synapses_to_threshold,
)
from flyloop.connectome.schema import Connectome
from flyloop.connectome.synthetic import eye_columns, synthetic_connectome


def _pair(weight: float) -> Connectome:
    neurons = pd.DataFrame(
        {"id": [0, 1], "type": ["pre", "post"], "nt": ["acetylcholine"] * 2,
         "side": ["L", "L"]}
    )
    W = sp.csr_matrix(np.array([[0.0, weight], [0.0, 0.0]], dtype=np.float32))
    return Connectome(neurons, W, name="pair")


def test_peak_psp_factor_matches_analytic_value():
    # For tau_syn=5ms, tau_m=20ms the peak of the difference of exponentials
    # is a well defined fraction of the conductance step.
    assert peak_psp_factor(LIFParams()) == pytest.approx(0.1575, abs=1e-3)


def test_synapses_to_threshold_is_order_160():
    n = synapses_to_threshold(LIFParams())
    assert 150 < n < 175


def test_silent_network_stays_silent():
    c = synthetic_connectome(n_central=200)
    b = LIFBrain(c, seed=0)
    r = b.run(0.05)
    assert r.spike_counts.sum() == 0, "the model has no basal firing rate by design"


def test_reset_clears_the_external_drive():
    c = synthetic_connectome(n_central=200)
    b = LIFBrain(c, seed=0)
    b.drive.set(eye_columns(c)["L"], 200.0)
    assert b.run(0.05).spike_counts.sum() > 0
    b.reset()
    assert b.run(0.05).spike_counts.sum() == 0


# synapses_to_threshold() is a claim about a *single* presynaptic spike, so the
# drive here is slow enough (spikes ~200 ms apart, ten membrane time constants)
# that successive PSPs cannot summate. At faster rates a connection well below
# this size still fires its target, which is the whole point of the measure.
_ISOLATED_SPIKE_RATE = 5.0


def test_subthreshold_single_spike_does_not_fire():
    below = synapses_to_threshold(LIFParams()) * 0.5
    b = LIFBrain(_pair(below), seed=0)
    b.drive.set(np.array([0]), _ISOLATED_SPIKE_RATE)
    r = b.run(2.0)
    assert r.spike_counts[0] > 0, "the presynaptic neuron must actually spike"
    assert r.spike_counts[1] == 0


def test_suprathreshold_single_spike_does_fire():
    above = synapses_to_threshold(LIFParams()) * 1.5
    b = LIFBrain(_pair(above), seed=0)
    b.drive.set(np.array([0]), _ISOLATED_SPIKE_RATE)
    r = b.run(2.0)
    assert r.spike_counts[1] > 0


def test_temporal_summation_lowers_the_effective_threshold():
    """A connection too weak for one spike still fires its target at high rate."""
    below = synapses_to_threshold(LIFParams()) * 0.5
    slow = LIFBrain(_pair(below), seed=0)
    slow.drive.set(np.array([0]), _ISOLATED_SPIKE_RATE)
    fast = LIFBrain(_pair(below), seed=0)
    fast.drive.set(np.array([0]), 100.0)
    assert slow.run(2.0).spike_counts[1] == 0
    assert fast.run(2.0).spike_counts[1] > 0


def test_inhibition_has_the_opposite_effect():
    above = synapses_to_threshold(LIFParams()) * 1.5
    exc = LIFBrain(_pair(above), seed=0)
    inh = LIFBrain(_pair(-above), seed=0)
    for b in (exc, inh):
        b.drive.set(np.array([0]), _ISOLATED_SPIKE_RATE)
    assert exc.run(2.0).spike_counts[1] > 0
    assert inh.run(2.0).spike_counts[1] == 0


def test_axonal_delay_is_respected():
    p = LIFParams(t_delay=5e-3)
    b = LIFBrain(_pair(synapses_to_threshold(p) * 3), params=p, seed=0)
    b.drive.set(np.array([0]), 400.0)
    first_pre = first_post = None
    for i in range(400):
        spk = b.step()
        if 0 in spk and first_pre is None:
            first_pre = i
        if 1 in spk and first_post is None:
            first_post = i
    assert first_pre is not None and first_post is not None
    assert (first_post - first_pre) * p.dt >= p.t_delay


def test_refractory_period_caps_the_firing_rate():
    p = LIFParams()
    b = LIFBrain(_pair(0.0), params=p, seed=0)
    b.drive.set(np.array([0, 1]), 5000.0)  # drive far above saturation
    r = b.run(0.5)
    max_rate = 1.0 / p.t_refractory
    assert r.rate(np.array([0])) <= max_rate * 1.05


def test_population_rates_are_recorded_per_bin():
    c = synthetic_connectome(n_central=200)
    b = LIFBrain(c, seed=0)
    b.drive.set(eye_columns(c)["L"], 200.0)
    rec = {"LC4": c.population("LC4")}
    r = b.run(0.1, record=rec, bin_size=10e-3)
    assert r.population_rates["LC4"].shape == (10,)
    assert r.population_rates["LC4"].sum() > 0
