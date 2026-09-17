"""The run endpoint's request validation.

The service itself needs a connectome and is exercised by hand; what is worth
pinning here is the boundary, because it is the only place a browser can put
arbitrary values into a simulation.
"""

import pytest

from flyloop.app.server import Spec


def test_defaults_are_a_runnable_stimulus():
    s = Spec.from_json({})
    assert s.bearing == 35.0 and s.steps == 40 and s.body == "kinematic"


def test_fields_are_taken_from_the_request():
    s = Spec.from_json({"bearing": -70.0, "steps": 30, "odour": "control"})
    assert (s.bearing, s.steps, s.odour) == (-70.0, 30, "control")


def test_none_and_the_string_none_both_mean_no_odour():
    """The form sends "none"; a script is likelier to send null."""
    assert Spec.from_json({"odour": "none"}).odour is None
    assert Spec.from_json({"odour": None}).odour is None


def test_an_unknown_field_is_refused_rather_than_ignored():
    """Silently dropping a misspelled field runs a different stimulus than asked."""
    with pytest.raises(ValueError, match="unknown fields"):
        Spec.from_json({"bearng": 35.0})


def test_an_unknown_odour_is_refused():
    with pytest.raises(ValueError, match="unknown odour"):
        Spec.from_json({"odour": "lavender"})


def test_an_unknown_body_is_refused():
    with pytest.raises(ValueError, match="kinematic or physics"):
        Spec.from_json({"body": "hexapod"})


@pytest.mark.parametrize("steps", [0, -5, 10_000])
def test_step_counts_are_bounded(steps):
    """400 steps is 20 s of fly and two minutes of wall clock; beyond that a
    browser tab has silently asked for an hour of compute."""
    with pytest.raises(ValueError, match="steps must be"):
        Spec.from_json({"steps": steps})


@pytest.mark.parametrize("trials", [-1, 500])
def test_training_trial_counts_are_bounded(trials):
    with pytest.raises(ValueError, match="train_trials must be"):
        Spec.from_json({"train_trials": trials})


def test_the_cache_key_ignores_what_changes_per_run():
    """Bearing and odour must not force a 15 s rebuild of the rate model."""
    a = Spec.from_json({"bearing": 10.0, "odour": "control", "steps": 5})
    b = Spec.from_json({"bearing": -80.0, "odour": "none", "steps": 200})
    assert a.key() == b.key()


def test_the_cache_key_separates_what_does_not():
    base = Spec.from_json({})
    for field, value in (("gain", 50.0), ("coupling_hops", 4), ("body", "physics")):
        assert Spec.from_json({field: value}).key() != base.key(), field


def test_glomeruli_default_to_the_documented_pair():
    s = Spec.from_json({})
    assert s.trained_glomeruli == ("DM1", "DM4")
    assert s.control_glomeruli == ("DA1", "VA2")


def test_glomeruli_come_from_the_request():
    s = Spec.from_json({"trained_glomeruli": ["V", "DL5"]})
    assert s.trained_glomeruli == ("V", "DL5")


def test_a_single_glomerulus_may_be_sent_as_a_bare_string():
    """A hand-written request is likelier to send "DA1" than ["DA1"]."""
    assert Spec.from_json({"trained_glomeruli": "DA1"}).trained_glomeruli == ("DA1",)


def test_whitespace_and_repeats_are_tidied_rather_than_refused():
    """Order carries no meaning here -- an odour is a set of ORNs."""
    s = Spec.from_json({"trained_glomeruli": [" DM1 ", "DM1", "DM4", ""]})
    assert s.trained_glomeruli == ("DM1", "DM4")


def test_an_empty_odour_is_refused():
    with pytest.raises(ValueError, match="at least one glomerulus"):
        Spec.from_json({"trained_glomeruli": []})


def test_too_many_glomeruli_are_refused():
    """The whole antennal lobe at once is not an odour, it is every odour."""
    with pytest.raises(ValueError, match="at most"):
        Spec.from_json({"control_glomeruli": [f"G{i}" for i in range(20)]})


def test_identical_odours_are_refused():
    """Differential conditioning needs something unrewarded to compare against;
    with both odours the same, a null result would be guaranteed by the request
    rather than measured."""
    with pytest.raises(ValueError, match="nothing to be measured against"):
        Spec.from_json(
            {"trained_glomeruli": ["DM1", "DM4"], "control_glomeruli": ["DM4", "DM1"]}
        )


def test_an_unknown_training_odour_is_refused():
    with pytest.raises(ValueError, match="unknown train_odour"):
        Spec.from_json({"train_odour": "none"})


def test_the_cache_key_ignores_the_glomeruli():
    """Swapping an odour is a table lookup; it must not cost a 15 s rebuild."""
    a = Spec.from_json({"trained_glomeruli": ["V"], "control_glomeruli": ["DA1"]})
    assert a.key() == Spec.from_json({}).key()
