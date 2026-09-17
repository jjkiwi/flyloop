"""Named chemicals as glomerular input: the joins, and what they refuse."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from flyloop.brain.dopamine import concentration_reward
from flyloop.connectome.schema import Connectome

DATA_ROOT = Path("/home/user/yijieyin/connectome_data_prep")
needs_door = pytest.mark.skipif(
    not (DATA_ROOT / "data" / "DoOR").is_dir(), reason="needs a DoOR clone"
)


def _nose() -> Connectome:
    """Four ORNs across two glomeruli, enough to build an odour vector on."""
    rows = []
    for i, g in enumerate(("DC4", "DP1m", "DM1", "VM7d")):
        rows.append(
            {"id": i, "type": f"ORN_{g}", "nt": "acetylcholine", "side": "R"}
        )
    n = len(rows)
    return Connectome(
        pd.DataFrame(rows),
        sp.csr_matrix((n, n), dtype=np.float32),
        name="nose",
        meta={"matrix_kind": "inprop"},
    )


# ---------------------------------------------------------------- reward


def test_reward_is_proportional_to_concentration():
    for c in (0.0, 0.25, 0.5, 1.0):
        assert concentration_reward(c, scale=1.0) == pytest.approx(c)


def test_reward_is_capped_so_one_trial_cannot_invert_the_weights():
    """The plasticity rule multiplies by this and nothing downstream is
    normalised, so an uncapped high concentration is not strong learning, it is
    a depression step large enough to drive weights negative in one trial."""
    assert concentration_reward(50.0, scale=1.0) == 1.0
    assert concentration_reward(2.0, scale=1.0, cap=0.5) == 0.5


def test_negative_concentration_is_refused():
    with pytest.raises(ValueError, match="negative"):
        concentration_reward(-1.0)


# ---------------------------------------------------------- graded odour


def test_a_profile_becomes_a_graded_receptor_vector():
    """The difference from the binary odour: each glomerulus arrives at the
    strength the chemical evokes, not at 1.0."""
    from flyloop.experiments.odorants import profile_odour
    from flyloop.experiments.olfactory import olfactory_receptors

    c = _nose()
    r = olfactory_receptors(c)
    v = profile_odour(c, {"DC4": 0.9, "DP1m": 0.5}, r)
    types = c.neurons["type"].astype(str).to_numpy()[r]
    assert v[types == "ORN_DC4"] == pytest.approx(0.9)
    assert v[types == "ORN_DP1m"] == pytest.approx(0.5)
    assert v[types == "ORN_DM1"] == pytest.approx(0.0)


def test_concentration_scales_the_whole_pattern():
    """Same substance, half the amount: same pattern, half the drive. Without
    this a dose-response experiment varies nothing."""
    from flyloop.experiments.odorants import profile_odour
    from flyloop.experiments.olfactory import olfactory_receptors

    c = _nose()
    r = olfactory_receptors(c)
    full = profile_odour(c, {"DC4": 0.9, "DP1m": 0.5}, r, concentration=1.0)
    half = profile_odour(c, {"DC4": 0.9, "DP1m": 0.5}, r, concentration=0.5)
    assert np.allclose(half, full * 0.5)


def test_a_profile_of_absent_glomeruli_is_refused():
    """A silent zero here is an odour that looks unlearnable rather than absent."""
    from flyloop.experiments.odorants import profile_odour
    from flyloop.experiments.olfactory import olfactory_receptors

    c = _nose()
    with pytest.raises(KeyError, match="none of the profile"):
        profile_odour(c, {"VA7l": 1.0}, olfactory_receptors(c))


def test_negative_concentration_is_refused_for_odours():
    from flyloop.experiments.odorants import profile_odour
    from flyloop.experiments.olfactory import olfactory_receptors

    c = _nose()
    with pytest.raises(ValueError, match="negative"):
        profile_odour(c, {"DC4": 1.0}, olfactory_receptors(c), concentration=-0.5)


# --------------------------------------------------------------- the join


@needs_door
def test_door_resolves_a_chemical_to_glomeruli():
    from flyloop.experiments.odorants import D2HG_PROXY, door_profile

    p = door_profile(DATA_ROOT, D2HG_PROXY)
    assert p.name == D2HG_PROXY
    # The ionotropic acid pathway, which is why this chemical was chosen.
    assert "DC4" in p.index and p["DC4"] > 0.5
    assert {"DP1m", "DP1l", "VL2a"} <= set(p.index)


@needs_door
def test_ambiguous_sensillum_recordings_are_dropped_and_reported():
    """A row like "VL1+DP1l+VC5 via ac2" cannot say which glomerulus responded;
    spreading it across all three invents specificity the recording lacks."""
    from flyloop.experiments.odorants import D2HG_PROXY, door_profile

    p = door_profile(DATA_ROOT, D2HG_PROXY)
    assert all("+" not in g for g in p.index)
    assert 0.0 < p.attrs["dropped"] < 1.0


@needs_door
def test_an_unknown_chemical_says_how_many_are_known():
    from flyloop.experiments.odorants import door_profile

    with pytest.raises(KeyError, match="no chemical named"):
        door_profile(DATA_ROOT, "D-2-hydroxyglutarate")


@needs_door
def test_d2hg_itself_is_absent_from_every_odorant_dataset():
    """The claim that forces the proxy. D-2HG is a non-volatile diacid; flies
    smell volatiles, so no glomerular response to it has ever been recorded.
    If this ever fails, the proxy should be retired for the real thing."""
    from flyloop.experiments.odorants import door_chemicals

    names = [n.lower() for n in door_chemicals(DATA_ROOT)]
    assert not any("hydroxyglutar" in n or "2hg" in n for n in names)
