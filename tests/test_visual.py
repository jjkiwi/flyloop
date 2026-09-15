"""Stimuli in hex coordinates, and what a silent network does with inhibition."""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from flyloop.brain import LIFBrain
from flyloop.connectome.schema import Connectome
from flyloop.vision.hexstim import (
    centre_column,
    column_neurons,
    grating,
    hemifield,
    hex_coords,
    looming,
    receding,
    static_disc,
    stimulus_set,
)


def _hex_connectome(radius: int = 4) -> Connectome:
    """A toy eye: one L1 and one L2 per hex column, nothing downstream."""
    rows = []
    nid = 0
    for x in range(-radius, radius + 1):
        for y in range(-radius, radius + 1):
            for t, nt in (("L1", "glutamate"), ("L2", "acetylcholine")):
                rows.append(
                    {"id": nid, "type": t, "nt": nt, "side": "R", "hex1": x, "hex2": y}
                )
                nid += 1
    neurons = pd.DataFrame(rows)
    return Connectome(neurons, sp.csr_matrix((nid, nid)), name="toy-eye")


@pytest.fixture(scope="module")
def eye():
    return _hex_connectome()


def test_hex_coords_finds_every_column(eye):
    coords = hex_coords(eye, cell_type="L1", side="R")
    assert len(coords) == 81
    assert all("," in c for c in coords)


def test_missing_hex_metadata_fails_loudly():
    neurons = pd.DataFrame(
        {"id": [0], "type": ["L1"], "nt": ["glutamate"], "side": ["R"], "hex1": [np.nan]}
    )
    c = Connectome(neurons, sp.csr_matrix((1, 1)), name="nohex")
    with pytest.raises(KeyError, match="hex-assigned"):
        hex_coords(c)


def test_column_neurons_groups_both_lamina_cells(eye):
    cols = column_neurons(eye, cell_types=("L1", "L2"), side="R")
    assert len(cols) == 81
    assert all(len(v) == 2 for v in cols.values())


def test_looming_expands_monotonically(eye):
    coords = hex_coords(eye)
    s = looming(coords, n_time=4)
    assert s.sizes() == sorted(s.sizes())
    assert s.sizes()[0] == 1


def test_receding_is_looming_reversed(eye):
    coords = hex_coords(eye)
    forward = looming(coords, n_time=4).sizes()
    assert receding(coords, n_time=4).sizes() == list(reversed(forward))


def test_static_disc_never_changes_size(eye):
    s = static_disc(hex_coords(eye), n_time=6)
    assert len(set(s.sizes())) == 1, "a changing 'static' control is not a control"


def test_static_disc_is_present_from_the_first_frame(eye):
    """An abruptly appearing object is a looming-like transient, not a control."""
    s = static_disc(hex_coords(eye), n_time=6)
    assert s.frames[0] == s.frames[-1]


def test_hemifield_splits_the_field(eye):
    coords = hex_coords(eye)
    left = hemifield(coords, bright="L", n_time=2)
    right = hemifield(coords, bright="R", n_time=2)
    assert set(left.frames[0]).isdisjoint(right.frames[0])
    assert len(left.frames[0]) + len(right.frames[0]) == len(coords)


def test_grating_moves_without_changing_how_much_is_lit(eye):
    s = grating(hex_coords(eye), direction=+1, period=4, n_time=8)
    assert s.frames[0] != s.frames[2], "the grating must drift"
    assert max(s.sizes()) - min(s.sizes()) <= len(hex_coords(eye)) // 4


def test_gratings_drift_in_opposite_directions(eye):
    coords = hex_coords(eye)
    r = grating(coords, direction=+1, period=4, n_time=8)
    left = grating(coords, direction=-1, period=4, n_time=8)
    assert r.frames[1] != left.frames[1]


def test_stimulus_set_covers_the_protocol(eye):
    s = stimulus_set(hex_coords(eye), n_time=4)
    assert set(s) == {
        "looming", "receding", "static", "grating_R", "grating_L", "dark_L", "dark_R"
    }


def test_centre_column_is_central(eye):
    coords = hex_coords(eye)
    assert centre_column(coords) == "0,0"


def test_all_lit_columns_are_real_columns(eye):
    coords = set(hex_coords(eye))
    for s in stimulus_set(list(coords), n_time=4).values():
        for frame in s.frames:
            assert set(frame).issubset(coords), s.name


# --- the finding that background makes possible ---------------------------


def _inhibitory_pair() -> Connectome:
    neurons = pd.DataFrame(
        {
            "id": [0, 1],
            "type": ["L1", "Mi1"],
            "nt": ["glutamate", "acetylcholine"],
            "side": ["R", "R"],
        }
    )
    dense = np.zeros((2, 2), dtype=np.float32)
    dense[0, 1] = -400.0  # L1 inhibits Mi1, well above single-spike threshold
    return Connectome(neurons, sp.csr_matrix(dense), name="inh-pair")


def test_inhibition_transmits_nothing_in_a_silent_network():
    """Zero basal firing makes every inhibitory pathway mute.

    Measured on MaleCNS: driving L1 at 200 Hz leaves Mi1 at exactly 0.0 Hz.
    There is nothing to inhibit, so the fly's ON pathway carries no signal.
    """
    b = LIFBrain(_inhibitory_pair(), seed=0)
    b.drive.set(np.array([0]), 200.0)
    r = b.run(0.1)
    assert r.spike_counts[0] > 0, "the inhibitory neuron itself must fire"
    assert r.spike_counts[1] == 0


def test_background_lets_inhibition_carry_a_signal():
    c = _inhibitory_pair()
    quiet = LIFBrain(c, seed=0)
    quiet.drive.set_background(np.array([1]), 60.0)
    without = quiet.run(0.1).spike_counts[1]

    driven = LIFBrain(c, seed=0)
    driven.drive.set_background(np.array([1]), 60.0)
    driven.drive.set(np.array([0]), 200.0)
    with_l1 = driven.run(0.1).spike_counts[1]

    assert without > 0
    assert with_l1 < without, "driving the inhibitory cell must suppress its target"


def test_clear_keeps_the_background():
    b = LIFBrain(_inhibitory_pair(), seed=0)
    b.drive.set_background(None, 10.0)
    b.drive.set(np.array([0]), 100.0)
    b.drive.clear()
    assert b.drive.rates.min() == pytest.approx(10.0)


def test_reset_removes_the_background_too():
    b = LIFBrain(_inhibitory_pair(), seed=0)
    b.drive.set_background(None, 10.0)
    b.reset()
    assert b.drive.rates.max() == 0.0
