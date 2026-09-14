import numpy as np
import pytest

from flyloop.connectome.synthetic import synthetic_connectome
from flyloop.vision import (
    ColumnMap,
    CompoundEye,
    TemporalFilter,
    hex_directions,
    to_rates,
)


def test_interommatidial_angle_is_about_five_degrees():
    az, _el = hex_directions(721, fov_azimuth=150, fov_elevation=120)
    d = np.diff(np.sort(az))
    spacing = float(np.median(d[d > 1e-3]))
    assert 1.0 < spacing < 6.0


def test_directions_stay_inside_the_field_of_view():
    az, el = hex_directions(256, fov_azimuth=150, fov_elevation=120)
    assert np.abs(az).max() <= 75.0 + 1e-6
    assert np.abs(el).max() <= 60.0 + 1e-6


def test_eyes_look_in_opposite_directions():
    eye = CompoundEye(64, gaze_offset=45.0)
    assert eye.eyes["L"].azimuth.mean() < 0 < eye.eyes["R"].azimuth.mean()


def test_each_eye_sees_its_own_side():
    eye = CompoundEye(64)
    pano = np.zeros((90, 180))
    pano[:, 40:60] = 1.0  # a patch well to the left of straight ahead
    s = eye.sample(pano)
    assert s["L"].mean() > s["R"].mean()


def test_static_scene_adapts_away():
    eye = CompoundEye(64)
    pano = np.zeros((90, 180))
    pano[:, 40:60] = 1.0
    tf = TemporalFilter(tau=20e-3)
    out = None
    for _ in range(50):
        out = tf(eye.sample(pano), 10e-3)
    assert np.abs(out["L"]).max() < 1e-2


def test_filter_responds_to_change():
    eye = CompoundEye(64)
    tf = TemporalFilter(tau=50e-3)
    dark = np.zeros((90, 180))
    bright = np.ones((90, 180))
    for _ in range(30):
        tf(eye.sample(dark), 10e-3)
    out = tf(eye.sample(bright), 10e-3)
    assert out["L"].mean() > 0.5


def test_rates_are_non_negative_and_capped():
    r = to_rates(np.array([-5.0, 0.0, 0.1, 100.0]), max_rate=200.0, gain=400.0)
    assert r.min() >= 0.0
    assert r.max() <= 200.0


def test_column_map_refuses_to_invent_retinotopy():
    c = synthetic_connectome(n_columns=32)
    with pytest.raises(ValueError, match="ommatidia"):
        ColumnMap.from_connectome(c, CompoundEye(64))


def test_column_map_is_one_neuron_per_ommatidium():
    c = synthetic_connectome(n_columns=64)
    m = ColumnMap.from_connectome(c, CompoundEye(64))
    assert len(m.indices["L"]) == 64
    assert not set(m.indices["L"]) & set(m.indices["R"])
