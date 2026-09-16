"""Episodes on disk: the format a browser reads, and the claims it makes.

Every array is described by an offset, a shape and a dtype in JSON, and read
back with a typed-array view. If a stated offset is wrong by one byte the
viewer shows plausible-looking garbage rather than failing, so the round trip
is checked here rather than in a browser.
"""

import json

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from flyloop.app.episode import FORMAT, Episode, write_atlas
from flyloop.connectome.schema import Connectome


def _episode(frames: int = 6, types: int = 4, joints=None) -> Episode:
    rng = np.random.default_rng(0)
    return Episode(
        manifest={"connectome": "toy", "gain": 1.0},
        scalars=pd.DataFrame(
            {
                "t": np.arange(frames, dtype=np.float32) * 0.05,
                "turn": rng.normal(size=frames).astype(np.float32),
            }
        ),
        type_names=[f"T{i}" for i in range(types)],
        type_activity=rng.random((frames, types)).astype(np.float16),
        joints=joints,
        phases=[{"name": "behave", "start": 0, "stop": frames, "detail": ""}],
    )


def _read(out, manifest, name):
    """Read one array back exactly the way a browser would."""
    spec = manifest["arrays"][name]
    raw = (out / manifest["binary"]).read_bytes()
    count = int(np.prod(spec["shape"]))
    arr = np.frombuffer(raw, dtype=np.dtype(spec["dtype"]), count=count, offset=spec["offset"])
    return arr.reshape(spec["shape"])


# ------------------------------------------------------------- round trip


def test_arrays_read_back_exactly(tmp_path):
    ep = _episode()
    out = ep.save(tmp_path)
    m = json.loads((out / "episode.json").read_text())
    assert m["format"] == FORMAT
    assert m["frames"] == 6
    assert _read(out, m, "type_activity") == pytest.approx(ep.type_activity)
    assert _read(out, m, "scalar/turn") == pytest.approx(
        ep.scalars["turn"].to_numpy(np.float32)
    )


def test_every_offset_is_four_byte_aligned(tmp_path):
    """A typed array cannot view an unaligned offset without copying."""
    # An odd number of float16 entries is what makes padding necessary at all.
    out = _episode(frames=3, types=5).save(tmp_path)
    m = json.loads((out / "episode.json").read_text())
    for name, spec in m["arrays"].items():
        assert spec["offset"] % 4 == 0, f"{name} starts at {spec['offset']}"


def test_the_binary_is_exactly_as_long_as_the_index_claims(tmp_path):
    out = _episode(frames=7, types=3).save(tmp_path)
    m = json.loads((out / "episode.json").read_text())
    last = max(m["arrays"].values(), key=lambda s: s["offset"])
    end = last["offset"] + int(np.prod(last["shape"])) * np.dtype(last["dtype"]).itemsize
    assert (out / m["binary"]).stat().st_size >= end


# ------------------------------------------------------------------ joints


def test_a_body_without_legs_writes_no_joints(tmp_path):
    """Zero-filling would draw a corpse sliding across the floor."""
    out = _episode().save(tmp_path)
    m = json.loads((out / "episode.json").read_text())
    assert m["has_joints"] is False
    assert "joints" not in m["arrays"]


def test_joints_survive_the_round_trip_with_their_nans(tmp_path):
    """Training frames are NaN on purpose: on a rig the legs say nothing."""
    j = np.full((6, 42), 0.3, dtype=np.float32)
    j[:2] = np.nan
    out = _episode(joints=j).save(tmp_path)
    m = json.loads((out / "episode.json").read_text())
    back = _read(out, m, "joints")
    assert m["has_joints"] is True
    assert np.isnan(back[:2]).all()
    assert back[2:] == pytest.approx(0.3)


# ------------------------------------------------------------------- atlas


def _connectome(n: int = 40) -> Connectome:
    rng = np.random.default_rng(1)
    xyz = rng.uniform(1000, 90000, size=(n, 3))
    xyz[3] = np.nan  # one neuron with no located soma
    neurons = pd.DataFrame(
        {
            "id": np.arange(n),
            "type": [f"T{i % 7}" for i in range(n)],
            "nt": ["acetylcholine"] * n,
            "side": ["L", "R"] * (n // 2),
            "soma_x": xyz[:, 0],
            "soma_y": xyz[:, 1],
            "soma_z": xyz[:, 2],
        }
    )
    return Connectome(neurons, sp.csr_matrix((n, n)), name="toy")


def test_atlas_drops_unlocated_neurons(tmp_path):
    c = _connectome()
    write_atlas(c, tmp_path)
    m = json.loads((tmp_path / "atlas.json").read_text())
    assert m["neurons"] == len(c.neurons) - 1


def test_quantised_positions_come_back_close_enough_to_see(tmp_path):
    """int16 over the whole brain is about three voxels: invisible at any zoom."""
    c = _connectome()
    write_atlas(c, tmp_path)
    m = json.loads((tmp_path / "atlas.json").read_text())
    raw = (tmp_path / "atlas.bin").read_bytes()
    spec = m["arrays"]["position"]
    q = (
        np.frombuffer(
            raw, dtype=np.int16, count=int(np.prod(spec["shape"])), offset=spec["offset"]
        )
        .reshape(spec["shape"])
        .astype(np.float64)
    )
    back = (q - m["quantised_offset"]) * np.asarray(m["scale"]) + np.asarray(m["origin"])

    xyz = c.neurons[["soma_x", "soma_y", "soma_z"]].to_numpy(float)
    want = xyz[np.isfinite(xyz).all(axis=1)]
    assert np.abs(back - want).max() < 5.0  # voxels


def test_the_type_index_points_at_the_right_names(tmp_path):
    c = _connectome()
    write_atlas(c, tmp_path)
    m = json.loads((tmp_path / "atlas.json").read_text())
    raw = (tmp_path / "atlas.bin").read_bytes()
    spec = m["arrays"]["type_index"]
    idx = np.frombuffer(raw, dtype=np.uint16, count=spec["shape"][0], offset=spec["offset"])
    names = np.asarray(m["type_names"])[idx]

    xyz = c.neurons[["soma_x", "soma_y", "soma_z"]].to_numpy(float)
    kept = c.neurons["type"].to_numpy()[np.isfinite(xyz).all(axis=1)]
    assert names.tolist() == kept.tolist()


def test_too_many_cell_types_is_refused_rather_than_wrapped(tmp_path):
    """uint16 silently wraps at 65,536, which would mislabel every neuron."""
    n = 3
    neurons = pd.DataFrame(
        {
            "id": np.arange(n),
            "type": [f"T{i}" for i in range(n)],
            "nt": ["acetylcholine"] * n,
            "side": ["L"] * n,
            "soma_x": [1.0, 2.0, 3.0],
            "soma_y": [1.0, 2.0, 3.0],
            "soma_z": [1.0, 2.0, 3.0],
        }
    )
    c = Connectome(neurons, sp.csr_matrix((n, n)), name="toy")
    import flyloop.app.episode as mod

    real = np.unique

    def fake(arr, **kw):
        names, inverse = real(arr, **kw)
        return np.array([f"T{i}" for i in range(70000)]), inverse

    mod.np.unique = fake
    try:
        with pytest.raises(ValueError, match="uint16"):
            write_atlas(c, tmp_path)
    finally:
        mod.np.unique = real
