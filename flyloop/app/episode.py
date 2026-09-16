"""Episodes on disk, sized for a browser rather than for an analysis.

Real-time is not available and will not be: a control step costs 1.99 s -- 0.16 s
of rate model and 1.8 s of physics -- against the 0.05 s it simulates, so the
loop runs about 40x slower than the animal. An application therefore precomputes
and replays, and the only question is what to keep.

**Per-neuron activity is not an option.** 161,429 neurons at 200 frames is
194 MB per episode. Per *cell type* it is 6,875 live types in float16, or
2.6 MB, and the atlas that maps neurons to types and to positions is 1.05 MB
sent once for every episode. Nothing is lost that a viewer could have drawn:
the spatial panel colours somata, and every soma of one type carries that type's
value anyway.

**Two bodies, one format.** The kinematic stub is calibrated to NeuroMechFly
(13.7 mm/s, 4.2 rad/s) but has no legs, so its ``joints`` array is absent rather
than zero-filled. A viewer must check for it and fall back to drawing the body
as a marker; silently posing a fly with 42 zeros would show a corpse sliding
across the floor and look like a bug in the physics.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..connectome.schema import Connectome

#: Format version. Bump when a viewer written against the old one would break.
FORMAT = "flyloop-episode/1"

#: Activation below this counts as silence, and the type is left out entirely.
LIVE_THRESHOLD = 1e-4

#: Scalar per-frame columns every episode carries, whatever the body.
SCALARS = (
    "t",
    "reward",
    "depression",
    "modulation",
    "mbon_readout",
    "learned_component",
    "gain",
)
#: Present only while the fly is behaving; NaN during training.
BEHAVIOUR_SCALARS = (
    "x",
    "y",
    "theta",
    "distance",
    "bearing",
    "half_width",
    "turn",
    "forward",
)


@dataclass
class Episode:
    """One recorded run, ready to be written for a player."""

    manifest: dict
    scalars: pd.DataFrame
    type_names: list[str]
    type_activity: np.ndarray  # (frames, types) float16
    joints: np.ndarray | None = None  # (frames, 42) float32, physics body only
    phases: list[dict] = field(default_factory=list)

    def save(self, out_dir: str | Path) -> Path:
        """Write ``episode.json`` and ``episode.bin`` into ``out_dir``."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        blobs: list[tuple[str, np.ndarray]] = []
        for name in self.scalars.columns:
            blobs.append((f"scalar/{name}", self.scalars[name].to_numpy(np.float32)))
        blobs.append(("type_activity", self.type_activity))
        if self.joints is not None:
            blobs.append(("joints", self.joints.astype(np.float32)))

        index, offset, chunks = {}, 0, []
        for name, arr in blobs:
            data = np.ascontiguousarray(arr)
            index[name] = {
                "offset": offset,
                "shape": list(data.shape),
                "dtype": data.dtype.name,
            }
            chunks.append(data.tobytes())
            offset += data.nbytes
            # Keep every chunk 4-byte aligned so a typed array can view it
            # without copying, the same rule the geometry export follows.
            pad = (-offset) % 4
            if pad:
                chunks.append(b"\0" * pad)
                offset += pad
        (out / "episode.bin").write_bytes(b"".join(chunks))

        manifest = dict(self.manifest)
        manifest.update(
            {
                "format": FORMAT,
                "frames": int(len(self.scalars)),
                "type_names": self.type_names,
                "phases": self.phases,
                "arrays": index,
                "binary": "episode.bin",
                "has_joints": self.joints is not None,
            }
        )
        (out / "episode.json").write_text(json.dumps(manifest))
        return out


def record_episode(
    loop,
    *,
    train_trials: int = 8,
    behave_steps: int = 40,
    train_odour: str = "trained",
    behave_odour: str | None = "trained",
    progress: bool = False,
) -> Episode:
    """Run a :class:`~flyloop.hybrid.HybridLoop` and keep what a player needs.

    Both phases go into one episode on one timeline, because the point of
    showing them together is that the synapses carry across: training moves
    weights the behaving fly then uses.
    """
    c: Connectome = loop.c
    types = c.neurons["type"].astype(str).to_numpy()
    order = np.argsort(types, kind="stable")
    sorted_types = types[order]
    starts = np.flatnonzero(np.r_[True, sorted_types[1:] != sorted_types[:-1]])
    names = sorted_types[starts].tolist()
    groups = np.split(order, starts[1:])

    frames: list[np.ndarray] = []
    joints: list[np.ndarray] = []
    want_joints = loop.body_has_joints

    def on_frame(acts: np.ndarray, *, phase: str) -> None:
        peak = acts.max(axis=1)
        frames.append(np.array([peak[g].mean() for g in groups], dtype=np.float32))
        if want_joints:
            # Posture is only meaningful while the fly is walking; during
            # training it is standing on a rig and the legs say nothing.
            joints.append(
                loop.body.joint_angles()
                if phase == "behave"
                else np.full_like(joints[-1] if joints else np.zeros(42), np.nan)
            )

    loop.reset()
    loop.on_frame = on_frame

    if train_trials:
        loop.train(train_trials, odour_name=train_odour)
    if behave_steps:
        loop.run(behave_steps, odour_name=behave_odour, progress=progress)

    log = loop.log()
    activity = np.vstack(frames) if frames else np.zeros((len(log), len(names)))
    live = activity.max(axis=0) > LIVE_THRESHOLD
    activity, names = activity[:, live], [n for n, k in zip(names, live, strict=True) if k]

    cols = [c_ for c_ in SCALARS + BEHAVIOUR_SCALARS if c_ in log.columns]
    manifest = {
        "connectome": c.name,
        "neurons": int(len(c.neurons)),
        "gain": float(loop.learned.gain),
        "coupling_share": float(loop.learned.share),
        "target": {"x": loop.target.x, "y": loop.target.y, "radius": loop.target.radius},
        "control_dt": float(loop.dt),
        "body": type(loop.body).__name__,
        "train_odour": train_odour,
        "behave_odour": behave_odour or "none",
        "live_types": int(live.sum()),
        "all_types": int(len(live)),
    }
    return Episode(
        manifest=manifest,
        scalars=log[cols].astype(np.float32),
        type_names=names,
        type_activity=activity.astype(np.float16),
        joints=np.vstack(joints) if joints else None,
        phases=[
            {"name": p.name, "start": p.start, "stop": p.stop, "detail": p.detail}
            for p in loop.phases
        ],
    )


def write_atlas(c: Connectome, out_dir: str | Path) -> Path:
    """Write the static per-neuron map an episode's type activity is painted on.

    Positions are quantised to int16 in the dataset's own voxel space. That is
    about 3 voxels of error on a 90,000-voxel brain -- invisible at any zoom a
    browser will show, and it halves the file.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    xyz = c.neurons[["soma_x", "soma_y", "soma_z"]].to_numpy(float)
    keep = np.flatnonzero(np.isfinite(xyz).all(axis=1))
    pos = xyz[keep]
    lo, hi = pos.min(axis=0), pos.max(axis=0)
    scale = (hi - lo) / 65534.0
    quant = np.round((pos - lo) / scale).astype(np.int32) - 32767
    quant = quant.astype(np.int16)

    types = c.neurons["type"].astype(str).to_numpy()[keep]
    names, index = np.unique(types, return_inverse=True)
    if len(names) > 65535:
        raise ValueError(f"{len(names)} cell types will not fit a uint16 index")

    (out / "atlas.bin").write_bytes(quant.tobytes() + index.astype(np.uint16).tobytes())
    (out / "atlas.json").write_text(
        json.dumps(
            {
                "format": "flyloop-atlas/1",
                "connectome": c.name,
                "neurons": int(len(keep)),
                "type_names": names.tolist(),
                "origin": lo.tolist(),
                "scale": scale.tolist(),
                "quantised_offset": -32767,
                "arrays": {
                    "position": {"offset": 0, "shape": [len(keep), 3], "dtype": "int16"},
                    "type_index": {
                        "offset": quant.nbytes,
                        "shape": [len(keep)],
                        "dtype": "uint16",
                    },
                },
                "binary": "atlas.bin",
            }
        )
    )
    return out
