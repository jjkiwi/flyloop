"""The fly's body as geometry a browser can draw from logged joint angles.

Everything else in this project logs *numbers* about the body -- heading,
bearing, joint angles. Those are the honest record, but nobody can look at a
42-column CSV and see that the middle right leg is dragging. This module exports
the geometry those numbers refer to: the kinematic tree of NeuroMechFly v2 plus
its 69 surface meshes, in a form plain Three.js can load and pose without a
server, a physics engine, or a Python round trip.

**The only thing that makes this worth having is that it bends correctly.** A
rig that renders a recognisable fly but composes its rotations in the wrong
order produces a plausible-looking animation of a body that never existed, and
nothing downstream would catch it -- the picture is the output. So the exported
tree is checked against MuJoCo's own ``mj_kinematics`` in
``tests/test_fly_geometry.py``, body by body, for random poses, and it agrees to
about 1e-15 mm. That test is the deliverable; the JSON is a side effect.

**MuJoCo semantics that a naive reading of the MJCF gets wrong.** Body ``pos``
and ``quat`` are relative to the *parent body* frame, not the world. A hinge
rotates the body relative to that frame, about an anchor ``pos`` expressed in
the body's own coordinates, and the several joints on one body compose in
document order, each axis interpreted in the frame left by the joints before it
(this is exactly what ``mj_kinematics`` does: right-multiply the body quaternion
per joint, then re-derive the origin from the anchor). NeuroMechFly puts up to
three hinges on a coxa, so the order is load-bearing: yaw/pitch/roll applied in
the wrong sequence is wrong by tens of degrees at the tarsus, not by a rounding
error.

**Quaternion conventions.** MuJoCo writes ``(w, x, y, z)``; Three.js reads
``(x, y, z, w)``. The conversion happens once, at the serialisation boundary --
:func:`to_threejs_quat` on the way out, :func:`from_threejs_quat` on the way
back in -- and ``rig.json`` records ``"quaternion_order": "xyzw"`` so a reader
never has to infer it. Everything inside this module is MuJoCo order.

**Meshes are used raw, not as MuJoCo compiles them.** MuJoCo translates each
mesh onto its centre of mass and rotates it onto its principal axes -- that is
what ``mesh_vert`` holds -- and folds the inverse of that into the compiled geom
frame, so ``geom_xpos``/``geom_xmat`` already carry it. The two cancel: STL
vertices scaled by the asset's ``scale`` and placed with the geom's *MJCF*
``pos``/``quat`` land where MuJoCo draws them, to 3e-8 mm. Skipping the
recentring keeps the exporter independent of the MuJoCo compiler, and the test
that pins it poses the model first, so it would catch a rotation as well as an
offset.

**Units are millimetres**, matching the MJCF and the rest of this project. The
STL files are metres; the asset ``scale`` of 1000 converts them.

Normals are not exported. Recomputing them in the browser costs a few
milliseconds per load and saves about a third of the file; the meshes are
closed and consistently wound, so ``computeVertexNormals()`` is enough.
"""

from __future__ import annotations

import json
import struct
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

#: Which MJCF variant. FlyGym ships three; ``seqik`` is the one ``Fly()`` loads
#: by default (``config.yaml: paths.mjcf.seqik``) and therefore the one whose
#: joint names match a logged observation. The ``deepfly3d`` variant has a
#: different kinematic ordering -- the file names say so -- and the
#: ``capsuletarsus`` variant replaces the tarsus meshes with primitives.
MJCF_VARIANT = "seqik"

#: Bytes allowed for ``meshes.bin``. The task's ceiling is 8 MB for the whole
#: export; ``rig.json`` is ~60 kB, so this leaves a wide margin for it.
MESH_BUDGET_BYTES = 7_000_000

#: Bounds for the vertex-clustering search, as cells along a mesh's longest
#: bounding-box axis. Below ~16 the tarsi collapse into sticks; above 512 the
#: grid is finer than the mesh and clustering is a no-op.
_CLUSTER_MIN, _CLUSTER_MAX = 16, 512


# --------------------------------------------------------------- quaternions
#
# Written out rather than pulled from scipy: these run inside the forward
# kinematics, which is the thing being validated against MuJoCo, and a
# four-line function that can be read against mju_mulQuat is worth more here
# than a dependency whose conventions have to be checked anyway.


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product in MuJoCo's ``(w, x, y, z)`` order."""
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate ``v`` by ``q``, without building a matrix."""
    u = q[1:]
    return v + 2.0 * np.cross(u, np.cross(u, v) + q[0] * v)


def quat_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """Hinge rotation. ``axis`` need not be normalised; MuJoCo normalises too."""
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    return np.concatenate([[np.cos(0.5 * angle)], np.sin(0.5 * angle) * axis])


def to_threejs_quat(q: Sequence[float]) -> list[float]:
    """MuJoCo ``(w, x, y, z)`` -> Three.js ``(x, y, z, w)``."""
    w, x, y, z = (float(c) for c in q)
    return [x, y, z, w]


def from_threejs_quat(q: Sequence[float]) -> np.ndarray:
    """Three.js ``(x, y, z, w)`` -> MuJoCo ``(w, x, y, z)``."""
    x, y, z, w = (float(c) for c in q)
    return np.array([w, x, y, z])


def _vec(text: str | None, default: Sequence[float]) -> np.ndarray:
    if text is None:
        return np.array(default, dtype=float)
    return np.fromstring(text.strip(), sep=" ", dtype=float)


def _unit_quat(text: str | None) -> np.ndarray:
    """MJCF quaternions are not required to be normalised, and these are not.

    ``A1A2`` carries ``quat="0.991 0.0 -0.131 0.0"``, which is off unit length by
    3e-5. MuJoCo normalises at compile time; skipping it here would put every
    abdominal segment a micrometre out and, worse, would grow with depth.
    """
    q = _vec(text, (1.0, 0.0, 0.0, 0.0))
    return q / np.linalg.norm(q)


# --------------------------------------------------------------- the rig


@dataclass(frozen=True)
class Geom:
    """One mesh instance, placed in its body's frame."""

    mesh: str  # asset name, e.g. "mesh_Thorax"
    pos: np.ndarray
    quat: np.ndarray  # MuJoCo (w, x, y, z)


@dataclass(frozen=True)
class Body:
    """A node of the kinematic tree. ``pos``/``quat`` are parent-relative."""

    name: str
    parent: int  # index into FlyRig.bodies; -1 for the root
    pos: np.ndarray
    quat: np.ndarray  # MuJoCo (w, x, y, z)
    geoms: tuple[Geom, ...] = ()
    joints: tuple[int, ...] = ()  # indices into FlyRig.joints, document order


@dataclass(frozen=True)
class Joint:
    """A hinge. ``axis`` and ``pos`` are in the driven body's own frame."""

    name: str
    body: int  # index into FlyRig.bodies
    axis: np.ndarray
    pos: np.ndarray  # rotation anchor; zero for every joint in this model
    type: str = "hinge"


@dataclass
class FlyRig:
    """The kinematic tree, with enough of the MJCF to reproduce its kinematics.

    ``bodies`` is in depth-first document order, so a parent always precedes its
    children and one forward pass suffices -- in Python here, and in the browser
    from ``rig.json``.
    """

    bodies: list[Body]
    joints: list[Joint]
    meshes: dict[str, "MeshAsset"]
    source: Path | None = None
    joint_order: tuple[str, ...] | None = None
    metadata: dict = field(default_factory=dict)

    # -------------------------------------------------------------- lookup

    def body_index(self, name: str) -> int:
        for i, b in enumerate(self.bodies):
            if b.name == name:
                return i
        raise KeyError(f"no body named {name!r}")

    def joint_index(self, name: str) -> int:
        for i, j in enumerate(self.joints):
            if j.name == name:
                return i
        raise KeyError(f"no joint named {name!r}")

    # ---------------------------------------------------------- kinematics

    def _angles(self, angles: Mapping[str, float] | Sequence[float] | None) -> np.ndarray:
        """Accept either a name->angle map or a logged vector in joint order."""
        out = np.zeros(len(self.joints))
        if angles is None:
            return out
        if isinstance(angles, Mapping):
            for name, value in angles.items():
                out[self.joint_index(name)] = float(value)
            return out
        vec = np.asarray(angles, dtype=float).reshape(-1)
        if self.joint_order is None:
            raise ValueError(
                "this rig carries no joint order, so a bare vector cannot be "
                "placed; pass a {joint name: angle} mapping instead"
            )
        if vec.size != len(self.joint_order):
            raise ValueError(
                f"expected {len(self.joint_order)} angles in joint order, got {vec.size}"
            )
        for name, value in zip(self.joint_order, vec, strict=True):
            out[self.joint_index(name)] = float(value)
        return out

    def forward_kinematics(
        self, angles: Mapping[str, float] | Sequence[float] | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """World pose of every body, in the root body's frame.

        This is ``mj_kinematics`` for a tree of hinges, transcribed: place the
        body in its parent, then for each of its joints in document order rotate
        about the anchor, re-deriving the origin from the anchor so an
        off-centre hinge swings the body rather than spinning it in place. Every
        joint in this model anchors at the body origin, so that correction is a
        no-op here -- it is kept because the next MJCF may not be so tidy, and a
        silently wrong exporter is the failure mode this module exists to avoid.

        Returns ``(pos, quat)`` of shape ``(n_bodies, 3)`` and ``(n_bodies, 4)``,
        the quaternions in MuJoCo order. The root is the identity frame: the
        model has no free joint, so where the fly *is* comes from the log, not
        from here.
        """
        q_all = self._angles(angles)
        pos = np.zeros((len(self.bodies), 3))
        quat = np.zeros((len(self.bodies), 4))
        for i, body in enumerate(self.bodies):
            if body.parent < 0:
                p, r = np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])
            else:
                p, r = pos[body.parent], quat[body.parent]
            p = p + quat_rotate(r, body.pos)
            r = quat_mul(r, body.quat)
            for ji in body.joints:
                joint = self.joints[ji]
                anchor = p + quat_rotate(r, joint.pos)
                r = quat_mul(r, quat_axis_angle(joint.axis, q_all[ji]))
                p = anchor - quat_rotate(r, joint.pos)
            pos[i], quat[i] = p, r
        return pos, quat

    # ----------------------------------------------------- serialisation

    def to_dict(self) -> dict:
        """The tree as it goes into ``rig.json`` -- Three.js quaternion order."""
        return {
            "bodies": [
                {
                    "name": b.name,
                    "parent": b.parent,
                    "pos": [float(v) for v in b.pos],
                    "quat": to_threejs_quat(b.quat),
                    "joints": list(b.joints),
                    "geoms": [
                        {
                            "mesh": g.mesh,
                            "pos": [float(v) for v in g.pos],
                            "quat": to_threejs_quat(g.quat),
                        }
                        for g in b.geoms
                    ],
                }
                for b in self.bodies
            ],
            "joints": [
                {
                    "name": j.name,
                    "body": j.body,
                    "axis": [float(v) for v in j.axis],
                    "pos": [float(v) for v in j.pos],
                    "type": j.type,
                }
                for j in self.joints
            ],
            "joint_order": list(self.joint_order) if self.joint_order else None,
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> "FlyRig":
        """Rebuild a rig from ``rig.json``, so the export can be tested as read."""
        bodies = [
            Body(
                name=b["name"],
                parent=int(b["parent"]),
                pos=np.asarray(b["pos"], dtype=float),
                quat=from_threejs_quat(b["quat"]),
                geoms=tuple(
                    Geom(
                        mesh=g["mesh"],
                        pos=np.asarray(g["pos"], dtype=float),
                        quat=from_threejs_quat(g["quat"]),
                    )
                    for g in b.get("geoms", ())
                ),
                joints=tuple(int(i) for i in b.get("joints", ())),
            )
            for b in data["bodies"]
        ]
        joints = [
            Joint(
                name=j["name"],
                body=int(j["body"]),
                axis=np.asarray(j["axis"], dtype=float),
                pos=np.asarray(j["pos"], dtype=float),
                type=j.get("type", "hinge"),
            )
            for j in data["joints"]
        ]
        order = data.get("joint_order")
        return cls(
            bodies=bodies,
            joints=joints,
            meshes={},
            joint_order=tuple(order) if order else None,
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class MeshAsset:
    """An ``<asset><mesh>`` entry: where the STL is and what it is scaled by."""

    name: str
    path: Path
    scale: np.ndarray


# ------------------------------------------------------------------ parsing


def default_mjcf_path(variant: str = MJCF_VARIANT) -> Path:
    """Locate FlyGym's MJCF, rather than hard-coding a site-packages path."""
    try:
        import flygym_gymnasium
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "the body geometry lives in FlyGym's data directory:\n"
            "    pip install flygym-gymnasium\n"
            "or pass an explicit path to parse_mjcf()."
        ) from exc
    root = Path(flygym_gymnasium.__file__).parent / "data"
    names = {
        "seqik": "neuromechfly_seqik_kinorder_ypr.xml",
        "deepfly3d": "neuromechfly_deepfly3d_kinorder_ryp.xml",
        "seqik_simple": "neuromechfly_seqik_kinorder_ypr_capsuletarsus.xml",
    }
    if variant not in names:
        raise KeyError(f"unknown MJCF variant {variant!r}; have {sorted(names)}")
    return root / "mjcf" / names[variant]


def parse_mjcf(path: str | Path | None = None) -> FlyRig:
    """Read the MJCF into a :class:`FlyRig`.

    Only what the kinematics and the drawing need is read: the body tree, the
    hinges, the mesh assets and the geoms that instance them. Masses, contact
    parameters, actuators and sensors are deliberately dropped -- this rig is
    for looking at a logged trajectory, not for re-simulating one.

    Note that the tree here has 70 bodies while MuJoCo's *compiled* model of the
    same file reports 55: the file sets ``fusestatic="true"``, so bodies with no
    joints (thorax, wings, halteres, abdomen) are folded into their parents at
    compile time. Keeping them separate is the point of an exporter -- they are
    separate meshes to draw -- and it makes no difference to the kinematics,
    which is why the MuJoCo comparison test can only check the 55 that survive
    unless the model is built through FlyGym, which does not fuse.
    """
    path = Path(path) if path is not None else default_mjcf_path()
    root = ET.parse(path).getroot()

    meshes: dict[str, MeshAsset] = {}
    asset = root.find("asset")
    for mesh in [] if asset is None else asset.findall("mesh"):
        name = mesh.get("name")
        file = mesh.get("file")
        if name is None or file is None:
            continue
        meshes[name] = MeshAsset(
            name=name,
            path=(path.parent / file).resolve(),
            scale=_vec(mesh.get("scale"), (1.0, 1.0, 1.0)),
        )

    bodies: list[Body] = []
    joints: list[Joint] = []

    def walk(element: ET.Element, parent: int) -> None:
        for node in element.findall("body"):
            index = len(bodies)
            mine: list[int] = []
            for joint in node.findall("joint"):
                jtype = joint.get("type", "hinge")
                if jtype != "hinge":
                    # Slides and balls would need their own branch in the FK and
                    # in the browser; this model has none, so refuse rather than
                    # export something that poses wrongly.
                    raise NotImplementedError(
                        f"joint {joint.get('name')!r} is a {jtype}; only hinges are exported"
                    )
                mine.append(len(joints))
                joints.append(
                    Joint(
                        name=joint.get("name", f"joint_{len(joints)}"),
                        body=index,
                        axis=_vec(joint.get("axis"), (0.0, 0.0, 1.0)),
                        pos=_vec(joint.get("pos"), (0.0, 0.0, 0.0)),
                        type=jtype,
                    )
                )
            geoms = tuple(
                Geom(
                    mesh=g.get("mesh", ""),
                    pos=_vec(g.get("pos"), (0.0, 0.0, 0.0)),
                    quat=_unit_quat(g.get("quat")),
                )
                for g in node.findall("geom")
                if g.get("type") == "mesh" and g.get("mesh")
            )
            bodies.append(
                Body(
                    name=node.get("name", f"body_{index}"),
                    parent=parent,
                    pos=_vec(node.get("pos"), (0.0, 0.0, 0.0)),
                    quat=_unit_quat(node.get("quat")),
                    geoms=geoms,
                    joints=tuple(mine),
                )
            )
            walk(node, index)

    world = root.find("worldbody")
    if world is None:
        raise ValueError(f"{path} has no <worldbody>")
    walk(world, -1)
    return FlyRig(bodies=bodies, joints=joints, meshes=meshes, source=path)


# -------------------------------------------------------------- joint order


def observation_joint_order() -> tuple[str, ...]:
    """The 42 names a logged ``obs["joints"]`` vector is in, taken from FlyGym.

    ``Fly.__init__`` defaults both ``actuated_joints`` and ``monitored_joints``
    to ``preprogrammed.all_leg_dofs``, and ``Fly.get_observation`` reindexes the
    sensor readings by ``_monitored_joint_order`` into *monitored* order -- so
    with the defaults this project uses, column *i* of the observation is
    ``all_leg_dofs[i]``. Read from the package rather than reconstructed here:
    if FlyGym reorders its DoFs, the export should follow, and the test asserts
    the two still agree.

    The caveat that goes with that: a ``Fly`` built with a non-default
    ``monitored_joints`` logs a different order, and this export would then be
    describing a vector nobody produced.
    """
    try:
        from flygym_gymnasium import preprogrammed
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "the observation joint order is FlyGym's, not ours:\n"
            "    pip install flygym-gymnasium"
        ) from exc
    return tuple(preprogrammed.all_leg_dofs)


# --------------------------------------------------------------------- STL


def load_stl(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a binary STL into indexed geometry.

    STL is a triangle soup: every triangle carries its own three vertices, so a
    closed mesh stores each vertex about six times. Browsers want an index
    buffer, and deduplicating here rather than there roughly halves the wire
    size and lets the GPU reuse the post-transform vertex cache.

    Deduplication is by exact float32 bit pattern. That is safe for these files
    because they were written from a mesh that already had shared vertices --
    the same coordinates come back byte-identical -- and it is the conservative
    choice: a tolerance would weld distinct vertices that happen to sit close,
    which on the tarsi (0.2 mm segments) is a real risk.

    Returns ``(vertices (n, 3) float64, faces (m, 3) int64)`` in the STL's own
    units; the caller applies the MJCF asset scale.
    """
    path = Path(path)
    raw = path.read_bytes()
    if len(raw) < 84:
        raise ValueError(f"{path} is too short to be a binary STL")
    (count,) = struct.unpack("<I", raw[80:84])
    expected = 84 + 50 * count
    if len(raw) != expected:
        # An ASCII STL beginning "solid" would sail past the header check and
        # decode as noise, so the length is the discriminator, not the magic.
        raise ValueError(
            f"{path} is not a binary STL: header claims {count} triangles "
            f"({expected} bytes) but the file is {len(raw)} bytes"
        )
    # 50 bytes per triangle: 3 floats of normal, 9 of vertices, 2 of attribute
    # byte count. Slicing as bytes and re-viewing skips the normals without an
    # intermediate copy of the whole array.
    records = np.frombuffer(raw, dtype=np.uint8, count=50 * count, offset=84)
    records = records.reshape(count, 50)
    soup = records[:, 12:48].copy().view("<f4").reshape(count * 3, 3)
    vertices, inverse = np.unique(soup, axis=0, return_inverse=True)
    faces = np.asarray(inverse).reshape(count, 3).astype(np.int64)
    return vertices.astype(np.float64), faces


def apply_scale(
    vertices: np.ndarray, faces: np.ndarray, scale: Sequence[float]
) -> tuple[np.ndarray, np.ndarray]:
    """Scale a mesh, flipping triangle winding when the scale is a mirror.

    Thirty of NeuroMechFly's 69 mesh assets are the *other* side's STL with
    ``scale="1000 -1000 1000"`` -- the model ships one coxa and mirrors it. A
    mirror reverses orientation, so leaving the winding alone turns every left
    leg inside out: back-face culling hides it, and ``computeVertexNormals()``
    aims its normals into the mesh, so it lights as a dark hole. MuJoCo reverses
    the face order itself when the scale determinant is negative; this does the
    same thing, which is why the export can be drawn with the default
    ``FrontSide`` material.

    Baking the mirror into the vertices costs a duplicate copy of each mirrored
    part -- about half of ``meshes.bin``. The alternative, one buffer per
    distinct STL plus a negative per-geom scale, moves exactly this winding
    problem into the browser, where it has to be solved again in GLSL.
    """
    scale = np.asarray(scale, dtype=float)
    vertices = vertices * scale
    if float(np.prod(scale)) < 0.0:
        faces = faces[:, ::-1]
    return vertices, np.ascontiguousarray(faces)


# --------------------------------------------------------------- decimation


def cluster_decimate(
    vertices: np.ndarray, faces: np.ndarray, cells: int
) -> tuple[np.ndarray, np.ndarray]:
    """Vertex clustering: snap vertices to a grid, keep one point per cell.

    The cheapest decimator that preserves topology-ish appearance at a distance,
    which is all a 3 mm fly rendered at a few hundred pixels needs. It is not
    quadric simplification: it will flatten a thin feature whose two sides fall
    in the same cell, and it makes no attempt to preserve the silhouette. The
    grid is sized per mesh, from that mesh's own bounding box, so a 0.2 mm
    tarsus segment keeps the same *relative* detail as the thorax instead of
    being dissolved by a grid picked for a body twenty times its size.

    Triangles whose corners collapse into one or two cells are dropped; that is
    where most of the saving comes from.
    """
    lo = vertices.min(axis=0)
    extent = vertices.max(axis=0) - lo
    step = float(extent.max()) / max(int(cells), 1)
    if step <= 0.0:
        return vertices, faces
    grid = np.floor((vertices - lo) / step).astype(np.int64)
    _, inverse = np.unique(grid, axis=0, return_inverse=True)
    inverse = np.asarray(inverse).reshape(-1)
    n_clusters = int(inverse.max()) + 1
    # Cluster representative is the mean of its members, which keeps flat
    # regions flat; the nearest-to-centre alternative costs a second pass and
    # looks no better at this size.
    sums = np.zeros((n_clusters, 3))
    np.add.at(sums, inverse, vertices)
    counts = np.bincount(inverse, minlength=n_clusters).astype(float)
    new_vertices = sums / counts[:, None]
    new_faces = inverse[faces]
    keep = (
        (new_faces[:, 0] != new_faces[:, 1])
        & (new_faces[:, 1] != new_faces[:, 2])
        & (new_faces[:, 0] != new_faces[:, 2])
    )
    new_faces = new_faces[keep]
    if new_faces.size == 0:
        return vertices, faces
    used, remapped = np.unique(new_faces, return_inverse=True)
    return new_vertices[used], np.asarray(remapped).reshape(-1, 3).astype(np.int64)


def _packed_bytes(meshes: Mapping[str, tuple[np.ndarray, np.ndarray]]) -> int:
    return sum(v.shape[0] * 12 + f.shape[0] * 12 for v, f in meshes.values())


def _decimate_all(
    meshes: Mapping[str, tuple[np.ndarray, np.ndarray]], cells: int
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return {k: cluster_decimate(v, f, cells) for k, (v, f) in meshes.items()}


def _choose_resolution(
    meshes: Mapping[str, tuple[np.ndarray, np.ndarray]],
    *,
    budget_bytes: int | None,
    max_triangles: int | None,
) -> tuple[int, dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Largest grid resolution that still fits. Bisection, because each probe
    costs a full clustering pass over half a million triangles."""

    def fits(result: Mapping[str, tuple[np.ndarray, np.ndarray]]) -> bool:
        if budget_bytes is not None and _packed_bytes(result) > budget_bytes:
            return False
        if max_triangles is not None:
            if sum(f.shape[0] for _, f in result.values()) > max_triangles:
                return False
        return True

    lo, hi = _CLUSTER_MIN, _CLUSTER_MAX
    best_cells, best = lo, _decimate_all(meshes, lo)
    if not fits(best):
        raise ValueError(
            f"even at {lo} cells per axis the export does not fit; raise the budget"
        )
    while lo < hi:
        mid = (lo + hi + 1) // 2
        candidate = _decimate_all(meshes, mid)
        if fits(candidate):
            lo, best_cells, best = mid, mid, candidate
        else:
            hi = mid - 1
    return best_cells, best


# ------------------------------------------------------------------ export


def export_rig(
    out_dir: str | Path,
    *,
    decimate_to: int | None = None,
    mjcf: str | Path | None = None,
) -> dict:
    """Write ``rig.json`` and ``meshes.bin`` into ``out_dir``; return the manifest.

    ``decimate_to`` is a target *total triangle count*. ``None`` means "as much
    detail as fits in :data:`MESH_BUDGET_BYTES`" -- which is not the same as no
    decimation: the raw meshes index out at 8.6 MB, over the 8 MB ceiling, so
    the default path always clusters a little. Pass a number at or above the raw
    count to keep the meshes untouched, and check the resulting size yourself.

    ``meshes.bin`` is one flat buffer: for each mesh, ``vertexCount * 3``
    float32 positions followed by ``indexCount`` uint32 indices, both recorded
    in ``rig.json`` as byte offsets into the file. Every element is 4 bytes and
    offsets accumulate in multiples of 4, so the typed-array views the browser
    makes over the ``ArrayBuffer`` are aligned without any copying::

        const rig = await (await fetch("rig.json")).json();
        const buf = await (await fetch("meshes.bin")).arrayBuffer();
        const m = rig.meshes["mesh_Thorax"];
        const g = new THREE.BufferGeometry();
        g.setAttribute("position", new THREE.BufferAttribute(
            new Float32Array(buf, m.positionOffset, m.vertexCount * 3), 3));
        g.setIndex(new THREE.BufferAttribute(
            new Uint32Array(buf, m.indexOffset, m.indexCount), 1));
        g.computeVertexNormals();

    Positions are float32: the fly is 3 mm across and float32 resolves about
    2e-7 mm there, four orders below anything visible. The *kinematics* stay
    float64 in ``rig.json``, where the error would accumulate down a leg.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rig = parse_mjcf(mjcf)
    rig.joint_order = observation_joint_order()

    missing = [n for n in rig.joint_order if n not in {j.name for j in rig.joints}]
    if missing:
        raise ValueError(
            f"FlyGym names {len(missing)} actuated joints the MJCF does not have "
            f"({missing[:3]}...); the wrong XML variant is being exported"
        )

    used = sorted({g.mesh for b in rig.bodies for g in b.geoms})
    raw: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name in used:
        asset = rig.meshes[name]
        vertices, faces = load_stl(asset.path)
        raw[name] = apply_scale(vertices, faces, asset.scale)

    raw_vertices = sum(v.shape[0] for v, _ in raw.values())
    raw_faces = sum(f.shape[0] for _, f in raw.values())

    if decimate_to is not None and decimate_to >= raw_faces:
        cells, packed = None, dict(raw)
    else:
        cells, packed = _choose_resolution(
            raw,
            budget_bytes=None if decimate_to is not None else MESH_BUDGET_BYTES,
            max_triangles=decimate_to,
        )

    blob = bytearray()
    index: dict[str, dict] = {}
    for name in used:
        vertices, faces = packed[name]
        entry = {
            "vertexCount": int(vertices.shape[0]),
            "triangleCount": int(faces.shape[0]),
            "positionOffset": len(blob),
        }
        blob += np.ascontiguousarray(vertices, dtype=np.float32).tobytes()
        entry["indexOffset"] = len(blob)
        entry["indexCount"] = int(faces.size)
        blob += np.ascontiguousarray(faces, dtype=np.uint32).tobytes()
        index[name] = entry

    kept_vertices = sum(e["vertexCount"] for e in index.values())
    kept_faces = sum(e["triangleCount"] for e in index.values())
    manifest = {
        "format": "flyloop-fly-rig",
        "version": 1,
        "units": "mm",
        "quaternion_order": "xyzw",
        "source": str(rig.source),
        "meshes_file": "meshes.bin",
        "meshes": index,
        "decimation": {
            "method": None if cells is None else "vertex-clustering",
            "cells_per_longest_axis": cells,
            "vertices": {"before": raw_vertices, "after": kept_vertices},
            "triangles": {"before": raw_faces, "after": kept_faces},
            "triangle_fraction_kept": round(kept_faces / raw_faces, 4),
            "mesh_bytes": len(blob),
        },
        **rig.to_dict(),
    }

    (out_dir / "meshes.bin").write_bytes(bytes(blob))
    (out_dir / "rig.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return manifest


def load_exported_rig(out_dir: str | Path) -> FlyRig:
    """Read back a ``rig.json`` as a :class:`FlyRig`, meshes left on disk.

    The point of going through the file rather than reusing the in-memory rig is
    that it exercises the serialisation -- notably the quaternion reordering,
    which is the one place a sign or an axis swap can hide and still look like a
    fly.
    """
    data = json.loads((Path(out_dir) / "rig.json").read_text(encoding="utf-8"))
    rig = FlyRig.from_dict(data)
    rig.metadata = {k: v for k, v in data.items() if k not in ("bodies", "joints")}
    return rig
