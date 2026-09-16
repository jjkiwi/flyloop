"""The exported body rig, checked against the simulator it was taken from.

A geometry exporter is unusually easy to get wrong without noticing, because its
output is a picture: a rig with two hinges swapped still renders a fly, still
animates, and still looks like it is walking. So the load-bearing test here is
not that the file parses -- it is
:func:`test_exported_rig_matches_mujoco_forward_kinematics`, which poses the
exported rig and MuJoCo's own model with the same angles and compares every body
position. Everything else in this file exists to localise a failure once that
one goes red.
"""

import json
import os
import struct

import numpy as np
import pytest

from flyloop.viz.fly_geometry import (
    FlyRig,
    apply_scale,
    cluster_decimate,
    default_mjcf_path,
    export_rig,
    from_threejs_quat,
    load_exported_rig,
    load_stl,
    observation_joint_order,
    parse_mjcf,
    quat_mul,
    quat_rotate,
    to_threejs_quat,
)

# The MJCF and the meshes ship inside FlyGym, so without it there is nothing to
# export; and this machine has no EGL or OSMesa, so anything that touches MuJoCo
# has to say so before the first import.
pytest.importorskip("flygym_gymnasium", reason="the body geometry ships with FlyGym 1.x")
os.environ.setdefault("MUJOCO_GL", "disable")

#: The counts in the shipped seqik MJCF. Pinned rather than recomputed: if a
#: FlyGym upgrade changes the body plan, this file should fail loudly instead of
#: quietly exporting a different animal.
N_BODIES, N_JOINTS, N_MESH_ASSETS, N_ACTUATED = 70, 87, 69, 42


@pytest.fixture(scope="module")
def rig():
    return parse_mjcf()


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    """One export for the whole module -- it reads 23 MB of STL and clusters it."""
    out = tmp_path_factory.mktemp("rig")
    manifest = export_rig(out)
    return out, manifest


# ---------------------------------------------------------------- the tree


def test_parse_reads_the_whole_body_plan(rig):
    assert len(rig.bodies) == N_BODIES
    assert len(rig.joints) == N_JOINTS
    assert len({g.mesh for b in rig.bodies for g in b.geoms}) == N_MESH_ASSETS
    assert all(j.type == "hinge" for j in rig.joints)


def test_parents_precede_children(rig):
    """A browser applies the tree in one pass, so document order has to be valid."""
    assert rig.bodies[0].parent == -1
    for i, body in enumerate(rig.bodies):
        assert body.parent < i


def test_every_joint_points_back_at_its_body(rig):
    for i, body in enumerate(rig.bodies):
        for ji in body.joints:
            assert rig.joints[ji].body == i
    listed = [ji for b in rig.bodies for ji in b.joints]
    assert sorted(listed) == list(range(N_JOINTS))


def test_joints_on_one_body_keep_document_order(rig):
    """The coxa carries three hinges and composing them out of order is wrong.

    Document order is yaw, pitch, roll for a coxa, and the MJCF relies on it:
    each axis is read in the frame the previous joints left behind.
    """
    coxa = rig.bodies[rig.body_index("LFCoxa")]
    assert [rig.joints[j].name for j in coxa.joints] == [
        "joint_LFCoxa_yaw",
        "joint_LFCoxa",
        "joint_LFCoxa_roll",
    ]


def test_body_quaternions_are_normalised(rig):
    """The MJCF's are not -- A1A2's is off unit length by 3e-5."""
    for body in rig.bodies:
        assert np.linalg.norm(body.quat) == pytest.approx(1.0, abs=1e-12)


# --------------------------------------------------------- quaternion order


def test_threejs_quaternion_conversion_round_trips():
    q = np.array([0.5, -0.5, 0.5, 0.5])
    assert to_threejs_quat(q) == [-0.5, 0.5, 0.5, 0.5]
    assert from_threejs_quat(to_threejs_quat(q)) == pytest.approx(q)


def test_quaternion_helpers_agree_with_a_known_rotation():
    """Guards the (w, x, y, z) convention itself, not just its round trip."""
    q = np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)])  # +90 deg about z
    assert quat_rotate(q, np.array([1.0, 0.0, 0.0])) == pytest.approx([0, 1, 0], abs=1e-12)
    twice = quat_mul(q, q)
    assert quat_rotate(twice, np.array([1.0, 0.0, 0.0])) == pytest.approx([-1, 0, 0], abs=1e-12)


# ---------------------------------------------------------------- the STL


def test_load_stl_matches_the_files_own_triangle_count(rig):
    asset = rig.meshes["mesh_Thorax"]
    header_count = struct.unpack("<I", asset.path.read_bytes()[80:84])[0]
    vertices, faces = load_stl(asset.path)
    assert faces.shape == (header_count, 3)
    assert faces.min() >= 0 and faces.max() < len(vertices)


def test_load_stl_deduplicates_the_triangle_soup(rig):
    """A closed mesh repeats each vertex about six times; indexing is the saving."""
    vertices, faces = load_stl(rig.meshes["mesh_Thorax"].path)
    assert len(vertices) < faces.size / 3
    # No vertex left unreferenced, or the index buffer is describing a different
    # mesh from the position buffer.
    assert len(np.unique(faces)) == len(vertices)


def test_load_stl_rejects_a_file_that_is_not_a_binary_stl(tmp_path):
    ascii_stl = tmp_path / "fake.stl"
    ascii_stl.write_text("solid fake\nfacet normal 0 0 1\nendsolid fake\n")
    with pytest.raises(ValueError, match="binary STL"):
        load_stl(ascii_stl)


def _signed_volume(vertices, faces):
    a, b, c = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


def test_mirrored_assets_keep_their_faces_pointing_outwards(rig):
    """Thirty assets are the other side's STL under a negative scale.

    Without the winding flip the left half of the fly is inside out: it vanishes
    under back-face culling and lights as a black hole where it does not. The
    signed volume is the cheap way to see it -- negative means inverted.
    """
    mirrored = rig.meshes["mesh_LFCoxa"]
    assert float(np.prod(mirrored.scale)) < 0.0
    vertices, faces = load_stl(mirrored.path)
    assert _signed_volume(*apply_scale(vertices, faces, mirrored.scale)) > 0.0
    plain = rig.meshes["mesh_RFCoxa"]
    vertices, faces = load_stl(plain.path)
    assert _signed_volume(*apply_scale(vertices, faces, plain.scale)) > 0.0


# ------------------------------------------------------------- decimation


def test_clustering_reduces_the_mesh_and_leaves_it_valid(rig):
    vertices, faces = load_stl(rig.meshes["mesh_Thorax"].path)
    coarse_v, coarse_f = cluster_decimate(vertices, faces, 24)
    assert len(coarse_v) < len(vertices)
    assert len(coarse_f) < len(faces)
    assert coarse_f.min() >= 0 and coarse_f.max() < len(coarse_v)
    # Clustering snaps vertices to a grid, so the hull may shrink by up to a
    # cell but must not wander outside the original bounding box.
    cell = (vertices.max(axis=0) - vertices.min(axis=0)).max() / 24
    assert (coarse_v.min(axis=0) >= vertices.min(axis=0) - cell).all()
    assert (coarse_v.max(axis=0) <= vertices.max(axis=0) + cell).all()


def test_clustering_keeps_triangles_non_degenerate(rig):
    _, faces = cluster_decimate(*load_stl(rig.meshes["mesh_RFTarsus3"].path), 20)
    assert (faces[:, 0] != faces[:, 1]).all()
    assert (faces[:, 1] != faces[:, 2]).all()
    assert (faces[:, 0] != faces[:, 2]).all()


# ---------------------------------------------------------------- the export


def test_export_fits_the_budget(exported):
    out, manifest = exported
    total = sum(p.stat().st_size for p in out.iterdir())
    assert total < 8_000_000, f"export is {total / 1e6:.2f} MB"
    triangles = manifest["decimation"]["triangles"]
    assert triangles["after"] < triangles["before"]
    assert 0.0 < manifest["decimation"]["triangle_fraction_kept"] <= 1.0


def test_mesh_index_covers_the_buffer_exactly(exported):
    """Every byte of meshes.bin is claimed by exactly one mesh, and the typed
    array views the browser makes over it are 4-byte aligned."""
    out, manifest = exported
    blob = (out / "meshes.bin").read_bytes()
    spans = []
    for entry in manifest["meshes"].values():
        assert entry["indexCount"] == entry["triangleCount"] * 3
        assert entry["positionOffset"] % 4 == 0
        assert entry["indexOffset"] % 4 == 0
        spans.append((entry["positionOffset"], entry["vertexCount"] * 12))
        spans.append((entry["indexOffset"], entry["indexCount"] * 4))
    spans.sort()
    cursor = 0
    for start, length in spans:
        assert start == cursor
        cursor += length
    assert cursor == len(blob) == manifest["decimation"]["mesh_bytes"]


def test_exported_indices_stay_inside_their_own_vertex_buffer(exported):
    out, manifest = exported
    blob = (out / "meshes.bin").read_bytes()
    for name, entry in manifest["meshes"].items():
        idx = np.frombuffer(
            blob, dtype="<u4", count=entry["indexCount"], offset=entry["indexOffset"]
        )
        assert idx.max() < entry["vertexCount"], name


def test_every_geom_names_a_mesh_that_was_exported(exported):
    _, manifest = exported
    exported_names = set(manifest["meshes"])
    for body in manifest["bodies"]:
        for geom in body["geoms"]:
            assert geom["mesh"] in exported_names


def test_rig_json_declares_its_conventions(exported):
    out, _ = exported
    data = json.loads((out / "rig.json").read_text())
    assert data["quaternion_order"] == "xyzw"
    assert data["units"] == "mm"
    assert data["meshes_file"] == "meshes.bin"


# --------------------------------------------------------------- joint order


def test_exported_joint_order_is_flygyms_own(exported):
    """The 42-vector in a log is ordered by FlyGym, so the export must not guess.

    ``Fly.__init__`` defaults ``actuated_joints`` and ``monitored_joints`` to
    ``preprogrammed.all_leg_dofs``, and ``get_observation`` writes the joint
    block in monitored order; with those defaults the observation's column *i*
    is ``all_leg_dofs[i]``. This asserts the exported order is that list,
    element for element, and that the attribute is the one being read.
    """
    from flygym_gymnasium import Fly, preprogrammed

    _, manifest = exported
    fly = Fly()
    assert fly.actuated_joints == list(preprogrammed.all_leg_dofs)
    assert fly.monitored_joints == list(preprogrammed.all_leg_dofs)
    assert list(manifest["joint_order"]) == fly.actuated_joints
    assert len(manifest["joint_order"]) == N_ACTUATED
    assert observation_joint_order() == tuple(fly.actuated_joints)


def test_joint_order_names_real_joints_of_the_exported_tree(exported):
    _, manifest = exported
    names = {j["name"] for j in manifest["joints"]}
    assert set(manifest["joint_order"]) <= names
    # Every actuated joint drives exactly one body, and no two share one -- a
    # duplicate would mean a logged angle silently overwriting another.
    driven = [
        j["body"] for j in manifest["joints"] if j["name"] in set(manifest["joint_order"])
    ]
    assert len(driven) == N_ACTUATED


# -------------------------------------------------------- forward kinematics


def test_rig_round_trips_through_json(exported):
    """FK from the file must equal FK from memory, or the export lost something.

    The quaternion reordering is the reason this is a separate test: a swapped
    component survives serialisation, renders a fly, and bends wrongly.
    """
    out, _ = exported
    from_disk = load_exported_rig(out)
    in_memory = parse_mjcf()
    in_memory.joint_order = observation_joint_order()
    rng = np.random.default_rng(7)
    angles = rng.uniform(-0.8, 0.8, size=N_ACTUATED)
    a_pos, a_quat = from_disk.forward_kinematics(angles)
    b_pos, b_quat = in_memory.forward_kinematics(angles)
    assert np.abs(a_pos - b_pos).max() < 1e-12
    assert np.abs(a_quat - b_quat).max() < 1e-12


def test_zero_pose_leaves_every_body_at_its_mjcf_offset(exported):
    """With all angles zero the tree is just the chain of body frames."""
    out, _ = exported
    rig = load_exported_rig(out)
    pos, quat = rig.forward_kinematics()
    for i, body in enumerate(rig.bodies):
        if body.parent < 0:
            continue
        expect = pos[body.parent] + quat_rotate(quat[body.parent], body.pos)
        assert pos[i] == pytest.approx(expect, abs=1e-12)


def test_a_vector_and_a_mapping_place_the_same_pose(exported):
    out, _ = exported
    rig = load_exported_rig(out)
    rng = np.random.default_rng(11)
    angles = rng.uniform(-0.5, 0.5, size=N_ACTUATED)
    by_name = dict(zip(rig.joint_order, angles, strict=True))
    assert rig.forward_kinematics(angles)[0] == pytest.approx(
        rig.forward_kinematics(by_name)[0], abs=1e-15
    )


def test_a_vector_of_the_wrong_length_is_refused(exported):
    out, _ = exported
    rig = load_exported_rig(out)
    with pytest.raises(ValueError, match="42 angles"):
        rig.forward_kinematics(np.zeros(41))


@pytest.mark.parametrize("body_name", ["RFCoxa", "LFCoxa", "LFTibia"])
def test_mesh_vertices_land_where_mujoco_draws_them(rig, body_name):
    """Right kinematics with the mesh hung in the wrong place is still wrong.

    The exporter ignores MuJoCo's mesh recentring -- MuJoCo shifts each mesh onto
    its centre of mass and rotates it onto its principal axes, and folds the
    inverse into the compiled geom frame, so ``geom_xpos``/``geom_xmat`` already
    include it and ``mesh_pos``/``mesh_quat`` must *not* be applied again. The
    claim being checked is that the two cancel, leaving raw STL vertices under
    the geom's MJCF transform already correct. Checked at a posed, non-identity
    configuration and on a mirrored asset, by matching the two world point clouds
    nearest-neighbour: ``mesh_vert`` is float32 and reordered, so the sets can be
    compared but the orders cannot.
    """
    mujoco = pytest.importorskip("mujoco")
    from scipy.spatial import cKDTree

    model = mujoco.MjModel.from_xml_path(str(default_mjcf_path()))
    data = mujoco.MjData(model)
    rng = np.random.default_rng(17)
    angles = {}
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        angles[name] = float(rng.uniform(-0.8, 0.8))
        data.qpos[model.jnt_qposadr[i]] = angles[name]
    mujoco.mj_forward(model, data)

    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, body_name)
    mid = model.geom_dataid[gid]
    start, count = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
    local = model.mesh_vert[start : start + count].astype(float)
    theirs = data.geom_xpos[gid] + local @ data.geom_xmat[gid].reshape(3, 3).T

    body = rig.bodies[rig.body_index(body_name)]
    (geom,) = body.geoms
    vertices, faces = load_stl(rig.meshes[geom.mesh].path)
    vertices, _ = apply_scale(vertices, faces, rig.meshes[geom.mesh].scale)
    pos, quat = rig.forward_kinematics(angles)
    i = rig.body_index(body_name)
    ours = np.array(
        [pos[i] + quat_rotate(quat[i], geom.pos + quat_rotate(geom.quat, v)) for v in vertices]
    )

    assert len(ours) == len(theirs)
    distance, _ = cKDTree(theirs).query(ours)
    # float32 mesh storage in MuJoCo puts the floor at about 3e-8 mm.
    assert distance.max() < 1e-5, f"{body_name}: worst vertex off by {distance.max():.2e} mm"


def test_exported_rig_matches_mujoco_on_the_uncompiled_file(exported):
    """Same check as the slow test, against MuJoCo compiling the MJCF directly.

    This one is cheap -- no FlyGym model assembly -- but it can only see 55 of
    the 70 bodies: the file sets ``fusestatic="true"``, so jointless bodies
    (thorax, wings, halteres, abdominal segments) are folded into their parents
    at compile time and have no ``xpos`` of their own.
    """
    mujoco = pytest.importorskip("mujoco")

    out, _ = exported
    rig = load_exported_rig(out)
    model = mujoco.MjModel.from_xml_path(str(default_mjcf_path()))
    data = mujoco.MjData(model)
    joint_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)
    ]
    rng = np.random.default_rng(3)
    worst = 0.0
    for _ in range(4):
        angles = {
            n: float(a)
            for n, a in zip(joint_names, rng.uniform(-0.9, 0.9, model.njnt), strict=True)
        }
        data.qpos[:] = 0.0
        for i, name in enumerate(joint_names):
            data.qpos[model.jnt_qposadr[i]] = angles[name]
        mujoco.mj_forward(model, data)
        pos, _ = rig.forward_kinematics(angles)
        for i in range(model.nbody):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
            if name == "world":
                continue
            worst = max(worst, float(np.abs(data.xpos[i] - pos[rig.body_index(name)]).max()))
    assert worst < 1e-4, f"max body position error {worst:.3e} mm"


@pytest.mark.slow
def test_exported_rig_matches_mujoco_forward_kinematics(exported):
    """The whole point of the exporter: pose it and MuJoCo identically, compare.

    Built through FlyGym rather than from the raw file so that nothing is fused
    away and all 70 bodies can be checked -- and so that what is being validated
    is the model this project actually walks, actuators, spawn pose and all.

    The comparison is in the ``FlyBody`` frame. The fly is attached to the arena
    with a free joint, so its world position is wherever it was spawned; the rig
    deliberately does not model that (where the fly *is* comes from the log),
    and comparing world coordinates would only be measuring the spawn.
    """
    import mujoco
    from flygym_gymnasium import Fly, Simulation

    out, _ = exported
    rig = load_exported_rig(out)
    fly = Fly()
    sim = Simulation(flies=[fly], cameras=[])
    physics = sim.physics
    model, data = physics.model, physics.data

    hinges = [i for i in range(model.njnt) if model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE]
    prefix = f"{fly.name}/"
    root = model.name2id(prefix + "FlyBody", "body")
    # Every body of the fly, by the name it has in our rig.
    fly_bodies = [
        (i, model.id2name(i, "body")[len(prefix) :])
        for i in range(model.nbody)
        if (model.id2name(i, "body") or "").startswith(prefix)
    ]
    fly_bodies = [(i, n) for i, n in fly_bodies if n]
    assert len(fly_bodies) == N_BODIES

    rng = np.random.default_rng(5)
    worst = 0.0
    for _ in range(5):
        angles = rng.uniform(-0.9, 0.9, size=N_ACTUATED)
        for i in hinges:
            data.qpos[model.jnt_qposadr[i]] = 0.0
        for name, value in zip(rig.joint_order, angles, strict=True):
            physics.named.data.qpos[prefix + name] = value
        physics.forward()
        frame = data.xmat[root].reshape(3, 3)
        origin = data.xpos[root]
        pos, _ = rig.forward_kinematics(angles)
        for i, name in fly_bodies:
            local = frame.T @ (data.xpos[i] - origin)
            worst = max(worst, float(np.abs(local - pos[rig.body_index(name)]).max()))
    assert worst < 1e-4, f"max body position error {worst:.3e} mm"


def test_from_dict_rejects_a_rig_it_cannot_pose():
    """A rig with no joint order cannot be handed a bare logged vector."""
    bare = FlyRig.from_dict(
        {
            "bodies": [{"name": "root", "parent": -1, "pos": [0, 0, 0], "quat": [0, 0, 0, 1]}],
            "joints": [],
            "joint_order": None,
        }
    )
    with pytest.raises(ValueError, match="no joint order"):
        bare.forward_kinematics(np.zeros(3))
