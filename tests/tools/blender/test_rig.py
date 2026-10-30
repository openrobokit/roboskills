"""A motion's world moves become Blender pose matrices, and render_motion's steps, without Blender in tests."""

import numpy as np
from scipy.spatial.transform import Rotation

from roboskills.tools.blender import blender, rig


def rigid(rng, count):
    out = np.tile(np.eye(4), (count, 1, 1))
    out[:, :3, :3], out[:, :3, 3] = Rotation.random(count, rng).as_matrix(), rng.normal(size=(count, 3))
    return out


def test_bases_pose_the_bones_where_the_moves_take_them():
    rng = np.random.default_rng(0)
    parents = np.array([-1, 0, 1, 1])
    rests, armature = rigid(rng, 4), rigid(rng, 1)[0] @ np.diag([0.01, 0.01, 0.01, 1])  # Mixamo's scale
    moves = rigid(rng, 8).reshape(2, 4, 4, 4)
    moves[:, 3] = moves[:, 1]  # a bone the motion does not name moves with its parent
    pose = rig.bases(moves, rests, parents, armature)
    posed = np.empty_like(pose)
    for b, p in enumerate(parents):  # Blender's chain
        above = posed[:, p] @ np.linalg.inv(rests[p]) if p >= 0 else np.eye(4)
        posed[:, b] = above @ rests[b] @ pose[:, b]
    assert np.allclose(armature @ posed, moves @ armature @ rests, atol=1e-8)
    assert np.allclose(pose[:, 3], np.eye(4))


def test_a_still_motion_keeps_the_rest_pose():
    rng = np.random.default_rng(1)
    rests, turns = rigid(rng, 3), np.tile(np.eye(3), (1, 3, 1, 1))
    joints = rng.normal(size=(3, 3))
    moves = rig.rigid(turns, joints[None], joints)
    pose = rig.bases(moves, rests, np.array([-1, 0, 1]), rigid(rng, 1)[0])
    assert np.allclose(pose, np.eye(4))


def test_rigid_turns_a_bone_about_its_rest_joint():
    turn = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    move = rig.rigid(turn[None, None], np.array([[[1.0, 1, 0]]]), np.array([[1.0, 0, 0]]))[0, 0]
    assert np.allclose(move[:3, :3] @ [1.0, 0, 0] + move[:3, 3], [1, 1, 0])  # the joint lands on its posed place
    assert np.allclose(move[:3, :3] @ [2.0, 0, 0] + move[:3, 3], [1, 2, 0])  # and what hung off it turned with it


def test_render_motion_bakes_renders_fresh_then_encodes(monkeypatch, tmp_path):
    np.savez(tmp_path / "motion.npz", fps=30.0)
    (tmp_path / "out/frames").mkdir(parents=True)
    (tmp_path / "out/frames/0000.png").write_text("stale")
    runs = []
    monkeypatch.setattr(blender.subprocess, "run", lambda cmd, **kw: runs.append([str(x) for x in cmd]))
    blender.render_motion(tmp_path / "c.glb", tmp_path / "motion.npz", tmp_path / "v.mp4", tmp_path / "out", "eevee")
    assert not (tmp_path / "out/frames").exists()
    assert runs[0][3].endswith("rig.py") and runs[0][4] == "animate" and runs[0][-1] == str(tmp_path / "out/scene.glb")
    assert runs[1][3].endswith("motion_studio.py") and "--final" in runs[1] and runs[1][-1] == "eevee"
    assert runs[2][-1].endswith("video.mp4") and runs[3][-1].endswith("compare.mp4")
