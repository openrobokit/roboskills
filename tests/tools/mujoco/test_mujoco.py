"""The MANO hand's IK reproduces keypoints of known poses, warm-started frame to frame, and holds unseen frames; DROID's
recorded robot replays on the Panda and 2F-85 with the gripper's linkage closed."""

import json

import numpy as np
import pytest

from roboskills.eval.droid.read import read_robot
from roboskills.tools.mujoco.mujoco import FRANKA, KEYPOINTS, MANO, fit_hand


mujoco = pytest.importorskip("mujoco")


def test_fit_hand_recovers_known_keypoints_and_holds_unseen_frames():
    model = mujoco.MjModel.from_xml_path(str(MANO))
    data, (low, high) = mujoco.MjData(model), model.jnt_range.T
    poses = []
    for i in range(20):  # a hand turning and moving while its fingers flex
        q = np.r_[[0.3 + 0.003 * i, -0.2, 0.15, 2.0 + 0.01 * i, -0.7, 1.2], np.zeros(model.nq - 6)]
        q[6:] = low[6:] + (high[6:] - low[6:]) * (0.3 + 0.2 * np.sin(i / 7 + np.arange(model.nq - 6)))
        poses.append(q)

    def keypoints(q):
        data.qpos[:] = q
        mujoco.mj_kinematics(model, data)
        return np.array(
            [
                data.xpos[model.body(n).id] if kind == "body" else data.site_xpos[model.site(n).id]
                for kind, n in KEYPOINTS
            ]
        )

    joints = np.array([keypoints(q) for q in poses])
    joints[[0, 10, 11]] = np.nan  # unseen
    fitted = fit_hand(joints)
    assert max(np.abs(keypoints(q) - j).max() for q, j in zip(fitted, joints) if np.isfinite(j).all()) < 1e-4
    np.testing.assert_array_equal(fitted[0], fitted[1])  # before the first seen frame: the first seen pose
    np.testing.assert_array_equal(fitted[11], fitted[9])  # a gap holds the last seen pose


def test_read_robot_replays_droid_joints_with_a_closed_gripper_linkage(tmp_path):
    arm = np.array([[0, -0.4, 0.2, -2.4, -0.1, 1.9, 0.4], [0.3, 0.2, 0.0, -1.8, 0.2, 2.2, -0.5]])
    np.savez(tmp_path / "robot.npz", joint_position=arm, gripper_position=np.array([0.0, 1.0]))
    world = np.eye(4)
    world[:3, 3] = [0.3, 0.6, 0.6]
    (tmp_path / "camera.json").write_text(json.dumps(dict(extrinsics=dict(T_base_from_camera=world.tolist()))))
    qpos, read = read_robot(tmp_path / "robot.npz", tmp_path / "camera.json")
    np.testing.assert_array_equal(read, world)
    model = mujoco.MjModel.from_xml_path(str(FRANKA))
    data, flange = mujoco.MjData(model), []
    assert qpos.shape == (2, model.nq)
    for data.qpos[:] in qpos:
        mujoco.mj_forward(model, data)
        connect = data.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY
        assert np.abs(data.efc_pos[connect]).max() < 2e-3  # the 2F-85's four-bar linkage holds, open and closed
        flange.append(data.body("base_mount").xpos.copy())
    assert np.linalg.norm(flange[1] - flange[0]) > 0.1  # the arm moved
