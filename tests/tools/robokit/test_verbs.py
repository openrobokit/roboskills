"""The targets action builds from what the agent passes in (heuristic_grasp, grasp_poses, place_pose), without a planner, robot
or simulator."""

from types import SimpleNamespace

import numpy as np
import pytest
from robokit.xform.numpy import rot_tl_to_tf_mat, transform_points
from scipy.spatial.transform import Rotation

from roboskills.tools.robokit import action


HAND = SimpleNamespace(kind="jaw", open=[1.0], closed=[0.0])
MODEL = SimpleNamespace(robot=SimpleNamespace(arms={"left": SimpleNamespace(hand=HAND)}),
                        info={"left": SimpleNamespace(M=SimpleNamespace(root=SimpleNamespace(T_world_base=np.eye(4))))})  # fmt: skip


def test_place_sets_the_held_object_upright_at_the_spot_and_heading_from_any_hand_pose():
    rng = np.random.default_rng(27)
    shape = rng.uniform([-0.05, -0.02, -0.006], [0.05, 0.02, 0.006], (1000, 3))  # in the object frame
    closed = rot_tl_to_tf_mat(
        Rotation.from_euler("xyz", [110, 20, 13], degrees=True).as_matrix(), np.array([0.02, -0.01, 0.014])
    )
    T_tcp_obj = np.linalg.inv(closed)  # the object frame at the origin when the TCP closed at `closed`
    for angles in ([0, 0, 0], [30, 20, 70], [-45, -10, 165]):
        T = rot_tl_to_tf_mat(Rotation.from_euler("xyz", angles, degrees=True).as_matrix(), np.array([0.2, 0.1, 0.9]))
        P = transform_points(shape, T @ T_tcp_obj)
        for yaw in (0, 45, 179):
            p = action.place_pose(MODEL, "left", T, T_tcp_obj, P, [0.3, -0.1], yaw=yaw, hover=0.04, table_z=0.75)
            contact = p.tcp.copy()
            contact[2, 3] += action.PUSH
            X = contact @ T_tcp_obj
            assert np.allclose(X[:2, 3], [0.3, -0.1]) and np.isclose(X[2, 2], 1.0)
            assert abs((np.degrees(np.arctan2(X[1, 0], X[0, 0])) - yaw + 90) % 180 - 90) < 1e-6
            now = T @ T_tcp_obj  # it turns the least way round: at most 90 deg from its heading now
            turn = np.degrees(np.arctan2(X[1, 0], X[0, 0]) - np.arctan2(now[1, 0], now[0, 0]))
            assert abs((turn + 180) % 360 - 180) <= 90 + 1e-6
            assert np.isclose(np.percentile(transform_points(shape, X)[:, 2], 0.5), 0.75)
            assert np.allclose(p.ee_pre[:3, 3] - contact[:3, 3], [0, 0, 0.04]) and p.hand.tolist() == [1.0]
    with pytest.raises(ValueError, match="shape is unknown"):
        action.place_pose(MODEL, "left", np.eye(4), np.eye(4), np.zeros((5, 3)), [0.3, -0.1], table_z=0.75)
    with pytest.raises(ValueError, match="no surface height"):
        action.place_pose(MODEL, "left", np.eye(4), np.eye(4), shape, [0.3, -0.1])


def test_grasp_pose_approaches_along_tcp_x_with_the_hand_open():
    T = rot_tl_to_tf_mat(Rotation.from_euler("y", 90, degrees=True).as_matrix(), np.array([0.5, 0.0, 0.8]))  # x down
    p = action.heuristic_grasp(MODEL, "left", T, [0.2])
    np.testing.assert_allclose(p.ee_pre[:3, 3] - T[:3, 3], [0, 0, action.PRE_GRASP], atol=1e-9)
    assert p.verb == "grasp" and p.hand_pre.tolist() == [1.0] and p.hand.tolist() == [0.2]
    assert action.heuristic_grasp(MODEL, "left", T).hand.tolist() == [0.0]
    with pytest.raises(ValueError, match=r"hand \(1,\)"):
        action.heuristic_grasp(MODEL, "left", T, [0.1, 0.2])


def test_grasp_poses_keep_the_scene_clear_of_the_object_and_rotate_by_variant(monkeypatch):
    seen = {}

    def candidates(S, arm, P, tz, obstacles, target, avoid, axis, cone, seed):
        seen.update(obstacles=np.concatenate([obstacles, avoid]), axis=axis)
        return [action.Pose("grasp", arm, rank=k) for k in range(3)]

    monkeypatch.setitem(__import__("sys").modules, "fake_backend", SimpleNamespace(candidates=candidates))
    monkeypatch.setenv("GRASP_BACKEND", "fake_backend")
    P = np.array([[0.5, -0.01, 0.8], [0.5, 0.01, 0.8]])
    scene = np.vstack([P, [[0.5, 0.0, 0.7], [0.8, 0.0, 0.9]]])  # the object, the table, something beside it
    avoid = np.array([[0.0, 0.5, 0.9]])
    poses = action.grasp_poses(MODEL, "left", P, scene, 0.75, approach="side", avoid=avoid, variant=1)
    assert [p.rank for p in poses] == [1, 2, 0]
    np.testing.assert_allclose(seen["obstacles"], [[0.8, 0.0, 0.9], [0.0, 0.5, 0.9]])
    np.testing.assert_allclose(seen["axis"], [1.0, 0.0, 0.0], atol=1e-6)  # from the arm's base toward the object
    with pytest.raises(ValueError, match="no object points"):
        action.grasp_poses(MODEL, "left", np.zeros((0, 3)), scene, 0.75)
    monkeypatch.delenv("GRASP_BACKEND")
    with pytest.raises(ValueError, match="heuristic_grasp"):
        action.grasp_poses(MODEL, "left", P, scene, 0.75)
