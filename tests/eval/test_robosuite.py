"""CaP-X Robosuite runtime: action layout, trajectory following, step budget and arm naming, without the simulator."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from roboskills.eval.interface import Trajectory
from roboskills.eval.robosuite.robot import (
    SETTLE,
    TOL,
    Robosuite,
    action,
    command,
    finger_q,
    opening,
    sides,
    stacked,
)


def fake(arms=("left", "right"), lag=1.0, limit=1000, lead=0.0):
    """A runtime over a fake env whose arm joints move `lag` of the way to each target per step and whose
    fingers go straight to the gripper's sign: open or closed."""
    r = object.__new__(Robosuite)
    data = SimpleNamespace(qpos=np.zeros(9 * len(arms)), qvel=np.zeros(9 * len(arms)))
    r.order = list(arms)
    r.qpos = {a: np.arange(7) + 9 * i for i, a in enumerate(arms)}
    r.fingers = {a: ([7 + 9 * i, 8 + 9 * i], [7 + 9 * i, 8 + 9 * i]) for i, a in enumerate(arms)}

    def step(a):
        for i, s in enumerate(arms):
            q = data.qpos[r.qpos[s]]
            data.qpos[r.qpos[s]] = q + lag * (a[8 * i : 8 * i + 7] - q)
            data.qpos[r.fingers[s][0]] = finger_q([1.0 if a[8 * i + 7] < 0 else 0.0])

    r.env = SimpleNamespace(step=step, sim=SimpleNamespace(data=data))
    r.dt, r.steps, r.step_limit, r.ended, r.instruction = 0.05, 0, limit, False, "test"
    r.lead = {a: np.full(7, lead) for a in arms}
    r.cmd = {a: (np.zeros(7), np.ones(1)) for a in arms}
    r.actions, r.frames, r._frame = [], [], lambda: None
    return r


def test_physical_scene_is_denied_before_reading_simulator_state():
    robot = object.__new__(Robosuite)
    robot.information = "observations"
    with pytest.raises(PermissionError, match="observations mode"):
        robot.scene()


def ramp(n=10, to=0.5, dt=0.05):
    t = np.arange(1, n + 1) * dt
    return t, np.linspace(0, to, n + 1)[1:, None] * np.ones(7)


def test_action_orders_robots_and_maps_jaw_commands_to_gripper_signs():
    q = {"left": np.arange(7.0), "right": -np.arange(7.0)}
    a = action(["left", "right"], q, {"left": np.ones(1), "right": np.zeros(1)})
    np.testing.assert_allclose(a, np.r_[np.arange(7.0), -1.0, -np.arange(7.0), 1.0])  # absolute joints, open -1
    assert action(["right"], {"right": q["right"]}, {"right": np.array([2.0])})[-1] == -1.0  # clipped
    assert action(["right"], {"right": q["right"]}, {"right": np.zeros(0)}).shape == (7,)  # the sponge: joints only


def test_jaw_opening_is_the_pad_gap_and_reads_closed_on_nothing_as_not_gripping():
    gripping = lambda u: 0.03 < float(u[0]) < 0.95  # the hold rule agent_as_policy/SKILL.md gives (closed 0, open 1)

    for u in (0.0, 0.3, 1.0):
        np.testing.assert_allclose(opening(finger_q([u])), [u])
    assert finger_q([1.0])[0] == pytest.approx(0.04) and finger_q([0.0])[0] > 0  # pads meet before the joints do
    assert gripping(opening([0.00105, -0.00105])) is False  # closed on nothing (measured travel 0.0021 m)
    assert gripping(opening([0.021, -0.021])) is True  # a 4 cm cube
    assert gripping(opening([0.039, -0.039])) is False  # open


def test_command_starts_from_the_current_target_and_leads_by_the_velocity():
    t, Q = np.array([0.5, 1.0]), np.array([[1.0], [1.0]])
    assert command(t, Q, np.zeros(1), 0.25, 0.0, 0.05)[0] == pytest.approx(0.5)
    assert command(t, Q, np.zeros(1), 0.25, 0.2, 0.05)[0] == pytest.approx(0.5 + 0.2 * 2.0)  # 2 rad/s
    assert command(t, Q, np.zeros(1), 0.6, 0.2, 0.05)[0] == pytest.approx(1.0)  # at rest: no lead
    assert command(t, np.zeros((2, 0)), np.zeros(0), 0.25, 0.0, 0.05).shape == (0,)  # a hand with no command


def test_follow_streams_one_arm_and_holds_the_other_and_its_hand():
    r = fake()
    r.cmd["right"] = (np.full(7, 0.3), np.zeros(1))
    t, Q = ramp()
    f = r.follow(Trajectory(t, {"left": Q}, {}, {}))
    A = np.asarray(r.actions)
    assert f.steps == 10 and not f.done and not f.stopped  # perfect tracking: nothing to settle
    np.testing.assert_allclose(A[:, :7], Q)
    np.testing.assert_allclose(A[:, 8:15], 0.3)
    np.testing.assert_allclose(A[:, [7, 15]], [[-1.0, 1.0]] * 10)
    np.testing.assert_allclose(r.cmd["left"][0], Q[-1])
    np.testing.assert_allclose(f.obs.q["left"], Q[-1])


def test_follow_feeds_the_controller_lag_forward_then_settles_on_the_last_waypoint():
    r = fake(lag=0.5, lead=0.2)
    t, Q = ramp()
    f = r.follow(Trajectory(t, {"left": Q}, {}, {}))
    A = np.asarray(r.actions)
    np.testing.assert_allclose(A[1:9, 0], Q[1:9, 0] + 0.2 * 1.0)  # 0.5 rad over 0.5 s
    assert 10 < f.steps <= 10 + SETTLE
    np.testing.assert_allclose(A[10:, :7], np.broadcast_to(Q[-1], (f.steps - 10, 7)))
    assert np.abs(f.obs.q["left"] - Q[-1]).max() < TOL


def test_follow_moves_a_hand_alone_and_waits_for_the_fingers():
    r = fake()
    n = 5
    f = r.follow(Trajectory(np.arange(1, n + 1) * 0.05, {}, {"right": np.zeros((n, 1))}, {}))
    A = np.asarray(r.actions)
    assert f.steps == n and (A[:, 15] == 1.0).all() and (A[:, 7] == -1.0).all()
    assert f.obs.hand["right"][0] == pytest.approx(0.0) and r.cmd["right"][1][0] == 0.0


def test_stop_ends_the_motion_and_holds_the_measured_posture():
    r = fake(lag=0.5)
    calls = []
    t, Q = ramp()
    f = r.follow(Trajectory(t, {"left": Q}, {}, {}), stop=lambda o: calls.append(o.steps) or len(calls) == 3)
    assert f.stopped and f.steps == 3 and calls == [1, 2, 3]
    np.testing.assert_allclose(r.cmd["left"][0], f.obs.q["left"])
    assert np.abs(f.obs.q["left"] - Q[-1]).max() > 0.1


def test_the_step_limit_ends_the_episode_mid_motion():
    r = fake(limit=4)
    t, Q = ramp()
    f = r.follow(Trajectory(t, {"left": Q}, {}, {}))
    assert f.steps == 4 and f.done and r.ended and f.obs.steps == f.obs.step_limit == 4
    assert r.follow(Trajectory(t, {"left": Q}, {}, {})).steps == 0 and len(r.actions) == 4


def test_arms_are_named_by_side_in_capx_scene_camera():
    # CaP-X's two-arm agentview (MuJoCo camera quaternion w, x, y, z) as an OpenCV pose; robots on opposed sides.
    w, x, y, z = 0.653, 0.271, 0.271, 0.653
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat([x, y, z, w]).as_matrix() @ np.diag([1.0, -1.0, -1.0])
    T[:3, 3] = [1.5, 0.0, 2.5]
    robot0, robot1 = [0.0, -0.56, 0.912], [0.0, 0.56, 0.912]
    assert sides(T, [robot0, robot1]) == ["left", "right"]
    assert sides(T, [robot1, robot0]) == ["right", "left"]
    assert sides(T, [robot0]) == ["right"]


def test_restack_layout_puts_the_second_cube_on_the_first():
    cube = lambda name, h: SimpleNamespace(
        name=name, horizontal_radius=h, bottom_offset=[0, 0, -h], top_offset=[0, 0, h]
    )
    sampler = SimpleNamespace(
        mujoco_objects=[cube("cubeA", 0.02), cube("cubeB", 0.02)],
        reference_pos=np.array([0.0, 0.0, 0.8]),
        z_offset=0.01,
        _sample_x=lambda r: 0.1,
        _sample_y=lambda r: 0.05,
        _sample_quat=lambda: np.array([1.0, 0, 0, 0]),
    )
    placed = stacked(sampler)
    np.testing.assert_allclose(placed["cubeA"][0], [0.1, 0.05, 0.83])
    np.testing.assert_allclose(placed["cubeB"][0], [0.09, 0.05, 0.83 + 0.02 + 0.02 + 0.01])


@pytest.mark.parametrize(("z", "expected"), [((0.05, 0.09), False), ((-0.09, -0.05), True)])
def test_restack_success_rejects_both_cubes_held_up(z, expected):
    r = object.__new__(Robosuite)
    r.task = "cube_restack"
    data = SimpleNamespace(
        xpos=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, z[0]], [0.0, 0.0, z[1]]]), xmat=np.tile(np.eye(3).ravel(), (3, 1))
    )
    r.env = SimpleNamespace(
        _check_success=lambda: True,
        cubeA_body_id=1,
        cubeB_body_id=2,
        sim=SimpleNamespace(data=data, model=SimpleNamespace(body_name2id=lambda name: 0)),
    )
    assert r._success() is expected
