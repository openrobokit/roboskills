"""RoboCasa365 runtime: base kinematics and action, the arm model cut, trajectory following and the step budget, without
the simulator."""

import contextlib
import xml.etree.ElementTree as ET
from types import SimpleNamespace

import numpy as np
import pytest

from roboskills.eval.interface import Trajectory
from roboskills.eval.robocasa365 import TASKS, launch
from roboskills.eval.robocasa365.robot import (
    BASE_TOL,
    BLOCKED,
    EPS,
    GAIN,
    SETTLE,
    Kinematics,
    RoboCasa,
    arm_xml,
    base_action,
    finger_q,
    rot,
    wrap,
)


LO, HI = np.array([-1.0, -1.0, -1.5]), np.array([1.0, 1.0, 1.5])  # the Omron base's velocity actuator ranges


def rotz(a):
    R = np.eye(3)
    R[:2, :2] = rot(a)
    return R


def controller(a, theta):
    """What RoboCasa's legacy base controller and the velocity actuators make of a base action: joint velocities."""
    g = np.array(a, float)
    g[:2] = rot(theta) @ g[:2]
    return np.clip((HI + LO) / 2 + (HI - LO) / 2 * g, LO, HI)


def fake(limit=1000, moves=True, lag=1.0):
    """A runtime over a fake env: base joints integrate the velocities the controller makes of each action (no
    friction), arm joints move `lag` of the way to their targets, fingers go to the gripper's sign."""
    r = object.__new__(RoboCasa)
    data = SimpleNamespace(qpos=np.zeros(12), qvel=np.zeros(12), site_xmat=np.zeros((1, 9)))
    r.bq = r.bv = np.arange(3)
    r.qpos, r.fingers = np.arange(3, 10), ([10, 11], [10, 11])
    r.kin = Kinematics([1.0, 2.0], np.pi, [-0.21, 0.0])
    r.split = {"right": (0, 7), "right_gripper": (7, 8), "base": (8, 11), "torso": (11, 12)}
    r.vlo, r.vhi, r.friction, r.torso = LO, HI, np.zeros(3), np.array([0.0])
    r.base_ctrl, r.center = SimpleNamespace(init_ori=rotz(np.pi)), 0
    data.site_xmat[0] = rotz(np.pi).ravel()

    def step(a):
        q = data.qpos[r.qpos]
        data.qpos[r.qpos] = q + lag * (a[:7] - q)
        data.qpos[r.fingers[0]] = finger_q([1.0 if a[7] < 0 else 0.0])
        theta = np.arctan2(data.site_xmat[0, 3], data.site_xmat[0, 0]) - np.pi
        if moves:
            data.qpos[r.bq] += controller(a[8:11], theta) * r.dt
        data.site_xmat[0] = rotz(np.pi + data.qpos[2]).ravel()

    r.env = SimpleNamespace(step=step, sim=SimpleNamespace(data=data))
    r.dt, r.steps, r.step_limit, r.ended, r.instruction = 0.05, 0, limit, False, "test"
    r.lead, r.base_cmd = np.zeros(7), np.zeros(3)
    r.cmd = {"right": (np.zeros(7), np.ones(1))}
    r.actions, r.frames, r._frame = [], [], lambda: None
    r._gl, r._sync = contextlib.nullcontext, lambda full=False: None
    return r


def ramp(start, goal, n=40, dt=0.05):
    t = np.arange(1, n + 1) * dt
    s = ((1 - np.cos(np.pi * t / t[-1])) / 2)[:, None]
    return t, np.asarray(start, float) + s * (np.asarray(goal, float) - np.asarray(start, float))


def test_cohort_and_launch_reject_unknown_tasks():
    assert len(TASKS) == 5 and all(1500 <= limit <= 3000 for limit, _ in TASKS.values())
    with pytest.raises(ValueError, match="Unsupported"):
        launch.make("NoSuchTask")


def test_base_pose_and_joints_are_inverse_and_the_base_turns_about_its_hinge():
    kin = Kinematics([10.0, 10.0], np.pi, [-0.21, 0.0])
    q = np.array([-10.6, 4.4, 0.7])
    np.testing.assert_allclose(kin.joints(kin.pose(q)), q, atol=1e-12)
    turned = kin.pose(q + [0.0, 0.0, 1.0])
    np.testing.assert_allclose(turned[:2], kin.pose(q)[:2])  # turning in place: only the heading changes
    np.testing.assert_allclose(kin.pose(np.zeros(3)), [10.21, 10.0, np.pi])  # the hinge 0.21 m behind, J turned by pi
    path = kin.joints([kin.pose(q)[:2].tolist() + [0.7 + np.pi - 2 * np.pi]] * 2, near=0.7)  # a heading a turn off
    np.testing.assert_allclose(path[:, 2], 0.7)


def test_base_action_inverts_the_controller_and_feeds_friction_forward():
    friction = np.array([0.25, 0.25, 0.167])
    for theta in (0.0, 0.8, -2.5):
        v = np.array([0.3, -0.2, 0.5])
        np.testing.assert_allclose(
            controller(base_action(v, theta, LO, HI, friction), theta), v + friction * np.sign(v)
        )
    tiny = np.array([0.001, 0.0, -0.003])  # near rest the friction feed-forward fades out
    np.testing.assert_allclose(
        controller(base_action(tiny, 0.3, LO, HI, friction), 0.3), tiny + friction * tiny / EPS, atol=1e-15
    )
    np.testing.assert_allclose(controller(base_action([5.0, 0, -9.0], 0.0, LO, HI, friction), 0.0), [1.0, 0.0, -1.5])


def test_arm_model_is_the_subtree_at_the_origin_with_its_own_actuators_and_meshes():
    source = """<mujoco><compiler angle="degree"/>
      <asset><mesh name="wheel" file="/w.stl"/><mesh name="link" file="/l.stl"/></asset>
      <worldbody><body name="robot0_base" pos="10 10 0" quat="0 0 0 1">
        <body name="mobilebase0_base"><joint name="slide" type="slide"/><geom type="mesh" mesh="wheel"/>
          <body name="robot0_link0" pos="0.1 0 0.7" quat="0.7 0 0 0.7"><geom type="mesh" mesh="link"/>
            <body name="robot0_link1"><joint name="robot0_joint1"/><site name="grip"/></body></body></body></body>
      </worldbody>
      <actuator><motor name="arm" joint="robot0_joint1"/><velocity name="drive" joint="slide"/></actuator>
      <sensor><force name="ft" site="grip"/><force name="other" site="nowhere"/></sensor>
      <contact><exclude body1="robot0_base" body2="robot0_link0"/></contact></mujoco>"""
    xml = arm_xml(source, "robot0_link0")
    (root,) = xml.find("worldbody")
    assert root.get("name") == "robot0_link0" and root.get("pos") == "0 0 0" and root.get("quat") is None
    assert xml.find("compiler").get("angle") == "radian"
    assert [a.get("name") for a in xml.find("actuator")] == ["arm"]
    assert [s.get("name") for s in xml.find("sensor")] == ["ft"]
    assert [m.get("name") for m in xml.find("asset")] == ["link"] and xml.find("contact") is None
    ET.tostring(xml)


def test_a_base_path_reaches_its_end_with_the_arm_riding_along():
    r = fake()
    start = r.kin.pose(np.zeros(3))
    t, B = ramp(start, start + [0.5, -0.3, np.pi / 2])
    f = r.follow(Trajectory(t, {}, {}, {}, base=B))
    o = r.obs(cams=False)
    assert not f.stopped and f.steps <= len(t) + SETTLE
    assert np.linalg.norm(o.base[:2] - B[-1, :2]) < BASE_TOL[0] and abs(wrap(o.base[2] - B[-1, 2])) < BASE_TOL[1]
    A = np.asarray(r.actions)
    np.testing.assert_allclose(A[:, :7], 0.0)  # the arm held its targets
    np.testing.assert_allclose(A[:, 11:], [[0.0, -1.0]] * len(A))  # torso at its reset height, arm mode
    np.testing.assert_allclose(r.base_cmd, r.kin.joints(B[-1]))


def test_headings_are_reported_wrapped_and_driven_the_way_the_path_turns():
    r = fake()
    start = r.kin.pose(np.zeros(3))
    assert np.isclose(r.obs(cams=False).base[2], -np.pi) or np.isclose(r.obs(cams=False).base[2], np.pi)
    t, B = ramp(start, start + [0.0, 0.0, -0.4])  # through +-pi
    r.follow(Trajectory(t, {}, {}, {}, base=B + [0.0, 0.0, -2 * np.pi]))  # the same path a turn lower
    assert np.isclose(r.obs(cams=False).base[2], np.pi - 0.4, atol=BASE_TOL[1])
    assert abs(r.env.sim.data.qpos[2] + 0.4) < BASE_TOL[1]  # the joint turned 0.4 rad, not a whole turn more


def test_a_blocked_base_ends_the_motion_and_holds_where_it_is():
    r = fake(moves=False)
    start = r.kin.pose(np.zeros(3))
    t, B = ramp(start, start + [1.0, 0.0, 0.0])
    t2, Q = ramp(np.zeros(7), np.full(7, 0.5))
    f = r.follow(Trajectory(t, {"right": Q}, {}, {}, base=B))
    assert not f.stopped and f.steps < len(t)  # ended early, short of the path
    off = np.flatnonzero(np.linalg.norm(B[:, :2] - start[:2], axis=1) > BLOCKED[0])[0]
    assert f.steps <= off + BLOCKED[2] + 1
    np.testing.assert_allclose(r.base_cmd, np.zeros(3))
    np.testing.assert_allclose(r.cmd["right"][0], f.obs.q["right"])  # the arm holds where it got to


def test_an_arm_path_holds_the_base_and_settles_on_the_last_waypoint():
    r = fake(lag=0.5)
    t, Q = ramp(np.zeros(7), np.full(7, 0.3), n=10)
    f = r.follow(Trajectory(t, {"right": Q}, {"right": np.zeros((10, 1))}, {}))
    A = np.asarray(r.actions)
    assert 10 < f.steps <= 10 + SETTLE
    np.testing.assert_allclose(A[:, 8:11], 0.0)
    np.testing.assert_allclose(A[-1, 7], 1.0)  # closing
    np.testing.assert_allclose(r.cmd["right"][0], Q[-1])
    assert f.obs.hand["right"][0] == pytest.approx(0.0)


def test_stop_and_the_step_limit_end_a_motion():
    r = fake(limit=6)
    start = r.kin.pose(np.zeros(3))
    t, B = ramp(start, start + [0.4, 0.0, 0.0])
    calls = []
    f = r.follow(Trajectory(t, {}, {}, {}, base=B), stop=lambda o: calls.append(o.steps) or len(calls) == 3)
    assert f.stopped and f.steps == 3 and calls == [1, 2, 3]
    np.testing.assert_allclose(r.base_cmd, r.env.sim.data.qpos[:3])
    f = r.follow(Trajectory(t, {}, {}, {}, base=B))
    assert f.done and r.ended and r.steps == 6
    assert r.follow(Trajectory(t, {}, {}, {}, base=B)).steps == 0


def test_holding_still_corrects_the_base_toward_its_target():
    r = fake()
    r.env.sim.data.qpos[:3] = [0.01, -0.02, 0.0]
    r._hold()
    np.testing.assert_allclose(controller(r.actions[-1][8:11], 0.0), GAIN * np.array([-0.01, 0.02, 0.0]), atol=1e-12)
