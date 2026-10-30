"""BEHAVIOR-1K runtime without the simulator: BDDL goals in words, the gripper, the robot model, and `follow` over a fake
simulator (arms, base waypoints with feedback, a blocked base, the hand alone, the step limit, the moving mount)."""

import json
import math
import xml.etree.ElementTree as ET
from types import SimpleNamespace

import numpy as np
import pytest

from roboskills.eval.behavior import INSTANCES, TASKS
from roboskills.eval.behavior.goal import describe
from roboskills.eval.behavior.hand import T_EE_TCP, TCP_Z, finger_q, model, opening
from roboskills.eval.behavior.robot import DT, GRASP, Behavior
from roboskills.eval.interface import Trajectory


def problem(init, goal):
    return f"(define (problem p) (:domain behavior-1k) (:objects) (:init {init}) (:goal {goal}))"


def test_goals_in_words_name_objects_by_category_room_and_support():
    honey = problem("(ontop jar__of__honey.n.01_1 countertop.n.01_1) (inroom cabinet.n.01_1 entryway) "
                    "(inroom countertop.n.01_1 kitchen) (inroom floor.n.01_1 living_room)",
                    "(and (inside ?jar__of__honey.n.01_1 ?cabinet.n.01_1))")  # fmt: skip
    assert (
        describe(honey) == "Put the jar of honey (on the countertop in the kitchen) inside the cabinet in the entryway."
    )
    scanner = problem("(ontop scanner.n.02_1 table.n.02_1) (not (toggled_on scanner.n.02_1)) (ontop laptop.n.01_1 "
                      "table.n.02_1) (inroom table.n.02_1 private_office)",
                      "(and (toggled_on ?scanner.n.02_1) (nextto ?scanner.n.02_1 ?laptop.n.01_1))")  # fmt: skip
    assert describe(scanner) == ("Switch on the scanner (on the table in the private office) and put the scanner next to "
                                 "the laptop (on the table in the private office).")  # fmt: skip
    batteries = problem("(inroom cabinet.n.01_1 kitchen)", "(and (exists (?cabinet.n.01 - cabinet.n.01) (forall "
                        "(?battery.n.02 - battery.n.02) (inside ?battery.n.02 ?cabinet.n.01))))")  # fmt: skip
    assert describe(batteries) == "Put every battery inside one cabinet (the same cabinet for all)."
    glass = problem("(inroom sink.n.01_1 bathroom) (inroom floor.n.01_2 bedroom) (ontop water_glass.n.02_1 sink.n.01_1) "
                    "(ontop water_glass.n.02_2 sink.n.01_1) (ontop ashcan.n.01_1 floor.n.01_2)",
                    "(and (forall (?water_glass.n.02 - water_glass.n.02) (inside ?water_glass.n.02 ?ashcan.n.01_1)))")  # fmt: skip
    assert describe(glass) == "Put every water glass (on the sink in the bathroom) inside the ashcan in the bedroom."
    closed = problem(
        "(inroom electric_refrigerator.n.01_1 kitchen)", "(and (not (open ?electric_refrigerator.n.01_1)))"
    )
    assert describe(closed) == "Close the electric refrigerator in the kitchen."


def test_cohort_maps_seeds_to_public_test_instances_with_challenge_limits():
    assert INSTANCES == {"1": 301, "2": 302, "3": 303}
    assert len(TASKS) == 5 and all(limit > 0 for limit, _ in TASKS.values())
    assert TASKS["turning_on_radio"][0] == int(2149.64 * 1.5)  # 1.5x the mean human demonstration, as the challenge


def test_gripper_opening_is_finger_travel_and_tcp_frame_is_right_handed():
    for u in (0.0, 0.4, 1.0):
        np.testing.assert_allclose(opening(finger_q([u])), [u])
    assert finger_q([1.0]) == pytest.approx([0.05, 0.05])
    R = T_EE_TCP[:3, :3]
    assert np.linalg.det(R) == pytest.approx(1.0)
    np.testing.assert_allclose(R[:, 0], [0, 0, -1])  # approach: along the fingers, the gripper link's -z
    np.testing.assert_allclose(R[:, 1], [0, 1, 0])  # squeeze: the fingers' travel
    assert T_EE_TCP[2, 3] == TCP_Z < 0


def test_model_keeps_the_torso_and_everything_below_with_visual_collisions(tmp_path):
    src = tmp_path / "r.urdf"
    links = ["base_link", "torso_link4", "left_arm_link1", "zed_link"]
    xml = ['<robot name="r">']
    for name in links:
        xml.append(f'<link name="{name}"><visual name="v"><geometry><mesh filename="m/{name}.obj"/></geometry>'
                   f'<material name="x"/></visual><collision><geometry><mesh filename="missing.obj"/></geometry>'
                   "</collision></link>")  # fmt: skip
    for parent, child, kind in [("base_link", "torso_link4", "revolute"), ("torso_link4", "left_arm_link1", "revolute"),
                                ("torso_link4", "zed_link", "fixed")]:  # fmt: skip
        xml.append(f'<joint name="{child}_j" type="{kind}"><parent link="{parent}"/><child link="{child}"/>'
                   '<axis xyz="0 0 1"/><limit lower="-1" upper="1" effort="1" velocity="1"/></joint>')  # fmt: skip
    src.write_text("".join(xml) + "</robot>")
    root = ET.parse(model(src, tmp_path / "out.urdf")).getroot()
    assert [link.get("name") for link in root.findall("link")] == ["torso_link4", "left_arm_link1", "zed_link"]
    assert [j.get("name") for j in root.findall("joint")] == ["left_arm_link1_j", "zed_link_j"]
    arm = root.find("link[@name='left_arm_link1']")
    (collision,) = arm.findall("collision")  # the visual mesh, absolute, named, without its material
    assert collision.find("geometry/mesh").get("filename") == str(tmp_path / "m/left_arm_link1.obj")
    assert collision.get("name") == "left_arm_link1_collision0" and collision.find("material") is None
    assert not root.find("link[@name='torso_link4']").findall("collision")  # visual only


class FakeSim:
    """The server's wire: arms move `lag` of the way to their targets per step, the base follows its target with the
    controller's per-step displacement unless a wall at x = `wall` stops it, fingers go straight to their target."""

    def __init__(self, lag=1.0, base_lag=0.6, wall=None, limit=1000, tilt=0.0, tilt_rate=0.0):
        self.q = {s: np.zeros(7) for s in ("left", "right")}
        self.f = {s: np.full(2, 0.05) for s in ("left", "right")}
        self.base, self.steps, self.limit, self.lag, self.base_lag, self.wall = (
            np.zeros(3),
            0,
            limit,
            lag,
            base_lag,
            wall,
        )
        self.rows, self.lost, self.tilt, self.tilt_rate = [], None, tilt, tilt_rate

    def state(self):
        c, s = math.cos(self.base[2]), math.sin(self.base[2])
        mount = np.eye(4)
        mount[:2, :2] = [[c, -s], [s, c]]
        mount[:3, 3] = [self.base[0], self.base[1], 1.2]
        return dict(q={k: v.copy() for k, v in self.q.items()}, fingers={k: v.copy() for k, v in self.f.items()},
                    base=self.base.copy(), mount=mount, steps=self.steps, step_limit=self.limit,
                    done=self.steps >= self.limit, tilt=self.tilt, stopped=getattr(self, "stopped", None))  # fmt: skip

    def request(self, op, rows=(), **_):
        assert op == "act"
        self.stopped = None
        for row in rows:
            self.rows.append(row)
            for s, q in row.get("q", {}).items():
                self.q[s] += self.lag * (np.asarray(q) - self.q[s])
            for s, u in row.get("u", {}).items():
                self.f[s] = finger_q([u])
            if "base" in row:  # a drive (else the base holds where it is)
                if self.wall is not None and self.base[0] >= self.wall - 1e-9:
                    self.stopped = "contact"  # as the simulator: a drive stops at the step after a new contact
                if not self.stopped:
                    target = np.asarray(row["base"], float)
                    self.base += self.base_lag * (target - self.base)
                    if self.wall is not None:
                        self.base[0] = min(self.base[0], self.wall)
            self.steps += 1
            self.tilt += self.tilt_rate
        return self.state()

    def close(self):
        pass


def fake(**kw):
    r = object.__new__(Behavior)
    r.sim, r.lock = FakeSim(**kw), __import__("threading").Lock()
    r.step_limit, r.steps, r.ended, r.dt, r.instruction = r.sim.limit, 0, False, DT, "test"
    r.T_mount = np.eye(4)
    r.arms = {s: SimpleNamespace(T_world_base=r.T_mount) for s in ("left", "right")}
    r.frames, r.actions, r.requests = [], [], []
    r._take(r.sim.state())
    r.cmd_q = {s: np.zeros(7) for s in ("left", "right")}
    r.cmd_u = {s: np.ones(1) for s in ("left", "right")}
    r.cmd_base = np.zeros(3)
    return r


def path(n=15, q=None, base=None, hand=None):
    t = np.arange(1, n + 1) * DT
    return Trajectory(t=t, q=q or {}, hand=hand or {}, T={}, base=None if base is None else np.asarray(base, float))


def test_follow_streams_one_arm_and_holds_the_other_and_the_base():
    r = fake()
    r.cmd_q["right"] = np.full(7, 0.3)
    Q = np.linspace(0, 0.5, 16)[1:, None] * np.ones(7)
    f = r.follow(path(q={"left": Q}))
    np.testing.assert_allclose(r.obs(cams=False).q["left"], 0.5, atol=1e-6)
    assert all(row["q"]["right"] == [0.3] * 7 and "base" not in row for row in r.sim.rows)  # the base holds itself
    assert f.steps == len(r.sim.rows) >= 15 and not f.stopped


def test_follow_drives_the_base_to_world_waypoints_and_moves_the_mount():
    r = fake()
    mount = r.arms["left"].T_world_base
    B = np.c_[np.linspace(0, 1.0, 31)[1:], np.zeros(30), np.linspace(0, math.pi / 2, 31)[1:]]
    r.follow(path(n=30, base=B))
    o = r.obs(cams=False)
    assert np.hypot(*(o.base[:2] - [1.0, 0.0])) < 0.005 and abs(o.base[2] - math.pi / 2) < math.radians(0.5)
    assert r.arms["right"].T_world_base is mount  # one array, updated in place
    np.testing.assert_allclose(mount[:2, 3], o.base[:2])
    np.testing.assert_allclose(mount[:2, 0], [math.cos(o.base[2]), math.sin(o.base[2])], atol=1e-9)


def test_robot_mask_is_the_latest_observations_own_segmentation():
    r = fake()
    mask = np.zeros((4, 4), bool)
    mask[1:3, 1:3] = True
    cam = dict(
        rgb=np.zeros((4, 4, 3), np.uint8), depth=np.ones((4, 4), np.float32), K=np.eye(3), T=np.eye(4), robot=mask
    )
    r.cams, r.ended = {"head": cam}, True  # an ended episode observes from its last reply
    first = r.obs()
    np.testing.assert_array_equal(r.robot_mask("head", first), mask)
    second = r.obs()
    assert r.robot_mask("head", first) is None  # not the frame it was segmented with
    assert r.robot_mask("head", second) is not None and r.robot_mask("left_wrist", second) is None


def test_base_targets_lead_the_path_and_keep_yaw_unwrapped():
    start, t = np.array([0.0, 0.0, 3.1]), np.array([0.5, 1.0])
    B = np.array([[0.5, 0.0, 3.3], [1.0, 0.0, 3.5]])  # past +pi: unwrapped, not wrapped back to -pi
    np.testing.assert_allclose(Behavior._base_at(start, 0.25, t, B), [0.25, 0.0, 3.2])
    B2 = B - [0, 0, 2 * math.pi]  # the same headings a turn lower: shifted to start nearest the current yaw
    np.testing.assert_allclose(Behavior._base_at(start, 1.0, t, B2), [1.0, 0.0, 3.5])


def test_a_blocked_base_ends_the_motion_and_holds_where_it_stopped():
    r = fake(wall=0.3)
    B = np.c_[np.linspace(0, 2.0, 61)[1:], np.zeros(60), np.zeros(60)]
    f = r.follow(path(n=60, base=B))
    assert f.steps < 30  # the drive stops when the base touches the wall, well before the path's 60 steps
    np.testing.assert_allclose(r.cmd_base, [0.3, 0.0, 0.0], atol=1e-6)
    assert r.sim.rows[0].get("drive") and all("base" in row for row in r.sim.rows)
    r.follow(path(n=10, q={"left": np.full((10, 7), 0.1)}))
    assert "base" not in r.sim.rows[-1]  # an arm motion afterwards leaves the base to hold itself


def test_a_robot_leaning_further_stops_any_motion():
    r = fake(tilt=4.0)  # leaning on something already: it still moves
    f = r.follow(path(n=20, q={"left": np.linspace(0, 1, 21)[1:, None] * np.ones(7)}))
    assert f.steps >= 20
    r = fake(tilt_rate=0.75)
    f = r.follow(path(n=60, q={"left": np.linspace(0, 1, 61)[1:, None] * np.ones(7)}))
    assert f.steps == 8 and not f.stopped  # the first chunk, then the motion ends where the arm is
    np.testing.assert_allclose(r.cmd_q["left"], r.obs(cams=False).q["left"])


def test_the_hand_alone_opens_quickly_holds_a_close_for_the_grasp_and_the_step_limit_ends_motion():
    r = fake()
    f = r.follow(path(n=10, hand={"left": np.zeros((10, 1))}, q={"left": np.zeros((10, 7))}))
    assert f.steps == GRASP and r.obs(cams=False).hand["left"][0] == pytest.approx(0.0)  # held for assisted grasping
    f = r.follow(path(n=10, hand={"left": np.ones((10, 1))}))
    assert f.steps <= 4 and r.obs(cams=False).hand["left"][0] == pytest.approx(1.0)
    r = fake(limit=20)
    f = r.follow(path(n=60, q={"left": np.linspace(0, 1, 61)[1:, None] * np.ones(7)}))
    assert f.done and r.ended and r.steps == 20 and len(r.sim.rows) == 20


def test_a_replay_sends_the_recorded_requests_again_and_holds_the_last_command():
    """OmniGibson's physics depends on the renders between steps too, so the replay sends what the episode sent."""
    r = fake()
    rows = [{"q": {"left": [0.1] * 7, "right": [0.2] * 7}, "u": {"left": 0.5, "right": 1.0}}] * 3
    r._send(rows[:2])
    r._send(rows[2:])
    again = fake()
    obs = again.replay([json.loads(json.dumps(m)) for m in r.requests])
    assert again.sim.rows == r.sim.rows and again.requests == r.requests and obs.steps == 3
    assert np.allclose(again.cmd_q["left"], 0.1) and np.allclose(again.cmd_u["left"], 0.5)
