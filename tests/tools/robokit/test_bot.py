"""`Tools`, the stateless bot tools, with the robot, the model and the planner faked."""

from types import SimpleNamespace

import numpy as np
import pytest
from robokit.xform.numpy import rot_tl_to_tf_mat

from roboskills.eval.interface import Follow
from roboskills.tools.robokit.action import Pose
from roboskills.tools.robokit.bot import ENDED, Tools


def bot_with(q=0.0, hand=1.0):
    """Tools on two arms whose TCPs sit at their joint value along x; the live obs reads joints `q`, hand `hand`."""
    bot = Tools.__new__(Tools)
    obs = SimpleNamespace(q={a: np.full(1, q) for a in ("left", "right")}, hand={a: np.full(1, hand) for a in ("left", "right")},
                          base=None, body=None, steps=3, step_limit=100, instruction="stack")  # fmt: skip
    bot.robot = SimpleNamespace(arms=dict.fromkeys(obs.q), ended=False, obs=lambda cams=True: obs,
                                follow=lambda traj, stop=None: Follow(steps=7, done=False, stopped=False, obs=obs))  # fmt: skip
    bot.model = SimpleNamespace(robot=bot.robot, fq=lambda a, o: o.q[a], tcp=lambda a, fq: rot_tl_to_tf_mat(tl=np.r_[fq, 0, 1.0]))  # fmt: skip
    bot.out = None
    return bot, obs


class FakePlan:
    """An arm plan made at joints `q_start` of two paths, recording its outcome like a Plan."""

    def __init__(self, q_start=0.0, verb="move"):
        self.pose = Pose(verb, "left", tcp=rot_tl_to_tf_mat(tl=np.array([0.0, 0, 1.0])))
        self.q_start, self.paths, self.tcp_end = {"left": np.full(1, q_start)}, ["in", "close"], self.pose.tcp
        self.steps, self.ends_q, self.followed = None, None, []

    def record(self, follows, ended):
        self.steps, self.ended, self.stopped = sum(f.steps for f in follows), ended, False


def test_status_reports_the_live_robot_with_raw_tcp_poses():
    bot, _ = bot_with(q=0.2)
    s = bot.status()
    assert s["steps"] == 3 and s["instruction"] == "stack" and s["ended"] is False and s["base"] is None
    assert np.allclose(s["tcp"]["left"][:3, 3], [0.2, 0, 1.0]) and s["tcp"]["left"].shape == (4, 4)


def test_plan_starts_from_the_live_joints_with_the_table_and_scene_and_takes_the_first_reachable():
    bot, obs = bot_with()
    seen = []
    bot.planner = SimpleNamespace(plan=lambda *a: seen.append(a) or [None, "second", "third"])
    poses = [Pose("move", "left", rank=k) for k in range(3)]
    scene = np.zeros((2, 3))
    assert bot.plan(poses, table=(0.75, None), scene=scene) == "second" and seen == [(poses, obs, (0.75, None), scene)]
    bot.planner = SimpleNamespace(plan=lambda *a: [None, None])
    assert bot.plan(poses) is None


def test_execute_runs_a_plan_once_and_reports_its_outcome():
    bot, _ = bot_with()
    followed = []
    follow = bot.robot.follow
    bot.robot.follow = lambda traj, stop=None: followed.append(traj) or follow(traj)
    with pytest.raises(ValueError, match="returned None"):
        bot.execute(None)
    p = FakePlan()
    out = bot.execute(p)
    assert followed == ["in", "close"] and out["steps"] == 14 and out["short"] is None  # one follow per hand command
    assert out["tcp_error_m"] == pytest.approx(0.0) and out["tcp_error_deg"] == pytest.approx(0.0)
    with pytest.raises(ValueError, match="has run"):
        bot.execute(p)
    bot.robot.ended = True
    with pytest.raises(RuntimeError) as e:
        bot.execute(FakePlan())
    assert str(e.value) == ENDED and followed == ["in", "close"]


def test_execute_refuses_a_plan_made_before_the_arm_moved():
    bot, _ = bot_with(q=0.03)
    with pytest.raises(ValueError, match="moved since"):
        bot.execute(FakePlan(q_start=0.0))
    bot.execute(FakePlan(q_start=0.015))  # within 0.02 rad: still its start
