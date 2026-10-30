"""A mobile base and a legged body: the drive and posture Poses the agent builds, planned from the live measurement,
without a simulator."""

from types import SimpleNamespace

import numpy as np
import pytest

from roboskills.eval.interface import Base, Body
from roboskills.tools.robokit.action import Pose
from roboskills.tools.robokit.planner import Planner


def planner_of(base=None, body=None):
    """A Planner on a fake robot with no arms."""
    robot = SimpleNamespace(arms={}, base=None if base is None else Base(vmax=0.5, wmax=1.0),
                            body=None if body is None else Body(lo=0.4, hi=0.75, vmax=0.2))  # fmt: skip
    return Planner(SimpleNamespace(robot=robot))


def now_of(base=None, body=None):
    return SimpleNamespace(q={}, hand={}, base=None if base is None else np.array(base, float), body=body)


def test_drive_reaches_the_goal_within_the_base_limits():
    now = now_of(base=[1.0, 2.0, 0.0])
    (plan,) = planner_of(base=True).plan([Pose("drive", None, base=np.array([2.0, 2.0, np.pi / 2]))], now)
    (path,) = plan.paths
    assert path.q == {} and path.hand == {} and path.t[0] > 0 and path.body is None and plan.q_start == {}
    np.testing.assert_allclose(path.base[-1], [2.0, 2.0, np.pi / 2])
    speed = np.linalg.norm(np.diff(np.r_[[now.base], path.base], axis=0)[:, :2], axis=1) / np.diff(np.r_[0, path.t])
    assert speed.max() <= 0.5 + 1e-6


def test_fixed_arms_cannot_drive_and_a_robot_without_legs_cannot_crouch():
    with pytest.raises(ValueError, match="no base"):
        planner_of().plan([Pose("drive", None, base=np.zeros(3))], now_of())
    with pytest.raises(ValueError, match="height="):
        planner_of().plan([Pose("posture", None, height=0.5)], now_of())


def test_posture_lowers_the_pelvis_within_the_body_range():
    planner = planner_of(body=True)
    (plan,) = planner.plan([Pose("posture", None, height=0.5)], now_of(body=0.75))
    (path,) = plan.paths
    assert path.q == {} and path.base is None and np.isclose(path.body[-1], 0.5)
    assert (np.abs(np.diff(np.r_[0.75, path.body])) / np.diff(np.r_[0, path.t])).max() <= 0.2 + 1e-6
    deepest = planner.plan([Pose("posture", None, height=0.1)], now_of(body=0.75))[0].paths[0].body[-1]
    assert np.isclose(deepest, 0.4)  # clipped to the deepest squat
