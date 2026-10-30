from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
import pytest

from roboskills.tools.robokit.planner import Planner


def test_shared_robot_model_does_not_share_the_arms_target_or_joint_mask(monkeypatch):
    """One planner and obstacle grid per (robot, arm, size, table or not) in the process: each arm its own target link
    and joint mask, goal IK that keeps no clearance from the scene, and the grid in the planner's scene."""
    pytest.importorskip("robokit")
    from robokit.helpers import motion_plan

    @dataclass
    class Stage:
        lock_frames: tuple = (0,)

    @dataclass
    class Config:
        max_retries: int = 0
        stage_retries: int = 0
        cuda_graph: str = "none"
        stages: tuple = (Stage(),)
        goal_ik: object = field(
            default_factory=lambda: SimpleNamespace(solver=SimpleNamespace(num_candidates=[128, 128, 4]))
        )

    class Provider:
        def __init__(self, config, robot, link, **kwargs):
            self.link, self.scene, self.config = link, kwargs["scene"], config

        def set_active_joint_mask(self, mask):
            self.mask = mask

        def warmup(self, size):
            self.size = size

    monkeypatch.setattr(motion_plan, "MotionPlanner", Provider)
    monkeypatch.setattr(motion_plan.presets, "offline_pose_lbfgs", Config(), raising=False)
    shared = SimpleNamespace(desc="shared.urdf", T=np.eye(4), names=["left_joint", "right_joint"])
    model = SimpleNamespace(device="cpu", info={name: SimpleNamespace(M=shared, idx=[i])
                            for i, name in enumerate(("left", "right"))}, robot=SimpleNamespace(
                                arms={name: SimpleNamespace(ee_link=name + "_ee") for name in ("left", "right")}))  # fmt: skip
    monkeypatch.setattr(Planner, "_planners", {})
    planner, bare, tables = Planner(model), None, []
    planner._robot = lambda a, fq, z: tables.append(z)  # RoboKit's robot is not needed to build or share them
    (left, _), (right, _) = planner._planner("left", 16, bare), planner._planner("right", 16, bare)
    assert left is not right and (left.link, right.link) == ("left_ee", "right_ee")
    assert left.mask == [True, False] and right.mask == [False, True]
    assert left.config.stages[0].lock_frames == (0, -1) and left.size == 16  # a pose goal held at its goal-IK answer
    assert planner._planner("left", 16, bare)[0] is left
    import warp as wp

    with_floor, grid = planner._planner("left", 16, 0.8)
    assert tables[-1] == 0.8 and with_floor is not left  # the robot's spheres are cut at the table given
    assert with_floor.config.goal_ik.task("ik_collision").margin == 0.0  # 2.5 cm pushed a TCP off a pose near the table
    assert Planner(model)._planner("left", 16, 1.0)[0] is with_floor  # another episode: the same planner and grid
    grid.update(points=np.array([[0.0, 0.0, 0.8]], np.float32), reset=True)  # what _obstacles writes before a solve
    points = wp.array([[0.0, 0.0, 0.85]], dtype=wp.vec3, device="cpu")
    offsets = wp.array([0, 1], dtype=wp.int32, device="cpu")
    assert with_floor.scene.query_sdf(points, offsets, distance_only=True).numpy()[0] == pytest.approx(0.05, abs=0.012)
