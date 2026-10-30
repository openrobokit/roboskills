"""Physical snapshots agree with native frames and exclude nonphysical labels."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from roboskills.eval.libero.robot import Libero
from roboskills.eval.physical import CollisionScene, layout_fingerprint


@pytest.fixture
def mujoco():
    pytest.importorskip("trimesh")
    return pytest.importorskip("mujoco")


def test_observation_guard_precedes_every_native_read():
    robot = object.__new__(Libero)
    robot.information = "observations"
    with pytest.raises(PermissionError):
        robot.scene()  # No env or simulator exists to read.


def test_calibration_checks_description_limits_in_arm_order(tmp_path):
    robot = object.__new__(Libero)
    arm = SimpleNamespace(ee_link="hand", lo=np.array([-3.0, -1.0]), hi=np.array([3.0, 1.0]), T_world_base=np.eye(4))
    robot.arms, robot.directory, robot.site = {"right": arm}, tmp_path, 0
    position = np.array([0.2, 0.3, 0.4])
    data = SimpleNamespace(get_body_xmat=lambda name: np.eye(3), get_body_xpos=lambda name: position,
                           site_xpos=position[None])  # fmt: skip
    robot.env = SimpleNamespace(sim=SimpleNamespace(data=data))
    spec = SimpleNamespace(actuated_joint_limits=np.array([[-1, 1], [-2, 2], [-3, 3]]))
    loaded = SimpleNamespace(root=arm, rk=SimpleNamespace(spec=spec))
    model = SimpleNamespace(
        info={"right": SimpleNamespace(M=loaded, idx=[2, 0], ee=0)},
        fq=lambda arm: np.zeros(3),
        links=lambda a, q: arm.T_world_base[None],
        tcp=lambda a: arm.T_world_base,
    )
    robot.calibrate(model)
    np.testing.assert_allclose(arm.T_world_base[:3, 3], position)
    assert json.loads((tmp_path / "calibration.json").read_text())["tcp_error_m"] == 0
    arm.lo[0] = -2  # Reject a native limit that differs from the loaded description.
    with pytest.raises(ValueError, match="joint limits"):
        robot.calibrate(model)


def test_physical_surface_pose_joint_and_exclusions(mujoco):
    model = mujoco.MjModel.from_xml_string("""
    <mujoco><worldbody>
      <body name="robot"><geom type="box" size=".1 .1 .1"/>
        <body name="finger"><geom type="sphere" size=".03"/></body>
      </body>
      <body name="cabinet" pos="1 2 3" euler="0 0 90">
        <geom type="box" size=".2 .3 .4"/>
        <body name="drawer" pos="0 0 .5">
          <joint name="drawer_slide" type="slide" axis="1 0 0" range="0 .4"/>
          <geom type="box" size=".1 .2 .05"/>
          <site name="reward_target" pos="9 9 9"/>
        </body>
      </body>
      <body name="goal_region"><geom type="box" size="1 1 1" contype="0" conaffinity="0"/></body>
    </worldbody></mujoco>""")
    data = mujoco.MjData(model)
    data.qpos[0] = 0.15
    mujoco.mj_forward(model, data)
    scene = CollisionScene(model, [model.body("robot").id]).snapshot(data)
    rows = {row["id"]: row for row in scene["objects"]}
    assert set(rows) == {"cabinet", "drawer"}
    assert "reward_target" not in json.dumps(scene)
    drawer = rows["drawer"]
    np.testing.assert_allclose(np.asarray(drawer["pose"])[:3, 3], [1, 2.15, 3.5])
    np.testing.assert_allclose(drawer["lower"], [0.8, 2.05, 3.45])
    np.testing.assert_allclose(drawer["upper"], [1.2, 2.25, 3.55])
    joint = drawer["joints"][0]
    assert joint["kind"] == "slide" and joint["qpos"] == [0.15]
    np.testing.assert_allclose(joint["axis"], [0, 1, 0], atol=1e-12)
    before = np.asarray(drawer["points"])
    data.qpos[0] = 0.25
    mujoco.mj_forward(model, data)
    after = next(row for row in CollisionScene(model, []).snapshot(data)["objects"] if row["id"] == "drawer")
    np.testing.assert_allclose(np.asarray(after["points"]) - before, np.tile([0, 0.1, 0], (len(before), 1)), atol=1e-12)


def test_compiled_mesh_transform_is_not_applied_twice(mujoco):
    model = mujoco.MjModel.from_xml_string("""
    <mujoco><asset>
      <mesh name="offset" scale="2 3 4" vertex="1 2 3  2 2 3  1 4 3  1 2 6"/>
    </asset><worldbody>
      <body name="object" pos=".1 .2 .3" euler="0 0 90">
        <geom type="mesh" mesh="offset" pos=".4 .5 .6"/>
      </body>
    </worldbody></mujoco>""")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    points = np.asarray(CollisionScene(model, []).snapshot(data)["objects"][0]["points"])
    source = np.array([[1, 2, 3], [2, 2, 3], [1, 4, 3], [1, 2, 6]]) * [2, 3, 4]
    local = source + [0.4, 0.5, 0.6]
    expected = local @ np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]) + [0.1, 0.2, 0.3]
    np.testing.assert_allclose(points.min(0), expected.min(0), atol=2e-6)
    np.testing.assert_allclose(points.max(0), expected.max(0), atol=2e-6)


def test_adapter_snapshot_uses_physics_only(mujoco):
    model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
      <body name="robot"><geom size=".1"/></body>
      <body name="gripper"><geom size=".1"/></body>
      <body name="box"><geom type="box" size=".1 .2 .3"/></body>
    </worldbody></mujoco>""")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    # Match robosuite's model-name adapter without any task state or checker.
    proxy = SimpleNamespace(body_name2id=lambda key: model.body(key).id)
    for key in (
        "nbody",
        "ngeom",
        "njnt",
        "body_parentid",
        "geom_bodyid",
        "geom_contype",
        "geom_conaffinity",
        "geom_type",
        "geom_size",
        "body",
        "jnt_bodyid",
    ):
        setattr(proxy, key, getattr(model, key))
    robot = object.__new__(Libero)
    robot.information = "privileged"
    robot.env = SimpleNamespace(
        sim=SimpleNamespace(model=proxy, data=data),
        robots=[
            SimpleNamespace(
                robot_model=SimpleNamespace(root_body="robot"), gripper=SimpleNamespace(root_body="gripper")
            )
        ],
    )
    assert [row["id"] for row in robot.scene()["objects"]] == ["box"]


def test_layout_identity_ignores_robot_and_visual_markers_but_detects_object_motion(mujoco):
    model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
      <body name="robot"><joint name="arm" type="slide"/><geom size=".1"/></body>
      <body name="object"><joint name="item" type="slide"/><geom size=".1"/></body>
      <body name="marker"><geom size=".1" contype="0" conaffinity="0"/></body>
    </worldbody></mujoco>""")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    roots = [model.body("robot").id]
    before = layout_fingerprint(model, data, roots)
    data.qpos[0] = 0.2
    mujoco.mj_forward(model, data)
    assert layout_fingerprint(model, data, roots) == before
    data.qpos[1] = 0.02
    mujoco.mj_forward(model, data)
    assert layout_fingerprint(model, data, roots) != before
