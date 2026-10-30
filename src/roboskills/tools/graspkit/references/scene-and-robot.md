# Import a scene and robot

Paths below are relative to the selected GraspKit checkout unless stated otherwise.

## Scene contract

GraspKit consumes `SceneDescription`, not a live simulator scene or a USD stage.
Export the supplied scene's table, target and relevant obstacles into its native
description. Convert USD meshes with their accumulated transforms, `metersPerUnit`
and up axis applied. Preserve geometry and poses consistently: bake each transform
once. Use metres and Z-up world coordinates; quaternions are **wxyz**.

| Field | Meaning |
|---|---|
| `scene_id` | Unique across both splits; also the observation/grasp join key |
| `target_actor_id` | Key of the grasped mesh in `meshes` |
| `boxes["table"]` | `BoxDescription`: full side lengths, centre position, quaternion |
| `meshes[actor_id]` | `MeshDescription`: collision mesh, scale, world pose, optional visual mesh |
| msgpack root | `{"train": {id: scene}, "test": {id: scene}}` |

Use actual convex collision parts, e.g. CoACD output that preserves separate
pieces, for simulation. A raw concave visual mesh is not a substitute. Keep
visual and collision geometry aligned. Other obstacles have their own actor IDs.
Articulated objects need simulator support beyond this static mesh/box contract.

The following is a **serialization example**, to put in the task's artifact
converter. `GK_RUN` is an absolute ignored run directory. Supply collision meshes
at the two paths first; this example assumes each mesh's origin is at its bottom,
and the table top is world Z=0. Replace those poses with the exported scene poses.

```python
import os
from pathlib import Path

from graspkit.scene.scene_description import BoxDescription, MeshDescription, SceneDescription

out = Path(os.environ["GK_RUN"])
out.mkdir(parents=True, exist_ok=True)
scenes = {"train": {}, "test": {}}
for split, object_name in (("train", "object_a"), ("test", "object_b")):
    sid = f"{object_name}_000"
    collision = out / "objects" / object_name / "collision.glb"
    assert collision.is_file(), collision
    scenes[split][sid] = SceneDescription(
        scene_id=sid,
        target_actor_id="object",
        boxes={"table": BoxDescription("table", (1.0, 1.0, 0.1), (0.0, 0.0, -0.05))},
        meshes={
            "object": MeshDescription(
                actor_id="object",
                collision_mesh_path=str(collision),
                scales=(1.0, 1.0, 1.0),
                position=(0.0, 0.0, 0.0),
                quaternion=(1.0, 0.0, 0.0, 0.0),
            )
        },
    )
path = str(out / "scenes.msgpack")
SceneDescription.save_scene_descriptions_msgpack(scenes, path)
loaded = SceneDescription.load_scene_descriptions_msgpack(path, return_splits=True)
assert set(loaded["train"]).isdisjoint(loaded["test"])
assert all(s.target_actor_id in s.meshes for group in loaded.values() for s in group.values())
```

Two scenes only check plumbing. Build enough diverse scenes for actual learning.
Split objects before pose/view augmentation when measuring object generalization;
different IDs or poses of the same object do not establish object-disjoint data.
GraspKit calls the second split `test`; if used for checkpoint selection, treat it
as development data and reserve a separate untouched final evaluation set.

## Robot contract

Find an existing `configs/robot/` entry with `build_robot_config(name)` and an
experiment targeting that embodiment. `robot_config` is a `RobotConfig` object,
not a `--robot_config <name>` string. `build_experiment_config(name)` returns a
registered experiment; copying a module without registering it is insufficient.

| Training target | Required configuration |
|---|---|
| Floating hand/gripper | `robot_config.base_type="floating"`; actions contain root XYZ + rxyz + actuated hand joints |
| Fixed arm + hand | `robot_config.base_type="fixed"`; actions contain its actuated joints |
| Floating policy, fixed-arm evaluation | Floating target plus matching `traj_builder.fixed_base_robot_config`, `floating_base_robot_config`, and `convert_to_fixed=True` |

The fixed robot must carry the **same end effector** when converting grasps.
A fixed robot used only to anchor a camera does not establish arm reachability.

For a new embodiment, implement its reusable support in GraspKit/RoboKit:

- Supply URDF/MJCF, mesh references, joint limits/mimics, `init_qpos`, collision
  spheres and self-collision exclusions. Match actuated joint ordering to
  `GraspRobot.from_config(config).actuated_joint_names`.
- Set palm, hand-base, fingertip and contact link names; provide contact points or
  `contact_candidates_config`, and `ee_coord_spec` in its declared link frame.
  Wrong axes or contact links can produce plausible-looking unusable grasps.
- Register the config in `configs/robot/__init__.py`. Reuse an existing experiment
  for that embodiment, or add a reusable embodiment preset and register it in
  `configs/experiment/__init__.py`; keep task scene paths/recipes in artifacts.
- Ensure the selected evaluator can instantiate the robot with matching joints,
  controllers and collision geometry. ManiSkill can generate a config-based agent
  for supported URDF robots; an MJCF robot needs the MuJoCo backend. Author a
  specialized agent only when the generic one cannot represent its control.

Read `docs/reference/robot_hands.md`, `configs/robot_config.py`, and the closest
working robot/experiment before implementing new support.

## Camera contract

Use the camera intrinsics and extrinsics that will produce deployment observations.
For the calibrated renderer, pose JSON accepts `T_base_camera` (camera-to-anchor
4×4 transform, or a stack of them), or `extrinsic` which it inverts. Intrinsic JSON
contains `{"intrinsic": [fx, fy, cx, cy, height, width, depth_scale]}`.
The calibrated pose uses OpenCV camera axes; the renderer's
`compute_camera_world_pose_7d` in `utils/xform_utils.py` converts it to SAPIEN.
Convert a USD camera rotation to that input convention before writing the JSON.

`camera_frame="world"` uses that pose in world coordinates;
`camera_frame="robot"` composes it with the fixed robot's initial base pose.
Use `point_cloud_frame="world"` with world-frame grasp labels. Verify the crop
bounds contain the object, the segmentation contains target points, and the
camera sees the intended surfaces. Re-render after changing geometry, poses,
cameras or crops.
