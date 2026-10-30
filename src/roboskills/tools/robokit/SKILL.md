---
name: robokit
description: Robot kinematics, collision-aware motion planning and the stateless robot verbs (`bot`) an agent calls on an adapter's Robot, on RoboKit; RoboKit's SimMPC for contact stages; a rigid object's pose and scale fitted to its masks (RoboKit's mask alignment).
---

# RoboKit

**Runs in** this Python (the mask fit in its own `uv run`): `robokit[mjwarp]`, a pip dependency pinned to a commit
in `pyproject.toml`, imported where used. GraspKit and the simulators are installed separately at the versions their
adapters require.

**Calls** (modules here; read their docstrings, [README.md](README.md) has the `bot` verbs):

- `robot_model.RobotModel(robot)` on a `roboskills.eval.interface.Robot`: per arm `fq`, `tcp(arm, q)` (4, 4), surface
  points; `write_collision_spheres(rk, path)` for a model without a sphere file.
- `planner.Planner(model).plan(poses, robot_now, table=None, scene=None)`: one `Plan` per `action.Pose`, or None where
  RoboKit finds no path; poses of one arm and shape are one RoboKit batch.
- `bot.Tools(robot, sam, out=None)`: the agent's `bot` (status, observe, segment, table, fuse, describe, register,
  carried, heuristic_grasp, grasp_poses, place_pose, plan, execute); `sam` is a caller-owned
  `roboskills.tools.sam3.sam3.Sam`.
- Real2sim agents: `robokit.helpers.sim_mpc` and `robokit.terms` (their docstrings) on MuJoCo Warp.
- Real2sim hands: `retarget.retarget_hand` (HaWoR's hand on a floating RoboKit hand preset, default Sharpa, in the
  scene: `retarget/`; the wrist steadied by `--steady` and `--calm`, 0 for neither) and `mpc.track_with_mpc` (SimMPC
  tracks that and the object; `engine.rollout` delivers `mpc/`); `python -m roboskills.tools.robokit {retarget,mpc}`.
- `mask_alignment.fit_object_to_masks(mesh, poses, masks, depth, out, label=1, scale=False)`: a rigid mesh's pose (and
  scale) per frame fitted to label `label` of masks/<frame>.png, other labels occluding it, by RoboKit's hand-eye mask
  alignment (the mesh a one-link robot, its pose the extrinsic; RGB preset from the start alone); the depth of the
  object's origin stays. Runs in `uv run` of [`mask_alignment.py`](mask_alignment.py) (this RoboKit pin, torch 2.9.1
  cu126, a prebuilt nvdiffrast wheel; nothing compiles) on `$CUDA_VISIBLE_DEVICES`; writes out/poses.npz
  (`camera_from_object`, `scale`, `frames`, soft silhouette `iou` and `iou_start`). About 0.1 s a frame at 640x480,
  0.5 s at 1440x1080, after a few seconds of start-up.
- `mask_alignment.fit_shape(masks, depth, shape, out, frame=0)`: a shape folder (`mesh.ply`, `pose.json`) refit with
  scale at `frame`, written in the same layout plus `fit.json` (`scale`, `iou`, `iou_start`); real2sim's `fit`.

**Returns.** A `Plan`'s `paths` are the robot's `Trajectory`s, one per segment, sampled every 0.02 s; `execute` returns
`steps`, `ended`, `stopped`, `q`, `hand`, `tcp`, `tcp_error_m`, `tcp_error_deg` and `short`. Frames are world metres
and radians unless named; jaw readings are 0 closed .. 1 open.

**GPU memory:** one planner and obstacle grid per arm and batch size, 0.2 GB at the batch of 16 used here (0.8 GB at
64, 12 GB at 1024). SimMPC with `MjWarpSim(..., cpu_execute=True)` costs about 5-10 s per simulated second.

**Known failures:** `plan` gives None for an unreachable pose; `execute` refuses a plan that already ran or whose start
is over 0.02 rad from the live joints; planning keeps clear of the table and observed points but not yet of a held
load; `grasp_poses` raises without a `$GRASP_BACKEND` (build the grasp with `heuristic_grasp`). The mask fit is
local: a start far from the mask (IoU near 0) has nothing to follow, and a symmetric object keeps its starting turn
about its axis.

**Upstream:** [RoboKit](https://github.com/openrobokit/robokit).
