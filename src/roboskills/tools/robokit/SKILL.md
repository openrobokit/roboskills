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
  points; `write_collision_spheres(rk, path)` for a model without a sphere file (`fit_spheres` per link: spheres centred
  inside the mesh, each reaching about 2 cm past it, at most 16 a link before the slack grows; a hand link with no
  collision mesh uses its visual one).
- `planner.Planner(model).plan(poses, robot_now, table=None, scene=None)`: one `Plan` per `action.Pose`, or None where
  RoboKit finds no path; poses of one arm and shape are one RoboKit batch.
- `bot.Tools(robot, sam, out=None)`: the agent's `bot` (status, observe, segment, table, fuse, describe, register,
  carried, heuristic_grasp, grasp_poses, place_pose, twist_pose, plan, execute); `sam` is a caller-owned
  `roboskills.tools.sam3.sam3.Sam`.
- `action.twist_pose(T_tcp, axis, point, deg, arm)`: a `twist` Pose turning a held object about an axis, its arc a
  waypoint every `TWIST_STEP` deg (each solved onto it) the planner joins into one contact run, slow (`SLOW["twist"]`) unless given a `speed` (1.0: a move's, for a pour).
- Real2sim agents: `robokit.helpers.sim_mpc` and `robokit.terms` (their docstrings) on MuJoCo Warp.
- Real2sim hands: `retarget.retarget_hand` (HaWoR's hand on a floating RoboKit hand preset, default Sharpa, in the
  scene: `retarget/`; the wrist steadied by `--steady` and `--calm`, 0 for neither) and `mpc.track_with_mpc` (SimMPC
  tracks that and the object; `engine.rollout` delivers `mpc/`); `python -m roboskills.tools.robokit {retarget,mpc}`.
- A video's person or a text's motion on a humanoid: `humanoid.retarget_person(smplx, out, hands=None,
  robot="g1_inspire")`, GVHMR's `smplx.npz` (and WiLoR's `hands.npz`, read by its `usable`) or Kimodo's SMPL-X
  `_amass.npz` on a RoboKit humanoid named by its preset, `<body>` or an assembly `<body>_<hand>`. RoboKit's: the
  body's online retargeter with the robot's `<body>_aligned` (else `<body>`) preset, each frame started from the one
  before alone (no far restarts) and without its per-frame step clamp (it lags fast moves then catches up); the hand's
  `HandRetargeterOnline` on the hand preset's tracked links and its end effector's palm coordinate spec, with the
  character fingers' terms (the Inspire preset's own online config smooths away the small bends). Ours: the feet the
  person plants (`feet.down`) held level on their spots on a second solve (`feet.plant`, put into the person's joints
  through the preset's target formula run backwards, the feet held as firmly as the root), the feet's soles from the
  lowest mesh under the links tracking the toes, the hands without WiLoR relaxed. The wrists are the body's: WiLoR's
  hand turns, often 100 degrees and more from the body's wrists, lie past a robot wrist's range, and following them
  flipped the wrists and elbows for a frame. Writes `out/motion.npz` (`fps`, `qpos` root xyz wxyz then `joint_names`,
  `link_names` with their world `turns` and `positions`, `planted`, one row per video frame) and `out/robot.glb` (a
  node per link, for `tools/blender`'s `render_motion`); `python -m roboskills.tools.robokit humanoid`. About 15 s for
  400 frames once compiled (a new robot compiles about 100 s). Run on a whole dance: g1_inspire (fingers retargeted;
  the Inspire fingers curl but neither spread sideways nor bend back past straight) and h1 (its ankles only pitch, so
  its planted feet sit less flat); on its first 30 frames: g1, g1_mano, h1_2, gr3 and bhl (hands kept). Not run:
  g1_xhand and h1_inspire (RoboKit's assemblies have no whole-body URDF), k1, t1, adam_lite, pm01, talos, tienkung and
  toddy (their assets would not download at this RoboKit pin). RoboKit's `FullBodyRetargeter` is for the G1 with
  Inspire hands holding an object and needs torch.
- `feet`: the planted feet both motion_gen's characters and `humanoid`'s robots use: `level` (the floor under the
  person), `down` (which heels and toes stand: GVHMR's `feet_still`, else height and speed), `plant` and `step`.
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
`steps`, `ended`, `stopped`, `refused`, `q`, `hand`, `tcp`, `tcp_error_m`, `tcp_error_deg` and `short`. Frames are world metres
and radians unless named; jaw readings are 0 closed .. 1 open.

**GPU memory:** one planner and obstacle grid per arm and batch size, 0.2 GB at the batch of 16 used here (0.8 GB at
64, 12 GB at 1024). SimMPC with `MjWarpSim(..., cpu_execute=True)` costs about 5-10 s per simulated second.

**Known failures:** `plan` gives None for an unreachable pose; `execute` refuses a plan that already ran or whose start
is over 0.02 rad from the live joints; planning keeps clear of the table and observed points but not yet of a held
load; `grasp_poses` raises without a `$GRASP_BACKEND` (build the grasp with `heuristic_grasp`); `Tools.grasp_backend`
holds the backend's `$GRASP_*` settings (None: none), and without one the API listing says `grasp_poses` is not
available. The mask fit is
local: a start far from the mask (IoU near 0) has nothing to follow, and a symmetric object keeps its starting turn
about its axis.

**Upstream:** [RoboKit](https://github.com/openrobokit/robokit).
