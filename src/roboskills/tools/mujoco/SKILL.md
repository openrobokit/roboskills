---
name: mujoco
description: Build MuJoCo scenes around an articulated MJCF actor, replay per-step actuator controls under native physics with contact feedback, query saved states and render them, through roboskills.tools.mujoco.engine.
---

# MuJoCo

**Runs in** this Python: `mujoco>=3.5` (and MuJoCo Warp through `robokit[mjwarp]`), imported only when a function
runs. Rendering needs `MUJOCO_GL=egl` (and `MUJOCO_EGL_DEVICE_ID`) set before Python starts, and `ffmpeg` on `PATH`
for H264. IK and SimMPC come from robokit, not from here.

**Scene from perception:** `python -m roboskills.tools.mujoco --video V --masks masks/ --depth depth/depth.npz
--shape shape/ --collision collision/ --out scene/ [--gravity gravity/gravity.json --hands hands/hands.npz --track
poses.npz --support pedestal]` ([`mujoco.py`](mujoco.py)): a table plane fitted under the object (seeded by GeoCalib's
gravity, or without it by the object's lowest point) sets up; the object rests on a table box with its convex parts.
With `--support pedestal`, as in Do as I Do, there is no table: up is GeoCalib's, and a pedestal holds the object at
each end of its track where HaWoR's joints are 0.1 m or more from it. The hand (none without `--hands`) is Do as I
Do's MANO MJCF ([`mano/`](mano/right.xml), MIT, contacts off) whose qpos `fit_hand` fits per frame to HaWoR's 21
joints (damped least squares on MuJoCo's Jacobians, warm-started, unseen frames held); `scene.xml`, `scene.mjb`,
`trajectory.npz` (hand and object qpos per source frame, the object's last), `world.json`, `preview.jpg` beside the
source and, with `--track`, `mujoco_ext1.mp4`. With a DROID clip's `--robot robot.npz --camera camera.json`
([`eval/droid/read.py`](../../eval/droid/read.py)), Menagerie's Panda and Robotiq 2F-85 ([`franka/`](franka/panda_2f85.xml),
Apache-2.0 and BSD-2-Clause) replay its joints instead of the hand, in its base frame from its calibrated camera.

**Calls** (`roboskills.tools.mujoco.engine`; the real2sim workflow's SKILL.md gives the body schema and shows how
they combine):

- `build(path, actor_xml=None, bodies=(), cameras=(), option=None)` extends the actor MJCF with named primitive
  compounds (a body is a dictionary of MJCF attributes plus `free`, geom dictionaries in `parts`, `joints`, `sites`,
  `cameras` and recursive `children`; sizes are MuJoCo half sizes), sets `<option>` attributes (default
  `timestep=0.002`), writes `path` and returns it.
- `load(scene)` returns `(model, data)` for XML or MJB.
- `rollout(scene, out, ctrl, *, fps, frames, initial=None, camera=None, size=(640, 480), video=True, preview=12,
  source=None)` steps `ctrl` (one row of native actuator controls per simulation step) from `initial`
  (`qpos`, `qvel`) in CPU MuJoCo and saves the state at each of `frames` source frames.
- `measure(out, rows, objects={label: ("body"|"site", name)}, reference=None, projection=None)` and
  `render(out, *, camera=None, fps=30, size=(640, 480), video=True, preview=12, source=None)` read saved states only.

**Returns.** `rollout` writes into a new `out`: `scene.mjb`; `trajectory.npz` with the saved states,
`source_frame_indices` = arange(frames) and `source_time` (s); `diagnostics.json` with `frames`, `physical_seconds`,
`max_penetration_m`, `warnings`, `contact_frames` (per frame: named `geoms` pairs with `normal_force_N` and
`force_world_N` on `geoms[1]`), `scene_sha256`, `state_sha256` and `seconds`; `preview.jpg` (rows tiled with
`source`); with `video`, `mujoco_ext1.mp4` (H264) and `render.json` (`scene_sha256`, `state_sha256`, ...). `measure`
returns per saved row each label's `xyz` and 3x3 `rotation` in the reference's frame (world without one) and its
`pixel` under a 3x4 world-to-pixel `projection` (None at or behind the camera). Units: metres, seconds, newtons.

**GPU memory:** rendering only, about 0.1 GiB; physics runs on the CPU.

**Known failures:** controls outside an actuator's `ctrlrange` are refused (no clipping); `rollout` refuses an
existing `out`.

**Upstream:** [MuJoCo documentation](https://mujoco.readthedocs.io/).
