---
name: real2sim
description: Reconstruct a robot or human demonstration video as a native MuJoCo scene, trajectory and render that looks like the source and is physics.
---

# Real2sim

Reconstruct a source demonstration as one native MuJoCo delivery that looks right and is physics. The brief supplies
permitted inputs, target assets, clocks and constraints. Never read another case, previous reconstruction or
comparison output. Do not launch nested model CLIs.

- **Looks right:** each frame of your render is similar to the source frame at the same index: the camera's view, the
  table and background, every object's shape, size, color and place, the actor's pose and the motion's timing.
  Reviewers place the two side by side at uniform frames and at your event frames.
- **Physics:** free objects move only by contact with the actor, which its own actuators drive (requirements below).
  A check re-steps MuJoCo from each saved state with its ctrl and compares the objects with the next saved state.

## Default: two reconstructions, then one

1. **Start the code workflow first, in the background**, in a workspace W under your output:
   `python -m roboskills.workflows.real2sim --workspace W --video SOURCE --object "NAME" --tools sam3 moge3 geocalib
   sam3d hawor align track coacd mujoco retarget mpc --support pedestal --retry` (NAME: the object the actor
   handles), its output to a log. It takes about 15 minutes, mostly on the GPU, and
   writes a folder per tool ([README.md](README.md)): masks, depth, the object's mesh and pose per frame, the hand, a
   scene and, for a hand target, `retarget/` and `mpc/`, a physics rollout (scene.mjb, trajectory.npz with qpos,
   qvel, ctrl and time per source frame, mujoco_ext1.mp4). Each tool's status.json says whether its check passed.
   Its rollout is physics but often looks less like the source: its camera, depth and object size can be off. A
   robot target runs `--tools pnp` on a workspace its reader prepares (README.md).
2. **Meanwhile, reconstruct the scene yourself from the source frames alone**, without the workflow's outputs or
   tools: the camera, the table and background, every object's shape, size, color and texture, the actor and its
   motion. Render it from your camera and compare it with the source frame by frame.
3. **When the workflow is done, make one delivery that looks right and is physics.** Compare both reconstructions
   with the source and decide, part by part (camera, geometry, appearance, object motion, actor motion), what to
   keep. Physics comes from a native rollout in which objects move only by contact: the workflow's own (its controls
   replayed in a scene with your appearance, where the geometry they touch is unchanged), its tools rerun on your
   parts (`--tools ... --redo ...` on a copy of W), or SimMPC run yourself (below).
4. **Check before you deliver:** the re-step check above, and the render beside the source at uniform and event
   frames.

These are defaults, not rules: use any tool at any step, change the order, or skip a step that does not help. Wait
for every process you start before you finish. The request gives an absolute Unix deadline; check `time.time()`
before long calls, and by 75% of the budget start the final native delivery. Keep terminal output to selected
frames, extrema and paths.

Pass `source=<source video>` to `rollout` or `render`: its `preview.jpg` then shows source | native | 50% blend for
each sampled frame. In the blend, table edges, the actor, objects and contacts should overlap; doubled outlines are
registration or timing errors.

## Native tools

Import `roboskills.tools.mujoco.engine`; read only a specific definition when
blocked. The brief gives installed Python and asset paths.

- `build(path, actor_xml=None, bodies=(), cameras=(), option=None)` extends MJCF
  while preserving assets/defaults/actuators. Bodies contain MJCF attributes plus
  `free`, geom dictionaries in `parts`, `joints`, `sites`, `cameras` and recursive
  `children`. Sizes are MuJoCo half sizes. `option` sets MJCF `<option>`
  attributes (default `timestep=0.002`); for SimMPC use
  `option=dict(integrator="implicitfast", density=0, viscosity=0)`. Put every
  model setting in the scene file; never edit a loaded model in Python, or replay
  diverges from your plan. Use complete XML for complex articulation, multiple
  actor copies or includes with different asset paths.
- `load(scene)` returns `(model, data)` for XML or MJB.
- `measure(candidate_dir, rows, objects={label: ("body"|"site", name)},
  reference=label, projection=P)` returns, per saved row, each label's `xyz` and
  3x3 `rotation` in the reference's frame (world without one) and its `pixel`
  under a 3x4 world-to-pixel `P = K @ [R | t]` (None behind the camera), without
  stepping. Use it for grasp drift, support and projected landmarks instead of
  writing loaders.
- `rollout(scene, candidate_dir, ctrl, fps=F, frames=N, initial=dict(qpos=q0,
  qvel=v0), camera="source", size=(W,H), video=False, preview=12, source=video)`
  steps `ctrl` (one row of native actuator controls per simulation step, at least
  `round((N-1)/F/timestep)` rows) in CPU MuJoCo, saves the state at each source
  frame and returns diagnostics. Controls stay inside native ctrlrange; nothing is
  clipped. `report["contact_frames"][i]` lists named `geoms` pairs at source frame
  i with summed `normal_force_N` and `force_world_N` (on `geoms[1]`). Query
  relevant pairs at event frames only. Set `size` to the source aspect ratio.
- `render(candidate_dir, camera="source", fps=F, video=True, preview=12,
  source=video)` re-renders its saved states.
- `python -m roboskills.tools.moge3 --video V --out DIR` runs isolated MoGe depth
  for every frame into `DIR/depth.npz` (`depth`, `valid`, `K`, `frames`; `--frame`
  sets the reference frame whose field of view the others reuse).

`rollout` refuses overwrite; use distinct candidate directories. It writes
`scene.mjb`, `trajectory.npz`, `diagnostics.json`, `preview.jpg` and, with
`video=True`, `mujoco_ext1.mp4` plus `render.json` binding model/state hashes.

robokit IK (GPU, batched over key frames; joint order is
`robot.spec.actuated_joint_names`, so map it to actuators by name). Parse every
actor with its world link kept, so link poses are MuJoCo world body poses;
floating hands and grippers with offset joint anchors load too. `PixelTask` adds
image residuals in pixels for link origins seen by the source camera (intrinsic `K`,
4x4 world-to-camera `T_cw`), for landmarks whose 3D position is unknown:

```python
import copy
from robokit.robo import Robot, RobotSpec
from robokit.helpers.ik import IKSolver, presets
from robokit.terms import PixelTask

robot = Robot(spec=RobotSpec.parse(actor_xml, keep_mjcf_world_link=True))
ik = IKSolver(presets.basic, robot=robot, target_links=["tool_link"], device="cuda")
q = ik.solve_numpy(targets).q.numpy()  # targets (K, n_links, 7) as xyz + wxyz
links = [robot.link_names.index(name) for name in ("wrist", "elbow")]
pixels = PixelTask(
    frame_index=links, uv=observed, intrinsic=K, T_camera_world=T_cw, weight=0.05, device="cuda"
)  # uv (frames, 2, 2)
ik = IKSolver(copy.deepcopy(presets.basic).add(pixels), robot=robot, target_links=["tool_link"], device="cuda")
```

`weight` is per pixel (0.05 makes 4 px cost like 1 cm of a target). Compiling a
solver's GPU kernels can take a minute for a new robot or residual size; solve all
key frames in one batch, keep landmark counts fixed and reuse the solver.

SimMPC, planning on MuJoCo Warp and executing in CPU MuJoCo (times in steps):

```python
from robokit.helpers.sim_mpc import SimMPC, SimMPCConfig
from robokit.helpers.sim_mpc.mjwarp import MjWarpSim
from robokit.terms import ControlTask, GraspTask, StateTask

config = SimMPCConfig(
    num_samples=256,
    horizon_steps=250,
    node_steps=50,
    ctrl_steps=125,
    max_iterations=3,
    min_improvement=0.005,
    patience=1,
)
pad = config.horizon_steps + config.ctrl_steps
extend = lambda x: np.concatenate([x, np.repeat(x[-1:], pad, 0)])  # every reference: T + 1 + pad rows
weight = np.zeros((T + 1, model.nv))  # per-dof tracking weights
weight[grasp:, obj_dof : obj_dof + 3], weight[grasp:, obj_dof + 3 : obj_dof + 6] = 1.0, 0.1  # per m, per rad
weight[:, actor_dofs] = 0.03  # weak pull toward the reference actor joints in qpos_ref
costs = [
    ControlTask(0.03, extend(ctrl_ref), norm=True),
    StateTask(extend(qpos_ref), extend(qvel_ref), extend(weight), pose_tolerance=0.005),  # qvel_ref: diffs
    GraspTask(contact_body, obj_body, extend(offset), extend(carrying * 1.0), tolerance=0.005, object_frame=False),
]  # offset: contact body minus object, world frame
sim = MjWarpSim(model, start, nconmax=200, njmax=1000, cpu_execute=True)  # start: MjData after mj_forward
(executed,), plans = SimMPC(sim, extend(ctrl_ref), ctrl_std, costs, config).solve(T)
engine.rollout(scene, out, executed["ctrl"][:T], fps=F, frames=N, initial=dict(qpos=start.qpos, qvel=start.qvel))
```

`ctrl_std` is per actuator in control units, small for arm joints (about 0.015
rad) and zero for commands you fix; keep `ctrl_ref` inside native ctrlrange.
Replaying the executed controls from the same state reproduces SimMPC's states.
Samples run in parallel on the GPU, so solve time grows with `horizon_steps`,
`max_iterations` and the motion length, not with `num_samples`: this config costs
about 2-3 s per simulated second, so keep 256 or more samples and solve the
complete sequence rather than shrinking the search. Solve a prefix first only to
check the grasp. Raise `nconmax`/`njmax` if MuJoCo Warp reports overflow. If
SimMPC cannot run the scene, replay interpolated reference controls instead and
say so in `result.json`.

## Physical and visual requirements

- Interacting movable props, including receivers resting on a table, have free
  joints and plausible mass/friction; stillness is not anchoring evidence. Hinges
  and slides remain physical. No carried-object welds, pose resets/animation,
  reference forces or mocap attachments. Execution may change actor controls,
  never steer object poses; SimMPC costs only choose controls. Rendering may
  restore saved states.
- Preserve openings, thin walls, insertion depth and asymmetric ends in collision
  geometry (compounds or convex decomposition in one object frame). Do not change
  hardware scale or suppress actor-object collisions.
- Dropping, unintended receiver movement, excessive penetration, implausible force,
  unstable controls and solver warnings are failures, never suppressed. Keep the
  source actor visible and continuous; no source-pixel backplates.
- Native validity, task outcome and source fidelity are separate; occlusion is
  uncertainty, not success.

## Delivery

Under `deliver/` retain `scene.mjb` with XML/assets or the builder; `trajectory.npz`
with actual `qpos`, `qvel`, `ctrl`, `time` and complete `source_frame_indices=
arange(N)`; unchanged diagnostics; and H264/yuv420p `mujoco_ext1.mp4` with N frames
at source display FPS, all written by one `rollout(..., video=True)` into
`deliver/`. Optional `deliver/result.json` (under 2 KB) records your task
assessment, events, estimated versus measured quantities, time mapping, anchored
supports, known limits and evidence paths. A completed turn, movie or low residual
does not certify success. End with a short factual status.
