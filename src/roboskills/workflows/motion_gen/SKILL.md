---
name: motion-gen
description: Turn a text, image or video request into a rendered motion video, choosing the models (Kimodo, GVHMR, Mixamo characters, RoboKit robots), writing the scripts, rendering, looking at the frames and revising.
---

# Motion generation

Turn the request into a rendered motion video. The request is plain text; files it refers to are in `inputs/` or at
the paths it gives. Work in the current directory or, at the repository root, in a new `runs/motion_gen/NAME/`: your
scripts, intermediate files and revisions all go there. Only the result is judged, so choose freely how to get there.
Start from the tools below; do not reuse files from other runs.
For an explicitly requested rerender, reuse the supplied saved scene and motion as declared inputs. Change only
presentation; preserve motion arrays, prop animation, frame count and fps, and verify this in the delivery notes.

## Approach

- Choose the motion source that fits: text-to-motion (Kimodo), motion recovered from a video (GVHMR), or motion
  you author directly (keyframes, inverse kinematics). Choose the body the request names: a robot from RoboKit, a
  Mixamo character (the character skill), or SMPL-X.
- If the request implies a scene, build it at real scale with the objects the action needs. The motion must respect
  it: support, contact and clearance, not just passing near it.
- Generate contact together with the body. When a hand must reach, hold or place something, give the generator
  that hand's pose at the contact frames. Decide where the body stands and which way it faces at each contact, and
  constrain the root position and heading there: a person faces what they work on. Reaching, stepping and turning
  then come out as one coordinated motion. Arms the task does not use hang relaxed and swing naturally. A generated
  hand lands near its constrained pose, not on it (centimetres off, its turn sometimes far off): settle the contact
  afterwards.
- Without a video to follow, grasp where a person would hold the object for the task (a bottle by its body, a mug by
  its handle), palm close and fingers wrapped. If the hand cannot oppose its thumb there, let the thumb brace instead
  of moving the grasp somewhere unnatural. How a hand can hold something depends on its size against the object's: a
  person whose fingers cannot reach around an object takes it in both hands or supports it from below. Resolve small
  overlaps by adjusting the grasp, not by opening it. After a release, the arm returns to a relaxed pose along a
  natural path.
- Edits made frame by frame, even smoothed afterwards, make limbs stiff or shaky; edits over the whole clip keep the
  source's timing and arcs.
- Try a short piece first to settle coordinates, scale, facing and timing before the full run.
- Render the complete motion with a wide view and close views of hands, contacts and feet. Look at the frames
  (read the images), not only at numbers. Judge the arms as you would a person's: shoulder, elbow and torso share
  the movement, and paths curve rather than snap. Look at hands and contacts from other angles than the video's
  camera too: what that camera cannot see (an object's far side, depth) always looks right from its own view.
  Confirm from the images that a hand really holds what it carries; a small distance alone does not show it.
- Frames where a hand or a held object leaves the video's picture, or is hidden, carry no observation of it: the
  motion there is yours to complete, as a person would naturally continue it, and only a camera that sees it shows
  whether it is.
- Check each piece as you build it: a value you estimated (an offset, a size, a position) is a guess until you have
  looked at and measured what it produces. Before the final render, check the complete candidate including the
  actual props. Inspect source-matched closeups and dense windows around the worst jumps, overlaps and placement
  drift. Repair visible defects and recheck contacts after smoothing.
- Prefer the best visually faithful checkpoint. A lower jerk or collision number does not justify losing the
  action, flattening its timing, moving the camera, changing prop geometry or dropping support constraints.
- For a photo, use the attached image before generating and visually compare the candidate's arms, legs and
  stance with it. A numerical hip-height change does not establish pose fidelity.
- For a video with several people, maintain person identities across frames, and use one actual camera estimate
  for every person. Common coordinates must agree throughout the clip; aligning only the first frame is insufficient.
  Verify source screen order and front/back spacing at the start, middle and end.
- Verify prop observations against the source before fitting: masks must contain the object rather than clothing.
  Use segmentation (SAM 3) or reviewed sparse state keyframes when colours are ambiguous.

## Acceptance

The result is judged by looking at it, on two things:

- It matches the video wherever the video is clear: poses (the hands' turns and fingers included), timing, where
  the hands and feet are, and what touches what: the same part on the same thing, held the same way, from the frame
  a touch starts to the frame it ends. Without a video, it does what was asked.
- It moves naturally, as a person would (a robot as its body allows), and its interactions are plausible. Where
  the video cannot tell, choose what is plausible.

And it stays within what a body can do: every joint within a person's range and turning continuously from frame to
frame, nothing passing through the body, an object or the scene, feet bearing weight where they stand. A joint
twisted or bent past a person's range, or snapping round between two frames, is a defect to repair, never a
limitation to record; so is any defect you can see where the video is clear.

## Deliver

- `final/`: the video as `final/video.mp4` (H.264), a contact sheet of stills, and `notes.md` with what you built, the
  commands that reproduce it, the revisions, and the problems that remain, measured on the delivered motion itself.
  Save the full motion as final/motion.npz with fps and a per-frame turns, positions or qpos array so delivery can
  verify video coverage. Keep each revision in `revisions/NN/` with its video and one line on what changed and why.
  Say plainly that playback is kinematic unless you simulated physics.
- Before you finish, confirm with `ffprobe` that `final/video.mp4` decodes without errors and has `motion.npz`'s frame
  count and fps.
- Deliver when you find nothing left to repair, or at a given deadline the best reviewed candidate with what remains
  listed. A deadline is a limit, not a reason to keep working. The rendered video is the delivery: a repair that
  leaves too little time to render the whole clip costs more than it fixes, so keep the final render its time. Stop
  only your own verified processes; preserve other agents, shared services and tmux servers.

## Compute

Use only the GPU(s) in `CUDA_VISIBLE_DEVICES`; if `MOTION_GEN_GPU_GB` is set, the card is shared, so keep
your total GPU memory under it. Bound every process you start.

## Tools

Each tool has a setup file in `runs/motion_gen/env/` (`kimodo.sh`, `gvhmr.sh`, `blender.sh`, `character.sh`,
`wilor.sh` and `sam3.sh`), written by `src/roboskills/workflows/motion_gen/setup.sh`: if that folder is missing, run
it once from the repository root.
Source it in the shell that runs the tool; it names the Python to use and shows a minimal verified call. Each tool
has its own Python environment: keep a tool's scripts, inputs and outputs here, and do not install packages into it;
make your own venv here if you need more packages. The
repository's own environment (`uv run` from the repository root) has RoboKit, NumPy, SciPy and trimesh.
Reuse these installations; do not rebuild an environment. Inspect a tool's API through the Python its setup file
names (for example `inspect.getfile` and `inspect.signature`): another environment may hold a different version.
Read video frames with ffmpeg (raw frames through a pipe) rather than OpenCV's `VideoCapture`, which wraps the same
decoder with less control and on a busy shared machine reads at an erratic pace.

**Kimodo** (`kimodo.sh`): text-to-motion, with a person's coordination and timing. `Kimodo-G1-RP-v1` generates
the Unitree G1 body directly (29 joints, no fingers), 30 fps, about 10 s per call.
With the human model `kimodo.sh` names (`MOTION_KIMODO_HUMAN_MODEL`, SMPL-X),
`$KIMODO_PYTHON -m kimodo.scripts.generate "PROMPT" --model "$MOTION_KIMODO_HUMAN_MODEL" --duration 5 --output
work/body` also writes `work/body_amass.npz`, the AMASS SMPL-X file the character skill reads. Period-separated phase
prompts with a space-separated duration string generate a longer coherent action in one call. If the environment sets
`TEXT_ENCODER_MODE=local`, Kimodo encodes locally and needs no encoder service. Otherwise its text encoder is a
service (about 15 GB of GPU memory, ready after about 40 s): `kimodo.sh` says whether one is already running or how
to start one; stop one you started when
done. Output `STEM.csv`: per frame root xyz, root quaternion
wxyz and the 29 joint angles, Z-up, +X forward (MuJoCo frame). `STEM.npz`: joints, rotations and foot contacts in
Kimodo's frame, Y-up, +Z forward, +X to the robot's left. `--constraints FILE.json` uses Kimodo's frame and starts at
root (0, 0) facing +Z, e.g. `[{"type": "root2d", "frame_indices": [0, 119], "smooth_root_2d": [[0, 0], [0.5, 2.0]]}]`;
`right-hand`, `left-hand`, `right-foot` and `left-foot` constraints take a whole G1 pose per constrained frame:
`{"type": "right-hand", "frame_indices": [...], "local_joints_rot": (N, 34, 3) axis-angle, "root_positions": (N, 3),
"smooth_root_2d": (N, 2)}`. They also pin the root position and heading on those frames, and the hand lands within
a few centimetres. The human model takes the same constraint types with its own SMPL-X joints (`fullbody` keyframes
included, see `kimodo/constraints.py`). Build G1 ones from G1 qpos rows, such as an inverse-kinematics pose at the contact:
`MujocoQposConverter(build_skeleton(34)).qpos_to_motion_dict(qpos, 30)` (from `kimodo.exports.mujoco` and
`kimodo.skeleton.registry`) gives `local_rot_mats`, `root_positions` and `smooth_root_pos` (use its x and z);
`kimodo.geometry.matrix_to_axis_angle` converts the rotations. The G1 hand joint is `right_wrist_yaw_skel`. The models
were trained on clips of at most 10 s: plan a longer action as phases on one task clock and check the joins.

**GVHMR** (`gvhmr.sh`): recovers SMPL-X body motion from a video, about 25 s and 7 GB per 4 s clip.
`gvhmr.recover(video, static_camera=True, device="cuda", output_root=..., output_dir=...)` returns a result whose
`save_npz` writes `world_*` and `camera_*` `global_orient`, `body_pose` (21 joints, no hands), `betas`, `transl`
at 30 fps. The world frame is Y-up in metres with the ground near y = 0; `transl` is not the pelvis position, so run
the SMPL-X model for joints (only SMPL-X body models are installed). With several people it keeps the largest track;
to choose a person, write `<output_dir>/preprocess/bbx.pt` with that person's per-frame `bbx_xyxy` and `bbx_xys`
before calling it, and use a fresh output directory per person.

**WiLoR** (`wilor.sh`): a video's hands, which GVHMR does not give. Per frame its detector gives hand
boxes and left or right, and per box the model gives MANO's `global_orient` and `hand_pose` (15 joints, each in its
parent), `pred_keypoints_3d` (21) and `pred_cam`. A visible hand's shape and turn are good; a small, blurred or
covered one is often missed or read flipped. It reads each hand from a crop of the frame it is given, so give it the
video's frames at full resolution: scaled-down or recompressed frames lose the fingers.

**SAM 3** (`sam3.sh`): an object's masks from a text prompt (a noun phrase such as "umbrella"), one image at a time:
there is no propagation through a video, so detect frame by frame (about 1 s per 1080p frame for two prompts) and
follow the object from frame to frame by its position. A small or fast-moving prop is sometimes missed, or ranked
below a wrong detection, at the default threshold of 0.3; a second phrasing and a lower threshold (0.2) often find
it. A mask says where the object is in the picture; its depth, and its turn about an axis it is symmetric about, it
cannot tell.

**Characters** (`character.sh`): a person's SMPL-X motion (a video's GVHMR, or Kimodo's) onto a Mixamo character, and
its joints and skin measured against an object. `character.sh` sets `CHARACTER_SKILL`, the path of its SKILL.md:
read that file before using it.

**Blender** (`blender.sh`): the `bpy` module, for FBX characters and rendering. Set the scene fps before importing an
animated GLB. For several actors and props in Blender 4.5, export their intended active actions together with
`export_animation_mode="ACTIVE_ACTIONS", export_merge_animation="ACTION"`; remove conflicting imported NLA
tracks only after preserving the intended action. Reimport at the target fps and verify every actor and prop
at the start, middle and end before rendering. Mixamo bone names carry a prefix that varies by file
(`mixamorig:Hips`, `mixamorig2:Hips`); match on the part after the colon. An imported Mixamo armature object is
rotated +90 degrees about X and scaled 0.01, so armature space is Y-up in centimetres, with a T-pose bind facing -Y.
Render with Cycles on the GPU (about 1.5–2 s per frame, whatever the samples); EEVEE (about 0.6 s a frame after about
35 s compiling shaders) and Workbench (about 0.15 s) open their GL context on the first of your GPUs.

Run a final render in the foreground with a bounded timeout; its completed invocation is the receipt. Avoid
background jobs and PID polling: `pgrep -f` can match its own command, and `kill -0` also succeeds for zombies and
can wait forever after a render has finished. Reuse completed frame files when resuming. When two render GPUs are
explicitly allocated, split saved frame indices into even/odd sets and run one renderer on each card, then wait for
those exact child processes and combine the numbered PNGs.

**Shared renderer** (`$MOTION_STUDIO`, set by `blender.sh`): render the final video with it; preview however you
like (`--engine eevee` or `workbench` draws the same camera faster). `$BLENDER_PYTHON "$MOTION_STUDIO" SCENE.glb
MOTION.npz --out DIR` renders an animated GLB scene with GPU denoising, loading the scene once, so a few samples per
pixel (`--samples`, default 32) already give a clean frame; `--frames` renders chosen saved frames, `--final` every
frame (render the full final clip once). Each run writes
numbered PNGs, a contact sheet and `render.json`. MOTION contains fps, turns (for frame count) and positions (all
characters' joints, or a robot's links, for framing) named by `joint_names` or `link_names`; optionally per-frame
`camera_poses` [T, 4, 4] (world from camera; x right, y down, z ahead) and `camera_intrinsics` [T, 3, 3] for an
`image_width` x `image_height` picture, `studio_bounds` [2,3] to keep relevant props in the default framing, and
`studio_camera_direction` [3] pointing horizontally from the scene toward the default camera. Cameras:
- by default one fixed camera frames everything that moves over the clip, from `--view AZ,EL`: degrees around the
  vertical from studio_camera_direction (else from the video camera's side), counterclockwise seen from above, and
  above the ground (default 0,22; a negative azimuth as `--view=-90,10`). Choose a front three-quarter view that shows
  the action and keep it fixed across revisions; lower elevations keep a long path's actor large;
- `--camera source` draws from the video's camera at the video's aspect;
- `--frame-on NAMES` frames only the named joints or links, GLB objects or `prop`; with `--follow` the camera re-frames
  on them every frame, a close-up that moves with the hands, from any `--view`;
- `--camera-from DIR` reuses an earlier render's camera, so revisions compare from the same place.
Render each view once and put views and the source video side by side with ffmpeg. Write SCENE.glb from your Blender
scene with `bpy.ops.export_scene.gltf(filepath=..., export_format="GLB", export_animations=True)` after setting the
scene fps; it takes seconds. Choose presentation from the request: studio is the default for isolated motion;
`--environment outdoor` preserves authored ground/prop colours and supplies daylight, and `--environment warm` an
interior palette with window light (`--kitchen` adds a kitchen backdrop where the scene has no room). Build the
scene's appropriate ground and visible markings; do not turn an outdoor activity into a white studio. Keep the
selected environment fixed across revisions. `--exposure` adjusts brightness in stops (default 0). Keep subjects and
props visibly distinct from the background using authored materials; a white robot on a white floor needs tonal
contrast. Presentation changes must preserve checked motion. A scene without a character plays its animation from
frame 0. Bake all scene animations into the GLB at that same fps. Preview from the open side of a room: walls must not
obscure the interaction. Keep reference obstacles present when changing the camera. Align actor and prop keys to
the imported action's actual first frame. Verify exported geometry against saved poses at contact frames; a
matching object origin alone is insufficient.

**RoboKit** (prepared runtime): robot models and kinematics. A robot's inverse kinematics and motion planning use
RoboKit's solvers (`robokit.helpers.ik`, `robokit.helpers.motion_plan`), not one you write. The current native SDK uses
`IKSolver(config, robot=robot, target_links=links, device="cuda:0")` and
`solve_numpy(targets, init_state=seed, rest_state=seed)`. Targets are xyz+wxyz `[T,L,7]`;
convert matrices with `robokit.xform.numpy.tf_mat_to_pose7`; batch all saved frames. Seed with
`robot.state(q=wp.from_numpy(q_init.astype(np.float32), dtype=wp.float32, device="cuda:0"))`. Without
`enable_T_world_base` the solver puts the robot's base at the world origin, so world targets and a world scene are
read from the wrong place. A base that moves through the clip (a humanoid's pelvis) is given per frame and held there:
`cfg.enable_T_world_base = True`, `cfg.base_active_mask = (0.0,) * 6` (1.0 frees that axis of x, y, z, roll, pitch,
yaw), and `T_world_base=` the base poses as xyz+wxyz (`robokit.utils.warp_utils.wp_vec7`) in the seed and rest states;
targets and the scene then stay in world coordinates, as in `09_humanoid_offline.py`.

Use `robokit.helpers.motion_plan.presets.build_offline_ik_config(num_candidates=1, num_samples=4,
num_refine_samples=4, cuda_graph_mode="none")` for an initial small IK population;
increase it only on failed targets. Its optimizer list can use `dataclasses.replace(opt, linear_solver="cg")`
for each `cfg.solver.optimizer` entry: the default fused dense Cholesky can exceed device shared memory
on larger full-body models. Verify a nontrivial reachable target before the complete clip; a zero-error rest
pose does not establish that a GPU kernel ran. The solver keeps the robot clear of a scene it is given, on every
frame at once, which an edit to one stretch of the motion does not; it does not choose the path for you (over a
rim rather than through it), and its spheres can miss small parts such as fingers:
`IKSolver(..., scene=WarpScene(...))` with the scene's surfaces as `robokit.geom` boxes or meshes, the robot loaded
with its collision spheres (`Robot.load(urdf, load_collision_spheres=True, collision_spheres_path=...)`; a robot
module's `COLLISION_SPHERE_PATH`). `robokit/examples/motion_plan/09_humanoid_offline.py` plans G1 hand goals around
scene boxes with both feet planted. What the robot touches stays in its scene: a contact rests on the surface rather
than opening a hole in it. That example gives the foot spheres zero margin so standing on the floor costs nothing;
`robokit/examples/grasp/01_basic.py` keeps the grasped object in the collision scene and pulls chosen contact points
onto its surface. The spheres approximate the links, so keep the preset's clearance margin, and check clearance on the
native link surfaces (the robot's geometry queries).
Inspect a different SDK's signature once rather than mixing this runtime with the repository venv.

`Robot.load(urdf, load_meshes=True)` loads native link surfaces from
`robot.spec.link_visual_geometries[name].to_mesh()`; FK is
`robot.forward_kinematics_via_matrix_numpy(q, T_world_base=base, device="cpu")`.

`robokit.assets.robots.assembly.g1_inspire` has the G1 with Inspire hands (`URDF_PATH`, `REST_Q`, finger joints
`R_index_proximal_joint` and so on); `robokit.robo.Robot` gives forward kinematics and `robokit.helpers.ik` inverse
kinematics. SMPL-X body models are in `~/.robokit/body_models/smplx/`.
