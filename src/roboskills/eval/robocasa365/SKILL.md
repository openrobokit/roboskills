---
name: robocasa365
description: Run RoboCasa365 kitchen tasks with RoboCasa's PandaOmron mobile manipulator (MuJoCo) behind the shared Robot contract, including its base, using the pinned RoboCasa/robosuite environment and assets.
---

# RoboCasa365 (PandaOmron kitchens)

The runtime for five of [RoboCasa365](https://github.com/robocasa/robocasa)'s
kitchen tasks with RoboCasa's PandaOmron: a Franka Panda arm on a torso lift on
an Omron holonomic base. RoboCasa builds the kitchen, the task, the robot's
start and the success check; this runtime adds cameras, control and the robot's
facts for the shared skills, including `Robot.base` for `bot.drive`. It follows
the setup of EFE's RoboCasa365 benchmark (pins, assets, the 13-channel
standard-mode action, robot cameras) but generates scenes from seeds instead of
EFE's frozen instances, and runs none of EFE's or Robo Use's code.

| Module | Role |
|---|---|
| `launch.py` | `make(task, seed="1", out=..., runtime=None, image=None, gpu=None)`: the shared launcher's entrypoint; starts the episode's simulator as a server ([eval/serve.py](../serve.py)) and returns its `Robot`. `episode(...)` builds the `RoboCasa` in the server. |
| `robot.py` | `RoboCasa`: seeded scenes, cameras and robot mask, arm/gripper/base control, `follow`, native scoring and terminal files; `replay(actions)` steps a recorded `actions.npy`. |
| `__init__.py` | The cohort: task names, step limits and RoboCasa's success conditions. |
| `ROBOT.md` | Arm, hand, base and how to drive it, reach, heights, cameras, budget and success, for the agent. |
| `setup.sh`, `pyproject.toml`, `uv.lock` | Pinned host environment, RoboCasa checkout and assets. |
| [`docker/robocasa365`](../../../../docker/robocasa365/Dockerfile) | The simulator image, `ghcr.io/openrobokit/roboskills-robocasa365:0.2.0`: `setup.sh` in a separate builder, with an independent Ubuntu runtime, in layers under 10 GB; runtimes pull it ([docker/README.md](../../../../docker/README.md)). |

## Prepare the runtime

Run from the repository root on a CUDA/EGL host with `uv` and `curl`:

```bash
bash src/roboskills/eval/robocasa365/setup.sh   # ROBOCASA=<checkout> to move RoboCasa from runs/vendor/robocasa
export MUJOCO_GL=egl
```

`setup.sh` syncs the venv (`.venv` here), installs the repository into it in
editable mode, and clones RoboCasa into `runs/vendor/robocasa` (gitignored).
RoboCasa joins the venv's path through a `robocasa.pth` entry, its
dependencies being in `pyproject.toml`. An editable install would build from
its manifest, which names every asset file: minutes locally, and far longer on
a network home. The script then downloads RoboCasa's asset archives into that
checkout's `robocasa/models/assets`, where RoboCasa reads them, checking each
against its SHA-256. These are EFE's six archives (`images/robocasa365/
fetch_assets.sh`): textures, AI-generated textures, Lightwheel fixtures, and
Objaverse, AI-generated and Lightwheel objects, 11 GB downloaded and 23 GB
unpacked. Each unpacked archive leaves a marker, so reruns skip it and take
under a second. A first run took 4–6 min (66 MB/s).
RoboCasa and robosuite also get their default private macro files, which stops
their warnings on every import.

Pins: RoboCasa `robocasa/robocasa@4f8a2980` (1.0.1), robosuite
`ARISE-Initiative/robosuite@5ce6643f` (1.5.2), MuJoCo 3.3.1, NumPy 2.2.5,
Python 3.11. These are EFE's RoboCasa365 pins. The rest of the host matches the
Robosuite runtime: PyTorch 2.11 CUDA 12.6, Warp 1.18 and RoboKit 1.0.0rc1.
RoboCasa installs without its dataset and training extras (LeRobot, Tianshou).
Robosuite's teleoperation, IK-controller and GUI OpenCV
dependencies are excluded too.

## Runtimes

`make` starts the simulator as a server and returns its `Remote`
([eval/serve.py](../serve.py)), so the launcher never holds the simulator.
`runtime=native` (the default, or `$ROBOSKILLS_RUNTIME`) runs the server with
this venv. `runtime=docker` runs it in
`ghcr.io/openrobokit/roboskills-robocasa365:0.2.0` (`image=` to override):
`setup.sh`'s venv, RoboCasa checkout and assets, with the host's `roboskills`
package mounted at `/src/roboskills` and `out` at its own path. `runtime=apptainer` runs a sandbox
of the same image from `~/.roboskills/images/roboskills-robocasa365-0.2.0`, built from the registry when missing. The
host keeps this venv either way, for the skills (RoboKit, SAM). In a
container the server copies the arm model, with its
meshes, into `out` (`model/right/`, `robot/assets/`), where the host loads
them. Besides the Robot contract the server offers `replay`.

```bash
docker pull ghcr.io/openrobokit/roboskills-robocasa365:0.2.0   # once per host; runtime=docker also pulls it when missing
```

`setup.sh` runs in a build stage that reads only `setup.sh`, `pyproject.toml`
and `uv.lock`, so other edits here keep its cache; it took 11 min. The image (33.4 GB) copies its result in layers under GHCR's 10 GB
per layer: Python (0.09 GB), the venv (8.13 GB), the checkout without its
assets (0.09 GB), the other assets (textures, generated textures, fixtures:
3.19 GB), the Objaverse and Lightwheel objects (7.9 GB), and the
AI-generated objects by name, a–h (6.43 GB) and i–z (6.55 GB). The build stage
makes everything readable to the caller, whom the container runs as
(`HOME=/tmp`). RoboCasa writes each fixture's and object's edited XML beside it
while it builds a kitchen, then deletes it, so the asset folders are writable
too; those writes stay in the container's own layer.

Check a runtime without a model: a no-op episode must fail, a recorded
successful episode's native actions (`replay`) must succeed, and the runtimes
must end at the same joints.

```bash
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl src/roboskills/eval/robocasa365/.venv/bin/python \
  -m roboskills.eval.check roboskills.eval.robocasa365.launch --runtimes native,docker \
  --out runs/check/robocasa365 --oracles ORACLES.json
```

`ORACLES.json` is your own manifest of recorded successful episodes ([format](../check.py)).

The oracle is an agent's successful `NavigateKitchen` seed-1 episode, recorded
on 2026-10-21 with this `robot.py`: 335 native steps, the base driving on 314
of them. `replay` steps them as recorded. The arm and hand then hold the last
action's targets and the base stands where it ended, for `finish`'s 10 still
steps.

Verified on a Quadro RTX 6000 on 2026-10-21: in both runtimes the
no-op fails and the oracle succeeds (335 steps, 345 with `finish`'s), ending at
the same joints to 0.0 rad; the whole check takes about 3.5 min. Driven through
the server, natively and in Docker alike, four cameras gave RGB-D and robot
masks (19 % of each view) and a 0.3 m drive back ended 2.5 mm from its goal. The
host's `T_world_base` moved 302.469 mm in place with the base, and RoboKit's TCP
on the served model with it. A `stop` callback ended an arm motion after 5
checks. The two runtimes agreed on
every number. `PickPlaceCounterToSink` seed 1's recording (206 steps, a grasp
and a short base move) also replays to success natively.

The manifest's 5 tasks, each with an agent's recorded success (CloseDrawer, DeliverBrewedCoffee, NavigateKitchen,
PickPlaceCounterToSink, TurnOnMicrowave), all pass in both runtimes (2026-10-22).

## Tasks

| Task | Steps | RoboCasa success (checked by `finish`) | Seeds 1, 2, 3: layout/style |
|---|---|---|---|
| `NavigateKitchen` | 2000 | the arm's shoulder within 0.2 m of RoboCasa's parking spot at the named fixture, heading within ~11° | 29/24, 51/16, 49/42 |
| `DeliverBrewedCoffee` | 3000 | mug and plate on the dining counter, centres within 0.3 m, the gripper 0.25 m from the mug | 29/15, 53/22, 51/9 |
| `PickPlaceCounterToSink` | 2000 | the object's centre in the sink basin, the gripper 0.25 m from it | 29/24, 42/9, 49/42 |
| `CloseDrawer` | 1500 | the drawer at most 5 % open | 29/24, 51/16, 49/42 |
| `TurnOnMicrowave` | 1500 | the start button touched with the door closed, the gripper 0.15 m from it | 29/24, 51/16, 49/42 |

The two mobile tasks need the base. In `NavigateKitchen` the target's parking
spot is 1.7–2.7 m from the start. In `DeliverBrewedCoffee` the plate is 3.0–5.3
m from the mug, which sits on the coffee machine where the robot starts. The
three atomic tasks start at their fixture. Their targets lie 0.47–0.81 m from
the shoulder, near the arm's 0.84 m top-down reach, so a small base move can
help. All five are RoboCasa365 tasks at the pin, and EFE's list holds three of
them (CloseDrawer, TurnOnMicrowave, DeliverBrewedCoffee).

Step limits allow about 40 tool calls at 20 Hz. Observation is free. An arm
move costs roughly 20–60 steps and a pick (approach, close, lift) or a placement
roughly 60–200. A drive costs about 60 steps per metre and 50 per quarter turn
on the shared planner's start-stop profile at 0.5 m/s and 1 rad/s, plus a few
settling. The atomic tasks get 1500 steps, 40 calls at about 35 steps each. The
pick-and-place and navigation tasks get 2000, for a few metres of driving and
looking around on top. `DeliverBrewedCoffee` gets 3000, EFE's cap for
RoboCasa365 episodes, for 3–5 m of driving with a mug in hand. EFE's
demonstration-tracking references take 85–2189 steps for its 50 tasks.

Seed `N` is RoboCasa's environment seed: `numpy.random.default_rng(N)` picks
the layout and style from all 60 × 60 (minus the task's exclusions), the
robot's start, the fixtures' state and the objects. `random` and `numpy.random`
are seeded with `N` too. RoboCasa removes duplicate counter-top regions through
a Python set of XML elements, which iterates in memory-address order, so a
seed gave one of several scenes from process to process (seen with
`NavigateKitchen`, seed 3). The runtime replaces that `set` in
`robocasa.models.fixtures.counter` with an insertion-ordered dict. All 15
cohort episodes now rebuild the same initial state in fresh processes; their
full `qpos` hashes were compared twice. Reset steps are not charged.

## Control and frames

One budget step is one native 20 Hz control step with EFE's 13-channel action:
seven absolute arm joint targets, the gripper, three base velocities, the torso,
and the hybrid-mode flag (-1). It goes through RoboCasa's composite controller:
robosuite `HYBRID_MOBILE_BASE`, with RoboCasa's `JOINT_VELOCITY_LEGACY` base
controller.

- **Arm**: robosuite `JOINT_POSITION`, absolute (kp 1500, damping ratio 1, no
  interpolation). `follow` streams the trajectory interpolated per step, adds
  the controller's steady lag (kd/kp = 0.052 s times the target velocity), and
  settles on the last waypoint for at most 30 steps. The arm and torso
  controllers are built in delta mode and switched to absolute after the reset.
  RoboCasa settles each new scene with ten zero actions, and in absolute mode
  zero means the arm's zero posture.
- **Gripper**: robosuite's sign-driven `GRIP`; jaw command `u` → `1 - 2u`.
- **Torso**: held at its reset height (0, its lowest) with absolute targets.
- **Base**: three joints move it in a fixed joint frame: forward and side
  slides, then a yaw hinge whose axis is 0.21 m behind the arm's mount. `Obs.base`
  is that hinge axis with the base's heading, which is the middle of the
  footprint and the point the base turns about. `Trajectory.base` waypoints map
  to joint targets exactly (`Kinematics`), with whole turns taken to start
  nearest the joint's yaw. Each step commands the path's joint velocity plus
  6/s times the measured joint error. The command is mapped through the legacy
  controller's heading rotation and the velocity actuators' ranges
  (`base_action`). The joints' dry friction (250 N, 250 N·m) stalls the
  kv = 1000/1500 velocity servos below 0.25 m/s / 0.17 rad/s, so that much is
  fed forward and faded out near zero velocity. A base more than 10 cm or 10°
  behind its path for 10 steps is blocked: the motion ends and the robot holds
  where it is. Without `path.base` the base holds its last target with the same
  feedback. `Base(vmax=0.5, wmax=1.0, radius=0.44)`: the actuators' 1 m/s and
  1.5 rad/s, less friction and feedback headroom. The radius bounds the
  0.70 × 0.50 m footprint about the hinge axis.
- **Arm model**: `robot0.xml` is the robot's MJCF cut to the Panda's
  `robot0_link0` subtree (arm and gripper; no base, no torso) at the origin.
  `Arm.T_world_base` is `robot0_link0`'s live world pose. It is updated in place
  on every `obs` and at the end of every `follow`, so it moves with the base
  and torso. The server sends it with every reply, and the host's copy is
  updated in place too.

## Observations

`robot0_agentview_left`, `robot0_agentview_center`, `robot0_agentview_right`
(on the torso, behind and above the arm) and `robot0_eye_in_hand`. Each gives
512 × 512 RGB, depth in metres (robosuite's `get_real_depth_map`), pinhole K
from `cam_fovy`, and the OpenCV-frame pose at its current place. Robosuite's
renderer hides collision geoms (group 0), so the cameras see visual meshes only.
`robot_mask` comes from one segmentation render per camera: every geom under
`robot0_base` (base, torso, arm, gripper), grown 2 px over anti-aliased edges.
The skills use it instead of the arm model, so the clouds lose the base and
torso the model lacks. `Obs.hand` is the measured jaw opening; `table_z` is not
set.

Rendering is robosuite's EGL context on the GPU. A caller may observe and move
the robot from threads of its own (the server calls it from one, the simulator
test from several), and an EGL context is current in one thread at a time. The runtime makes the context current for a whole observation, a whole
`follow` (for its video frames) or `finish`, then releases it. Nested uses
share one hold under a re-entrant lock. The reset also leaves it released.
Making it current and releasing it cost about 30 ms each on a busy GPU, so a
per-render release would triple the observation time. One observation (four
RGB-D renders plus four segmentation renders) takes 0.12 s on a busy RTX
A6000 and 0.14 s on an RTX 3090 or Quadro RTX 6000. The first takes 0.4–0.9 s. A control step costs
30–63 ms of CPU physics. A reset
takes 5–14 s, and 23 s with a cold cache.

## Run an episode

```bash
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl src/roboskills/eval/robocasa365/.venv/bin/python \
  -m roboskills.workflows.agent_as_policy.run \
  --adapter roboskills.eval.robocasa365.launch \
  --out runs/robocasa365-agent -- task=CloseDrawer seed=1 out=runs/robocasa365-agent
```

Perception requires the SAM 3 weights and their isolated worker. Agent episodes
also require the subscribed model CLI.
`out` receives the server's `sim.log` and `robot/` (`robot0.xml`). `finish` writes `native_result.json` (task, seed,
layout, style, steps, step limit, success, instruction, dt), `actions.npy`
(N × 13 native actions) and `video.mp4` (the center camera every 10 steps)
into the directory it is given, the launcher's `env0/`.

## Verify

```bash
# offline: base kinematics and action, arm model cut, following, budget
python -m pytest tests/eval/test_robocasa365.py
# in the simulator (EGL host): every cohort task, each seed
MUJOCO_GL=egl ROBOCASA365_SEEDS=1,2,3 uv run --no-sync --project src/roboskills/eval/robocasa365 --with pytest \
  python -m pytest -s tests/eval/test_robocasa365_sim.py
```

The simulator test builds each episode's `RoboCasa` in its own process
(`launch.episode`, as the server does) and resets it in one thread, then
observes from two others. It checks four cameras with sane depth and a robot mask. It drives the
base straight, sideways and turning, taking the other way when a fixture blocks
it, and requires each move within 2 cm and 2°. It moves the arm and requires
it within 0.01 rad. Before and after the moves, `Obs.base` and
`Arm.T_world_base` must equal the simulator's poses, and RoboKit's
`RobotModel(robot).tcp("right")` must match the grip site within 5 mm and 2°.
Finally it checks `finish`'s files. All 15 cohort episodes (seeds 1–3) passed
on 2026-10-21, in 1.3–2 min per seed on one GPU. Smooth drives
settled within 2.9–3.4 mm, and turns within 0.05–0.35 mm and 0.06–0.21°. One
side step pressed against a wall settled at 10.7 mm. On the way, a base following
the shared planner's 0.5 m/s profile trailed its path by 10–14 mm; a step to full
speed, by about 20 mm. The arm reached within 0.00025 rad. `Obs.base` and
`T_world_base` matched the simulator to 0.0 mm, and RoboKit's TCP matched the
grip site to 0.000 mm and 0.04°, before and after driving.

## Known issues

- Kitchens are tight. The base's rear sweeps 0.44 m when it turns, and
  RoboCasa parks it against a counter, often beside a wall. Turns and side
  steps from the start pose are often blocked until the robot backs away.
  `drive` plans no detour.
- The arm model has no base or torso, and `robot_mask` removes both from the
  clouds, so arm planning does not avoid the robot's own base. The base's front
  edge is 0.15 m ahead of the shoulder and below 0.38 m; keep the hand above it
  when working low in front.
- The base stops within about 3–4 mm of a smooth path's end. Its dry friction
  leaves that much where the feedback no longer overcomes it.
- MuJoCo's process-wide mesh cache stays on. EFE turned it off for bit-exact
  rebuilds within one process; this runtime builds one scene per process.
