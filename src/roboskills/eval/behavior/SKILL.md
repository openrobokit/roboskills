---
name: behavior
description: Run BEHAVIOR-1K episodes (the 2026 BEHAVIOR Challenge's R1Pro mobile manipulator on OmniGibson 3.9.2 / Isaac Sim 5.1, in the official image) behind the shared Robot contract with a mobile base, and prepare or troubleshoot the image, dataset and host environment.
---

# BEHAVIOR-1K

The runtime for any workflow or policy: a 2026 BEHAVIOR Challenge task instance in OmniGibson, the R1Pro behind
[`roboskills.eval.interface`](../interface.py) with a mobile `Base`. The simulator runs in the official image, one
process per episode; the launcher side (roboskills, RoboKit, SAM) runs in this runtime's venv on the host.

| Module | Role |
|---|---|
| `launch.py` | `make(task, seed, out, gpu, ...)`: the simulator in the image on one GPU (docker; native inside the image; apptainer), the robot; `close()` |
| `robot.py` | `Behavior`: arms' joint targets with lead and settle, base waypoints tracked with feedback (blocked base ends the motion), hand settle, cameras, `finish` |
| `sim/server.py` | image side: the challenge's `Evaluator` with the instance loaded, the 30 Hz control loop, the BDDL check, rendering; RoboDojo's socket wire |
| `hand.py` | the gripper's facts (TCP, travel), the RoboKit arm model from the robot assets' URDF |
| `goal.py` | `describe(problem)`: a BDDL goal as a sentence, for tasks the challenge gives no sentence |
| `check.py` | the end-to-end check below (`check.json`) |
| `__init__.py` | `TASKS` (the cohort, step limits, sentences), `INSTANCES` (seed -> instance), `data()` |
| `ROBOT.md` | the robot, base, cameras, budget and goal predicates, for the agent |
| `setup.sh`, `pyproject.toml`, `uv.lock` | host venv, image pull, dataset download |

## Pins

| Component | Pin |
|---|---|
| Simulator image | `docker.io/stanfordvl/behavior:3.9.2@sha256:2702900192741904e69a573aa7869ea51c592586f9791050796efc37faa893fb` (BEHAVIOR-1K `b1979916`, OmniGibson 3.9.2, Isaac Sim 5.1.0, Python 3.11; 28.5 GB, image ID `f379e08c1dc6`) |
| Dataset | BEHAVIOR's own downloader in that image: `omnigibson-robot-assets` 3.8.2, `behavior-1k-assets` 3.9.0, `2026-challenge-task-instances`, `omnigibson.key` (35 GB; the BEHAVIOR Data Bundle licence: no redistribution) |
| Host venv | Python 3.10, PyTorch 2.11 CUDA 12.6, Warp 1.18, RoboKit 1.0.0rc1 (the Robosuite runtime's rev) |

The image is used as published: `launch.py` starts it with `sleep`, makes OmniGibson's `.kit` target directory writable
and the app-data volume the caller's (as root, in that container only), and runs the simulator as the caller.

## Prepare the runtime

```bash
export BEHAVIOR_DATA=$PWD/runs/vendor/behavior-data
bash src/roboskills/eval/behavior/setup.sh                   # venv; image pull on a docker host
bash src/roboskills/eval/behavior/setup.sh --accept-license  # elsewhere: also downloads the dataset (~10 min at 140 MB/s, 20 more unpacking)
```

A dockerd running in an apptainer may bind only `/home` of the host's paths: bind mounts must then live under `/home` (a host `/tmp` path
mounts an empty root-owned directory), which is why app data lives in docker volumes.

GPU: an RTX-class card with RT cores (Quadro RTX 6000 verified; not A100); 5-6.6 GB of GPU memory and about 7 GB of
host RAM per simulator (resident), 8 CPU threads (`OMP_NUM_THREADS`). Startup: 8-9 minutes on a cold app-data slot
(Kit compiles shaders), 3.5-5 on a warm one; the scene is the challenge's partial load of the task's rooms.

## Run an episode

```bash
CUDA_VISIBLE_DEVICES=5 src/roboskills/eval/behavior/.venv/bin/python -m roboskills.workflows.agent_as_policy.run \
  --adapter roboskills.eval.behavior.launch --out runs/behavior/glass-1 \
  -- task=dispose_of_glass seed=1 gpu=5 out=runs/behavior/glass-1/server
```

`gpu` is the physical card the container gets; `CUDA_VISIBLE_DEVICES` keeps the host side's RoboKit/Warp on the same
card. `seed` 1-3 is the challenge's public test instance 301-303 (`mode=public_test`, its first three). `out` holds the
robot model (`r1pro_arms.urdf`) and `behavior_sim.log`; the launcher's episode directory gets
`native_result.json`, `actions.npy` (19 per step: each arm's 7 joint targets and opening, the base's world target) and
`video.mp4` (the head camera, a frame per 10 steps).

## Check a host (smoke test)

```bash
CUDA_VISIBLE_DEVICES=5 BEHAVIOR_DATA=... src/roboskills/eval/behavior/.venv/bin/python \
  -m roboskills.eval.behavior.check task=turning_on_radio seed=1 gpu=5 out=runs/behavior-check/turning_on_radio-1
```

It boots the episode in one thread and drives it from another, then writes `check.json`: startup, observation
latency, GPU memory, each camera (depth valid, the floor back-projected through K and the pose to z = 0), the robot
model's TCP against the simulator's `eef_link` and finger midpoint (before and after the base moves), base waypoints
(forward, back, sideways, turns) against their targets, `Obs.base` and `T_world_base` against the simulator, both arms
to joint targets, 4 m straight ahead into whatever is there, and `finish`.

Verified 2026-10-21 for every cohort task and instance (15 episodes; Quadro RTX 6000s, docker):
each started headless on one GPU, reset, and gave three RGB-D cameras with every depth pixel valid (98-100 %) whose
back-projected floor lies at z = 0.000 to -0.006 m (the scenes' floor tops). Base waypoints (0.3 m forward and back,
0.2 m sideways and back, 60 degree turns, 0.2/0.1 m with a 30 degree turn) ended within 0.5 mm and 0.3 degrees; `Obs.base`
matched the simulator exactly and `T_world_base` matched `torso_link4` to 0.0 mm. RoboKit's `RobotModel.tcp` matched the
simulator's `eef_link` to 0.0 mm and 0.013 degrees for both arms, before and after driving and turning, with the finger
midpoint 0.0 mm off the TCP's axis. Both arms reached joint targets within 0.0004 rad. Driving 4 m ahead stopped at the
first touch (after 0-1.9 m; re-checked with the no-push guard, the robot leaning 2.5 degrees at most) and held there. `finish` wrote `video.mp4`, `actions.npy` and
`native_result.json`.

| | turning_on_radio | store_honey | composting_waste | installing_a_scanner | dispose_of_glass |
|---|---|---|---|---|---|
| startup, warm app data (s) | 266-286 | 212-233 | 216-226 | 251 | 247 |
| startup, cold (s) | 537 | | | 480-490 | 469-551 |
| control step without rendering (ms) | 46 | 38-39 | 37-38 | 36-39 | 33-37 |
| `obs()` with cameras and robot masks (s) | 0.18-0.40 | 0.15-0.23 | 0.16-0.23 | 0.18-0.21 | 0.18-0.33 |
| GPU memory (GB) | 5.7-5.9 | 4.9-5.1 | 5.1-5.3 | 5.7-6.4 | 5.5-5.7 |

The first observation is the slowest; the robot's masks add about 40 ms (the simulator's side: 0.14 s median, 0.10
without). `obs(cams=False)` takes 0.2 ms. The shared `bot` verbs (`Tools`) ran on it: `drive` 0.3 m in 27 steps, a 45 degree turn in
45, both within 0.1 mm and 0.01 degrees; `move` of each arm 0.25 m forward and 0.35 m up, reached to 0.7 mm in 53 steps;
`hand` closed (15 steps, held for the assisted grasp) and opened (6); `home` 53 per arm.

## Check without a model

```bash
CUDA_VISIBLE_DEVICES=0 BEHAVIOR_DATA=... src/roboskills/eval/behavior/.venv/bin/python -m roboskills.eval.check \
  roboskills.eval.behavior.launch --oracles ORACLES.json --runtimes docker \
  --out runs/check/behavior -- gpu=0
```

`ORACLES.json` is your own manifest of recorded successful episodes ([format](../check.py)).

[eval/check.py](../check.py) finishes an episode at once (the no-op must fail) and hands an oracle to `replay`. The
manifest holds no-ops only for now: an episode's `actions.npy` does not reproduce it, because OmniGibson's physics also
depends on the renders between steps. Replaying one recording's rows twice ends at identical joints, yet both runs
missed the radio the recorded episode turned on (2026-10-22, docker). So `finish` also saves `requests.jsonl`, every
act and observe in order, and `replay` sends those again. Two such replays of a 560-step episode used its 560 steps,
and their last frames are within a mean 2.2 of 255 grey levels of the original's. An agent's successful episode
recorded with it becomes an oracle by adding its `requests.jsonl` to the manifest. Replaying recorded agent transcripts
without a model (their turns' code run again, which made HumanoidArena's oracles) did not produce one here: all three
turning_on_radio transcripts tried missed the radio (2026-10-22).

## The cohort

`TASKS`: five of the challenge's 100 tasks, chosen for single-step pick/place/open/switch verbs: one or two goal
facts about one to four small objects, no cooking, cleaning, slicing or particles, each needing the base to move, and
every object and target within reach of the arms with the torso held at the challenge's reset pose. That pose sets the
reach: a hand gets from about 0.53 m (pointing down, in front of the base) to 1.5 m above the floor, so tasks with
objects on the floor or on low furniture are out (`picking_up_trash`'s cans on the floor, `installing_a_modem`'s
modem on a 0.45 m coffee table, the `collecting_aluminum_cans` floor cans). Sizes and heights below are instance 301's
(task objects' bounding boxes in the loaded scene); the other instances move the objects, not the furniture.

| Task | Scene | Goal (BDDL) and where things are | Step limit |
|---|---|---|---|
| `turning_on_radio` | house_double_floor_lower | the radio (14 x 32 x 24 cm, on a 0.42 m living-room table, its top at 0.66 m) switched on: a press | 3224 |
| `store_honey` | Rs_int | the honey jar (8 x 15 cm) from the 1.08 m kitchen counter inside the entryway cabinet (0.84 m tall, four doors): open, pick, place | 10150 |
| `composting_waste` | Rs_int | a half banana and a half pomegranate from the 1.08 m kitchen counter inside the kitchen ashcan (0.74 m, a lid) | 5457 |
| `installing_a_scanner` | office_cubicles_right | the scanner (28 x 34 x 6 cm, on a 0.79 m private-office table) switched on and next to the laptop on the same table | 6387 |
| `dispose_of_glass` | hotel_suite_large | four water glasses (8 x 16 cm) from the 0.79 m bathroom counter inside the 0.42 m ashcan in the bedroom | 13792 |

Step limits are the challenge's own: its evaluator times out at 1.5 times the mean human demonstration (30 Hz
steps), and the server reads that number from the evaluator. The instruction is the challenge's sentence where it has
one (`turning_on_radio`), else `goal.describe` of the task's `problem0.bddl`: the goal facts in order, each object
named by its category, the room `:init` puts it in and what it starts on, e.g. "Put every water glass (on the sink in
the bathroom) inside the ashcan in the bedroom."

## Control and scoring

- Robot config: the challenge's `eval/r1pro.yaml` (assisted grasping, 250 kg base, head aperture 40) with
  OmniGibson's primitives controllers (absolute joint positions for arms, trunk and fingers; a position-mode
  holonomic base), which the challenge allows. The trunk holds its reset pose: `torso_link4` is the arm model's root.
- Each step sends each arm's joint targets, each gripper's opening (fully open goes past the joint limit, which
  releases an assisted grasp) and, while the base drives, a world [x, y, yaw] base target; the server turns that into
  the base controller's displacement from the measured pose (at most 3 cm and 0.05 rad), so the base tracks waypoints
  with feedback (0.5 m/s, 0.8 rad/s). Otherwise the displacement is zero: the base holds where it is and pushes
  nothing back.
- The base never pushes. As in the challenge, the R1Pro hangs from six virtual joints whose roll, pitch and height are
  free (OmniGibson drives only x, y and yaw, each up to 1000 N or N m), so a base driven into furniture tips over: the
  first agent sweep's tips were all drives into chairs, a desk, a wall or a sink, the base pushing 1000 N for tens of
  steps, the robot pitching over within 5-10 steps of the first contact (replayed with the tilt logged). So the
  server stops a drive (its target becomes the base's pose) at the step after the robot touches anything other than
  the floor, a floor covering or what it holds (what it touched when the drive began counts once the base has moved
  10 cm or 10 degrees: it may drive away from it, not shove it along), when the base leans 4 degrees more than at the
  drive's start (driving rocks it up to about 3), or when it has not moved for 3 steps while touching something; the
  host stops any motion whose lean grows 4 degrees. A stopped drive reports
  reached False. The yaw jumps came after the tips: a robot on its side has no meaningful heading, and the old
  hold, which drove the base back to its last target, spun it about its tilted axis.
- The BDDL goal is checked every step (the challenge's `PredicateGoal`); the episode ends when it holds or at the
  step limit, and nothing moves after. `finish` holds 10 steps and reports `success` (the goal held at some step),
  `success_step` and `goal_at_end`.
- Cameras: the head (ZED) and both wrist (RealSense) cameras, RGB and linear depth, made at 720 x 720 (the challenge's
  head resolution; its wrists are 480). Resizing a camera after it exists fails in this Isaac Sim's Replicator
  ("Invalid NodeObj" detaching an annotator), so the wrist cameras keep 720 with the same field of view.
- `robot_mask(cam, obs)`: Replicator's instance-id segmentation, attached to each camera's render product so it is
  drawn with the same frame as the RGB-D, of every mesh under the robot's prim (base, torso, head, arms, grippers),
  grown 2 pixels; None for an observation other than the latest. The arm model is only `torso_link4` and below, so
  without the mask the base and lower torso (about 70 % of each wrist camera's pixels at reset) would stay in the
  clouds as obstacles and objects.
- Only robot state, rendered RGB-D with calibration and the robot's mask, the instruction and the BDDL text (what the
  instruction comes from) reach the host; object poses, predicates and scores stay in the simulator.

## Agent results: the turn cap

`agent_as_policy` (Opus 5.5) on the cohort, 2026-10-21, Quadro RTX 6000s:

| Turn cap | Episodes | Successes | Model use | Wall clock per episode |
|---|---|---|---|---|
| 50 | 5 tasks x seeds 1-3 | 1/15 (`turning_on_radio` s1) | $48.45 | 19-62 min |
| 150 | 5 tasks x seed 1 | 2/5 (`turning_on_radio`, `store_honey`) | $41.15 | 25-102 min |

At 50 turns, 14 of 15 episodes ran out of turns with at most 74 % of their steps used. At 150 none reached the cap:
the agent ended them at 53-114 turns. `store_honey` succeeded at turn 79 (5138 steps). The three failures were
manipulation, not turns: a half banana knocked off the counter where no arm reaches (`composting_waste`), a glass
dropped on the floor and the bedroom ashcan knocked over (`dispose_of_glass`), and the step limit (6108 of 6387)
while pushing the scanner beside the laptop (`installing_a_scanner`). Use at least 150 turns here. For comparison,
EFE's BEHAVIOR tasks give an agent 7200 s of wall clock and no step limit (`tasks/behavior-1k/README.md`); every
episode above fit in that.

## Known issues

- One episode per simulator process: a new episode boots a new one (3.5-5 minutes warm).
- The first steps after a boot run slower (a second or two per 30 steps) while caches fill; then 33-47 ms per step
  without rendering. An observation with cameras renders three times (about 115 ms) and segments the robot.
- The head camera's near-horizontal rays meet walls at its own height (1.40 m) at every distance. The skills' table
  height therefore counts only up-facing points, from 0.15 m above the floor for a mobile robot. The task surfaces:
  the radio's table 0.42 m, the kitchen counter 1.08 m, the scanner's table and the bathroom counter 0.79 m.
- `turning_on_radio`: the radio shows no visible change when on (the challenge notes the same).
- Apptainer support follows RoboDojo's but is untested here: build a sandbox of the image at
  `~/.roboskills/images/behavior-3.9.2` (`$BH_SANDBOX`) first.
