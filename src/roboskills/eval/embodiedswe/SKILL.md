---
name: embodiedswe
description: Run EmbodiedSWE-Bench episodes (RoboBench presets on Isaac Lab 2.3.2 / Isaac Sim 5.1: a Franka Panda, a fixed G1 humanoid with three-finger hands, a walking G1) behind the shared Robot contract with cameras and robot state only, and prepare or troubleshoot the image, checkout and host environment.
---

# EmbodiedSWE-Bench

The runtime for any workflow or policy: a RoboBench preset (`suite.scene.robot.mode`) from EmbodiedSWE, its robot
behind [`roboskills.eval.interface`](../interface.py). The simulator runs in its image, one process per episode; the
launcher side (roboskills, RoboKit, SAM) runs in this runtime's venv on the host. What reaches the agent is
rendered RGB-D with calibration, the robot's mask in each camera, joint (and pelvis) state, the robot's geometry and the
scene's own description; object poses, contacts, rubric values and success stay in the simulator until `finish`.

| Module | Role |
|---|---|
| `launch.py` | `make(task, seed, out, gpu, ...)`: the simulator in the image on one GPU (docker or apptainer; native inside the image, `roboskills.eval.runtime`), the robot; `close()` |
| `robot.py` | `EmbodiedSWE`: arms' joint targets (led by the servo lag, then settled), hands, the walking G1's pelvis waypoints, cameras, `finish`, `replay` (a recorded episode's native actions) |
| `sim/server.py` | image side: the preset's env with our cameras, `GradedEnv` with its suite's grader, the control loop and the walking law, the robot model from its USD; the shared wire (`roboskills.eval.wire`) |
| `hand.py` | the Panda gripper's and the Dex3 hands' facts (TCP, postures), the Dex3 index fingertip |
| `check.py` | the end-to-end smoke test below (`check.json`, `cam_<name>.png`) |
| `__init__.py` | `TASKS` (the cohort, budgets, fixed cameras), `EGO` (robot cameras), `SEEDS`, `vendor()` |
| `ROBOT.md` | the robots, hands, cameras, budgets and what each task counts as done, for the agent |
| `setup.sh`, `pyproject.toml`, `uv.lock` | host venv, image build, checkout and assets |

## Pins

| Component | Pin |
|---|---|
| EmbodiedSWE | `046f7bf48865f3d3b48e64614350777fb5d5d163` (github.com/EmbodiedSWE/EmbodiedSWE), assets `62fab5b70ccb0103c60c4ca39236abae2cc8e900` of the HF dataset `EmbodiedSWE/robobench-assets` (its `assets_manifest.json`, sha256-checked at every build) |
| Simulator image | `ghcr.io/openrobokit/roboskills-embodiedswe:0.2.0`: [docker/embodiedswe](../../../../docker/embodiedswe/Dockerfile), independent Isaac Sim 5.1 image with pip `isaaclab==2.3.2`, the pinned IsaacLab assets source and torch 2.7. Isaac Sim 5.1 needs an NVIDIA driver below 590 |
| Host venv | Python 3.10, PyTorch 2.11 CUDA 12.6, Warp 1.18, RoboKit 1.0.0rc1 (BEHAVIOR's lock) |

## Prepare the runtime

```bash
export ESWE_VENDOR=$PWD/runs/vendor/EmbodiedSWE
bash src/roboskills/eval/embodiedswe/setup.sh   # venv; image pull on a docker host; checkout and assets
```

The checkout with its assets is 2.6 GB, the venv at `src/roboskills/eval/embodiedswe/.venv` of the checkout running it. The checkout is mounted read-only at
`/bench` with `<checkout>/.cache` writable (the asset fetcher takes a lock there even when nothing is missing), and
`HF_HUB_OFFLINE=1`: a missing asset fails the build instead of downloading. A dockerd that sees only `/home` needs the
checkout, `out` and this source tree there. Kit's shader and OptiX caches are the docker volumes
`sk-rd-<uid>-kitcache` / `-optixcache` (RoboDojo's: the same Isaac Sim), warm after the first boot on a host.

Runtimes (`runtime=`, else `$ESWE_RUNTIME`): native when `/isaac-sim/python.sh` exists (inside the image, a Kubernetes
Pod: the checkout is the working directory), else docker when docker has the image or pulls it (`$ESWE_IMAGE`), else
apptainer, which runs a sandbox of the same image (`$ESWE_SANDBOX`, default
`~/.roboskills/images/roboskills-embodiedswe-0.2.0`, built from the registry when missing; a
small root disk needs `APPTAINER_TMPDIR` on a larger one) read-only behind a private writable layer, the checkout at `/bench` as in docker, Kit's caches on the host's disk with
RoboDojo's (`<tmp>/rd-<uid>/cache/gpu<gpu>`), Kit's data, logs and the home private to the run.

GPU: 4.7–6.5 GB per simulator; startup 56–130 s (scene build, room scenery, the robot model's export).

## Run an episode

```bash
CUDA_VISIBLE_DEVICES=5 ESWE_VENDOR=... src/roboskills/eval/embodiedswe/.venv/bin/python -m roboskills.workflows.agent_as_policy.run \
  --adapter roboskills.eval.embodiedswe.launch \
  --out runs/embodiedswe/clear-1 --max-turns 150 -- task=packing.clear_organic_objects.franka.joint seed=1 gpu=5 out=runs/embodiedswe/clear-1/server
```

`gpu` is the physical card the container gets; `CUDA_VISIBLE_DEVICES` keeps the host's RoboKit/Warp/SAM on it. `seed`
is ours (RoboBench has none): 1 or 2, passed to `build(seed=)` and `reset(seed=)` as `eval/grader/grade.py` does, one
env per process. `out` gets the robot model (`robot/<robot>.urdf` and its meshes) and
`embodiedswe_sim.log`; the episode directory gets `native_result.json` (success, the rubric's score and stages),
`actions.npy` (each step's native action vector) and `video.mp4` (the `front` camera, 3 frames per simulated second).

## Check a host (smoke test)

```bash
CUDA_VISIBLE_DEVICES=5 ESWE_VENDOR=... src/roboskills/eval/embodiedswe/.venv/bin/python \
  -m roboskills.eval.embodiedswe.check task=puzzle.coffee.franka.joint seed=1 gpu=5 out=runs/embodiedswe-check/coffee-1
```

It boots the episode in one thread and drives it from another, then writes `check.json` and each camera's first RGB,
depth and mask: startup, observation latency, GPU memory; each camera's depth under the robot's mask back-projected
through K and the pose against the robot model's surface (camera, depth, mask and model agree); every model link and
each TCP against the simulator's bodies; the arms to joint targets; each hand closed and opened; the walking G1's
pelvis along waypoints (0.4 m forward and back, 0.3 m each side, 45 and 30 degree turns, a turning walk), and how far it
drifts while its arms move; `finish` and its files.

Verified 2026-10-21 for every cohort task at seeds 1 and 2 (10 episodes; Quadro RTX 6000s, docker, driver
580) and `stack_blocks` at seed 1: each started headless on one GPU and gave RGB-D cameras with every depth pixel valid
(97–100 %; the rest lies beyond their 40 m range). The depth under each camera's robot mask, back-projected through K and
the camera pose, lies a median 1.7–2.0 mm from the robot model's surface: 90–94 % of those pixels within 1 cm for the
Panda (the rest is the mask's 2-pixel margin), 80–89 % for the G1's head camera (its pelvis and legs are masked but are
not in the arm model). Every model link matched its simulator body to 0.001 mm and 0.06 degrees, and each TCP to
0.0 mm and 0.04 degrees, also after the walks. Arms reached joint targets within 0.0085 rad (Panda, a 40-step path in
54 steps) and 0.0006–0.0095 rad (G1, in 40–46). The Panda's jaw closed to an opening of 0.003–0.006 in 20 steps and
opened to 0.99 in 14–18; the Dex3 hands reached a fist and straight within 0.02 rad (fixed G1, servo stiffness 20) and
0.001–0.018 rad (walking G1, 400) in 13–31 steps. `finish` wrote `video.mp4`, `actions.npy` and `native_result.json`.
On the shared runtime (`runtime=docker|apptainer`), 2026-10-22, classify_objects seed 1: docker and
apptainer each started in 75–77 s on 5.6–5.8 GB and gave the same cameras (depth 100 % valid, the mask a median
1.9–2.0 mm off the model), links, arm errors and hand times as each other and as above.

| | coffee | tool_packing | clear_organic_objects | classify_objects | fruit_delivery |
|---|---|---|---|---|---|
| startup, warm caches (s) | 75–77 | 75–78 | 127–139 | 56–69 | 79–112 |
| control step without rendering (ms) | 35–42 | 29–37 | 40–53 | 41–43 | 49–53 |
| `obs()` with cameras and robot masks (s) | 0.13–0.19 | 0.10–0.18 | 0.14–0.35 | 0.11–0.14 | 0.15–0.19 |
| GPU memory (GB) | 4.7–5.0 | 5.7–6.4 | 6.5–6.7 | 5.5–5.7 | 6.0–6.2 |

The walking G1, seeds 1 and 2 (waypoints at the `Base`'s 0.1 m/s and 0.3 rad/s, then up to 6 s settling): 0.3 m
sideways ended 1–7 cm off, 0.4 m forward 5 cm, 0.4 m back 10 cm, 45 and 30 degree turns within 1.3 degrees and 1 cm,
0.3 m forward while turning 30 degrees 3–6 cm and 0.8–2.2 degrees; standing while both arms moved it drifted 2.5 cm.
`Obs.base` matched the simulator's pelvis exactly and every arm's mount its torso to 0.0 mm. The policy alone
(constant commands for 3 s, seed 1): forward 0.2 m/s walked 0.11 m/s leaning up to 17 degrees, 0.4 m/s stumbled
(32 degrees, no progress); back 0.3 walked 0.25; sideways 0.2 and 0.3 walked 0.14 and 0.29; a turn on the spot at
0.3 and 0.6 rad/s turned 0.03 and 0.07 rad/s, the same 0.6 rad/s with 0.15 m/s sideways 0.71 rad/s; hip heights
0.55–0.78 left the pelvis at 0.723 m.

## Check without a model

```bash
ESWE_VENDOR=... PATH=$HOME/.local/bin:$PATH src/roboskills/eval/embodiedswe/.venv/bin/python -m roboskills.eval.check \
  roboskills.eval.embodiedswe.launch --oracles ORACLES.json \
  --runtimes docker,apptainer --out runs/check/embodiedswe [--only franka|g1] -- gpu=3
```

`ORACLES.json` is your own manifest of recorded successful episodes ([format](../check.py)).

Per manifest entry and runtime ([`roboskills.eval.check`](../check.py)): an episode finished at once (must fail) and
the oracle, a successful agent episode's `actions.npy` stepped again by `robot.replay` (must succeed); the runtimes'
final joints must agree. Native actions reproduce an episode: Isaac Lab's renders between steps (observations, video
frames) leave the physics alone, unlike OmniGibson's (BEHAVIOR replays its requests), and the walking G1's vector holds
its policy's command, so no request log is needed. Each recorded episode replayed in docker (2026-10-22)
ended with its verdict, score and stages, its video within render noise (a mean 1.5–6.6 of 255 per frame).

Image 0.1.1 (on RoboDojo's image), 2026-10-22, docker, `--only franka`: every no-op failed and both
oracles succeeded. Verified 2026-10-22 on Quadro RTX 6000s (driver 580), docker and apptainer, exit 0: `--only g1` on GPU 3
in 25 min, `--only franka` on GPU 2 in 42 min. Every no-op failed, every oracle succeeded, and docker's and apptainer's
final joints were identical (gap 0.0). Startup 51–82 s an episode; an apptainer's first on a GPU 119–221 s (Kit's caches
cold on the host's disk).

| Entry (seed) | Oracle | Steps | Docker | Apptainer |
|---|---|---|---|---|
| `puzzle.classify_objects.g1.joint` (2) | agent episode | 2797 | no-op fails, oracle succeeds | the same |
| `locomanip.fruit_delivery.g1.loco_joint` (1) | agent episode | 6024 | no-op fails, oracle succeeds | the same |
| `packing.clear_organic_objects.franka.joint` (1) | agent episode | 8806 | no-op fails, oracle succeeds | the same |
| `packing.tool_packing.franka.joint` (1) | agent episode | 10941 | no-op fails, oracle succeeds | the same |
| `puzzle.coffee.franka.joint` (1) | none: no agent episode has solved it | | no-op fails | the same |
| `puzzle.stack_blocks.g1.joint` (1) | none: no agent episode has solved it | | no-op fails | the same |

The 2026-10-23 image comparison on an RTX A5000 (driver 580) reproduced a native failure for the walking
`fruit_delivery` seed-1 action replay in both 0.1.1 and 0.2.0 (score 0), despite the earlier Quadro result above.
Do not assume that this recorded walking policy succeeds on every host; retain the native verdict. See
[the image validation report](../../../../docker/validation.md).

## The cohort

| Task (preset) | Robot | What is done | Sub-goals | Step limit |
|---|---|---|---|---|
| `packing.clear_organic_objects.franka.joint` | Panda | every organic on the table (4–5 of 5) in the crate, no clutter in it | 5 | 15300 |
| `packing.tool_packing.franka.joint` | Panda | stapler, scissors, knife in drawers 1–3 of a chest, all drawers shut | 9 | 26100 |
| `puzzle.coffee.franka.joint` | Panda | capsule in, cover shut, mug under the spout, START, brew, mug back on the tray | 7 | 20700 |
| `puzzle.classify_objects.g1.joint` | G1, pelvis fixed | three cubes on their colour's zones | 3 | 9900 |
| `locomanip.fruit_delivery.g1.loco_joint` | G1, walking | the lemon carried around a table onto the far plate | 6 | 18000 |
| `puzzle.stack_blocks.g1.joint` | G1, pelvis fixed | four cubes in one tower on the pad (the fallback for the walking task) | 4 | 12600 |

Success is the suite's grader (`robobench/suites/<suite>/grader`, loaded in the simulator only) on the final state,
after `finish` holds still 30 steps: `GradedEnv.verdict()`'s `success`, with its `score` (the rubric's peak) and stage
values in `native_result.json`. The instruction is the scene's own `describe()` (not `env.describe()`, whose robot
paragraph describes the action vector). Audited for poses: none gives coordinates; they give sizes (the crate, the
cubes, the plate, the coffee machine), counts, the machine's spout facing -y, and the clear-organics list of every
organic the task may hold (an episode samples 4–5; ROBOT.md says so).

## Adaptations (a different condition from the paper's)

- Control: the presets' `.joint` modes (joint position targets at the robots' own PD; the G1's walking task
  `loco_joint`: arm joints plus the frozen AGILE walking policy's [vx, vy, wz, hip height]), not the paper's and our
  agentic_sim's `osc`. One control step is RoboBench's 0.02 s joint period at the scenes' 120 Hz physics: 2 substeps,
  1/60 s (so the G1's walking policy runs at 60 Hz, not the 50 Hz it was measured at in RoboBench's docstring).
- Observations: cameras and robot state only. Cameras are ours (RoboBench's scenes declare none for these tasks):
  TiledCameras with RGB, `distance_to_image_plane` and `instance_id_segmentation_fast`, added to the scene's and the
  robot's assets before the build (as `vla/eval/sim.py` does); the robot's mask is every mesh under its prim.
- Budget: RoboBench has no step limit; ours is 45 s per sub-goal plus 30 s (`TASKS`). Seeds 1 and 2 are ours.
- The first 30 steps hold the reset posture before the first observation (objects spawn a few cm up and land).
- The G1's waist holds its reset posture (the arms' mount stays fixed on the torso), and its hip-height input stays
  0.72: the walking policy holds the pelvis at 0.723 m whatever height it is given (0.55–0.78 tried), so the walking
  G1 has a `Base` but no `Body`.
- Walking: the host's pelvis waypoints become the policy's command in the server, at the control rate: the path's
  velocity plus 1.5/s times the pose error, at least 0.15 m/s while off by more than 1 cm, side steps (0.15 m/s,
  alternating every 0.75 s) while turning with nothing to translate, within |vx| 0.25, |vy| 0.3, |wz| 0.6.

## Agent results

`agent_as_policy` (Opus 5.5, 150-turn cap) on `packing.clear_organic_objects.franka.joint` seed 1 (4 organics on the
table), 2026-10-21: not solved, score 0.5 (2 of 4 organics in the crate), the agent ending it at turn 30
with 12002 of 15300 steps used; $3.27 of model use, 31 minutes. It found every item with SAM 3 and put the lime in on
a fourth try. Two produce items (the small lemon, later the lemon) lodged on the outside of a finger, stuck through
wrist flips and jaw cycles for 16 turns; pinches and grasps on the orange and lemons often closed on nothing
(round fruit rolling out of a stiff jaw). Its last move parked the stuck lemon low inside the crate, where it counted.

## Known issues

- One episode per simulator process; a new episode boots a new one.
- Produce can lodge on a Panda finger (PhysX holds it in the finger's collider) and stay there through flips and jaw
  cycles; seen twice in the clear_organic_objects agent episode. The joint mode's stiff servos press harder into
  objects than the paper's compliant `osc`.
- Control steps cost 29–53 ms of wall clock (two PhysX substeps, the grader's rubric every step, the walking policy);
  an observation with three cameras 0.1–0.35 s.
- The Panda's 400/80 joint servos trail a target by about 0.18 s: `follow` leads the arm's targets by 0.15 s and then
  settles (up to 40 steps) on the last waypoint.
- Walking is slow and imprecise: drives end 1–10 cm from their goal (forward and backward worst), so `drive` often
  reports `reached: false` with a few centimetres to go. Forward and backward steps lean the pelvis 12–17 degrees.
  RoboBench's docstring measured 76–94 % translation tracking for this policy at 50 Hz; in this preset it runs at
  60 Hz and forward walking is the weakest (about half the command).
- `tool_packing`'s Panda binding has no randomization (its jitters are 0): seeds 1 and 2 are the same scene.
- clear_organic_objects: the description lists all five organics the binding may sample; each episode puts 4 or 5 on
  the table and parks the rest off-camera.
- Fixed views see past the table into the room's scenery (kitchen, factory hall), whose depth reaches tens of metres.
