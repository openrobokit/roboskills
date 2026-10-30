---
name: humanoidarena
description: Run HumanoidArena episodes (the Unitree G1 29-DoF with Dex3 hands under the frozen GEAR-SONIC whole-body controller, Isaac Sim 6.1) behind the shared Robot contract, walking and squatting with GEAR-SONIC's kinematic planner, and prepare or troubleshoot its image, Apptainer sandbox, assets and host environment.
---

# HumanoidArena

The runtime for any workflow or policy: one official HumanoidArena test episode, the G1 behind
[`roboskills.eval.interface`](../interface.py) with a walking `Base` and a `Body` (the pelvis height). The simulator
runs in this runtime's image (Isaac Sim 6.1, Isaac Lab patched for 6.1, onnxruntime), one process per episode, with
upstream HumanoidArena, its assets and the GEAR-SONIC models mounted from the vendor directory; the launcher side
(roboskills, RoboKit, SAM) runs in this runtime's venv on the host.

| Module | Role |
|---|---|
| `launch.py` | `make(task, seed, out, gpu, ...)`: the simulator on one GPU in a [`runtime.Server`](../runtime.py) (docker as root; an Apptainer sandbox where docker or its image is missing; natively inside the image), a cache slot, the robot; `close()` |
| `robot.py` | `HumanoidArena`: arm joint targets with settle, binary hands, walking to world waypoints with feedback (a blocked walk ends the motion), the pelvis height with correction, the camera, `replay` (a recorded episode), `finish` |
| `sim/server.py` | image side: the task's env with its official test config and episode seed, SONIC through upstream's provider (semantic40), the arms' sag integrated away, the success check and fall rule, rendering; the shared [wire](../wire.py) |
| `sim/gait.py` | the legs' reference: GEAR-SONIC's kinematic planner (walk, turn, squat) resampled to 50 Hz and cross-faded, commanded from the measured pelvis |
| `sim/fixes.py` | Isaac Sim 6.1 fixes (RAX's, Apache-2.0): Kit shader rename, DDS stub, static-collider physics restart, OpenDoor door reset, football goal mesh |
| `hand.py` | the Dex3 TCP and press point, the binary hand command, the RoboKit arm model |
| `check.py` | the end-to-end check below (`check.json`) |
| `__init__.py` | `TASKS` (the cohort, step limits, instructions), `GROUPS` and `episode_seed` (seed -> official test episode), `vendor()` |
| `ROBOT.md` | the robot, walking, posture, camera, budget and each task's success, for the agent |
| [`docker/humanoidarena`](../../../../docker/humanoidarena/Dockerfile) | the image's Dockerfile (adapted from RAX's), Isaac Lab's 6.1 patch (RAX's) and pinned runtime requirements |
| `setup.sh`, `pyproject.toml`, `uv.lock` | host venv, image pull, vendor download |

## Pins

| Component | Pin |
|---|---|
| HumanoidArena | `William-wAng618/HumanoidArena` `68479287` (`release/open-source-prep`) and its simulation assets (Google Drive archive, sha256 `d0a0c6be…447b`, 2.7 GB) |
| GEAR-SONIC | Hugging Face `nvidia/GEAR-SONIC` `6733128a`: `model_encoder.onnx`, `model_decoder.onnx`, `observation_config.yaml`, `planner_sonic.onnx` (774 MB, the planner's V2 interface), sha256-checked |
| G1 description | cuRobo `4ea77366` (v0.8.0): Unitree's `g1_29dof_with_hand_rev_1_0.urdf` and meshes (BSD-3-Clause), the arm model's source; upstream's own G1 URDF is an older revision (shoulders 1 cm lower, another Dex3) |
| Image | `ghcr.io/openrobokit/roboskills-humanoidarena:0.2.0` ([docker/humanoidarena](../../../../docker/humanoidarena/Dockerfile)): `nvcr.io/nvidia/isaac-sim:6.1.0@sha256:0c16dd67…9080`, PyTorch 2.11 cu128, Isaac Lab `yuechen0614/IsaacLab` `afca7b09` with `isaaclab-isaacsim61.patch`, onnxruntime-gpu 1.22.0; 30 GB (cuRobo and the CUDA toolkit of RAX's image are left out) |
| Host venv | the BEHAVIOR runtime's: Python 3.10, PyTorch 2.11 cu126, RoboKit 1.0.0rc1 |

## Prepare the runtime

```bash
export HA_VENDOR=$PWD/runs/vendor/humanoidarena
bash src/roboskills/eval/humanoidarena/setup.sh --accept-nvidia-eula   # venv, image (docker hosts), vendor (3.7 GB)
```

setup.sh pulls the image; `launch.py` also pulls it when missing. The vendor directory must be readable by the
container's root on a network home (setup.sh makes it world-readable). Hosts without docker, or whose image store is
too small, run the image as an Apptainer sandbox at `~/.roboskills/images/roboskills-humanoidarena-0.2.0`
(`$HA_SANDBOX`; `HA_RUNTIME=apptainer` forces it, `docker` the image, `native` the image's own python when this process
runs inside it, as a Kubernetes Pod; setup.sh with `HA_RUNTIME=apptainer` skips the image), which `launch.py` builds
from the registry when missing; the legacy `roboskills-humanoidarena-6.1` sandbox predates release 0.2.0. Under
Apptainer `launch.py` binds the
slot's directory (the home, `/cache`, Kit's cache, `/var/tmp`) and private Kit data, logs, `/tmp` and `/dev/shm`
(upstream's image writer maps shared memory) to host directories.

Per simulator: about 3-7 GB of GPU memory, 15 GB of host RAM, 8 CPU threads.

## Run an episode

```bash
CUDA_VISIBLE_DEVICES=0 HA_VENDOR=... src/roboskills/eval/humanoidarena/.venv/bin/python -m roboskills.workflows.agent_as_policy.run \
  --adapter roboskills.eval.humanoidarena.launch \
  --out runs/humanoidarena/boxing-1 --max-turns 150 -- task=boxing_bag seed=1 gpu=0 out=runs/humanoidarena/boxing-1/server
```

`seed` 1 and 2 are the official test episodes of groups 0 and 1, repeat 0 (`episode_seed`: upstream's
`_derive_episode_seed`, the last 4 bytes of sha256("<gym id>|<group>|<repeat>")). The episode directory gets
`native_result.json`, `actions.npy` (20 per step: each arm's 7 joint targets and hand, the walk target, the pelvis
height), `requests.jsonl` (every act and observe sent, in order) and `video.mp4` (the front camera, a frame per 25
steps).

## Check a host (smoke test)

```bash
CUDA_VISIBLE_DEVICES=0 HA_VENDOR=... src/roboskills/eval/humanoidarena/.venv/bin/python \
  -m roboskills.eval.humanoidarena.check task=sit_sofa seed=1 gpu=0 out=runs/humanoidarena-check/sit_sofa-1
```

It boots the episode in one thread and drives it from another, then writes `check.json`: startup, observation
latency, GPU memory, the camera (depth, the floor back-projected through K and the pose to z = 0, the robot's mask),
the arm model's palm links and torso against the simulator's (before and after squatting and walking), both arms to
joint targets and back, both hands, the pelvis to 0.6 m and back, walks to world waypoints while the task's budget
lasts, `Obs.base` against the simulator's pelvis, and `finish`.

## Check without a model

```bash
CUDA_VISIBLE_DEVICES=0 HA_VENDOR=... src/roboskills/eval/humanoidarena/.venv/bin/python -m roboskills.eval.check \
  roboskills.eval.humanoidarena.launch --oracles ORACLES.json \
  --runtimes docker,apptainer --out runs/check/humanoidarena -- gpu=0
```

`ORACLES.json` is your own manifest of recorded successful episodes ([format](../check.py)).

[eval/check.py](../check.py) finishes an episode at once (the no-op must fail) and hands an oracle to `replay`: an
episode's `requests.jsonl` (its rows and the renders between them, sent again in order) or its `actions.npy` (its rows
alone, 10 per request).

Verified 2026-10-22 on an RTX 6000 Ada (docker and Apptainer): every no-op fails (boxing_bag 1, opendoor 2, sit_sofa
1, warehouse_navigation 2, football_single 1), and both oracles succeed in both runtimes at identical final joints (gap
0): sit_sofa 1 (1481 steps) and warehouse_navigation 2 (1626). Each oracle is the `requests.jsonl` of a model-free
replay of an agent's successful transcript (its turns' code run again), which succeeded on its own. The recorded
episodes themselves do not reproduce: boxing_bag 1 failed from its `actions.npy` and from requests rebuilt from its
sim log, and the transcripts of boxing_bag 1, opendoor 2 and sit_sofa 2 replayed to failure, so those stay no-op only.

## Control

- The action is HumanoidArena's semantic40 frame at 50 Hz, through upstream's SONIC provider (encoder mode 0, its
  latest frame held, as HumanoidArena's VLA baselines drive it): the pelvis orientation relative to the episode's
  initial heading, the 29 joint angles, two binary hands. SONIC tracks the joints and the pelvis orientation; the
  robot moves over the floor only by stepping. The arms' 14 joints are the agent's; the legs, waist and pelvis
  orientation come from the gait.
- Walking: GEAR-SONIC's kinematic planner generates whole-body motion at 30 Hz from 4 context frames and a command
  (mode, speed, movement and facing direction, height); `gait.py` resamples it to 50 Hz and cross-fades each new plan
  into the running reference over 8 frames, starting 2 frames ahead, as GR00T's deployment does. Each control step the
  server compares the measured pelvis with the row's world [x, y, yaw] target: farther than 4 cm (6 cm when standing)
  it slow-walks toward it at 1.2 m/s per metre of distance (0.15-0.5 m/s) facing the target's heading, planning again
  every second (sooner when the direction turns 30 degrees or the speed changes 0.1 m/s); otherwise it turns on the
  spot to a heading more than 4 degrees off (8 when standing), or stands. The host leads the waypoints by 0.6 s and
  settles until the pelvis is within 5 cm and 5 degrees and the reference stands, or stops closing in for 2 s;
  `Base.tol` (reached) is 8 cm, 5 degrees. Replanning every 0.4 s (and on every small change of direction near the
  target) left the robot stepping in place 10-15 cm short; giving the planner its goal position (its "specific
  target" inputs, shifted by the reference's drift from the measured pelvis) left it 20-45 cm short. SONIC sees only
  the reference's latest frame, as HumanoidArena's VLA baselines drive it; its trained future window was not needed.
- Posture: a pelvis height below 0.74 m plans the planner's squat to it; the host corrects the command by what is
  left (the squat stops a few cm high) up to three times, holds it through later motions until raised again, and
  walking stands up. `Body(lo=0.5, hi=0.79–0.80)`.
- Arms: SONIC holds an arm 2-9 degrees off its targets (more at the wrists); while a target rests, the server adds
  6 % of the remaining error to an offset each step (at most 0.3 rad per joint), and the host settles until the joints
  stop closing in.
- A walk stops (the gait stands) at the step after a robot body other than the feet touches something new (30 N), or
  the pelvis leans 20 degrees more than at the walk's start (walking backward sways it 14); the host also stops one
  that makes no progress for 2 s once it is under way.
- Success and the fall rule are upstream's: compute_success_mask (Football: the reward the manager computed after the
  step, positive, as RAX reads it), checked every step after upstream's reward sync; a fall (tilt over 75 degrees, or
  over 60 with 50 N on a critical body, for 5 steps) ends the episode.
- The camera is upstream's front camera (640 x 480, fx = fy = 243.2, on `torso_link`, pitched 47.6 degrees down, 1.27
  m up standing), rendered from a copy under the environment root placed on the measured torso before each render
  (RAX's way): RGB, depth along the optical axis (0 nothing), Replicator's instance segmentation for the robot's mask
  (grown 2 pixels), and the pose it was placed at (its own pose buffers lag a frame).
- Only robot state reaches the host: arm and Dex3 joints, the pelvis [x, y, yaw] and height (the simulator's, as
  BEHAVIOR and RoboCasa365 give their base; RAX gives its agents no position at all), the torso link's pose, the lean,
  whether the gait is still moving, the camera; object poses, contacts, rewards and success stay in the simulator.

## Verified

2026-10-21. check.py on the 5 tasks x seeds 1 and 2 with this code: an RTX 3090 (Apptainer sandbox) for
VisNavi, Boxing and OpenDoor seed 1, an RTX 6000 Ada (docker) for the rest; earlier
rounds on an A5000 (docker). Seeds 1 and 2 differ only in object layouts, so a task's two checks repeat its
robot motions exactly (the physics is deterministic on one GPU).

- Startup 52-66 s with warm caches on a local-disk sandbox, 118-126 s (Ada) and 170-185 s (A5000) under docker; about
  190 s cold.
- A control step: 70-100 ms (SONIC and 4 physics steps 55-87 ms, the success check 1-6 ms but VisNavi's 21 ms, a
  video frame every 25 steps 6-9 ms, the gait 3 ms, a plan 25-35 ms each); 43-47 ms on the Ada alone; 300-400 ms on a
  GPU another job keeps at 100 %. `obs()` 0.1-0.3 s; GPU memory 4-9 GB.
- Camera: every depth pixel valid; the floor back-projects to z = 0.0013 m; the robot's mask covers 9 % standing.
- The arm model's palm links match the simulator's to 0.0 mm and 0.02 degrees, its root the torso link to 0.0 mm,
  before and after squatting and walking. Arm joints end within 0.03-0.08 rad of their targets after a motion and
  its settle; hands reach their binary poses within 0.024 rad.
- Posture: 0.6 m reached at 0.55-0.62 m and back up to 0.78-0.79 m (about 270 steps); probes reached 0.52 m
  (commanded 0.45); the pelvis leans forward 15-20 degrees squatting. No fall in any run.
- Walking (34 walks of 0.3-0.58 m and 60 degree turns): ending 1-13 cm (median 5) and 0.2-6 degrees off, within the
  8 cm, 5 degree `reached` tolerance in 22 of 34, in 220-350 steps each. Longer probes: 1 m forward 2-5 cm
  off (within 5 cm after 235 steps), 0.5 m sideways 3-13 cm, 90 degree turns 1-3 degrees and 3-11 cm (the turn on the
  spot shifts the pelvis); the feet shuffle 0-5 cm standing still afterwards.
- `agent_as_policy` (Opus 5.5, an A5000): boxing_bag seed 1 succeeded in 8 turns and 713 of 900 steps ($0.21:
  find the green ball, walk 1.2 m beside the red bag, reach the right palm to it), sit_sofa seed 1 in 6 turns and
  1301 of 2000 steps ($0.27: two walks around the boxes, then a third backing toward the seat ended the episode as
  the official check's seated-geometry fallback (two sit bodies over the seat, within its height band) held, before
  `posture` was called).

## Known issues

- One episode per simulator process.
- Walking accuracy is a few cm; turns on the spot drift up to 10 cm; a walk costs 230-400 steps of a 900-2000 step
  budget.
- Some dockerds mangle `-v path:path:ro` (it mounted `<path>o`): `launch.py` uses `--mount`; it also needs
  `docker build --network host`.
- Hosts that end a session's processes past about 10-16 GB of memory kill the simulator (15 GB resident) during boot
  under Apptainer; run it under docker there.
- Upstream logs PhysX's triangle-mesh fallbacks (ankle links), one MDL compile error (the warehouse's traffic cone
  material) and OmniHub reconnect warnings; all harmless.
