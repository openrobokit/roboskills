---
name: robosuite
description: Run CaP-X's seven Robosuite tasks (single-arm and two-arm Panda, MuJoCo) behind the shared Robot contract, using the pinned robosuite fork and host environment.
---

# Robosuite (CaP-X tasks)

The runtime for CaP-X's Robosuite tasks: cube lifting, cube stack, cube restack,
spill wipe, nut assembly, two-arm lift and two-arm handover. It rebuilds CaP-X's
scenes, resets, controller, step limits and success checks
([capgym/cap-x](https://github.com/capgym/cap-x) `53e9966`,
`capx/envs/simulators/robosuite_*.py`) on CaP-X's robosuite fork. It runs no
CaP-X code and exposes none of CaP-X's privileged APIs.

| Module | Role |
|---|---|
| `launch.py` | `make(task, seed="1", out=..., runtime=None, image=None, gpu=None)`: the shared launcher's entrypoint; starts the episode's simulator server and returns its `Robot`. `episode(...)` builds the `Robosuite` in the server; `close()`. |
| `robot.py` | `Robosuite`: CaP-X's scenes and seeded resets, cameras, robot facts, joint control, `follow`, native scoring and terminal artifacts; `replay(actions)` steps a recorded `actions.npy`. |
| `__init__.py` | Task names, CaP-X step limits and natural-language instructions. |
| `ROBOT.md` | Arms and their names, gripper/sponge, TCP, cameras, control rate and step budget, for the agent. |
| `setup.sh`, `pyproject.toml`, `uv.lock` | Pinned robosuite fork and isolated host environment. |
| [`docker/robosuite`](../../../../docker/robosuite/Dockerfile) | The simulator image, `ghcr.io/openrobokit/roboskills-robosuite:0.2.0`: `setup.sh` in a separate builder, with an independent Ubuntu runtime; runtimes pull it ([docker/README.md](../../../../docker/README.md)). |

## Prepare the runtime

Run from the repository root on a CUDA/EGL host with `uv`:

```bash
bash src/roboskills/eval/robosuite/setup.sh
export MUJOCO_GL=egl
```

The benchmark source is CaP-X's robosuite fork, `uynitsuj/robosuite` at
`97292732ed909ac3ae116579fb768607034a4dbd` (robosuite 1.5.1 with CaP-X's changes
to Stack, Lift, Wipe, the wipe arena, the hammer and two-arm handover). Robo Use
pins the same commit for CaP-X `53e9966`. It installs from git with its assets,
so no separate checkout is needed. The host uses Python 3.10, MuJoCo 3.5.0
(CaP-X's version), NumPy 1.26.4, PyTorch 2.11
CUDA 12.6, Warp 1.18's CUDA 12 build and RoboKit 1.0.0rc1. Robosuite's
teleoperation, IK-controller and GUI OpenCV dependencies are excluded.

The CUDA 12 Warp wheel is pinned in `pyproject.toml`: PyPI's default Warp 1.18
wheel requires a CUDA 13 driver.

`python=PATH` selects this runtime's pinned simulator interpreter through the
server wire. `Remote.reset(seed)` retains the server while native reset rebuilds
randomized object sizes, physics, controllers and joints. Soft reset changes seed
semantics by retaining prior object sizes; use the native hard reset. `finish`
writes evidence; shutdown calls `close`.

Explicit `information=privileged` enables read-only `scene()` collision surfaces
and object poses for scripted data generation. Observation mode rejects it before
reading simulator state. Both modes use the same control and native scoring.
The MuJoCo geometry reader is shared with LIBERO in `../physical.py`.

## Runtimes

The simulator runs in a server of its own (`roboskills.eval.serve`); `make`
returns the host's end of its socket. `runtime=` (default
`$ROBOSKILLS_RUNTIME`, else `native`) chooses where the server runs:

- `native`: a process of the launcher's Python, so this host needs `setup.sh`'s
  venv. Robosuite renders on EGL device `$MUJOCO_EGL_DEVICE_ID`, else the first
  of `$CUDA_VISIBLE_DEVICES` (the physical GPU index).
- `docker`: a container of `ghcr.io/openrobokit/roboskills-robosuite:0.2.0` on
  one GPU, as the caller, with this host's `roboskills` package mounted at `/src/roboskills`. The
  server copies each arm's model, with its
  meshes, into the episode directory (`model/<arm>/`, `assets/`). The launcher still runs in the host
  venv, which plans grasps.
- `apptainer`: a sandbox of the same image in `~/.roboskills/images/`, built from
  the registry when missing (not yet verified here).

```bash
docker pull ghcr.io/openrobokit/roboskills-robosuite:0.2.0   # once per host; runtime=docker also pulls it when missing
```

Check the runtimes without a model: in each, a no-op episode (finished at once)
must fail and the oracle, a successful agent episode's `actions.npy` stepped
again by `replay`, must succeed with the same final joints:

```bash
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl src/roboskills/eval/robosuite/.venv/bin/python -m roboskills.eval.check \
  roboskills.eval.robosuite.launch --oracles ORACLES.json \
  --runtimes native,docker --out runs/check/robosuite
```

`ORACLES.json` is your own manifest of recorded successful episodes ([format](../check.py)).

`replay` holds the last recorded action, so `finish` adds ten still steps to
the recording's own ten. Verified 2026-10-21 on a Quadro RTX 6000:
the check passes with `two_arm_lift-s1.npy` and with `cube_stack-s1.npy`
(`task=cube_stack seed=1`), final joints identical across the runtimes (gap 0);
camera RGB-D, intrinsics, poses and robot facts are also bit-identical between
them. Natively, recorded `two_arm_handover` seed 2 and `nut_assembly` seed 1
episodes replay to success too. Video frames rendered before the first camera
observation are noise: every episode's first frame, and all of a replay's.

The manifest's 5 tasks, each with an agent's recorded success (cube_lifting, cube_stack, nut_assembly,
two_arm_handover, two_arm_lift), all pass in both runtimes (2026-10-22).

## Tasks

| Task | Robosuite env | Steps | CaP-X success (checked by `finish`) |
|---|---|---|---|
| `cube_lifting` | `Lift` | 1500 | the red cube more than 4 cm above the table |
| `cube_stack` | `Stack` | 1500 | the red cube on the green one, released |
| `cube_restack` | `Stack`, 4 cm cubes, the green cube starting on the red one | 1500 | the red cube on the green one, released, and not both cubes held up |
| `spill_wipe` | `Wipe` (sponge) | 4000 | every dirt marker wiped |
| `nut_assembly` | `NutAssemblySquare` | 1000 | the square nut on the square peg, the gripper away |
| `two_arm_lift` | `TwoArmLift`, opposed | 5000 | the pot more than 10 cm above the table |
| `two_arm_handover` | `TwoArmHandover`, opposed, hammer on the table | 5000 | the right arm holds the hammer's handle, the left holds nothing, the hammer more than 10 cm up |

Seed `N` reproduces the layout of Robo Use's `capx-<task>-sN` task: one
`numpy.random.default_rng(N)` drives robosuite's robot start noise and object
samplers. It is set after construction and before the reset. The same seed
always gives the same layout; CaP-X itself draws a new layout every trial. Each
reset then does what CaP-X's does. Single-arm tasks turn the wrist by π and
take 50 free physics steps. `nut_assembly` folds the arm out of view and takes
100 steps holding it. The two-arm tasks move `agentview` to CaP-X's raised view.
Reset steps are not charged to the budget. The two-arm arms are robot0 = `left`
and robot1 = `right`, named by their bases' sides in `agentview`.

Control is CaP-X's absolute `JOINT_POSITION` controller at 20 Hz (kp 1500,
kd 400; linear interpolation, except `two_arm_lift`). `follow` streams the
trajectory's joint targets, interpolated per step. It adds the controller's
steady lag (kd/kp times the target velocity) and settles on the last waypoint
for at most 30 steps. Arms and hands outside the trajectory hold their last
command. A jaw command `u` becomes the gripper action `1 - 2u`.

These choices differ from CaP-X:

- Observations are RGB-D from `agentview` and the wrist cameras, not CaP-X's
  `robot0_robotview`/`birdview` RGB.
- The runtime enforces the step limit itself (robosuite `ignore_done`), so
  `nut_assembly`'s reset steps do not shorten its 1000-step budget.
- Success is judged as CaP-X judges the state its program leaves: ten still
  steps after the agent finishes (Robo Use's rule), not at the step limit.

## Run an episode

```bash
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl src/roboskills/eval/robosuite/.venv/bin/python \
  -m roboskills.workflows.agent_as_policy.run \
  --adapter roboskills.eval.robosuite.launch --out runs/robosuite-native \
  -- task=cube_stack seed=1 out=runs/robosuite-native
```

Add `runtime=docker` after `--` to run the simulator in the image.

Perception requires the SAM 3 weights and its isolated worker. Agent episodes
also require the subscribed model CLI.

By default only rendered RGB-D, camera calibration, robot proprioception and robot geometry
reach the agent. Explicit privileged `scene()` additionally supplies read-only physical geometry. `robot_mask` returns None, so the skills remove the robot using
its model. Task annotations and native scores stay hidden; `finish` alone scores. The runtime renders from whichever thread calls it, making
robosuite's EGL context current only for each render. Each episode directory
receives `robot<i>.xml`, the robot-only descriptions placed by `T_world_base`,
and the server's `sim.log`. Its terminal
files are `native_result.json`, `actions.npy` and `video.mp4`.
