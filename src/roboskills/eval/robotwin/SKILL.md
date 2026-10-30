---
name: robotwin
description: Run native RoboTwin 2.0 episodes (the aloha-agilex robot's two 6-DoF arms in SAPIEN) behind the shared Robot contract, and prepare the pinned checkout, assets and host environment.
---

# RoboTwin 2.0

The runtime for any workflow or policy: RoboTwin's task envs, cameras and expert checks behind
[`roboskills.eval.interface`](../interface.py).

| Module | Role |
|---|---|
| `launch.py` | `make(task, seed, information, out, runtime, image, gpu)`: the shared launcher's entrypoint, the episode's simulator in a server of its own (`episode`, [eval/serve.py](../serve.py)). |
| `robot.py` | `RoboTwin`: one task env reset per test episode as RoboTwin's eval does, 14-D joint-target actions, RGB-D cameras, robot pixels, latched native success; `replay(actions)` steps a finished episode's `actions.npy`; `scene()` for explicitly privileged development episodes. |
| `state.py` | Physical SAPIEN collision snapshots for explicitly privileged episodes; no artifact imports. |
| `hand.py` | The jaw's command-to-finger map (`Hand.model_q`), importable without the simulator. |
| `ROBOT.md` | Arms, grippers, cameras, control and step budget, for the agent. |
| `setup.sh`, `pyproject.toml`, `uv.lock` | Pinned RoboTwin checkout outside `src/`, its clean-config assets and the host environment. |
| [`docker/robotwin`](../../../../docker/robotwin/Dockerfile) | The simulator image, `ghcr.io/openrobokit/roboskills-robotwin:0.2.0`: `setup.sh` in a separate builder, with an independent Ubuntu runtime; runtimes pull it ([docker/README.md](../../../../docker/README.md)). |

## Prepare the runtime

Run from the repository root on a CUDA host with `uv`:

```bash
export ROBOTWIN="$PWD/runs/vendor/RoboTwin"
bash src/roboskills/eval/robotwin/setup.sh
```

The benchmark is `RoboTwin-Platform/RoboTwin` at `6dde5715` with the `demo_clean` config (the embodiment and object
assets; the 11 GB background textures only matter for randomized configs). The host uses Python 3.10, SAPIEN
3.0.0b1, PyTorch 2.9.1 CUDA 12.6 and CuRobo 0.7.8 (RoboTwin's expert check) and RoboKit
1.0.0rc1. Importing `robot.py` (in the simulator's server) changes the working directory to `$ROBOTWIN`, where
RoboTwin resolves its assets and configs.

`python=PATH` starts this pinned simulator interpreter through the server wire;
its Warp remains at 1.12 for CuRobo's `warp.torch` import. `Remote.reset(seed)`
rebuilds the native episode scene while retaining the SAPIEN engine and static
robot facts; shutdown closes the native environment.

## Run an episode

```bash
PYTHONUTF8=1 CUDA_VISIBLE_DEVICES=0 src/roboskills/eval/robotwin/.venv/bin/python \
  -m roboskills.workflows.agent_as_policy.run --adapter roboskills.eval.robotwin.launch \
  --out runs/robotwin-agent -- task=click_bell seed=100000
```

RoboTwin's eval tries seeds from 100000 × (1 + its seed) upward and skips those its expert cannot solve. Only rendered RGB-D,
camera calibration, robot pixels, proprioception and robot geometry reach the agent. RoboTwin would end an
episode at the first step whose check passes; here the check is latched and only `finish` reads it, writing
`video.mp4` and `actions.npy`. Add `runtime=docker` after `--` to run the simulator in its image (below).

## Runtimes

`make` starts the simulator as a server and returns its `Remote` ([eval/serve.py](../serve.py)): the launcher never
holds the simulator. `runtime=native` (the default, or `$ROBOSKILLS_RUNTIME`) runs the server with this venv and the
caller's environment (`ROBOTWIN`, `PYTHONUTF8=1`). `runtime=docker` runs it in
`ghcr.io/openrobokit/roboskills-robotwin:0.2.0` (`image=` to override): this `setup.sh`'s venv, checkout and assets,
with the host's `roboskills` package mounted at `/src/roboskills` and `out` at its own path; `runtime=apptainer` runs a sandbox of the same
image from `~/.roboskills/images/roboskills-robotwin-0.2.0`, built from the registry when missing. SAPIEN renders through Vulkan, which NVIDIA's container
toolkit provides. The host keeps this venv either way, for the skills (RoboKit, SAM); the server writes the
URDF with its planning joint limits into `out` (in a container, with its meshes), where the host loads it. The
privileged `scene()` reads physical collision geometry through the packaged `state.py`; it requires no artifact checkout.

```bash
docker pull ghcr.io/openrobokit/roboskills-robotwin:0.2.0   # once per host; runtime=docker also pulls it when missing
```

The image's venv takes two layers and the checkout with its assets a third, each under GHCR's 10 GB. PyTorch and
CuRobo install as prebuilt wheels, so no CUDA toolkit is needed.

Check a runtime without a model: a no-op episode must fail, a recorded successful episode's native actions
(`replay`) must succeed, and the runtimes must end at the same joints.

```bash
ROBOTWIN="$PWD/runs/vendor/RoboTwin" PYTHONUTF8=1 CUDA_VISIBLE_DEVICES=0 src/roboskills/eval/robotwin/.venv/bin/python \
  -m roboskills.eval.check roboskills.eval.robotwin.launch --runtimes native,docker --out runs/check/robotwin \
  --oracles ORACLES.json
```

`ORACLES.json` is your own manifest of recorded successful episodes ([format](../check.py)).

Verified 2026-10-21 on an RTX 3090: in both runtimes the no-op fails and the oracle (an agent's successful
episode, 26 recorded actions) succeeds, ending at identical joints (gap 0.0) with pixel-identical head-camera videos.
The image is 16.1 GB in layers of 6.9, 2.6 and 5.7 GB. A scripted turn through `Remote` (cameras and robot masks,
planned and straight moves, the gripper, a `stop` callback) gave identical results in-process, native and in Docker.

The manifest's 5 tasks, each with an agent's recorded success (click_bell, handover_block, open_laptop,
place_empty_cup, stack_blocks_two), all pass in both runtimes (2026-10-22).
