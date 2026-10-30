---
name: dexjoco
description: Run native DexJoCo MuJoCo episodes with Panda/Allegro hands behind the shared Robot contract, using the pinned benchmark and host environment.
---

# DexJoCo

The runtime for native single-arm and bimanual RGB-D episodes.

| Module | Role |
|---|---|
| `launch.py` | `make(task, seed, out, vendor, runtime, image, gpu)`: the shared launcher's entrypoint, the episode's simulator in a server of its own (`episode`, [eval/serve.py](../serve.py)). |
| `robot.py` | `DexJoCo`: cameras, robot proprioception/geometry, native control, `follow`, native task scoring and terminal artifacts; `replay` steps a recorded `actions.npy`. |
| `__init__.py` | Supported task names, native step limits and external vendor-path resolution. |
| `ROBOT.md` | Panda arms, Allegro finger order, TCP, cameras, control rate and native step accounting. |
| `setup.sh`, `pyproject.toml`, `uv.lock` | Pinned benchmark checkout and isolated host environment. |
| [`docker/dexjoco`](../../../../docker/dexjoco/Dockerfile) | The simulator image, `ghcr.io/openrobokit/roboskills-dexjoco:0.2.0`: `setup.sh` in a separate builder, with an independent Ubuntu runtime; runtimes pull it ([docker/README.md](../../../../docker/README.md)). |

## Prepare the runtime

Run from the repository root on a CUDA/EGL host with `uv`:

```bash
export DEXJOCO_VENDOR="$PWD/runs/vendor/dexjoco"
bash src/roboskills/eval/dexjoco/setup.sh  # the checkout and this venv
uv pip install --python src/roboskills/eval/dexjoco/.venv/bin/python --no-deps -e .
export MUJOCO_GL=egl
```

An existing prepared checkout is selected by `DEXJOCO_VENDOR`, or the explicit
`vendor` argument (a path the server sees). Keep the checkout outside `src/`. An
installed wheel contains the runtime, its documentation and setup files; it does
not contain simulator assets, experiment code, model weights or rollout outputs.

The benchmark source is `brave-eai/dexjoco` at
`8d23b0fab23b17a58c4b55f3942e17013aaf8267`. The host uses Python 3.10,
MuJoCo 3.4.0, PyTorch 2.11 CUDA 12.6, Warp 1.18 and RoboKit 1.0.0rc1.

## Runtimes

`make` starts the simulator as a server and returns its `Remote` ([eval/serve.py](../serve.py)): the launcher never
holds the simulator. `runtime=native` (the default, or `$ROBOSKILLS_RUNTIME`) runs the server with this venv and the
caller's environment (`DEXJOCO_VENDOR`, `MUJOCO_GL=egl`; on a multi-GPU host `MUJOCO_EGL_DEVICE_ID` is the physical
GPU's index). `runtime=docker` runs it in `ghcr.io/openrobokit/roboskills-dexjoco:0.2.0` (`image=` to override): this
`setup.sh`'s venv and checkout (`DEXJOCO_VENDOR=/opt/roboskills/runs/vendor/dexjoco`), with the host's `roboskills`
package mounted at `/src/roboskills` and `out` at its own path; `runtime=apptainer` runs a sandbox of the same image from
`~/.roboskills/images/roboskills-dexjoco-0.2.0`, built from the registry when missing. The host keeps this venv either way, for the skills (RoboKit,
SAM). In a container the server copies each arm's MJCF, with its meshes, into `out/model/<arm>/`, where the
host loads it.

```bash
docker pull ghcr.io/openrobokit/roboskills-dexjoco:0.2.0   # once per host; runtime=docker also pulls it when missing
```

Check a runtime without a model: a no-op episode must fail, a recorded successful episode's native actions
(`replay`) must succeed, and the runtimes must end at the same joints.

```bash
CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 src/roboskills/eval/dexjoco/.venv/bin/python \
  -m roboskills.eval.check roboskills.eval.dexjoco.launch --runtimes native,docker --out runs/check/dexjoco \
  --oracles ORACLES.json
```

`ORACLES.json` is your own manifest of recorded successful episodes ([format](../check.py)).

Verified on a Quadro RTX 6000, 2026-10-21: the no-op holds to the native end (1000 steps) and fails; the oracle,
an agent episode's 834 recorded actions, succeeds at step 834 natively and in Docker, with identical final joints.
Physics depends only on the seed and the actions (resets draw from the seeded global NumPy RNG; native `time.sleep` is
disabled). The image is 9.56 GB, its `setup.sh` layer 8.56 GB uncompressed (the venv's PyTorch/CUDA wheels); the
exported arm model has the vendor model's kinematics. Across the two runtimes depth, robot masks and a `follow` ended
by the host's `stop` are identical; RGB differs by at most one intensity level. Apptainer is unverified.

The manifest's 4 tasks, each with an agent's recorded success (bimanual_unlock_ipad, click_mouse, hammer_nail,
pick_bucket), all pass in both runtimes (2026-10-22).

## Run an episode

The generic workflow can use this runtime independently of the recorded recipe:

```bash
CUDA_VISIBLE_DEVICES=0 src/roboskills/eval/dexjoco/.venv/bin/python \
  -m roboskills.workflows.agent_as_policy.run \
  --adapter roboskills.eval.dexjoco.launch --out runs/dexjoco-native \
  -- task=hammer_nail seed=20000 out=runs/dexjoco-native
```

Perception requires the SAM 3 weights and its existing isolated worker. Agent
episodes also require the subscribed model CLI.

Only rendered RGB-D, camera calibration, robot proprioception and robot geometry
reach the agent. Robot segmentation removes the robot's pixels; task/object state
and native scores remain hidden. Native success is latched at the first native
termination and reported only by `finish`. The adapter writes native results,
actions and video, then closes the renderers and environment. Preserve incomplete
episodes rather than overwriting them.
