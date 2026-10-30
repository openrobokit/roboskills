---
name: libero
description: Run native LIBERO and LIBERO-Pro episodes (one Panda arm in MuJoCo) behind the shared Robot contract, prepare the pinned SDKs and host environment, and evaluate frozen policies on small matched development comparisons with auditable success and error counts.
---

# LIBERO and LIBERO-Pro

The runtime for any workflow or policy: native MuJoCo physics, LIBERO's predicates and RGB-D cameras behind
[`roboskills.eval.interface`](../interface.py).

| Module | Role |
|---|---|
| `launch.py` | `make(suite, task, seed, out, profile, runtime, image, gpu)`: the shared launcher's entrypoint, the episode's simulator in a server of its own (`episode`, [eval/serve.py](../serve.py)). |
| `robot.py` | `Libero`: any LIBERO-Pro suite and task, seeded procedural reset, native joint-position control, `follow`, cameras, native scoring, receipts; `profile_settings` for the recorded native profiles. |
| `hand.py` | The native Panda gripper's axes, width and finger map. |
| `../physical.py` | Shared MuJoCo collision surfaces for privileged episodes; host-only layout fingerprints. |
| `ROBOT.md` | Arm, gripper, cameras, control rate and step budget, for the agent. |
| `setup.sh`, `env.sh`, `pyproject.toml`, `uv.lock` | Pinned SDK checkouts outside `src/`, their config, and the host environment. |
| [`docker/libero`](../../../../docker/libero/Dockerfile) | The simulator image, `ghcr.io/openrobokit/roboskills-libero:0.2.0`: `setup.sh` in a separate builder, with an independent Ubuntu runtime; runtimes pull it ([docker/README.md](../../../../docker/README.md)). |

Suites are LIBERO-Pro's benchmark names: `libero_spatial`, `libero_object`, `libero_goal`, `libero_10`, `libero_90`
and their perturbations, e.g. `libero_goal_swap` (object positions), `libero_goal_task` (task), `libero_goal_lan`
(language), `libero_goal_object` and `libero_goal_env`. `task` is the suite's task index.

## Prepare the runtime

Run from the repository root on a CUDA/EGL host with `uv`:

```bash
export LIBERO_VENDOR="$PWD/runs/vendor/libero"
bash src/roboskills/eval/libero/setup.sh
```

The SDKs are `uynitsuj/LIBERO-PRO` at `47aaa803` and CaP-X's robosuite 1.4 fork `Max-Fu/robosuite` at `a498b087`,
both on `PYTHONPATH` from `env.sh` (bash). The host uses Python 3.10, MuJoCo 3.3.0 (the matched profiles check it),
PyTorch 2.11 CUDA 12.6, Warp 1.18 and RoboKit 1.0.0rc1.

## Run an episode

```bash
bash -c 'source src/roboskills/eval/libero/env.sh && CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 "$LIBERO_PYTHON" \
  -m roboskills.workflows.agent_as_policy.run --adapter roboskills.eval.libero.launch \
  --out runs/libero-agent \
  -- suite=libero_goal_swap task=0 seed=1 out=runs/libero-agent'
```

`make` uses the `matched-v3` profile: the reference-matched reset, controller and 800×512 cameras, 990 policy
steps at 20 Hz, an episode that ends when LIBERO's predicate first holds, and evaluation seeds 1-50. The robot's
base is placed by measured forward kinematics when the episode starts. Only rendered RGB-D, camera calibration,
robot proprioception and robot geometry reach the agent; `finish` writes `native.json`, `actions.npy` and
`video.mp4`.

## Runtimes

`make` starts the simulator as a server and returns its `Remote` ([eval/serve.py](../serve.py)): the launcher never
holds the simulator. `runtime=native` (the default, or `$ROBOSKILLS_RUNTIME`) runs the server with this venv and
`env.sh`'s environment. `runtime=docker` runs it in `ghcr.io/openrobokit/roboskills-libero:0.2.0` (`image=` to
override): this `setup.sh`'s venv and SDK checkouts, with the host's `roboskills` package mounted at `/src/roboskills` and `out` at its own
path; `runtime=apptainer` runs a sandbox of the same image from `~/.roboskills/images/roboskills-libero-0.2.0`, built from the registry when missing. The
host keeps this venv either way, for the skills (RoboKit, SAM). In a container the server copies the
robot's model, with its meshes, into `out`, where the host loads them.

```bash
docker pull ghcr.io/openrobokit/roboskills-libero:0.2.0   # once per host; runtime=docker also pulls it when missing
```

Check a runtime without a model: a no-op episode must fail, a recorded successful episode's native actions
(`replay`) must succeed, and the runtimes must end at the same joints.

```bash
bash -c 'source src/roboskills/eval/libero/env.sh && CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 "$LIBERO_PYTHON" \
  -m roboskills.eval.check roboskills.eval.libero.launch --runtimes native,docker --out runs/check/libero \
  --oracles ORACLES.json'
```

`ORACLES.json` is your own manifest of recorded successful episodes ([format](../check.py)).

Verified 2026-10-21 on a Quadro RTX 6000: in all three runtimes the no-op fails and the oracle (an agent's
successful episode, 821 recorded actions) succeeds, ending at identical joints (gap 0.0). The sandbox came from
`apptainer build --sandbox ~/.roboskills/images/roboskills-libero-0.2.0 docker-daemon://<image>`. The image is
11.1 GB, its venv layer 10.1 GB uncompressed. A scripted agent turn through `Remote` (find, a grasp, its camera
check) ran natively and in Docker; in Docker the grasp lifted the wine bottle.

The manifest's 2 tasks, each with an agent's recorded success (libero_goal_swap 2, libero_spatial_swap 0), all pass
in both runtimes (2026-10-22).

## Freeze the comparison

Record each policy's source path, SHA-256, origin and development history before
execution. Evaluation has no model calls, selection or policy repairs. Existing
research results cannot establish a shared-workflow improvement by themselves.

Declare the same suite, task ID, native instruction, seeds and scoring for both
conditions. The development campaigns use task zero in `libero_goal_task`,
`libero_object_task`, `libero_spatial_task` and `libero_10_task`; suite filenames
are misleading, so verify the instruction retained in each native receipt.

Match the initialization and execution settings: procedural seeded reset, no
saved-state restoration, ten separately recorded settling ticks, 1000 policy
ticks at 20 Hz, 600-second episode deadline, controller/profile, camera setup
and information access. Record SDK commits, runtime identity, dependency pins,
perception/grasp configuration, worker allocation and actual controller settings.
Verify whether settling consumes the native horizon: configured `max_steps`
can differ from the remaining policy ticks. Record both counts explicitly.
Reset/layout fingerprints help detect differing initial conditions; equal seeds
alone do not guarantee equal layouts or deterministic execution.

Use development seeds >=51; official seeds 1–50 are outside this runner's scope.
Check prior protocols and receipts before declaring seeds fresh: the CLI rejects
the source run's declared cohorts, but does not discover other runs automatically.
Repetitions of one seed are repeated executions, not additional independent layouts.

## Read and report evidence

Use native receipts, not authored stage flags. Native `success` means success at
any step; `final_success` means the terminal native checker passed. The frozen
check wrapper scores **final native success** and retains the any-step value.
Report both if the reference uses a different convention.

For every condition show final successes / planned cases, completed cases,
infrastructure errors, runtime and source hash. Preserve policy exceptions,
timeouts, missing receipts and native failures separately; none may increase the
reported rate. A check with `valid_score=false` is incomplete or invalid, not a
completed benchmark result. Keep receipts, initial/final images, videos, actions,
reset hashes and original logs under ignored `runs/`.

Four tasks × three seeds is a **12-case development comparison**. It does not
reproduce the reported 80-task benchmark or establish statistical equivalence.
Diagnose mismatches before further evaluation; task-specific policy repairs are
outside this skill.
