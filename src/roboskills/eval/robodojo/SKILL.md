---
name: robodojo
description: Run RoboDojo episodes (two ARX X5 arms in Isaac Sim 5.1, or 6.0 in a Kubernetes Pod) behind the shared Robot contract, and prepare or troubleshoot the pinned simulator image, host environment and benchmark assets.
---

# RoboDojo

The RoboDojo runtime for any workflow or policy: the simulator in its Isaac Sim
container, one `Robot` per env and the eval suites.

| Module | Role |
|---|---|
| `launch.py` | `make_vector(task, layouts, gpu, out, ...)`: the sim container on one GPU, one clocked robot per env (`obs=official`: RoboDojo's RGB-only eval observation); `close()`; `replay(run, gpu)`: a finished batch again, from its event log |
| `robot.py` | `make(...)`: one env's `Robot` (`roboskills.eval.interface`); `Clocked` for vector envs; `checkpoint()`/`restore()` repeat the seeded reset and replay recorded actions (30-55 s per restore) |
| `layouts.py` | `check_layout(task, layout, seed, mode, suite)`: `legacy` (10 tasks) and `full5` (42 tasks, eval layouts 0-4); `development_layouts(value, stop=30)` validates a declared development cohort |
| `evaluation.py` | `rgb_contract(robot)` records inferred geometry provenance; `check_rgb_evidence(contract, arrays)` checks saved RGB inputs without starting a simulator |
| `rgbd.py`, `cams.json`, `depth_worker.py` | depth, intrinsics and camera poses for RoboDojo's official RGB-only observation (one MoGe-2 worker per process, in its own `uv run` environment) |
| `sim/` | container side: `server.py` (one env per app, the wire), `vector.py` (one scene, one env per layout), `replay.py` (a finished batch from its event log), `state.py` (privileged geometry); its image is [docker/robodojo](../../../../docker/robodojo/Dockerfile) |
| `grasp_backend.py` | the learned ARX X5 grasp policy as `$GRASP_BACKEND` (`roboskills.eval.robodojo.grasp_backend`): GraspKit's `graspkit.policies.arx_x5_server`, started on first use in a venv with GraspKit at 9eb02584 or later (`$GRASP_POLICY_PYTHON`); the published's checkpoints come from GraspKit's private HF dataset (an HF token) unless `GRASP_POLICY_CKPT` / `GRASP_POLICY_RANKER` give local ones |
| `ROBOT.md` | arms, hands, cameras, control rate and budget, for the agent |
| `recipes/` | each task's official RoboDojo wiki recipe (capability, description, process-score ladder; from XPolicyLab `408b99d9`), appended by `robot.py` to the episode instruction as `Task notes:` in every mode |
| `setup.sh`, `pyproject.toml`, `uv.lock` | the RoboDojo checkout (`$ROBODOJO_VENDOR`) and the host venv |

```bash
GRASP_BACKEND=roboskills.eval.robodojo.grasp_backend GRASP_POLICY_PYTHON=... \
CUDA_VISIBLE_DEVICES=0 src/roboskills/eval/robodojo/.venv/bin/python -m roboskills.workflows.agent_as_policy.run \
  --adapter roboskills.eval.robodojo.launch --out runs/rd-general_pickup \
  -- task=general_pickup layouts=10,11 gpu=0 mode=dev suite=full5
```

Use `serve` on the simulator host and `run --conn OUT/conn.json` on the agent host
to split them. The vector sim mounts the directory holding `roboskills` and the
checkout read-only, and `out` read-write. Episodes save no `video.mp4` unless `MANIP_VIDEO=1`.

The vector sim logs its batch to `out/sim_events.jsonl`: every physics tick's commands, every rendered observation and
each episode's end (outcome, final joints, object poses). An env's own `actions.npy` cannot reproduce its episode:
the envs share one clock, and an env keeps moving on the ticks where it is idle. Replay a finished batch with the same
image and RoboDojo checkout:

```bash
src/roboskills/eval/robodojo/.venv/bin/python -m roboskills.eval.robodojo.launch replay runs/rd-general_pickup gpu=0
```

It writes `replay/env<i>.mp4` (the head camera at each rendered observation) and `replay/replay.json`: each episode's
logged and replayed success and score, and the largest gaps in final joints (rad) and object positions (m).

That replay is RoboDojo's oracle; [eval/check.py](../check.py) adds the no-op (an episode finished at once must fail):

```bash
src/roboskills/eval/robodojo/.venv/bin/python -m roboskills.eval.check roboskills.eval.robodojo.launch \
  --oracles ORACLES.json --runtimes docker --out runs/check/robodojo -- gpu=0
```

`ORACLES.json` is your own manifest of recorded successful episodes ([format](../check.py)).

Verified 2026-10-22 on an RTX 3090 (docker): the no-op fails for cover_blocks, put_bottles_into_dustbin and
stack_blocks, and replays of the agents' recorded eval batches of those three tasks (15 episodes, 6 successes) give
every logged outcome again, with final joints and object positions identical (gap 0).

Image 0.2.0, verified 2026-10-23 on an RTX A5000 (driver 580, Docker): the `stack_blocks` recorded batch
(layouts 0–4, seed 2) reproduces all five outcomes (three successes, two failures), with zero joint/object-position
gaps, as does 0.1.0. The candidate's wire adapter also returns RGB-D/calibration and fails an immediate no-op. Use
this runtime's own pinned host venv; other runtimes pin an older RoboKit without the ARX X5 assembly.

Reuse an existing verified image, environment,
and asset checkout when possible. Run long setup and simulator jobs in tmux.

## Development comparisons

Reuse this runtime and its cohort/RGB checks across workflows.

Declare source versions, observation mode, layouts and native horizons before
execution. `development_layouts` accepts distinct indices 10–29 by default;
an explicit exclusive `stop` permits another finite development range. This
validation does not establish unseen scenes: inspect actual layout-file overlap
and prior receipts. Official layouts remain subject to `check_layout`.

Before scaling, verify native reset, a rendered task camera and one physical
action with a terminal native receipt. Frozen-script execution needs no model.
For RGB-only runs, save `rgb_contract(robot)` as `observation-contract.json`;
validate it against the saved `obs0.npz` with `check_rgb_evidence`. The latter
rejects disclosed native depth, camera calibration and object poses. Record
estimator/runtime versions separately; inferred geometry is not native depth.

Use native success, not authored completion messages. Preserve source, initial
observations, actions, model replies and terminal evidence outside expendable
worktrees. A missing terminal result remains unknown; collection may reattach
to its existing job, but must not silently launch another episode. Report the
planned denominator, native failures and interruptions separately; unknowns may
widen bounds, never increase a measured success rate. Comparison-specific
auditors decide completeness without repairing policies or rewriting receipts.

## Pinned components

| Component | Pin / source |
|---|---|
| RoboDojo source | `726e9aabfaa642203722eb126f5eaf0f37f3e1ad` |
| Hugging Face assets | Dataset `RoboDojo-Benchmark/RoboDojo`, revision `91f76c28d93dd20c5fa46ce6a5a1d96a4f384acd` |
| Isaac Sim | 5.1; base image `nvcr.io/nvidia/isaac-sim@sha256:f3563cb2ba0c18af0b2fb321360dcb73a917b899f879e3213623d6bee484fa54` |
| IsaacLab | RoboDojo's pinned `third_party/IsaacLab` submodule |
| Host environment | Python 3.10; this skill's `pyproject.toml` and `uv.lock` |
| Simulator image | `ghcr.io/openrobokit/roboskills-robodojo:0.2.0` from [docker/robodojo](../../../../docker/robodojo/Dockerfile), pulled |

Keep host and container Python separate: the host runs the robot, RoboKit and
perception; `/isaac-sim/python.sh` inside Docker runs Isaac Sim. Do not install
Isaac packages into the host venv or upgrade dependencies to diagnose a missing asset.

RGB-only depth inference uses MoGe-2 ViT-L and PyTorch 2.9.1 CUDA 12.6 through `uv` in the depth worker's
declared environment (MoGe `b942f00b`; the weights download once into the Hugging Face cache).
The newer MoGe-3 source forces a conflicting CUDA-13 index; CUDA 13 also fails
on CUDA 12 drivers. Keep the depth worker's declared dependency environment.

## Prepare or reuse the runtime

Run commands from the chosen repository/worktree. Use that machine's own `$HOME`.

```bash
rd_repo=$(git rev-parse --show-toplevel)
rd_skill="$rd_repo/src/roboskills/eval/robodojo"
export PATH="$HOME/.local/bin:$PATH"
command -v uv
docker image inspect ghcr.io/openrobokit/roboskills-robodojo:0.2.0 --format '{{.Id}}'
```

For a fresh installation, the existing setup files are the entrypoints. The
checkout lives outside `src/`; every component reads it from `ROBODOJO_VENDOR`:

```bash
export ROBODOJO_VENDOR=/absolute/path/to/RoboDojo
bash "$rd_skill/setup.sh"
uv sync --project "$rd_skill" --frozen
uv pip install --python "$rd_skill/.venv/bin/python" --no-deps -e "$rd_repo"
docker pull ghcr.io/openrobokit/roboskills-robodojo:0.2.0   # once per host; launch.py also pulls it when missing
```

One image serves every user and host: the sim runs as whoever launches it, and each user's Kit and OptiX caches
stay in their own volumes. Hosts where Docker cannot pull it run the same image under Apptainer from a sandbox at
`~/.roboskills/images/roboskills-robodojo-0.2.0` (`$RD_SANDBOX`), which `launch.py` builds from the registry when
missing (`$RD_RUNTIME` overrides the choice). Older `roboskills-robodojo-sim` sandboxes predate 0.2.0 and must not be
renamed to the new release. The runtime binds the host's NGX library into the sandbox, which Apptainer's `--nvccli` does not mount and RoboDojo's DLAA needs.

Without root, install Apptainer unprivileged with its `tools/install-unprivileged.sh` (it needs `rpm2cpio`). On a
small root disk, point `TMPDIR` and `APPTAINER_TMPDIR` at a larger one: the sandbox build and the sim's caches use
them. `RD_RUNTIME=apptainer` also serves a Docker host whose image store is full.

For an existing prepared checkout, set `ROBODOJO_VENDOR` to its absolute path.
Do not run `setup.sh` over a shared installation to repair experiment assets.
The host venv can also be reused by its absolute Python path.

Record source commit, submodule commit, Docker image ID, asset revision and host
Python path in the run evidence. The image tag alone is not a content pin.

## Check assets before starting Isaac

`setup.sh` skips its asset download whenever `Assets/Eval_Layout` exists. That
directory's presence does **not** prove that the installation is complete.

- Verify requested layouts under
  `Assets/Eval_Layout/RoboDojo/arx_x5/<seed>/<task>_<layout>.json`.
- Preserve the preceding layout files too: the native seed manager resolves
  indices from its sorted task-layout list. Sparse downloads can change indexing.
- Verify the objects those layouts reference, including `metadata.json`,
  `description.json` and their USD assets. A layout download alone is insufficient.
- Restore missing paths from the pinned asset revision into a separate vendor
  copy; preserve shared assets and frozen layouts. Avoid in-place writes to
  hardlinked files.

The downloader used by `setup.sh` can also fill an incomplete private checkout:

```bash
uv run --no-project --with huggingface_hub python - <<'PY'
import os
from huggingface_hub import snapshot_download

snapshot_download(
    "RoboDojo-Benchmark/RoboDojo",
    repo_type="dataset",
    revision="91f76c28d93dd20c5fa46ce6a5a1d96a4f384acd",
    allow_patterns=["Assets/**"],
    local_dir=os.environ["ROBODOJO_VENDOR"],
)
PY
```

Narrow `allow_patterns` to known missing paths for a partial repair. Retain downloaded
file hashes and the pinned revision with the run.

## Launch checks

Each host keeps its runtime settings (`RD_REPO`, `ROBODOJO_VENDOR`, `GRASP_*`, `RD_RUNTIME`, the Claude token) in
`~/.roboskills/robodojo.env`; source it in the launch shell.
`RD_REPO` is a plain copy of the repository; update it with `git archive origin/main | ssh <host> tar -x -C <RD_REPO>`.
Do not `rsync --delete` main into a host's own checkout: it also deletes that host's other worktrees and `runs/`.
Final evals use `layouts=0,1,2,3,4 mode=eval suite=full5`, one task per free GPU.
Export `PATH="$HOME/.local/bin:$PATH"` in the actual worker/tmux launch shell:
SAM perception starts `uv` as a subprocess, even when Python uses an absolute path.
Use the selected GPU consistently for the worker and its simulator container.
`launch.py` already supplies GPU isolation (Docker, or NVIDIA's container CLI under
Apptainer), an 8 GiB shared-memory area, writes as the caller and persistent per-user
Kit/OptiX caches; reuse that launcher.

If using the learned grasp backend, retain its existing `GRASP_BACKEND`,
`GRASP_POLICY_PYTHON`, checkpoint and ranker settings. See
[grasp_backend.py](grasp_backend.py)
for those settings. Host perception also needs access to the SAM model used by
[sam3.py](../../tools/sam3/sam3.py).

Validate setup with one frozen script on one development layout.
Use a fresh `runs/` output directory. Current script trials accept layouts 10–29
at seed 2; final layouts 0–4 remain untouched. Check `vector_sim.log` for
`ready vector_sim num_envs=`, then camera/perception output and a completed native
receipt. No model call is needed to execute a frozen script.

A completed native failure can establish that setup works; it does not establish
that the task script works. Check for errors and void episodes before counting
task outcomes.

## Observed failures

| Symptom | Diagnosis / action |
|---|---|
| `Seed ... not found` | Missing development layouts or incomplete preceding layout files; restore pinned files. |
| Object metadata loads as `None` | Referenced object assets are absent; restore their full directories. |
| `FileNotFoundError: uv` | Worker launch shell omitted the host tool directory from `PATH`. |
| Camera annotator fails before script execution | Simulator startup failure; retain native logs, exclude from task counts. A successful boot alone does not verify cameras. |
| `camera ... renders black` at startup; before that check, `Failed to create NGX context` and some cameras black (mean RGB under 5, depth normal) for the sim's life | The container lacks NVIDIA's NGX (DLSS) library, and RoboDojo renders with DLAA. Docker's `NVIDIA_DRIVER_CAPABILITIES=all` mounts it; Apptainer's `--nvccli` has no NGX capability, so `launch.py` binds the host's `libnvidia-ngx.so.*`. Without it 14 of 18 Apptainer sims had black cameras on 10-20, on RTX 4090 and Quadro RTX 6000 alike; with it none of 74, every task included. `vector.py` refuses to start a batch whose first frame has a camera under mean RGB 10. |
| `OSError: [Errno 28] No space left on device` from Python logging under Apptainer | `--contain` makes /tmp the session directory, 16 MiB on some hosts; `launch.py` binds a private /tmp. |
| A sim that failed during startup stays up until the launcher's boot timeout (`zenity: not found`) | Kit's exit waits on a crash dialog; `vector.start` exits the process on any startup error. |
| Script executes but grasp misses, placement shifts, or target is unreachable | Motion/script failure; tune the script, not setup or the generic workflow. |
| `Physical object <name>_<n> is absent from the current scene` after a restore | Each native reset renumbers object ids; `restore()` maps them back by name and nearest position and reports `unmatched_objects`. |
| Client deadline expires while simulator remains active | The current RoboDojo callback does not enforce the request's wall-clock limit. Record an incomplete trial and inspect the owned container before retrying. |
| `nvidia-container-cli: detection error: nvml error: gpu requires reset` | A GPU on the host is stuck (`nvidia-smi` shows `ERR!`); NVML enumeration fails for every card, so neither Docker nor Apptainer can start a GPU container until an admin resets it (seen 10-19). Where `nvidia-smi -r` is `Not Supported`, it needs a reboot. |
| `vkCreateInstance ... ERROR_INCOMPATIBLE_DRIVER`, `no suitable CUDA GPU`, then camera annotators fail on `None` | The NVIDIA container toolkit is too old to mount `libnvidia-gpucomp.so` (10-19); `launch.py` mounts it. |
| Segfault in `librtx.scenedb` right after `app ready` | Driver 595.99.02 crashes Isaac Sim 5.1's RTX even with every driver library mounted (10-19). Driver 580.82.07 works: with no GPU jobs left, `sudo sh NVIDIA-Linux-x86_64-580.82.07.run -m=kernel` (it refuses while NVIDIA modules are loaded unless run interactively), then reboot. |

For an expired trial, save logs and partial receipts first, identify its exact
container from the worker/run evidence, and stop only that container and its
verified owned handler if needed. A client timeout alone does not release the GPU.
Preserve queued requests; do not start broad reruns as a setup check.
