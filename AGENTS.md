# AGENTS.md

Read a component's `README.md` or `SKILL.md` before using or changing it.

## Layout

| Path | Holds |
|---|---|
| `src/roboskills/workflows/<name>/` | A workflow's `SKILL.md` (frontmatter `name` and `description`, then the agent's instructions), its own logic and, when released, its launcher `run.py`. |
| `src/roboskills/tools/<name>/` | One model, simulator or library: `SKILL.md`, a `README.md` with an example on `assets/example_data/`, and any thin wire of ours. `tools/robokit/` also holds `bot` and the perception, grasping and planning under it. |
| `src/roboskills/eval/` | Benchmark runtimes, one folder each with `launch.py`, `ROBOT.md` and `SKILL.md`. `interface.py` is their contract, `serve.py` their server, `check.py` validates one without a model. |
| `src/roboskills/core/` | Code two or more launchers share. |
| `docker/` | One image per simulator or CUDA-compiling model, pushed to GHCR. |
| `assets/` | README media and `example_data/`. |
| `runs/` | Generated outputs, gitignored. |

## Rules

- Improve shared workflows and tools. Runs are outputs for diagnosis, never tuning targets.
- Never patch per-task scripts, policies, rewards, trajectories, prompts, configs, offsets, scene geometry or outputs to
  repair a demo, anywhere, and never move such a fix into shared code as a task-name check.
- Evaluate through the launcher: declared inputs, fixed settings, its normal agent loop and budget, a fresh run from the
  original inputs after each change. Compare versions on the same inputs and report time, revisions, quality and
  failures. A failure with no general fix gets recorded, not tuned.
- Keep workflows independent of tasks, benchmarks, datasets and embodiments, and validate changes across tasks.
  Workflows never import each other.
- Reuse a tool before adding one. Tools are general, with small interfaces, and take reusable code only after a task
  works. A model runs in `uv run` of its file's inline dependencies, or in an image only when it compiles CUDA extensions
  (`python -m roboskills.eval.runtime IMAGE COMMAND...` runs any image).
- `src/roboskills/` packages only workflows, tools, `eval/` and `core/`. A released launcher imports nothing outside it.
- Keep one implementation: no compatibility aliases, duplicate trees, export pipelines or hash manifests. Do not recreate
  `docs/`, `dev/` or `examples/`.
- Task scripts, adapters, experiment code, configs and results stay off `main`. When unsure, keep it out.
- Train and evaluate one learned policy per task unless asked otherwise.
- Update the README, `SKILL.md` or docstrings when interfaces, setup or verified knowledge change, and this file when the
  layout does.

## Git

Keep the main checkout clean: other sessions share it, so never edit or copy files into it; fast-forwarding is fine. Work
in `.worktrees/<slug>` on a `<you>/<topic>` branch from `origin/main`, and remove the worktree after merging. Open a pull
request only when the user wants it merged, and merge once CI and a local `make test` pass.

## Checks

`uv sync --locked --group dev`, then `make lint`, `make check`, `make check-loc` (each workflow under 1,000 lines of
Python), `make test` and `make check-package`. CI runs all but `make test`. Test shared code, never one task's artifact.
