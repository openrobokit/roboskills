---
name: agent-as-policy
description: Run agent_as_policy on a benchmark episode: a model controls the live robot through `bot`, one Python block per turn, and the native score judges it. Use to run a task, seed or layout as an agent policy.
---

# Agent as policy

A model plays one live episode per env through `bot` (`src/roboskills/workflows/agent_as_policy/README.md`); you run
it and report the result. Do not play the episode yourself. Paths are from the repository root.

1. **Runtime.** The benchmark's adapter is `roboskills.eval.<benchmark>.launch` (robodojo, robotwin, dexjoco, libero,
   robosuite, robocasa365, behavior, embodiedswe, humanoidarena). Read `src/roboskills/eval/<benchmark>/SKILL.md`:
   how to prepare its environment and GPU, and the keywords `make` (or `make_vector`) takes, e.g. task and seed.
   Prepare it if it is not ready.
2. **Run** in the background, its output to a log: `python -m roboskills.workflows.agent_as_policy.run --adapter
   roboskills.eval.<benchmark>.launch --out runs/agent_as_policy/NAME -- k=v ...`. `--model` picks the model
   (default claude-opus-5-5; `deepseek-*` for DeepSeek), `--max-turns` caps the turns. An episode takes minutes to an
   hour.
3. **Report** `runs/agent_as_policy/NAME/summary.json`: each env's outcome, turns, cost and seconds. A `void` env is
   an LLM failure, not a task failure: rerun it.
