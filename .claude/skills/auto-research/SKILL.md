---
name: auto-research
description: Run auto_research on a benchmark task: a model researches live development episodes with resets, submits a `solve(bot)` program, and the frozen program is checked on validation and test cases without a model.
---

# Auto research

A model researches a task in development episodes and writes a `solve(bot)` program; the best submission is frozen
and run on validation and test cases it never sees (`src/roboskills/workflows/auto_research/README.md`). You run it
and report the result. Do not research the task yourself. Paths are from the repository root.

1. **Runtime.** The benchmark's adapter is `roboskills.eval.<benchmark>.launch`. Read
   `src/roboskills/eval/<benchmark>/SKILL.md`: how to prepare its environment and GPU, and the keywords its `make`
   (or `make_vector`) takes. Prepare it if it is not ready.
2. **Cases.** Shared keywords go after `--`; each case adds `"k=v ..."`. Pick development cases (`--dev`, repeatable)
   and different test cases (`--test`, optionally `--val`), e.g. seeds or layouts, unless the request names them.
3. **Run** in the background, its output to a log: `python -m roboskills.workflows.auto_research.run --adapter
   roboskills.eval.<benchmark>.launch --out runs/auto_research/NAME --dev "seed=..." --test "seed=..." -- task=...`.
   `--max-resets` (5) and `--max-submits` (3) are its budgets, `--model` the model (`deepseek-*` for DeepSeek).
4. **Report** `runs/auto_research/NAME/summary.json`: the frozen solution's development, validation and test
   successes, turns and cost. A `void` research is an LLM failure; its best submission is still checked.
