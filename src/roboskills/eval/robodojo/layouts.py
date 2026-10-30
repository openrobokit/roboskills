"""Which RoboDojo episodes may run in which mode (`check_layout`), per eval suite, and the upstream fixes some tasks need.

suite="legacy": the 10-task eval set (from the old `skills/robodojo/scripts/run.py`).
suite="full5": five layouts of each of the 42 canonical tasks at seed 2, derived from `full_eval_set.json`. This is a
task-specific eval selection; the vendor's native benchmark uses 25 or 50 episodes per task and three seeds.
"""

import json
from pathlib import Path


TASKS = [
    "fasten_screws",
    "store_laptop_and_headphones",
    "organize_table",
    "put_bottles_into_dustbin",
    "stack_bowls",
    "insert_key",
    "swap_blocks",
    "solve_equation",
    "press_by_number",
    "general_pickup",
]
EVAL_SEED = 2  # the held-out eval set. Spent: seed 1 layouts 0-10; seed 2 layouts 50-54 (press_by_number 22, 50-53).
# press_by_number's 65 layouts repeat across seeds, shifted: its seed-2 0-9 are Level 1's seed-0 38-47, so it uses 55-59
EVAL_LAYOUTS = {t: [55, 56, 57, 58, 59] if t == "press_by_number" else list(range(5)) for t in TASKS}
FIXES = {"swap_blocks": "robodojo_pr48"}  # upstream RoboDojo PR #48 (a SpringButton's reset)

MANIFEST = json.loads((Path(__file__).resolve().parent / "full_eval_set.json").read_text())
FULL5_TASKS = tuple(row["name"] for row in MANIFEST["tasks"] if row["variant"] == "standard")
FULL5_LAYOUTS = tuple(range(5))
FULL5_SEED = 2
SUITES = ("legacy", "full5")


def development_layouts(value, *, stop=30):
    """Validate a declared development cohort without admitting official layouts 0–4.

    ``stop`` is exclusive. Extending the usual 10–29 range does not establish
    freshness: callers must check prior receipts and actual scene-file overlap.
    """
    if type(stop) is not int or not 11 <= stop <= 65:
        raise ValueError("Declare a finite development layout range")
    layouts = [int(x) for x in value.split(",")] if isinstance(value, str) else list(value)
    if (
        not layouts
        or len(set(layouts)) != len(layouts)
        or any(type(x) is not int or not 10 <= x < stop for x in layouts)
    ):
        raise ValueError(f"Use distinct development layouts 10-{stop - 1}; final layouts 0-4 are untouched")
    return layouts


def check_layout(task, layout, seed, mode, suite="legacy"):
    """Refuse other eval_seeds (0: Level 1's official layouts; 1: the spent eval set), the eval set outside eval mode,
    and eval mode outside the eval set."""
    if suite == "full5":
        if task not in FULL5_TASKS:
            raise ValueError(f"not a canonical RoboDojo task: {task}")
        if int(seed) != FULL5_SEED:
            raise ValueError(f"full5 requires seed {FULL5_SEED}")
        if (mode == "eval") != (
            int(layout) in FULL5_LAYOUTS
        ):  # dev layouts (e.g. agentic_sim's script search) never touch the eval set
            raise ValueError(f"full5: layouts {FULL5_LAYOUTS} in eval mode only, every other layout in dev mode")
        return
    if suite != "legacy":
        raise ValueError(f"unknown suite {suite}; suites: {SUITES}")
    if task not in TASKS:
        raise ValueError(f"unknown task {task}; tasks: {TASKS}")
    if seed != EVAL_SEED:
        raise ValueError(f"refusing: eval_seed {seed}; episodes run on eval_seed {EVAL_SEED} only")
    if (mode == "eval") != (seed == EVAL_SEED and layout in EVAL_LAYOUTS[task]):
        raise ValueError(
            f"refusing: a {mode} episode on eval_seed {seed} layout {layout} of {task} (the eval set: "
            f"eval_seed {EVAL_SEED}, layouts {EVAL_LAYOUTS[task]}, eval mode only)"
        )
