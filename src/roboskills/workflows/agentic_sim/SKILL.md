---
name: agentic-sim
description: Build the simulation task a natural-language instruction describes, in a simulator the agent chooses, then solve the frozen task with a second agent; native checks judge both.
---

# Agentic simulation

Build the simulation task the instruction describes, then have a fresh agent, which never sees how you solved it,
solve it. `check.py` judges both. Paths are from the repository root; NAME is a short name for the task.

1. **Interpreter.** PYTHON is the absolute path of `runs/agentic_sim/env/bin/python`. If it is missing, build it
   once: `bash src/roboskills/workflows/agentic_sim/setup.sh` (MuJoCo, PyBullet, Newton, Blender, robokit).
2. **Build** in `runs/agentic_sim/NAME/work`, as `src/roboskills/workflows/agentic_sim/build.md` says: `task.py`,
   a scripted `demo.py` and `notes.md`. Run its check there, `PYTHON -m roboskills.workflows.agentic_sim.check
   task.py check`, and fix the task until it passes and its images and videos look right. build.md's `sh` is your
   shell; there is no `finish()`: you are done when the check passes. Then write
   `runs/agentic_sim/NAME/request.json` as `{"instruction": "<the instruction>"}`.
3. **Solve** with a fresh agent: `python -m roboskills.workflows.agentic_sim.run --solve runs/agentic_sim/NAME
   --out runs/agentic_sim/NAME-solved --python PYTHON`. It freezes the task without `demo.py`, and a second agent
   writes `solution.py` until every seed is solved (about 5 to 30 minutes). Report both checks and
   `runs/agentic_sim/NAME-solved/result.json`.

Asked only to solve a task someone built, follow `src/roboskills/workflows/agentic_sim/solve.md` yourself instead,
in a new work directory, never changing the task's.
