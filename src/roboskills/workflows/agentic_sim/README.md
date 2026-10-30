# Agentic simulation

An agent writes a simulation task from an instruction, a second agent writes its solution, and native checks judge
both. The workflow imports no simulator: the agents choose one from the interpreter [`setup.sh`](setup.sh) builds.
Live episodes on benchmark robots are [`agent_as_policy`](../agent_as_policy/README.md); saved `solve(bot)` programs
checked on held-out cases are [`auto_research`](../auto_research/README.md).

## Tasks from text

Two agents, in separate conversations: one builds a task from an instruction, the other solves the frozen task.
Each writes Python that the harness runs in its work directory (`core.agent`, the loop agent_as_policy uses), with
`sh` for programs, `show` for images and `finish` for the check ([`act.md`](act.md)); `--model` picks `claude -p`
without tools or, for `deepseek-*`, the DeepSeek API. [`setup.sh`](setup.sh) builds their interpreter once: MuJoCo,
PyBullet and Newton, the Blender and RTX renderers, and robokit.

```sh
# The agent in this repo (the agentic-sim skill, SKILL.md): interactive, or one instruction
claude  # then: /agentic-sim A Franka Panda arm puts a red cube on a green square target.
claude -p "/agentic-sim A Franka Panda arm puts a red cube on a green square target."
codex exec '$agentic-sim A Franka Panda arm puts a red cube on a green square target.'
# The agent anywhere
claude -p "Follow the agentic-sim skill in github.com/openrobokit/roboskills: A Franka Panda arm puts a red cube on a green square target."
```

```sh
bash src/roboskills/workflows/agentic_sim/setup.sh  # once; makes runs/agentic_sim/env
PY=$PWD/runs/agentic_sim/env/bin/python             # absolute: the agents and checks run elsewhere
# Build the task, then solve the frozen task with a second agent
python -m roboskills.workflows.agentic_sim.run "A Franka Panda arm puts a red cube on a green square target." --out runs/agentic_sim/cube --python $PY
python -m roboskills.workflows.agentic_sim.run --solve runs/agentic_sim/cube --out runs/agentic_sim/cube-solved --python $PY
# DeepSeek instead of Claude; fewer rounds and hours
python -m roboskills.workflows.agentic_sim.run "Two YAM arms thread a rope through a ring." --out runs/agentic_sim/rope --python $PY --model deepseek-flash --rounds 3 --hours 1
```

- **Build** ([`build.md`](build.md)): the agent writes `task.py` (the whole scene and a robokit robot, in a simulator
  it picks) and a scripted `demo.py`. [`check.py`](check.py) passes the task when every seed builds, the seeds start
  differently, holding still never succeeds, the robot starts at robokit's rest pose, the start frame shows a scene,
  and `demo.py` solves at least one seed.
- **Solve** ([`solve.md`](solve.md)): the task is frozen (copied without `demo.py`, read-only, fingerprinted), and
  the agent writes `solution.py`, a `solve(env)` that acts only through `env.step`. `check.py --solution` passes it
  when it solves every seed within `max_steps` and the task is unchanged.

Both stages check 8 seeds (`--seeds`). Each `finish()` is a round; a failed check, or one without `notes.md`, goes
back to the same conversation, up to `--rounds` (4) within `--hours` (2). `OUT/result.json` records
every round, and `OUT/round-K/check/` holds that round's receipt and videos. Importing a robokit robot downloads its
assets from Hugging Face, and `check.py` imports the task's robots, so check access before a long build:
`$PY -c "import robokit.assets.robots.arms.franka_panda"`.

## Tests

`python -m pytest tests/workflows/agentic_sim` (CPU only). The launcher is
[`run.py`](run.py).
