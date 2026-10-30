# Agent as policy

The LLM controls one live episode. Each turn it writes Python against `bot`;
the launcher returns printed output, errors, the robot status (`bot.status()`, the `STATUS`
in `core.episode`) and the images the code passed to `show`. Variables persist. Native scores stay hidden from the LLM.

`bot` is a [`Tools`](../../tools/robokit/bot.py): stateless tools (observe, segment, fuse, heuristic_grasp,
place_pose, plan, execute, ...) the model chains; the model's own code keeps the objects and holds. Only `execute`
moves the robot, through `robot.follow`.
[eval/interface.py](../../eval/interface.py) defines the adapter contract.
[The RoboKit tools README](../../tools/robokit/README.md) covers the tools, their units, frame
conventions and adapter hooks. `finish()` is the launcher's (`core.episode.Episode`): it calls `robot.finish` once.

```sh
# The agent in this repo (the agent-as-policy skill, .claude/skills/agent-as-policy): interactive, or one instruction
claude  # then: /agent-as-policy Stack the blocks in RoboDojo, layouts 0-4.
claude -p "/agent-as-policy Stack the blocks in RoboDojo, layouts 0-4."
codex exec '$agent-as-policy Stack the blocks in RoboDojo, layouts 0-4.'
# The agent anywhere
claude -p "Follow the agent-as-policy skill in github.com/openrobokit/roboskills to stack the blocks in RoboDojo, layouts 0-4."
```

```bash
# RoboDojo stack_blocks on its five eval layouts, one env each (its environment: eval/robodojo/SKILL.md)
python -m roboskills.workflows.agent_as_policy.run --adapter roboskills.eval.robodojo.launch \
  --out runs/agent_as_policy/stack_blocks -- task=stack_blocks layouts=0,1,2,3,4 gpu=0 mode=eval suite=full5
# DeepSeek instead of Claude
python -m roboskills.workflows.agent_as_policy.run --adapter roboskills.eval.robodojo.launch --model deepseek-flash \
  --out runs/agent_as_policy/stack_blocks-ds -- task=stack_blocks layouts=0,1,2,3,4 gpu=0 mode=eval suite=full5

python -m roboskills.workflows.agent_as_policy.run --adapter A.py --out DIR -- k=v
python -m pytest tests/workflows/agent_as_policy tests/tools/robokit
```

Benchmark runtimes live in [`src/roboskills/eval/`](../../eval/interface.py): RoboDojo, DexJoCo, LIBERO,
Robosuite, RoboTwin, two with a mobile base driven by `bot.drive` (RoboCasa365 and BEHAVIOR-1K), EmbodiedSWE-Bench
(a Panda, a fixed G1 and a walking G1) and HumanoidArena, whose whole-body G1 also squats and sits with `bot.posture`. Each `launch.py` is an adapter (`--adapter roboskills.eval.<benchmark>.launch`) and each
`ROBOT.md` describes its robot, for every agent; each
`SKILL.md` covers its environment and an allocated GPU. RoboDojo also serves vector envs and eval sets.

The launcher, [`run.py`](run.py), plays an adapter's envs (one `make`
env or a `make_vector` batch) in one process with shared planning and segmentation workers, one agent per env through
[`core.agent`](../../core/agent.py) with [`SKILL.md`](SKILL.md).
`bot.py` executes actions. Shared environment setup releases its SAM worker and
adapter through one cleanup scope, including startup failures.

Each episode saves `log.jsonl` (each turn with its `llm_s` and, from `claude -p`,
token `usage`) and native `result.json`. Each run adds `summary.json` (each env's
outcome, turns, cost and seconds). The planner lock is let go while a robot observes, and the
prompt is the whole log, unchanged once sent, so the prompt cache covers every
earlier turn (`core.agent.append_only`: images drop in blocks of 20). LLM failures are marked `void`
and excluded from this workflow's success denominator; rerun them. Learned-policy
evaluation separately includes setup failures in its denominator.
