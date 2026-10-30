# Launchers

A launcher is the code around a model that makes it do one workflow's job: its loop, budgets, scoring and
records. The workflow itself (its `SKILL.md`, the agent's instructions, and the logic only it uses) and the tools it
calls are in `src/roboskills/workflows/` and `src/roboskills/tools/`. Launchers sit beside their workflow,
`roboskills.workflows.<workflow>.run`, and run from an installed package.

| Launcher | Job |
| --- | --- |
| [`agent_as_policy/run.py`](../workflows/agent_as_policy/run.py) | An agent plays live episodes through `bot`, one model turn per Python block. |
| [`auto_research/run.py`](../workflows/auto_research/run.py) | An agent researches development episodes with resets, then submits a `solve(bot)` program, frozen and checked on validation and test cases. |
| [`agentic_sim/run.py`](../workflows/agentic_sim/run.py) | An agent writes and repairs task scripts that native checks judge. |

| `core/` | Shared by the launchers |
| --- | --- |
| [`__init__.py`](__init__.py) | `skill(name)`: a workflow's `SKILL.md` without its frontmatter, the prompt a launcher gives its model. |
| [`llm.py`](llm.py) | One-shot `claude -p` and `codex exec` calls; a long system prompt goes through a file. Both one-shot clients accept named base64 image panels; `codex` runs without tools in a read-only sandbox, retains raw events when given a directory and rejects tool calls. |
| [`agent.py`](agent.py) | The turn loop, its append-only prompt view and the code runner. |
| [`episode.py`](episode.py) | Live episodes: the adapter's robots behind `Tools`, how an episode ends, the planner lock motions let go, the robot's ROBOT.md. |
| [`process.py`](process.py) | Adapter loading and process-group termination. |

Receipts and hashes (`save`, `file_digest`, `digest`) are [`roboskills.eval.records`](../eval/records.py):
benchmark runtimes and the MuJoCo tool write them at run time too.

## Layout

- A launcher's procedures and command line are one `run.py`: `python -m roboskills.workflows.<workflow>.run`.
- Code moves into `core/` only when two launchers use it. `core/` knows no simulator, robot or benchmark;
  launchers import `core/`, never each other; `core/` imports no launcher.
- No infrastructure: workflow and tool code never decide where jobs run or what they may spend.
- Robot verbs (`bot`) are a tool: [`tools/robokit/bot.py`](../tools/robokit/bot.py).

Tests: `python -m pytest tests/core tests/workflows`.

Task-solving agents may use current-run feedback and relevant logs, but must not retrieve prior solutions
or private evaluation data through older runs, other worktrees or Git history.

Codex transport reconnect diagnostics remain in raw receipts. A successful final
`turn.completed`, zero exit code and saved reply establish completion; a later
failed turn, missing delivery or tool invocation still rejects a one-shot call.

The one-shot Codex client always uses an empty neutral temporary working directory.
Codex exposes its cwd to the model even with tools disabled; method/task directory
names must not enter blind review or judging. Raw evidence is copied to the caller’s
retained directory after success or failure. Tool-using callers choose their own cwd.
