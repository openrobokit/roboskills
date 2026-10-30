"""The solve stage without a simulator or a model: check.py on solutions to a numpy task, and run.solution's loop."""

import json
import sys
import time
import types

from roboskills.workflows.agentic_sim import check as checks
from roboskills.workflows.agentic_sim import run


TASK = """
import numpy as np

class Env:
    robot = []
    max_steps = 20

    def __init__(self, seed):
        self.seed = seed

    def reset(self):
        self.x = 0.1 * self.seed

    def step(self, action=None):
        self.x += 0.0 if action is None else action

    def observe(self):
        return dict(robot_q={}, objects=dict(cube=[self.x, 0, 0, 1, 0, 0, 0]))

    def render(self):
        return np.zeros((8, 8, 3), np.uint8)

    def success(self):
        return self.x > 1.0

def make(seed):
    return Env(seed)
"""


def no_video(monkeypatch):
    monkeypatch.setitem(sys.modules, "imageio", types.ModuleType("imageio"))
    monkeypatch.setitem(sys.modules, "imageio.v2", types.SimpleNamespace(mimwrite=lambda path, frames, **_: None))


def test_a_solution_must_solve_every_seed_within_its_steps(tmp_path, monkeypatch):
    no_video(monkeypatch)
    (tmp_path / "task.py").write_text(TASK)
    solutions = {
        "pushes.py": "def solve(env):\n    while True:\n        env.step(0.1)\n",  # ended by the harness on success
        "slow.py": "def solve(env):\n    for _ in range(100):\n        env.step(0.01)\n",  # out of steps
        "crashes.py": "def solve(env):\n    env.step(0.5)\n    raise ValueError('lost the cube')\n",
        "swallows.py": "def solve(env):\n    try:\n        while True:\n            env.step(0.01)\n    except Exception:\n        pass\n",
    }
    receipts = {}
    for name, text in solutions.items():
        (tmp_path / name).write_text(text)
        receipts[name] = checks.solve(tmp_path / "task.py", tmp_path / name, tmp_path / name[:-3], seeds=3)
    assert receipts["pushes.py"]["passed"] and receipts["pushes.py"]["solved"] == 3
    assert [r["steps"] for r in receipts["pushes.py"]["episodes"]] == [11, 10, 9]
    assert receipts["slow.py"]["solved"] == 0 and receipts["slow.py"]["episodes"][0]["steps"] == 20
    crashed = receipts["crashes.py"]["episodes"][0]
    assert not crashed["solved"] and "lost the cube" in crashed["error"] and not receipts["crashes.py"]["passed"]
    assert receipts["swallows.py"]["episodes"][0]["steps"] == 20  # its own handler cannot catch the step limit


def test_unsolved_seeds_and_a_changed_task_go_back_to_the_same_conversation(tmp_path, monkeypatch):
    built = tmp_path / "built"
    (built / "work").mkdir(parents=True)
    (built / "work/task.py").write_text(TASK)
    (built / "work/demo.py").write_text("def solve(env):\n    pass\n")  # the builder's, withheld from the solver
    (built / "request.json").write_text(json.dumps(dict(instruction="push the cube past 1 m")))
    calls, checked = [], []

    def llm(system, messages, model, cost, effort):
        calls.append(messages)
        cost["usd"] += 0.25
        code = (
            "open('solution.py', 'w').write('def solve(env):\\n    pass\\n')\nopen('notes.md', 'w').write('approach')\n"
        )
        if len(calls) > 1:  # the second turn edits the frozen task, the third puts it back
            text = TASK + ("\n# easier\n" if len(calls) == 2 else "")
            code += f"import os\nos.chmod('../task/task.py', 0o644)\nopen('../task/task.py', 'w').write({text!r})\n"
        return f"```python\n{code}print(finish())\n```"

    def fake_check(python, task, out, seeds, solution=None):
        checked.append((task, solution))
        return dict(passed=len(checked) > 1, solved=seeds if len(checked) > 1 else 1, episodes=[])

    monkeypatch.setattr(run, "claude", llm)
    monkeypatch.setattr(run, "check", fake_check)
    record = run.solution(built, tmp_path / "s", python="py", rounds=3, seeds=4)
    assert record["passed"] and len(record["rounds"]) == 3 and record["cost_usd"] == 0.75
    assert [r["task_unchanged"] for r in record["rounds"]] == [True, False, True]
    assert [r["passed"] for r in record["rounds"]] == [False, False, True]  # the second's check passed, its task not
    messages = calls[-1]
    assert "push the cube past 1 m" in messages[0]["content"] and str(tmp_path / "s/task") in messages[0]["content"]
    assert "1/4 seeds" in messages[2]["content"] and "frozen task directory changed" in messages[4]["content"]
    assert checked[0] == (tmp_path / "s/task/task.py", tmp_path / "s/work/solution.py")
    assert not (tmp_path / "s/task/demo.py").exists()


def test_record_reports_every_round(tmp_path, monkeypatch):
    built = tmp_path / "built"
    (built / "work").mkdir(parents=True)
    (built / "work/task.py").write_text(TASK)
    (built / "request.json").write_text(json.dumps(dict(instruction="push")))
    monkeypatch.setattr(run, "claude", lambda *a, **k: "```python\nprint(finish())\n```")
    monkeypatch.setattr(run, "check", lambda *a, **k: dict(passed=False, solved=0, error="boom"))
    record = run.solution(built, tmp_path / "s", python="py", rounds=2, seeds=2)
    assert not record["passed"] and [r["error"] for r in record["rounds"]] == ["boom", "boom"]
    assert json.loads((tmp_path / "s/result.json").read_text())["end"] == "out of checks"


def test_sh_stops_a_command_at_its_timeout(tmp_path):
    space = run.Workspace(tmp_path, tmp_path, None, deadline=time.time() + 60, rounds=1, cost={"usd": 0.0})
    started = time.time()
    assert "[stopped: out of time]" in space.sh("echo started; sleep 30", timeout=1)
    assert time.time() - started < 10 and space.sh("pwd").split("\n")[1] == str(tmp_path)
