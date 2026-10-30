"""The build stage without a simulator or a model: check.py on a numpy task, and build's check-and-resume loop."""

import sys
import types

from roboskills.workflows.agentic_sim import check as checks
from roboskills.workflows.agentic_sim import run


TASK = """
import numpy as np

class Env:
    robot = ["fakebot"]
    max_steps = 20

    def __init__(self, seed):
        self.seed, self.x = seed, 0.1 * seed

    def reset(self):
        self.x = 0.1 * self.seed

    def step(self, action=None):
        self.x += 0.0 if action is None else 0.05

    def observe(self):
        return dict(robot_q=dict(j1=REST, j2=0.0), objects=dict(cube=[self.x, 0, 0, 1, 0, 0, 0]))

    def render(self):
        return np.tile(np.arange(64, dtype=np.uint8)[:, None, None] * 4, (1, 48, 3))

    def success(self):
        return self.x > GOAL

def make(seed):
    return Env(seed)
"""
DEMO = "def solve(env):\n    while True:\n        env.step(1)\n"  # ended by the harness on success


def fake_robot(monkeypatch):
    monkeypatch.setitem(sys.modules, "fakebot", types.SimpleNamespace(REST_Q=dict(j1=0.5)))
    monkeypatch.setitem(sys.modules, "imageio", types.ModuleType("imageio"))
    monkeypatch.setitem(sys.modules, "imageio.v2", types.SimpleNamespace(mimwrite=lambda path, frames, **_: None))


def test_a_task_that_holds_its_contract_passes(tmp_path, monkeypatch):
    fake_robot(monkeypatch)
    (tmp_path / "task.py").write_text("REST, GOAL = 0.5, 0.5\n" + TASK)
    (tmp_path / "demo.py").write_text(DEMO)
    receipt = checks.check(tmp_path / "task.py", tmp_path / "out", seeds=3)
    assert receipt["passed"], receipt
    assert receipt["demo"]["solved"] == 3 and (tmp_path / "out/start-2.png").exists()


def test_each_check_catches_its_failure(tmp_path, monkeypatch):
    fake_robot(monkeypatch)
    (tmp_path / "task.py").write_text("REST, GOAL = 0.3, -1\n" + TASK)  # off its rest pose; already succeeded
    receipt = checks.check(tmp_path / "task.py", tmp_path / "out", seeds=2)
    assert not receipt["checks"]["robot_at_rest"] and not receipt["checks"]["holding_still_fails"]
    assert receipt["checks"]["builds"] and receipt["checks"]["starts_differ"] and not receipt["passed"]
    assert not receipt["checks"]["demo_solves"] and "no demo.py" in receipt["demo"]["error"]
    (tmp_path / "far.py").write_text("REST, GOAL = 0.5, 9\n" + TASK)  # out of the robot's reach
    (tmp_path / "demo.py").write_text(DEMO)
    receipt = checks.check(tmp_path / "far.py", tmp_path / "far", seeds=2)
    assert receipt["demo"]["solved"] == 0 and not receipt["checks"]["demo_solves"] and not receipt["passed"]
    (tmp_path / "broken.py").write_text("def make(seed):\n    raise RuntimeError('no scene')\n")
    receipt = checks.check(tmp_path / "broken.py", tmp_path / "broken", seeds=2)
    assert not receipt["checks"]["builds"] and "no scene" in receipt["error"]


def test_failed_checks_go_back_to_the_same_conversation(tmp_path, monkeypatch):
    receipts = iter([dict(passed=True, checks={}), dict(passed=False, checks=dict(builds=False)),
                     dict(passed=True, checks={})])  # fmt: skip
    replies = iter(["```python\nprint(sh('echo hi'))\nprint(finish())\n```",  # passes, but without notes.md
                    "```python\nopen('notes.md', 'w').write('MuJoCo')\nprint(finish())\n```",  # its check fails
                    "```python\nprint(finish())\n```"])  # fmt: skip
    calls = []

    def llm(system, messages, model, cost, effort):
        calls.append((system, messages))
        cost["usd"] += 0.5
        return next(replies)

    monkeypatch.setattr(run, "claude", llm)
    monkeypatch.setattr(run, "robots", lambda python: "robokit.assets.robots.arms.franka_panda")
    monkeypatch.setattr(run, "check", lambda python, task, out, seeds: next(receipts))
    record = run.build("stack two cubes", tmp_path / "b", python="py", rounds=4)
    assert record["passed"] and record["end"] == "passed" and len(record["rounds"]) == 3 and record["cost_usd"] == 1.5
    assert [r["passed"] for r in record["rounds"]] == [False, False, True]
    system, messages = calls[-1]
    assert "- sh(command, timeout=600)" in system and "{api}" not in system and "{joint: position}" in system
    assert "stack two cubes" in messages[0]["content"] and "franka_panda" in messages[0]["content"]
    first, second = messages[2]["content"], messages[4]["content"]
    assert "[exit 0]\nhi" in first and "write notes.md" in first and "builds" in second
    assert (tmp_path / "b/work/notes.md").read_text() == "MuJoCo"  # the agent's code ran in its work directory
