"""Self-checks without a simulator or an LLM: python -m pytest tests/workflows/auto_research"""

import json
from types import SimpleNamespace

from roboskills.workflows.auto_research import run


ADAPTER = """
class Robot:
    def __init__(self, layout):
        self.layout, self.done = layout, False
    def obs(self, cams=True): pass
    def follow(self, motion): motion()
    def finish(self, out):
        return dict(success=self.done, score=float(self.done), reason="r", steps=1, layout=self.layout, truth="SECRET")
def make(layout="0", **kw): return Robot(layout)
"""


class FakeTools:
    def __init__(self, robot, sam, out=None):
        self.robot, self.sam, self.out, self.model = robot, sam, out, None

    def status(self):
        """steps"""
        return dict(steps=0, instruction=f"task on layout {self.robot.layout}")

    def execute(self, plan):
        """moves"""
        self.robot.follow(lambda: setattr(self.robot, "done", True))


def research(tmp_path, monkeypatch, replies, **kw):
    from roboskills.tools.robokit import bot
    from roboskills.tools.sam3 import sam3

    monkeypatch.setattr(bot, "Tools", FakeTools)
    monkeypatch.setattr(sam3, "Sam", lambda: SimpleNamespace(detect=lambda *a, **k: None, close=lambda: None))
    (tmp_path / "a.py").write_text(ADAPTER)
    seen, replies = [], iter(replies)

    def llm(system, messages, model, cost):
        seen.append(json.dumps(messages))
        return next(replies)

    dev = [dict(layout="10"), dict(layout="11")]
    summary = run.research(str(tmp_path / "a.py"), tmp_path / "out", {}, dev, test=[dict(layout="99")], llm=llm, **kw)
    return summary, seen


SOLUTION = "def solve(bot):\n    bot.execute(None)\n    print('moved')\n"


def test_reset_submit_freeze(tmp_path, monkeypatch):
    replies = [
        "```python\nx = 7\nprint(reset(case=1))\n```",
        "```python\nprint(x, bot.status()['instruction'], submit('import os'))\n```",
        f"```python\nprint(submit({SOLUTION!r}))\n```",
        "```python\nfinish()\n```",
    ]
    summary, seen = research(tmp_path, monkeypatch, replies)
    assert "'success': False" in seen[1] and "'case': 1" in seen[1]  # the ended case-0 episode's outcome
    assert "7 task on layout 11" in seen[2] and "invalid solution" in seen[2] and "'submits_left': 3" in seen[2]
    assert seen[3].count("'success': True") == 2 and "moved" in seen[3]  # both cases, fresh episodes
    assert not any("SECRET" in s or "99" in s for s in seen)  # no truth, no test case
    assert summary["agent_finished"] and summary["attempts"] == [
        1,
        2,
    ]  # case 0; case 1 from the reset, then after the submit
    assert summary["dev"] == dict(episodes=2, successes=2) and summary["test"]["successes"] == 1
    assert (tmp_path / "out/solution.py").read_text() == SOLUTION
    assert json.loads((tmp_path / "out/test/case0/env0/result.json").read_text())["layout"] == "99"


def test_budgets_and_no_submission(tmp_path, monkeypatch):
    replies = [
        "```python\nprint(reset())\nprint(reset(case=5))\n```",
        "```python\nprint(reset())\n```",
        "```python\nfinish()\n```",
    ]
    summary, seen = research(tmp_path, monkeypatch, replies, max_resets=1)
    assert "case is 0 .. 1" in seen[1] and "no resets left" in seen[2]
    assert summary["resets"] == 1 and "dev" not in summary and "test" not in summary  # nothing to freeze


def test_void_research(tmp_path, monkeypatch):
    from roboskills.tools.robokit import bot
    from roboskills.tools.sam3 import sam3

    monkeypatch.setattr(bot, "Tools", FakeTools)
    monkeypatch.setattr(sam3, "Sam", lambda: SimpleNamespace(detect=lambda *a, **k: None, close=lambda: None))
    (tmp_path / "a.py").write_text(ADAPTER)

    def llm(system, messages, model, cost):
        raise RuntimeError("429")

    summary = run.research(str(tmp_path / "a.py"), tmp_path / "out", {}, [dict(layout="1")], llm=llm)
    assert "429" in summary["void"] and (tmp_path / "out/dev0/attempt0/env0/result.json").exists()
