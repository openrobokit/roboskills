"""Auto research: one agent researches a task in live development episodes (agent_as_policy's turn loop), resets
them to rerun a case or switch to another, and submits `solve(bot)` programs, each run on fresh episodes of every
development case. The best submission is frozen and run on validation and test cases without a model.

    python -m roboskills.workflows.auto_research.run --adapter A --out DIR --dev "k=v ..." [--dev ...] \
        [--val "k=v ..."] [--test "k=v ..."] [--max-resets N] [--max-submits N] [--max-turns N] -- k=v ...

A case is the adapter's `make_vector` / `make` keywords: the ones after "--", updated by the case's own "k=v ..." pairs.
A development case gives one env; a validation or test case may give several (a batch, e.g. layouts=0,1,2,3,4). One
simulator runs at a time. The agent sees development outcomes only (success, score, reason, steps); validation and
test cases and their outcomes never reach it. Results: DIR/dev<i>/attempt<k>/, DIR/submit<k>/ (solution.py and each
case's run), DIR/solution.py (the frozen one), DIR/val/, DIR/test/ and DIR/summary.json.
"""

import argparse
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, closing
from pathlib import Path

import numpy as np

from roboskills.core import agent, skill
from roboskills.core.episode import OUTCOME, STATUS, Lock, end, locked, make_envs, robot, to_json, unlocked
from roboskills.core.llm import claude, deepseek
from roboskills.core.process import load
from roboskills.eval.records import save
from roboskills.workflows.auto_research.policy import VERBS, validate


PROMPT = skill("agent_as_policy") + "\n" + skill("auto_research")
SHOWN = ("success", "score", "reason", "steps")  # what a development outcome tells the agent


def outcome(result, keys=OUTCOME):
    return {k: v for k, v in (result or {}).items() if k in keys}


def solve(source, tools):
    """Run `source`'s solve(bot) on `tools`: (what it printed, the error tail if it raised). Only the public verbs and
    bot.model (as the live agent has them), math and numpy."""
    try:
        validate(source, public=VERBS | {"model"})
    except (SyntaxError, ValueError) as e:
        return f"ERROR: invalid solution: {e}"
    return agent.execute(source + "\nsolve(bot)\n", {"bot": tools, "np": np, "math": math})


def play(mod, kw, out, plan, sam, source):
    """`source` on every env of one case (one adapter call), at once: each env's outcome and printed output."""
    out.mkdir(parents=True, exist_ok=True)
    with make_envs(mod, out, kw, plan, sam) as (episodes, _):

        def one(episode):
            with plan:
                printed = solve(source, episode.tools)
                end(episode)
            save(episode.out / "result.json", episode.result, default=to_json)
            return dict(outcome(episode.result), output=printed[-1500:])

        with ThreadPoolExecutor(len(episodes)) as pool:
            return list(pool.map(one, episodes))


class Research:
    """The agent's `reset`, `submit` and `finish` over the development cases, and the live development episode.
    `result` is None until the research ends (the turn loop's episode)."""

    def __init__(self, mod, out, dev, plan, max_resets, max_submits):
        self.mod, self.out, self.dev, self.plan = mod, out, dev, plan
        self.resets, self.submits = max_resets, max_submits
        self.attempts, self.submissions, self.result, self.repl = [0] * len(dev), [], None, None
        self.cleanup, self.sam, self.case, self.episode = ExitStack(), None, 0, None

    def open(self, case):
        """A fresh episode of development case `case`, as the namespace's `bot`."""
        if self.sam is None:  # one SAM 3 for every episode of the research, closed at its end
            from roboskills.tools.sam3.sam3 import Sam

            self.sam = self.cleanup.enter_context(closing(Sam()))
            self.sam.detect = unlocked(locked(self.sam.detect, Lock()), self.plan)
        self.case, k = case, self.attempts[case]
        self.attempts[case] += 1
        out = self.out / f"dev{case}" / f"attempt{k}"
        out.mkdir(parents=True)
        envs = make_envs(self.mod, out, self.dev[case], self.plan, self.sam)
        self.stack = ExitStack()
        (self.episode,), _ = self.stack.enter_context(envs)
        if self.repl is not None:
            self.repl.ns["bot"] = self.episode.tools

    def close(self):
        """End the live episode (scored) and stop its simulator: its outcome."""
        with self.plan:
            end(self.episode)
        save(self.episode.out / "result.json", self.episode.result, default=to_json)
        self.stack.close()
        return outcome(self.episode.result, SHOWN)

    def reset(self, case=None):
        """End this episode and start a fresh one of development case `case` (default: this one again); your
        variables stay, the robot and objects start over. Returns the ended episode's outcome."""
        case = self.case if case is None else case
        if not 0 <= case < len(self.dev):
            raise ValueError(f"case is 0 .. {len(self.dev) - 1}")
        if self.resets < 1:
            raise RuntimeError("no resets left: submit a solution or finish()")
        self.resets -= 1
        ended = self.close()
        self.open(case)
        return dict(ended=ended, case=case, resets_left=self.resets, submits_left=self.submits)

    def submit(self, source):
        """Run `source` (Python defining solve(bot)) on a fresh episode of every development case. This episode ends
        first; you continue in a fresh episode of this case. Returns each case's outcome and the solution's output."""
        if self.submits < 1:
            raise RuntimeError("no submissions left: finish()")
        try:
            validate(source, public=VERBS | {"model"})
        except (SyntaxError, ValueError, TypeError) as e:  # invalid code spends nothing
            return dict(error=f"invalid solution: {e}", submits_left=self.submits)
        self.submits -= 1
        ended, k = self.close(), len(self.submissions)
        d = self.out / f"submit{k}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "solution.py").write_text(source)
        rows = [play(self.mod, kw, d / f"dev{i}", self.plan, self.sam, source)[0] for i, kw in enumerate(self.dev)]
        self.submissions.append(dict(source=source, cases=rows, successes=sum(bool(r.get("success")) for r in rows)))
        self.open(self.case)
        cases = [dict(outcome(r, SHOWN), output=r["output"]) for r in rows]
        return dict(ended=ended, cases=cases, submits_left=self.submits, resets_left=self.resets)

    def finish(self):
        """End the research; the best submission is your solution."""
        if self.result is None:
            self.result = dict(ended=self.close())
        return "research finished"

    def best(self):
        """The submission with the most development successes; the latest of equals."""
        return max(reversed(self.submissions), key=lambda s: s["successes"], default=None)


def cases(mod, base, specs, out, plan, sam, source):
    """`source` on each case of `specs` (one adapter call each, in turn): every env's outcome."""
    return [r for i, kw in enumerate(specs) for r in play(mod, {**base, **kw}, out / f"case{i}", plan, sam, source)]


def score(rows):
    return dict(episodes=len(rows), successes=sum(bool(r.get("success")) for r in rows))


def research(adapter, out, base, dev, val=(), test=(), model="claude-opus-5-5", max_turns=None, max_resets=5,
             max_submits=3, llm=claude):  # fmt: skip
    """Research on `dev`, then run the frozen solution on `val` and `test`; returns the summary. `base`: the adapter's
    keywords; each case updates them."""
    mod = load(adapter, "adapter")
    out.mkdir(parents=True, exist_ok=True)
    dev = [{**base, **kw} for kw in dev]
    plan, cost, t0 = Lock(), {"usd": 0.0}, time.monotonic()
    r = Research(mod, out, dev, plan, max_resets, max_submits)
    with r.cleanup:
        r.open(0)
        r.repl = repl = agent.Repl(r.episode.tools, r)
        repl.exec, repl.api = locked(repl.exec, plan), locked(repl.api, plan)
        # they end and start whole episodes, which take the planner lock themselves
        repl.ns.update(reset=unlocked(r.reset, plan), submit=unlocked(r.submit, plan), finish=unlocked(r.finish, plan))
        task = r.episode.tools.status()["instruction"]
        notes = f"{len(dev)} development cases (0 .. {len(dev) - 1}); you are in case 0. Resets: {max_resets}, "
        notes += f"submissions: {max_submits}.\n\n" + robot(mod)
        try:
            done, turns = agent.run(repl, task, out, lambda s, m: llm(s, m, model=model, cost=cost), max_turns,
                                    prompt=PROMPT, context=notes, initial_code=STATUS, status_code=STATUS)  # fmt: skip
            row = dict(agent_finished=done, turns=turns)
        except Exception as e:  # noqa: BLE001  (the LLM failed: the research is void)
            row = dict(agent_finished=False, void=repr(e)[:300])
        if r.result is None:
            r.finish()
        t_research = time.monotonic()
        best = r.best()
        summary = dict(**row, cost_usd=round(cost["usd"], 4), resets=max_resets - r.resets, attempts=r.attempts,
                       submissions=[dict(successes=s["successes"], cases=[outcome(c) for c in s["cases"]])
                                    for s in r.submissions],
                       research_s=round(t_research - t0, 1))  # fmt: skip
        if best is not None:
            (out / "solution.py").write_text(best["source"])
            summary["dev"] = score(best["cases"])
            for name, specs in (("val", val), ("test", test)):
                rows = cases(mod, base, specs, out / name, plan, r.sam, best["source"])
                summary[name] = dict(score(rows), envs=[outcome(x) for x in rows])
        summary["seconds"] = round(time.monotonic() - t0, 1)
    save(out / "summary.json", summary, default=to_json)
    return summary


def pairs(text):
    return dict(x.split("=", 1) for x in text.split())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--dev", action="append", type=pairs, required=True, help='one case: "k=v ...", repeatable')
    p.add_argument("--val", action="append", type=pairs, default=[])
    p.add_argument("--test", action="append", type=pairs, default=[])
    p.add_argument("--model", default="claude-opus-5-5")
    p.add_argument("--max-turns", type=int, help="default: no limit")
    p.add_argument("--max-resets", type=int, default=5)
    p.add_argument("--max-submits", type=int, default=3)
    argv = sys.argv[1:]
    cut = argv.index("--") if "--" in argv else len(argv)  # k=v after "--": every case's adapter keywords
    a = p.parse_args(argv[:cut])
    base = dict(x.split("=", 1) for x in argv[cut + 1 :])
    llm = deepseek if a.model.startswith("deepseek") else claude
    summary = research(a.adapter, a.out.resolve(), base, a.dev, a.val, a.test, a.model, a.max_turns, a.max_resets,
                       a.max_submits, llm)  # fmt: skip
    print(json.dumps(summary, default=to_json))


if __name__ == "__main__":
    main()
