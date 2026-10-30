"""Agent as policy: an agent plays live episodes through `bot`, one model turn per Python block.

    python -m roboskills.workflows.agent_as_policy.run --adapter A --out DIR [--max-turns N] -- k=v ...

The adapter (a file A.py, or a module such as roboskills.eval.robodojo.launch) defines `make_vector(out=DIR, **kw)
-> [Robot]` (one simulator, one robot per env; may define `close()`) or `make(**kw) -> Robot` (one env). A ROBOT.md
beside the adapter goes to every agent, unless the adapter is under artifacts/, which agents never see. Each robot
gets its own Tools and agent loop, all at once in this process: one SAM 3, one grasp policy. Two locks: the planner
lock (a turn's own GPU work: RoboKit, the grasp policy), let go while a robot executes, observes or SAM runs, and the
SAM lock (its one worker), so the envs move at once, plan while another segments, and the agents think at once.
Results: DIR/env<i>/result.json and log.jsonl, and DIR/summary.json (successes and voids, startup_s, episodes_s, and
each env's turns, cost and seconds). A void episode (its LLM failed: the model could not play it) is finished to free
the simulator but counts as neither success nor failure; rerun it. Each episode saves its result when it ends.
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from roboskills.core import agent, skill
from roboskills.core.episode import OUTCOME, STATUS, Lock, end, locked, make_envs, robot, to_json
from roboskills.core.llm import claude, deepseek
from roboskills.core.process import load
from roboskills.eval.records import save


PROMPT = skill("agent_as_policy")
FIRST = STATUS  # the first status


def summarize(rows, **timing):
    return dict(
        episodes=len(rows),
        voids=sum(bool(r.get("void")) for r in rows),
        successes=sum(bool(r.get("success")) and not r.get("void") for r in rows),
        **timing,
        envs=rows,
    )


def local(adapter, out, kw, model="claude-opus-5-5", max_turns=None, llm=claude):
    """Play every env of the adapter at once, one agent loop each; returns the summary. `llm(system, messages,
    model=, cost=)`: the backend (default `claude -p`), called by keyword; `cost` is this env's {"usd": running total}.
    An episode its agent leaves open is finished by the runner."""
    mod = load(adapter, "adapter")
    context = robot(mod)
    out.mkdir(parents=True, exist_ok=True)
    plan, t0 = Lock(), time.monotonic()
    with make_envs(mod, out, kw, plan) as (bots, _):
        t_ready = time.monotonic()

        def one(i, episode):
            tools = episode.tools
            repl, cost, t = agent.Repl(tools, episode), {"usd": 0.0}, time.monotonic()
            repl.exec, repl.api = locked(repl.exec, plan), locked(repl.api, plan)
            try:
                done, turns = agent.run(
                    repl,
                    tools.status()["instruction"],
                    tools.out,
                    lambda s, m: llm(s, m, model=model, cost=cost),
                    max_turns,
                    prompt=PROMPT,
                    context=context,
                    initial_code=FIRST,
                    status_code=STATUS,
                )
                row = dict(agent_finished=done, turns=turns)
            except Exception as e:  # noqa: BLE001  (the LLM failed, not the model: the episode is void)
                row = dict(agent_finished=False, error=repr(e))
            with plan:
                end(episode, row["error"][:300] if "error" in row else None)
            save(episode.out / "result.json", episode.result, default=to_json)
            result = {k: v for k, v in (episode.result or {}).items() if k in OUTCOME}
            return dict(env=i, **row, **result, cost_usd=round(cost["usd"], 4), seconds=round(time.monotonic() - t, 1))

        with ThreadPoolExecutor(len(bots)) as pool:
            rows = list(pool.map(one, range(len(bots)), bots))
        timing = dict(startup_s=round(t_ready - t0, 1), episodes_s=round(time.monotonic() - t_ready, 1))
        summary = summarize(rows, **timing)
        save(out / "summary.json", summary, default=to_json)
        return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", default="claude-opus-5-5")
    p.add_argument("--max-turns", type=int, help="default: no limit")
    argv = sys.argv[1:]
    cut = argv.index("--") if "--" in argv else len(argv)  # k=v after "--" go to the adapter's make_vector() or make()
    a = p.parse_args(argv[:cut])
    kw = dict(x.split("=", 1) for x in argv[cut + 1 :])
    llm = deepseek if a.model.startswith("deepseek") else claude
    print(json.dumps(local(a.adapter, a.out.resolve(), kw, a.model, a.max_turns, llm)))


if __name__ == "__main__":
    main()
