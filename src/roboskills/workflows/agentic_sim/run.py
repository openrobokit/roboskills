"""Agentic simulation: one agent builds the simulation task an instruction describes, and a second solves the frozen
task, each writing Python that the harness runs in its work directory (`core.agent`); native checks (check.py) judge
both. README.md describes the two stages and this command line."""

import argparse
import base64
import inspect
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from roboskills.core import agent, skill
from roboskills.core.llm import claude, deepseek
from roboskills.core.process import terminate
from roboskills.eval.records import file_digest, save


BUILD, SOLVE, ACT = (skill("agentic_sim", f"{name}.md") for name in ("build", "solve", "act"))
MEDIA = {".png", ".jpg", ".mp4", ".pyc"}  # what a frozen task directory may rewrite without changing the task


def robots(python):
    """robokit's robot modules in the simulators' interpreter ``python``, one per line, read from its source:
    importing one fetches its assets."""
    find = "import importlib.util; print(importlib.util.find_spec('robokit').origin)"
    root = Path(
        subprocess.run([str(python), "-c", find], check=True, capture_output=True, text=True).stdout.strip()
    ).parent
    return "\n".join(".".join(("robokit", *p.relative_to(root).with_suffix("").parts))
                     for p in sorted((root / "assets/robots").glob("*/*.py")) if p.stem != "__init__")  # fmt: skip


def check(python, task, out, seeds, timeout=3600, solution=None):
    """check.py's receipt on ``task`` (with ``solution``: on that solution to it), run with the simulators'
    interpreter ``python`` beside the file it judges; a failed receipt when check.py writes none."""
    out.mkdir(parents=True)
    command = [str(python), "-m", "roboskills.workflows.agentic_sim.check", str(task), str(out), "--seeds", str(seeds)]
    command += ["--solution", str(solution)] if solution else []
    with (out / "check.log").open("w") as log:
        try:
            subprocess.run(command, check=False, cwd=(solution or task).parent, stdout=log, stderr=subprocess.STDOUT,
                           timeout=timeout)  # fmt: skip
        except subprocess.TimeoutExpired:
            return dict(passed=False, error=f"check.py took over {timeout} s")
    if (receipt := out / "receipt.json").exists():
        return json.loads(receipt.read_text())
    return dict(passed=False, error=(out / "check.log").read_text()[-4000:])


class Workspace:
    """An agent's tools for `agent.run`, in ``work``: its code runs with ``work`` as the current directory, `sh` runs
    programs there, `show` puts image files in the turn's receipt, and each `finish` is a round that ``judge`` checks.
    The session ends when a round passes, ``rounds`` are spent or ``deadline`` (Unix time) passes."""

    def __init__(self, work, out, judge, deadline, rounds, cost):
        self.work, self.out, self.judge, self.deadline, self.rounds, self.cost = (
            work,
            out,
            judge,
            deadline,
            rounds,
            cost,
        )
        self.record, self.images, self.result = dict(passed=False, rounds=[], cost_usd=0.0), [], None
        self.started = time.time()
        self.ns = dict(sh=self.sh, show=self.show, finish=self.finish, status=self.status)

    def sh(self, command, timeout=600):
        """Run a shell command in the work directory; returns its exit code and output (stdout and stderr together).
        After ``timeout`` seconds, or at the deadline, it is stopped with every process it started."""
        process = subprocess.Popen(command, shell=True, cwd=self.work, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, start_new_session=True)  # fmt: skip
        try:
            output = process.communicate(timeout=max(1.0, min(timeout, self.deadline - time.time())))[0]
        except subprocess.TimeoutExpired:
            terminate(process)
            output = process.communicate()[0] + "\n[stopped: out of time]"
        return f"[exit {process.returncode}]\n{output}"

    def show(self, *paths):
        """Show PNG files (paths relative to the work directory) with this turn's output."""
        self.images += [(str(p), base64.b64encode((self.work / p).read_bytes()).decode()) for p in paths]

    def status(self):
        """Time and checks left."""
        left = self.rounds - len(self.record["rounds"])
        return f"{max(0, self.deadline - time.time()) / 60:.0f} min and {left} check(s) left"

    def finish(self):
        """Hand your work to the harness's check; returns its verdict. Passing, or spending the last check, ends the
        session; otherwise fix what failed and call it again."""
        k = len(self.record["rounds"])
        passed, fields, retry = self.judge(self.out / f"round-{k}" / "check")
        notes = (self.work / "notes.md").exists()
        row = dict(round=k, passed=passed and notes, seconds=round(time.time() - self.started), **fields)
        self.record["rounds"].append(row)
        self.record["cost_usd"] = round(self.cost["usd"], 4)
        save(self.out / "result.json", self.record)
        if row["passed"]:
            self.record.update(passed=True, check=f"round-{k}/check")
            self.result = "passed"
            return "Passed. The session is over."
        if k + 1 >= self.rounds:
            self.result = "out of checks"
        return retry if notes else retry + "\nAlso write notes.md: the check counts only with it."

    def exec(self, code):
        """(printed output, session over, images [(name, base64 PNG)] the code showed), as `agent.run` expects."""
        cwd = os.getcwd()
        os.chdir(self.work)
        try:
            out = agent.execute(code, self.ns)
        finally:
            os.chdir(cwd)
        if self.result is None and time.time() > self.deadline:
            self.result = "deadline"
        shown, self.images = self.images, []
        return out, self.result is not None, shown

    def api(self):
        return "\n".join(f"- {n}{inspect.signature(f)}: {f.__doc__}" for n, f in self.ns.items())


def converse(prompt, message, out, work, judge, *, hours, rounds, model, effort):
    """One agent session in ``work`` (`agent.run` on `claude`, or `deepseek` for deepseek-* models) within ``hours``;
    each `finish()` is a round: ``judge(out/round-<k>/check)`` gives (passed, the round's fields, a failure's reply),
    which goes back to the same conversation, up to ``rounds``. Every round goes to out/result.json."""
    deadline = (started := time.time()) + hours * 3600
    message += f"\n\nDeadline: {datetime.fromtimestamp(deadline, timezone.utc).isoformat(timespec='minutes')}"
    space = Workspace(work, out, judge, deadline, rounds, cost := {"usd": 0.0})
    llm = deepseek if model.startswith("deepseek") else claude
    system = prompt.replace("{", "{{").replace("}", "}}") + "\n\n" + ACT  # only ACT's {api} is a field
    record = space.record
    try:
        _, record["turns"] = agent.run(space, message, out, lambda s, m: llm(s, m, model=model, cost=cost, effort=effort),
                                       prompt=system, initial_code="print(status())", status_code="print(status())")  # fmt: skip
    except Exception as error:  # noqa: BLE001  (a failed model call ends the session; its rounds stay recorded)
        record["error"] = f"{type(error).__name__}: {error}"[:500]
    record.update(end=space.result, cost_usd=round(cost["usd"], 4), elapsed_s=round(time.time() - started))
    save(out / "result.json", record)
    return record


def build(instruction, out, *, python, hours=2.0, rounds=4, seeds=8, model="claude-opus-5-5", effort=None):
    """One tool-using agent builds the task ``instruction`` describes as out/work/task.py (build.md), in a simulator
    it chooses from ``python``'s (setup.sh), until check.py passes it (`converse`). Returns out/result.json's record."""
    out = Path(out).resolve()
    (work := out / "work").mkdir(parents=True, exist_ok=False)
    save(out / "request.json", dict(instruction=instruction, python=str(python), hours=hours, rounds=rounds,
                                    seeds=seeds, model=model, effort=effort))  # fmt: skip
    message = f"Instruction: {instruction}\n\nPYTHON (the simulators and robokit): {python}\n\nrobokit's robots:\n"

    def judge(directory):
        receipt = check(python, work / "task.py", directory, seeds)
        failed = json.dumps({k: v for k, v in receipt.items() if k in ("checks", "error", "demo", "episodes")})[:12000]
        retry = "The harness's checks on task.py failed (receipt below). Fix the task and finish when the checks pass."
        return receipt["passed"], dict(checks=receipt.get("checks"), error=receipt.get("error")), f"{retry}\n{failed}"

    return converse(
        BUILD, message + robots(python), out, work, judge, hours=hours, rounds=rounds, model=model, effort=effort
    )


def fingerprint(directory):
    """{file: sha256} of a task directory's files, leaving out its checks, caches and media."""
    names = (path.relative_to(directory) for path in sorted(Path(directory).rglob("*")) if path.is_file())
    return {str(name): file_digest(Path(directory) / name) for name in names
            if not {"check", "__pycache__"} & set(name.parts) and name.suffix not in MEDIA}  # fmt: skip


def solution(built, out, *, python, hours=2.0, rounds=4, seeds=8, model="claude-opus-5-5", effort=None):
    """One agent writes out/work/solution.py (solve.md) for the task `build` made in ``built``, frozen first: copied to
    out/task without demo.py, read-only and fingerprinted. Unsolved seeds or a changed task go back to the session
    (`converse`). Returns out/result.json's record."""
    built, out = Path(built).resolve(), Path(out).resolve()
    task, work = out / "task", out / "work"
    shutil.copytree(built / "work", task, ignore=shutil.ignore_patterns("check", "__pycache__", "demo.py"))
    for path in filter(Path.is_file, task.rglob("*")):
        path.chmod(path.stat().st_mode & ~0o222)
    work.mkdir()
    frozen = fingerprint(task)
    instruction = json.loads((built / "request.json").read_text())["instruction"]
    save(out / "request.json", dict(built=str(built), task=frozen, python=str(python), hours=hours, rounds=rounds,
                                    seeds=seeds, model=model, effort=effort))  # fmt: skip
    message = (f"Instruction the task was built from: {instruction}\n\nTASK_DIR (the frozen task): {task}\n\n"
               f"PYTHON (the task's simulator and robokit): {python}")  # fmt: skip

    def judge(directory):
        unchanged = frozen.items() <= fingerprint(task).items()  # every frozen file is still there, unchanged
        receipt = check(python, task / "task.py", directory, seeds, solution=work / "solution.py")
        failed = json.dumps({k: v for k, v in receipt.items() if k in ("solved", "error", "episodes")})[:12000]
        retry = (f"The harness's check of solution.py failed: {receipt.get('solved', 0)}/{seeds} seeds solved"
                 + ("" if unchanged else ", and the frozen task directory changed, which must never happen")
                 + f" (receipt below). Fix the solution and finish when every seed is solved.\n{failed}")  # fmt: skip
        fields = dict(solved=receipt.get("solved"), task_unchanged=unchanged, error=receipt.get("error"))
        return receipt["passed"] and unchanged, fields, retry

    return converse(SOLVE, message, out, work, judge, hours=hours, rounds=rounds, model=model, effort=effort)


def main():
    parser = argparse.ArgumentParser(description="Build the simulation task an instruction describes, with one agent; "
                                     "or, with --solve, solve a built one.")  # fmt: skip
    parser.add_argument("instruction", nargs="?")
    parser.add_argument("--solve", type=Path, help="a build's directory: write a solution to its task instead")
    parser.add_argument("--out", type=Path, required=True, help="new directory, e.g. runs/agentic_sim/NAME")
    parser.add_argument("--python", type=Path, required=True, help="the simulators' interpreter (setup.sh)")
    for flag, kind, default in (("--hours", float, 2.0), ("--rounds", int, 4), ("--seeds", int, 8),
                                ("--model", str, "claude-opus-5-5"), ("--effort", str, None)):  # fmt: skip
        parser.add_argument(flag, type=kind, default=default)
    a = parser.parse_args()
    if (a.instruction is None) == (a.solve is None):
        parser.error("give an instruction to build, or --solve BUILD to solve one")
    options = dict(python=a.python, hours=a.hours, rounds=a.rounds, seeds=a.seeds, model=a.model, effort=a.effort)
    record = solution(a.solve, a.out, **options) if a.solve else build(a.instruction, a.out, **options)
    print(json.dumps(dict(passed=record["passed"], rounds=len(record["rounds"]), cost_usd=record["cost_usd"])))
    raise SystemExit(0 if record["passed"] else 1)


if __name__ == "__main__":
    main()
