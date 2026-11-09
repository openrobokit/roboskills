"""Approve-ahead execution: whole plans the agent queues run back to back while it keeps planning from where the queue
will end; the operator approves each one in the panel before the executor reaches it.

An entry is one plan's `Trajectory`s, in order, and what it needs first (`holding`: arms whose hands must hold
something, the driver's `holds`). It starts where the entry before it ends, so the queue's `end` is where the next plan
starts. The executor runs the first entry once it is approved: each trajectory checked against the measured state (`Limits.check`)
and streamed (`RealRobot._stream`); a refused or stopped trajectory, a failed `holding`, Stop or the end of the episode
drops every entry after it, and the arms hold. The reviewer shows the first entry not yet approved, from where it will
start: Execute approves it, Reject drops it and every entry after it, Rest drops them all and parks the arms once the
running entry ends, Stop ends the episode.
"""

import threading

import numpy as np


PENDING = ("waiting", "approved", "running")  # waiting: for the operator's approval
MAX_WAIT = 60.0  # s `view` blocks at most


class Entry:
    def __init__(self, n, paths, holding, label, start, status, dof):
        self.id, self.paths, self.holding, self.label, self.status = n, list(paths), list(holding or []), label, status
        self.start, self.result = start, None
        self.end = {a: v.copy() for a, v in start.items()}  # where it leaves each arm: its joints, then its hand
        for path in self.paths:
            for a, Q in path.q.items():
                self.end[a][: dof[a]] = np.asarray(Q, float)[-1]
            for a, U in path.hand.items():
                self.end[a][dof[a] :] = np.asarray(U, float)[-1]

    def summary(self):
        return dict(id=self.id, label=self.label, status=self.status, result=self.result)


class Queue:
    def __init__(self, robot):
        self.robot, self.entries, self.cond = robot, [], threading.Condition()
        self.dof = {a: hw.n for a, hw in robot.hw.items()}
        self.cancel, self.closing = threading.Event(), False  # cancel: the review on show is out of date
        self.threads = [threading.Thread(target=self._execute, daemon=True, name="queue-executor")]
        if robot.review:
            self.threads.append(threading.Thread(target=self._review, daemon=True, name="queue-reviewer"))
        for t in self.threads:
            t.start()

    def busy(self):
        with self.cond:
            return any(e.status in PENDING for e in self.entries)

    def _end(self):
        """{arm: joints and hand} where the last pending entry leaves the arms, else where they are measured (not their
        targets: an arm that stopped short of its last target, held back or sagging, starts the next plan from where it
        is)."""
        last = next((e for e in reversed(self.entries) if e.status in PENDING), None)
        return {a: v.copy() for a, v in last.end.items()} if last else self.robot._read()

    def add(self, paths, holding=None, label=""):
        """Queue one plan's trajectories; returns its id. ValueError when it does not start where the queue ends."""
        r = self.robot
        if r.ended:
            raise ValueError("the episode is over: nothing is queued any more")
        unknown = sorted(set(holding or []) - set(r.hw))
        if unknown:
            raise ValueError(f"holding names no arm of this bench: {unknown} (its arms: {sorted(r.hw)})")
        with self.cond:
            start = self._end()
            for a, Q in paths[0].q.items() if paths else []:
                gap = np.abs(np.asarray(Q, float)[0] - start[a][: self.dof[a]]).max()
                if gap > r.limits.start_tol:
                    raise ValueError(f"the {a} arm's plan starts {gap:.2f} rad from where the queue ends: plan again")
            for path in paths:  # refused now, as `follow` refuses before review, not once the operator approved it
                q0 = {a: np.asarray(path.q[a], float)[0] if a in path.q else start[a][: self.dof[a]] for a in start}
                why = r.limits.check(path, q0, {a: start[a][self.dof[a] :] for a in start}, r.bodies, r.tcp)
                if why:
                    raise ValueError(f"refused: {why}")
            review = r.review and not r._small(paths, start)  # a small correction runs unreviewed, as in `follow`
            e = Entry(
                len(self.entries) + 1, paths, holding, label, start, "waiting" if review else "approved", self.dof
            )
            self.entries.append(e)
            self.cond.notify_all()
        self._show()
        return e.id

    def view(self, wait=0.0):
        """{entries: [{id, label, status, result}], end: {q, hand} where the queue ends, running}; `wait` s: first
        wait until an entry changes (one finishes, fails or is dropped) or the queue empties."""
        with self.cond:
            before = [e.status for e in self.entries]
            if wait and any(e.status in PENDING for e in self.entries):
                self.cond.wait_for(
                    lambda: [e.status for e in self.entries] != before, timeout=min(float(wait), MAX_WAIT)
                )
            end = self._end()
            return dict(entries=[e.summary() for e in self.entries], running=any(e.status in PENDING for e in self.entries),
                        end=self._state(end))  # fmt: skip

    def _state(self, v):
        """{q, hand} of {arm: joints and hand}."""
        return dict(q={a: x[: self.dof[a]] for a, x in v.items()}, hand={a: x[self.dof[a] :] for a, x in v.items()})

    def flush(self, why="flushed", after=None):
        """Drop every entry not yet running (after entry `after` only, when given); returns their ids."""
        with self.cond:
            gone = [e for e in self.entries if e.status in ("waiting", "approved") and (after is None or e.id > after)]
            for e in gone:
                e.status, e.result = "dropped", dict(why=why)
            if gone:
                self.cancel.set()
                self.cond.notify_all()
        self._show()
        return [e.id for e in gone]

    def stop(self):
        """Drop what waits and let the running entry end (the episode's end)."""
        self.flush("the episode ended")
        with self.cond:
            self.closing = True
            self.cond.notify_all()
        self.cancel.set()
        for t in self.threads:
            t.join(timeout=120)

    # ---- the executor and the reviewer

    def _execute(self):
        r = self.robot
        while True:
            with self.cond:
                self.cond.wait_for(lambda: self.closing or self._head("approved"), timeout=0.5)
                if self.closing:
                    return
                e = self._head("approved")
                if e is None:
                    continue
                e.status = "running"
            self._show()
            try:
                res = self._run(e)
            except Exception as exc:  # the entry fails, the executor goes on (else every later plan would wait forever)
                r._fault("error", f"queued plan {e.id}: {type(exc).__name__}: {exc}")
                res = dict(steps=0, why=f"{type(exc).__name__}: {exc}", **self._state(e.start))
            with self.cond:
                e.result, e.status = res, "failed" if res.get("why") else "done"
                self.cond.notify_all()
            if e.status == "failed":
                self.flush(f"entry {e.id} failed: {res['why']}", after=e.id)
            r._show()
            self._show()

    def _head(self, status):
        """The first pending entry, if its status is `status`."""
        e = next((e for e in self.entries if e.status in PENDING), None)
        return e if e is not None and e.status == status else None

    def _run(self, e):
        r, steps, why = self.robot, 0, None
        empty = [a for a in e.holding if not r.hw[a].holds()]
        if r._stopped():
            why = "the episode is over"
        elif empty:
            why = f"needs {', '.join(empty)} holding something: the hand is open, or closed on nothing"
        for path in [] if why else e.paths:
            f = r._run(path)
            steps += f.steps
            if f.refused or f.stopped or f.done or r.ended:
                why = f.refused or ("the episode is over" if f.done or r.ended else "stopped: the arm was held back")
                break
        return dict(steps=steps, why=why, **self._state(r._read()))

    def _review(self):
        r, op = self.robot, self.robot.operator
        while True:
            with self.cond:
                self.cond.wait_for(lambda: self.closing or self._waiting(), timeout=0.5)
                if self.closing:
                    return
                e = self._waiting()
                if e is None:
                    continue
                self.cancel.clear()
            answer = r._review(e.paths, e.start, e.label, cancel=self.cancel)
            with self.cond:
                if e.status != "waiting":  # dropped while on show
                    continue
                if answer == "approve":
                    e.status = "approved"
                    self.cond.notify_all()
            if answer == "reject":
                r._fault("rejected", f"the operator rejected queued plan {e.id}")
                self.flush("the person at the bench rejected it", after=e.id - 1)
            elif answer == "rest":
                self.flush(
                    "the person at the bench opened the hands and moved the arms to rest instead: observe and plan again"
                )
                with self.cond:
                    self.cond.wait_for(lambda: not any(x.status == "running" for x in self.entries))
                r._rest()
            elif answer == "stop":
                r._fault("operator", "stop pressed during review: the episode ends")
                r.ended = True
                self.flush("the episode is over")
            op.say(f"Running: {r.instruction}")
            self._show()

    def _waiting(self):
        """The first entry waiting for approval (each is approved in order)."""
        return next((e for e in self.entries if e.status == "waiting"), None)

    def _show(self):
        rows = [f"{e.id}. {e.label or 'plan'}: **{e.status}**" + (f" ({e.result['why']})" if e.result and e.result.get("why") else "")
                for e in self.entries[-12:]]  # fmt: skip
        self.robot.operator.show_queue("\n".join(rows) or "Nothing queued")
