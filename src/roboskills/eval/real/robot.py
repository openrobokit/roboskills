"""A real bench behind the shared Robot contract (eval/interface.py): `RealRobot`, which a bench's runtime subclasses with
its arms (`_arms`: drivers, models and limits; eval/yam).

One rig YAML (~/.config/roboskills/<bench>.yaml, see the bench's SKILL.md) describes the bench: each arm's connection,
model and base pose, each camera's serial and calibrated pose, the table height and the workspace box. A budget step is one
streamed command tick (`dt`, 50 Hz by default): steps count only the time the arms are being commanded, so observing,
segmenting and thinking cost none. Success is the operator's verdict, given after the agent's last turn; the agent
never sees it. Each arm's state is its `n` joints, then its hand command (a jaw's opening, a dexterous hand's joints).
"""

import json
import shutil
import threading
import time
from pathlib import Path

import numpy as np

from roboskills.eval.interface import Camera, Follow, Obs, Trajectory
from roboskills.eval.real import safety
from roboskills.eval.real.panel import Operator
from roboskills.eval.real.queued import Queue


SETTLE = 0.5  # s at most spent settling on a path's last waypoint
TOL = 0.01  # rad: an arm has settled on its target
STOP_EVERY = 5  # ticks between the host's stop checks
LATE = 3  # ticks a command may run late before the motion stops (the process stalled)
VIDEO_FPS = 5
REVIEW_FRAMES = 60  # arm poses along a plan the operator scrubs through
REVIEW_UNDER = 0.025  # m: a plan whose TCPs stay this near where they start, hands unchanged, runs unreviewed (a
#                       correction, as `align` makes); the rig's `review_under` overrides it (0: review every plan)
STILL_Q, STILL_U = 1e-3, 0.01  # rad, hand command: a path that moves nothing more than this is skipped (not reviewed)


def load_rig(path):
    import yaml

    rig = yaml.safe_load(Path(path).expanduser().read_text())
    rig["_path"] = str(Path(path).expanduser().resolve())
    return rig


class RealRobot:
    REMOTE = ("replay", "enqueue", "queued", "flush")
    NAME = "Real bench"  # the panel's title
    HAND_SPEED = 0.5  # hand command units per s (a jaw: the full stroke in 2 s); the rig's `hand_speed` overrides it

    def __init__(self, instruction, rig, out, step_limit=9000, vmax=0.5, arms=None, sim=False, port=8080, review=True,
                 turns=None):  # fmt: skip
        self.out, self.instruction, self.step_limit = Path(out), str(instruction), int(step_limit)
        self.review = bool(review)
        self.out.mkdir(parents=True, exist_ok=True)
        self.rig, self.sim = load_rig(rig), bool(sim)
        self.dt, self.table_z = 1.0 / float(self.rig.get("rate", 50)), float(self.rig.get("table_z", 0.0))
        self.hand_speed = float(self.rig.get("hand_speed", self.HAND_SPEED))
        self.review_under = float(self.rig.get("review_under", REVIEW_UNDER))
        names = list(self.rig["arms"]) if arms is None else [a for a in str(arms).split(",") if a]
        self.operator = Operator(port, f"{self.NAME}: {self.instruction}")
        if turns:  # the agent's folder: its turns on the panel
            self.operator.watch(turns)
        self.steps, self.ended, self.actions, self.faults, self.settled_at = (
            0,
            False,
            [],
            self.out / "faults.jsonl",
            0.0,
        )
        self.hw, self.bodies, self.arms, self.cameras, self.queue = {}, {}, {}, {}, None  # cameras: {name: Stream}
        self.motion = threading.RLock()  # held while anything streams to the arms: one motion at a time
        try:
            cams = {n: c for n, c in (self.rig.get("cameras") or {}).items() if c.get("arm", names[0]) in names}
            if cams:  # before the arms: starting a RealSense stalls every thread past a motor's timeout
                from roboskills.eval.real.cameras import Stream

                for n, c in cams.items():  # one that fails closes those already open (close, below)
                    self.cameras[n] = Stream(c["serial"], c.get("width", 640), c.get("height", 480), c.get("fps", 30),
                                             c.get("depth_scale", 1.0))  # fmt: skip
            self.cam_specs = cams
            self._arms(names, float(vmax))
            self.tcp = {a: arm.hand.T_ee_tcp for a, arm in self.arms.items()}
            self._recorder()
            for a, body in self.bodies.items():
                self.operator.add_arm(a, body.meshes())
            self._show()
        except BaseException:
            self.close(release=False)
            raise
        try:
            text = f"Connected and holding. Reset the scene, then press Start (Rest: to the rig's rest).\n\n{self.instruction}"
            while self.operator.first(text, dict(start=self.operator.start, rest=self.operator.rest)) == "rest":
                self._rest()
        except BaseException:  # Ctrl+C or SIGTERM before Start: park, then turn off
            self.close()
            raise
        self.operator.say(f"Running: {self.instruction}")
        self.queue = Queue(self)

    def _arms(self, names, vmax):
        """Connect the arms `names` of the rig: set `hw` {arm: driver}, `bodies` {arm: safety.Body}, `arms` {arm:
        interface.Arm} and `limits` (safety.Limits)."""
        raise NotImplementedError

    def _rest_of(self, a):
        """Where arm `a` parks when the rig gives no `rest`: all zeros (folded)."""
        return np.zeros(self.hw[a].n)

    def _T_world_cam(self, spec):
        """A fixed camera's pose: `T_world_cam`, or `T_base_cam` on the base of the arm `base` names."""
        if "T_world_cam" in spec:
            return np.asarray(spec["T_world_cam"], float)
        return self.arms[spec["base"]].T_world_base @ np.asarray(spec["T_base_cam"], float)

    def _split(self, v, a):
        n = self.hw[a].n
        return v[:n], v[n:]

    def _show(self):
        """The panel's 3D view: the arms where they stand and every fixed camera's newest depth cloud."""
        now = self._read()
        self.operator.pose_arms("live", {a: self.bodies[a].poses(*self._split(v, a)) for a, v in now.items()})
        for name, spec in self.cam_specs.items():
            frame = None if "arm" in spec or name not in self.cameras else self.cameras[name].peek()
            if frame is not None:
                self.operator.show_cloud(name, frame[0], frame[1], self.cameras[name].K, self._T_world_cam(spec))

    def _rest(self):
        """The operator's Rest: the hands open where they are (what they hold drops), then the arms go to the rig's
        `rest` on checked paths at half speed (motors stay on)."""
        self.operator.say("Opening the hands, then moving to rest")
        with self.motion:
            why = safety.park(self.hw, self.limits, self.bodies, self.tcp, self._rests(),
                              {a: self.arms[a].home for a in self.hw}, self.dt,
                              opened={a: self.arms[a].hand.open for a in self.hw}, hand_speed=self.hand_speed)  # fmt: skip
        if why:
            self._fault("rest", why)
        self.settled_at = time.monotonic()
        self._show()
        return why

    def _rests(self):
        return {a: np.asarray(self.rig["arms"][a].get("rest", self._rest_of(a)), float) for a in self.hw}

    def _closure(self, a, u):
        """How closed hand command `u` is, 0 (open) .. 1 (closed), averaged over its values."""
        h = self.arms[a].hand
        span = np.asarray(h.closed, float) - np.asarray(h.open, float)
        k = np.abs(span) > 1e-9
        return float(np.clip((np.asarray(u, float)[k] - np.asarray(h.open, float)[k]) / span[k], 0, 1).mean()) if k.any() else 0.0  # fmt: skip

    def _review(self, paths, start, label="", cancel=None):
        """Show `paths` (one plan's trajectories, in order) from `start` {arm: joints and hand} to the operator and
        wait: "approve", "reject", "rest", "stop" or, when `cancel` is set first, "cancel"."""
        qs, us, cur, duration, contact = {}, {}, {a: v.copy() for a, v in start.items()}, 0.0, 0
        for a, v in start.items():
            qs[a], us[a] = [self._split(v, a)[0]], [self._split(v, a)[1]]
        for path in paths:  # as `_stream` runs them: a hand change's waypoints all hold its end, from the hand before
            n = len(path.t)
            duration += max(
                self.limits.retime(path)[-1], self._hand_s(path, {a: self._split(v, a)[1] for a, v in cur.items()})
            )
            contact += 0 if path.contact is None else int(np.sum(path.contact))
            for a, v in cur.items():
                q0, u0 = self._split(v, a)
                Q = np.asarray(path.q[a], float) if a in path.q else np.tile(q0, (n, 1))
                U = np.asarray(path.hand[a], float) if a in path.hand else np.tile(u0, (n, 1))
                qs[a] += list(Q)
                us[a] += list(U)
                cur[a] = np.r_[Q[-1], U[-1]]
        m = len(qs[next(iter(qs))])
        frames = [{a: self.bodies[a].poses(qs[a][i], us[a][i]) for a in start}
                  for i in np.unique(np.linspace(0, m - 1, min(m, REVIEW_FRAMES)).round().astype(int))]  # fmt: skip
        moves = []
        for a in sorted({a for p in paths for a in (*p.q, *p.hand)}):
            (q0, u0), (q1, u1) = self._split(start[a], a), self._split(cur[a], a)
            p0, p1 = (self.bodies[a].ee_pose(q) @ self.tcp[a] for q in (q0, q1))
            moves.append(f"{a}: joints up to {np.degrees(np.abs(np.asarray(qs[a]) - q0).max()):.0f} deg, TCP "
                         f"{100 * np.linalg.norm(p1[:3, 3] - p0[:3, 3]):.1f} cm to z {100 * p1[2, 3]:.1f} cm, "
                         f"hand {100 * self._closure(a, u0):.0f}% -> {100 * self._closure(a, u1):.0f}% closed")  # fmt: skip
        kind = (
            "**Hand only** (the arms hold still)"
            if not any(p.q for p in paths)
            else f"**Plan to review**{f' ({label})' if label else ''}"
        )
        text = (f"{kind}: {duration:.1f} s in {len(paths)} part{'s' * (len(paths) > 1)}"
                f"{f' ({contact} waypoints in contact)' if contact else ''}; {'; '.join(moves)}. Orange: the "
                "plan from where it starts (scrub or Play); grey: where the arms stand. Execute runs it; Reject refuses it "
                "(and everything queued after it); Rest moves to rest instead.")  # fmt: skip
        for path in paths:
            for a, T in path.T.items():
                self.operator.show_path(a, (np.asarray(T) @ self.tcp[a])[:, :3, 3])
        self._show()
        return self.operator.review(text, frames, duration, cancel=cancel)

    # ---- observation

    def _read(self):
        return {a: hw.read() for a, hw in self.hw.items()}

    def _stopped(self):
        """Stop (or a verdict) pressed while no motion ran: the episode ends now, not at the next motion."""
        if self.operator.stop.is_set() and not self.ended:
            self._fault("operator", "stop or a verdict pressed: the episode ends")
            self.ended = True
        return self.ended

    def obs(self, cams=True) -> Obs:
        self._stopped()
        cameras = {}
        for name, s in self.cameras.items() if cams else ():
            spec = self.cam_specs[name]
            if "arm" in spec:  # on a wrist: where the arm stands now
                a = spec["arm"]
                T = self.bodies[a].ee_pose(self.hw[a].read()[: self.hw[a].n]) @ np.asarray(spec["T_ee_cam"], float)
            else:
                T = self._T_world_cam(spec)
            cameras[name] = Camera(*s.grab(self.settled_at), s.K, T)
        now = self._read()
        return Obs(cameras, {a: self._split(v, a)[0] for a, v in now.items()},
                   {a: self._split(v, a)[1] for a, v in now.items()}, self.steps, self.step_limit, self.instruction)  # fmt: skip

    def robot_mask(self, cam, obs):
        return None  # the skills mask the arms from the model and the calibration

    # ---- motion

    def _fault(self, kind, message):
        with open(self.faults, "a") as f:
            f.write(json.dumps(dict(time=time.time(), steps=self.steps, kind=kind, message=message)) + "\n")
        self.operator.say(f"{kind}: {message}")

    def _hold(self):
        for h in self.hw.values():
            h.hold()

    def follow(self, path: Trajectory, stop=None) -> Follow:
        if self._stopped():
            return Follow(0, True, False, self.obs(False))
        if self.queue is not None and self.queue.busy():
            why = "plans are queued: wait for them (bot.queued(wait=...)) or bot.flush() them first"
            return Follow(0, False, True, self.obs(False), refused=why)
        if self._still(path):
            return Follow(0, self.steps >= self.step_limit, False, self.obs(False))
        if self.review and not self._small([path], self._read()):
            refused = self._check(path)  # a path that cannot run is refused before the operator sees it
            if refused:
                return Follow(0, self.steps >= self.step_limit, True, self.obs(False), refused=refused)
            answer = self._review([path], self._read())
            if answer == "stop":
                self._fault("operator", "stop pressed during review: the episode ends")
                self.ended = True
                return Follow(0, True, True, self.obs(False))
            if answer == "rest":
                why = self._rest()
                return Follow(0, False, False, self.obs(False), refused="the person at the bench moved the arms to "
                              f"rest instead{f' (stopped short: {why})' if why else ''}: observe and plan again")  # fmt: skip
            if answer == "reject":
                self._fault("rejected", "the operator rejected the plan")
                return Follow(0, False, False, self.obs(False), refused="the person at the bench rejected this plan")
        with self.motion:  # checked again where the arms are now (they drift during a review; a Rest may have run)
            why = self._check(path)
            if why:
                return Follow(0, self.steps >= self.step_limit, True, self.obs(False), refused=why)
            return self._stream(path, stop)

    def _run(self, path):
        """The queue's `follow`: checked against the measured state, not reviewed (its plan was approved whole)."""
        if self._stopped():
            return Follow(0, True, False, self.obs(False))
        if self._still(path):
            return Follow(0, self.steps >= self.step_limit, False, self.obs(False))
        with self.motion:
            refused = self._check(path)
            if refused:
                return Follow(0, self.steps >= self.step_limit, True, self.obs(False), refused=refused)
            return self._stream(path, None)

    def _small(self, paths, start):
        """`paths` from `start` {arm: joints and hand} keep every TCP within `review_under` m of where it starts, hands
        unchanged."""
        for path in paths:
            for a, Q in path.q.items():
                p0 = (self.bodies[a].ee_pose(self._split(start[a], a)[0]) @ self.tcp[a])[:3, 3]
                if any(np.linalg.norm((self.bodies[a].ee_pose(q) @ self.tcp[a])[:3, 3] - p0) > self.review_under
                       for q in np.asarray(Q, float)):  # fmt: skip
                    return False
            if any(np.abs(np.asarray(U, float) - self._split(start[a], a)[1]).max() > STILL_U for a, U in path.hand.items()):  # fmt: skip
                return False
        return True

    def _still(self, path):
        """Nothing in `path` moves: each arm stays at its target and each hand command where it is (a planner's hold
        around a hand change whose hand is already there): it is skipped, so the operator never reviews a no-op."""
        for a, Q in path.q.items():
            if np.abs(np.asarray(Q, float) - self._split(self.hw[a].target, a)[0]).max() > STILL_Q:
                return False
        return all(np.abs(np.asarray(U, float) - self._split(self.hw[a].target, a)[1]).max() <= STILL_U
                   for a, U in path.hand.items())  # fmt: skip

    def _check(self, path):
        """Why `path` must not run from where the arms are measured now (None: it may), recorded as a fault."""
        now = self._read()
        why = self.limits.check(path, {a: self._split(v, a)[0] for a, v in now.items()},
                                {a: self._split(v, a)[1] for a, v in now.items()}, self.bodies, self.tcp)  # fmt: skip
        if why:
            self._fault("refused", why)
        return why

    def _stream(self, path, stop):
        """Stream `path` at the command rate from the arms' targets, slowed to vmax, the hands at hand_speed."""
        for a, T in path.T.items():
            self.operator.show_path(a, (np.asarray(T) @ self.tcp[a])[:, :3, 3])
        t = np.r_[0.0, self.limits.retime(path)]
        contact = np.r_[False, np.zeros(len(path.t), bool) if path.contact is None else np.asarray(path.contact, bool)]
        start = {a: hw.target.copy() for a, hw in self.hw.items()}
        rows = {a: np.vstack([self._split(start[a], a)[0], np.asarray(Q, float)]) for a, Q in path.q.items()}
        hands = {a: np.vstack([self._split(start[a], a)[1], np.asarray(U, float)]) for a, U in path.hand.items() if a in self.hw}  # fmt: skip
        end, settle = int(np.ceil(t[-1] / self.dt - 1e-9)), int(round(SETTLE / self.dt))
        hand_step = self.hand_speed * self.dt  # a hand command moves at most this per tick: it never snaps shut
        lag = int(np.ceil(self._hand_s(path, {a: self._split(v, a)[1] for a, v in start.items()}) / self.dt))
        before, stopped, t0 = self.steps, False, time.monotonic()
        try:
            for k in range(1, end + settle + lag + 1):
                if self.steps >= self.step_limit:
                    self.ended = True
                    break
                s = min(k * self.dt, t[-1])
                touching = bool(contact[min(np.searchsorted(t, s), len(t) - 1)])
                for a, hw in self.hw.items():
                    q0, u0 = self._split(hw.target, a)
                    q = safety.interp(s, t, rows[a]) if a in rows else q0
                    u = u0 + np.clip(safety.interp(s, t, hands[a]) - u0, -hand_step, hand_step) if a in hands else u0
                    hw.send(q, u, contact=touching and a in rows)
                self.actions.append(np.concatenate([hw.target for hw in self.hw.values()]))
                self.steps += 1
                late = time.monotonic() - (t0 + k * self.dt)
                if late < 0:
                    time.sleep(-late)
                elif late > LATE * self.dt:
                    self._fault("late", f"command tick {k} ran {late * 1000:.0f} ms late")
                    stopped = True
                    break
                meas = self._read()
                why = next((f"{a}: {r}" for a in rows if (r := self.limits.tracking(self._split(self.hw[a].target, a)[0],
                            self._split(meas[a], a)[0], touching))), None)  # fmt: skip
                why = why or next((f"{a}: the driver stopped" for a, hw in self.hw.items() if not hw.alive()), None)
                if why:
                    self._fault("guard", why)
                    stopped = True
                    break
                if self.operator.stop.is_set():
                    self._fault("operator", "stop pressed: the episode ends")
                    stopped = self.ended = True
                    break
                hands_done = all(
                    np.abs(self._split(self.hw[a].target, a)[1] - U[-1]).max() < 1e-9 for a, U in hands.items()
                )
                if (
                    k > end
                    and hands_done
                    and all(
                        np.abs(self._split(meas[a], a)[0] - self._split(self.hw[a].target, a)[0]).max() < TOL
                        for a in rows
                    )
                ):
                    break
                if stop is not None and k % STOP_EVERY == 0 and stop(self.obs(False)):
                    stopped = True
                    break
        except BaseException:
            self._hold()
            raise
        if stopped:
            self._hold()
        self.settled_at = time.monotonic()
        self._show()
        self.ended = self.ended or self.steps >= self.step_limit
        return Follow(self.steps - before, self.steps >= self.step_limit, stopped, self.obs(False))

    def _hand_s(self, path, u_from):
        """Seconds the hands' commands take along `path` from `u_from` {arm: command} at `hand_speed` (the largest
        travel of any one value)."""
        travel = []
        for a, U in path.hand.items():
            if a in u_from:
                U = np.asarray(U, float)
                travel.append(float((np.abs(U[0] - u_from[a]) + np.abs(np.diff(U, axis=0)).sum(0)).max()))
        return max(travel, default=0.0) / self.hand_speed

    # ---- the queue (eval/real/queued.py), served to the agent

    def enqueue(self, paths, holding=None, label=""):
        """Queue one plan's trajectories to run after what is queued (`holding`: arms whose hands must hold something
        first); returns its id."""
        return self.queue.add(paths, holding, label)

    def queued(self, wait=0.0):
        """The queue: {entries: [{id, label, status, result}], end: {q, hand}, running}; `wait` s for a change first."""
        return self.queue.view(wait)

    def flush(self):
        """Drop every queued plan not yet running; returns their ids."""
        return self.queue.flush("dropped by the agent")

    def _move(self, targets, speed):
        """Stream every arm to `targets` {arm: (n,)} along a straight joint line at `speed` rad/s, outside the budget,
        if the path passes the checks; returns why not (None: moved)."""
        with self.motion:
            now = {a: self._split(v, a)[0] for a, v in self._read().items()}
            far = max(np.abs(np.asarray(targets[a]) - now[a]).max() for a in targets)
            n = max(2, int(np.ceil(far / speed / self.dt)))
            s = np.linspace(0, 1, n + 1)[1:, None]
            line = Trajectory(t=s[:, 0] * n * self.dt, hand={}, T={},
                              q={a: now[a] + s * (np.asarray(targets[a]) - now[a]) for a in targets})  # fmt: skip
            why = safety.run_path(self.hw, self.limits, self.bodies, self.tcp, line, self.dt)
        if why:
            self._fault("refused", why)
        return why

    # ---- recording and the end

    def _recorder(self):
        """A video_<camera>.mp4 for each `record` camera (one name or a list; default the first fixed camera)."""
        names = self.rig.get("record") or next((n for n, c in self.cam_specs.items() if "arm" not in c), None)
        names = [names] if isinstance(names, str) else list(names or [])
        self.videos, self._rec = {n: self.out / f"video_{n}.mp4" for n in names if n in self.cam_specs}, None
        if not self.videos or not self.cameras:
            return
        import imageio.v2 as imageio

        stop = threading.Event()

        def run(cam, path):
            writer = None
            while not stop.wait(1.0 / VIDEO_FPS):
                frame = self.cameras[cam].peek()
                if frame is None:
                    continue
                if writer is None:  # H.264, which browsers and default players open (OpenCV's mp4v they do not)
                    writer = imageio.get_writer(path, fps=VIDEO_FPS, codec="libx264", macro_block_size=2)
                writer.append_data(np.ascontiguousarray(frame[0][::2, ::2]))  # half size
            if writer is not None:
                writer.close()

        threads = [
            threading.Thread(target=run, args=kv, daemon=True, name=f"recorder-{kv[0]}") for kv in self.videos.items()
        ]
        for thread in threads:
            thread.start()
        self._rec = (stop, threads)

    def replay(self, actions):
        """Stream recorded `actions.npy` rows (each arm's joints and hand, in arm order) at `dt`, then hold."""
        t0 = time.monotonic()
        for k, row in enumerate(np.asarray(actions, float)):
            i = 0
            for a, hw in self.hw.items():
                w = len(hw.target)
                hw.send(*self._split(row[i : i + w], a))
                i += w
            self.actions.append(row)
            self.steps += 1
            time.sleep(max(0.0, t0 + (k + 1) * self.dt - time.monotonic()))
        return self.obs(False)

    def finish(self, out) -> dict:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        if self.queue is not None:  # what waits is dropped; the running plan ends first
            self.queue.stop()
        self._hold()
        self.ended = True
        text = (
            "The episode is over: give the verdict (and a note), then the result is saved. Rest: fold the arms first."
        )
        while self.operator.first(text, dict(judged=self.operator.judged, rest=self.operator.rest)) == "rest":
            self._rest()
        verdict, note = self.operator.verdict, self.operator.note_text
        homed = self._move({a: arm.home for a, arm in self.arms.items()}, self.limits.vmax / 2)
        if self._rec:
            self._rec[0].set()
            for thread in self._rec[1]:
                thread.join(timeout=10)
            self._rec = None
            for path in self.videos.values():
                if path.exists():
                    shutil.copyfile(path, out / path.name)
        width = sum(len(hw.target) for hw in self.hw.values())
        np.save(out / "actions.npy", np.asarray(self.actions, float).reshape(len(self.actions), width))
        shutil.copyfile(self.rig["_path"], out / "rig.yaml")
        if self.faults.exists():
            shutil.copyfile(self.faults, out / "faults.jsonl")
        result = dict(success=verdict == "success", verdict=verdict, note=note, instruction=self.instruction,
                      steps=self.steps, step_limit=self.step_limit, dt=self.dt, arms=list(self.arms), sim=self.sim,
                      homed=homed is None, home_refused=homed)  # fmt: skip
        if verdict == "void":
            result["void"] = f"operator: {note or 'voided'}"
        (out / "native_result.json").write_text(json.dumps(result, indent=2))
        self.operator.say(
            f"Saved: {verdict}. When the run closes, the hands open, the arms go to rest and the drivers let go."
        )
        return result

    def close(self, release=True):
        """Open the hands where they are, park the arms at `rest` (`safety.park`), then close the drivers. If they
        cannot be parked (a refused path, an interrupt), hold them until the operator, supporting them, presses
        Release. `release=False`: close at once (a failed connect, where nothing has moved)."""
        if self.queue is not None:
            self.queue.stop()
        if self.hw and release:
            why = safety.park(self.hw, self.limits, self.bodies, self.tcp, self._rests(),
                              {a: self.arms[a].home for a in self.hw}, self.dt,
                              opened={a: self.arms[a].hand.open for a in self.hw}, hand_speed=self.hand_speed)  # fmt: skip
            if why:
                self._fault("park", why)
            while why:
                try:
                    self.operator.released(
                        f"Holding: {why}. Support the arms, then press Release to let the drivers go."
                    )
                    why = None
                except KeyboardInterrupt:  # only Release (or the e-stop) lets go
                    self._hold()
        if getattr(self, "_rec", None):
            self._rec[0].set()
        for s in self.cameras.values():
            s.close()
        self.cameras = {}
        for hw in self.hw.values():
            hw.close()
        self.hw = {}
        self.operator.close()
