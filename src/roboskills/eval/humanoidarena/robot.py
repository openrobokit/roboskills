"""HumanoidArena behind `roboskills.eval.interface.Robot`: the Unitree G1 (29 DoF) with Dex3 hands under the frozen
GEAR-SONIC controller, walking, in one official test episode served by `sim/server.py` (launch.py starts it).

Per 50 Hz control step the simulator takes each arm's 7 joint targets, each hand's binary command, and while the robot
walks a world [x, y, yaw] target for its pelvis (the gait steps toward it with feedback from the measured pelvis), or a
pelvis height to squat to. The arms hang off `torso_link`, which moves with walking, the waist and the pelvis height:
it is every arm model's root and `Arm.T_world_base` is its measured world pose, read after every reply (one array,
shared by both arms, updated in place). What reaches the agent: the front camera's rendered RGB-D with intrinsics and
pose and where it sees the robot (`robot_mask`), the measured arm, hand and pelvis state, the robot's geometry, the step
count and the instruction. `finish` alone reads the success check.
"""

import json
import math
import sys
import threading
from pathlib import Path

import numpy as np

from roboskills.eval.humanoidarena import TASKS, episode_seed
from roboskills.eval.humanoidarena import vendor as vendor_dir
from roboskills.eval.humanoidarena.hand import SIDES, URDF, T_ee_tcp, closed, model, press_point
from roboskills.eval.interface import Arm, Base, Body, Camera, Follow, Hand, Obs
from roboskills.eval.wire import Sim


DT = 0.02  # one budget step: one 50 Hz control step (SONIC's rate; 4 physics steps at 200 Hz)
VMAX = 1.0  # rad/s, arm joint speed for retiming
HAND_STEPS = 20  # steps a hand command is held: the Dex3 closes in about 0.3 s
BLOCKED = 0.35  # rad: an arm this far behind its commanded path is held back by something (stop the motion)
BASE_LEAD = 0.6  # s: walk targets lead the path by this much (the gait's lag behind a moving target)
BASE_TOL = (0.05, math.radians(5))  # m, rad: the pelvis has arrived at its last waypoint (settling aims at this)
REACHED = (0.08, math.radians(5))  # m, rad: a walk ending this close has reached its goal (stepping lands 2-10 cm off)
BASE_SETTLE = 300  # steps at most spent settling the pelvis on its last waypoint
SETTLE_STALL = 100  # steps settling may go without coming 1 cm closer
STUCK = (0.03, math.radians(3), 100)  # m, rad, steps: moved less than this over that many steps, short of the target
START = 100  # steps a walk may take to get going before it counts as stuck
BODY_TOL = 0.03  # m: the pelvis height has arrived
BODY_ITERS, BODY_WAIT, BODY_CMD_LO = 3, 25, 0.35  # pelvis-height corrections, steps each waits, lowest height commanded
HOLD = 10  # steps `finish` holds still before the verdict
CHUNK = 10  # steps per request


class HumanoidArena:
    def __init__(self, port, task, seed, out, frame_every=25, vendor=None):
        if task not in TASKS:
            raise ValueError(f"unknown task {task!r}; the cohort is {tuple(TASKS)}")
        self.task, self.seed, self.frame_every = task, str(seed), int(frame_every)
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.sim, self.lock = Sim(port), threading.Lock()  # the launcher calls from several threads, one at a time
        self.requests = []  # every act and observe sent, in order: what `replay` sends again (requests.jsonl)
        first = self._req(op="reset")
        facts = first["facts"]
        self.step_limit, self.steps, self.ended, self.dt = int(first["step_limit"]), 0, False, DT
        self.instruction = TASKS[task][4]
        self.T_mount = np.asarray(first["mount"], float).copy()  # both arms' T_world_base: one array, updated in place
        path = model(str(vendor_dir(vendor) / URDF), self.out / "g1_arms.urdf")
        self.arms = {}
        for s in SIDES:
            joints, hj = list(facts["arm_joints"][s]), list(facts["hand_joints"][s])
            lo = np.array([facts["lo"][j] for j in joints])
            hi = np.array([facts["hi"][j] for j in joints])
            opened, shut = np.asarray(facts["hand_open"][s], float), np.asarray(facts["hand_closed"][s], float)
            hand = Hand("dex", T_ee_tcp(s), open=opened, closed=shut, joints=hj, press="closed",
                        press_point=press_point(path, s, shut, hj))  # fmt: skip
            self.arms[s] = Arm(s, path, joints, f"{s}_hand_palm_link", self.T_mount, lo, hi,
                               np.asarray(first["q"][s], float), hand, vmax=VMAX)  # fmt: skip
        self.stand = float(first["body"])
        self.base = Base(vmax=0.35, wmax=0.6, radius=0.3, tol=REACHED[0], tol_yaw=REACHED[1])
        self.body = Body(lo=0.5, hi=round(self.stand, 3), vmax=0.15, tol=0.04)
        self.frames, self.actions = [], []
        self._take(first)
        self.cmd_q = {s: np.asarray(self.state["q"][s], float) for s in SIDES}
        self.cmd_u = {s: 0 for s in SIDES}
        self.cmd_body = None  # None: standing
        (self.out / "robot.json").write_text(json.dumps(dict(task=task, seed=self.seed, episode_seed=episode_seed(task, seed),
                                                             instruction=self.instruction, step_limit=self.step_limit,
                                                             urdf=path, stand=self.stand,
                                                             press_point={s: self.arms[s].hand.press_point.tolist() for s in SIDES}),
                                                        indent=1))  # fmt: skip

    # ---- the simulator

    def _req(self, **msg):
        with self.lock:
            if msg["op"] in ("act", "observe"):  # what moves the episode, and the renders in between
                self.requests.append(msg)
            return self.sim.request(**msg)

    def _take(self, rep):
        """Keep the latest state (and camera); move the arms' mount with the torso."""
        self.state = {k: rep[k] for k in ("q", "hand", "base", "body", "tilt", "stopped", "walking", "done")}
        self.state["base"] = np.asarray(rep["base"], float)
        self.T_mount[:] = np.asarray(rep["mount"], float)
        if "cams" in rep:
            self.cams = rep["cams"]

    def obs(self, cams=True):
        cs = {}
        if cams and not self.ended:
            self._take(self._req(op="observe"))
        if cams:
            cs = {n: Camera(np.array(v["rgb"]), np.where(v["depth"] > 0, v["depth"], np.nan).astype(np.float32),
                            np.asarray(v["K"], float), np.asarray(v["T"], float)) for n, v in self.cams.items()}  # fmt: skip
            self.masks = {n: (cs[n].rgb, np.asarray(v["robot"], bool)) for n, v in self.cams.items() if "robot" in v}
        return Obs(cs, {s: np.asarray(self.state["q"][s], float) for s in SIDES},
                   {s: np.asarray(self.state["hand"][s], float) for s in SIDES}, min(self.steps, self.step_limit),
                   self.step_limit, self.instruction, base=self.state["base"].copy(), body=float(self.state["body"]))  # fmt: skip

    def robot_mask(self, cam, obs):
        """Where camera `cam` sees the robot in `obs`: the simulator's instance segmentation of that very frame, grown 2
        pixels; None for an observation other than the latest one."""
        rgb, mask = getattr(self, "masks", {}).get(cam, (None, None))
        return mask if cam in obs.cams and obs.cams[cam].rgb is rgb else None

    # ---- acting

    def _row(self, base=None, body=None, drive=False):
        """One control step's command: the arms' and hands' last commands, the pelvis height held, and the walk target
        while the robot walks (without one, the gait stands where it is)."""
        row = {"q": {s: self.cmd_q[s].tolist() for s in SIDES}, "u": dict(self.cmd_u)}
        body = self.cmd_body if body is None else body
        if body is not None:
            row["body"] = float(body)
        if base is not None:
            row["base"] = [float(v) for v in base]
            if drive:
                row["drive"] = True
        return row

    def _send(self, rows):
        """Send rows (one per control step). After the episode's end nothing runs, but the steps count."""
        rows = rows[: max(0, self.step_limit - self.steps)]
        self.actions += [flat(r) for r in rows]
        self.steps += len(rows)
        if rows and not self.ended:
            self._take(self._req(op="act", rows=rows, render=False))
            self.ended = bool(self.state["done"])
        self.ended = self.ended or self.steps >= self.step_limit

    def replay(self, oracle):
        """Send a finished episode again: its requests (`requests.jsonl`), in order, the renders between its rows
        included; or its rows alone (`actions.npy`), CHUNK per request. The arms, hands and pelvis height then hold the
        last row's commands. Proprioception after."""
        if isinstance(oracle, np.ndarray):
            rows = [unflat(a) for a in oracle]
            oracle = [dict(op="act", rows=rows[i : i + CHUNK], render=False) for i in range(0, len(rows), CHUNK)]
        for msg in oracle:
            if self.ended:
                break
            if msg["op"] == "observe":
                self._take(self._req(op="observe"))
                continue
            rows = msg["rows"][: max(0, self.step_limit - self.steps)]
            if not rows:
                break
            self.actions += [flat(r) for r in rows]
            self.steps += len(rows)
            self._take(self._req(op="act", rows=rows, render=False))
            self.ended = bool(self.state["done"]) or self.steps >= self.step_limit
            last = rows[-1]
            self.cmd_q = {s: np.asarray(last["q"][s], float) for s in SIDES}
            self.cmd_u = {s: int(last["u"][s]) for s in SIDES}
            self.cmd_body = last.get("body")
        return self.obs(cams=False)

    def follow(self, path, stop=None):
        n0 = self.steps
        try:
            return self._follow(path, stop)
        finally:  # where the steps go: one line per motion, with the skill calls that asked for it
            f, who = sys._getframe(1), []
            while f is not None and len(who) < 4:
                file, name = f.f_code.co_filename, f.f_code.co_name
                if ("/agent_as_policy/" in file or file.endswith("/tools/robokit/bot.py")) and not name.startswith(
                    "_exec"
                ):
                    who.append(name)
                f = f.f_back
            print(f"steps {self.steps - n0} at {n0} by {'<'.join(who)}", flush=True)

    @staticmethod
    def _base_at(start, t, tt, B):
        """The base target at `t` s on the path from `start` (at 0 s) through rows B at times tt; the path's yaw is
        unwrapped and shifted by whole turns to start nearest `start`'s."""
        B = np.array(B, float)
        B[:, 2] = np.unwrap(B[:, 2])
        B[:, 2] += 2 * math.pi * round((start[2] - B[0, 2]) / (2 * math.pi))
        tt, B = np.r_[0.0, tt], np.vstack([np.asarray(start, float)[None], B])
        return np.array([np.interp(t, tt, B[:, j]) for j in range(3)])

    def _follow(self, path, stop=None):
        n0 = self.steps
        t = np.asarray(path.t, float)
        tt = np.arange(DT, t[-1] + 1e-9, DT)
        if len(tt) == 0 or tt[-1] < t[-1] - 1e-6:
            tt = np.r_[tt, t[-1]]
        B = None if path.base is None else np.asarray(path.base, float).reshape(len(t), 3)
        H = None if path.body is None else np.asarray(path.body, float).reshape(len(t))
        base0 = self.state["base"].copy()
        body0 = self.state["body"] if self.cmd_body is None else self.cmd_body
        u0 = dict(self.cmd_u)
        rows, qs = [], []
        for i, ti in enumerate(tt):
            for s in SIDES:
                if s in path.q:
                    Q = np.asarray(path.q[s], float)
                    q = np.array([np.interp(ti, np.r_[0.0, t], np.r_[Q[0, j], Q[:, j]]) for j in range(Q.shape[1])])
                    self.cmd_q[s] = q
                if s in path.hand:
                    k = min(int(np.searchsorted(t, ti - 1e-9)), len(t) - 1)
                    hand = self.arms[s].hand
                    self.cmd_u[s] = closed(path.hand[s][k], hand.open, hand.closed)
            body = None if H is None else float(np.interp(ti, np.r_[0.0, t], np.r_[body0, H]))
            base = None if B is None else self._base_at(base0, min(ti + BASE_LEAD, t[-1]), t, B)
            rows.append(self._row(base, body, drive=i == 0))
            qs.append({a: self.cmd_q[a].copy() for a in SIDES})
        if H is not None:
            self.cmd_body = None if H[-1] >= self.body.hi - 0.02 else float(H[-1])
        if B is not None:  # walking stands the robot up
            self.cmd_body = None
        still = all(np.ptp(np.asarray(Q, float), axis=0).max() < 1e-4 for Q in path.q.values())  # (a hand alone)
        stopped = blocked = False
        history = [self.state["base"].copy()]
        for i in range(0, len(rows), CHUNK):
            self._send(rows[i : i + CHUNK])
            if self.steps >= self.step_limit or self.ended:
                break
            last = min(i + CHUNK, len(rows)) - 1
            lag = max([np.abs(np.asarray(self.state["q"][a], float) - qs[last][a]).max() for a in path.q] or [0.0])
            history.append(self.state["base"].copy())
            if lag > BLOCKED and i + CHUNK < len(rows):
                blocked = True
                print(f"adapter: arm blocked ({lag:.2f} rad behind its path); stopped the motion there", flush=True)
            elif B is not None and self.state.get("stopped"):
                blocked = True
                why = "touched something" if self.state["stopped"] == "contact" else "began to tip"
                print(f"adapter: the robot {why} while walking; stopped the walk there", flush=True)
            elif B is not None and i >= START and self._stuck(history, rows[last]["base"]):
                blocked = True
                print("adapter: the walk made no progress; stopped it there", flush=True)
            if not blocked and stop is not None and i + CHUNK < len(rows) and stop(self.obs(cams=False)):
                stopped = True
            if stopped or blocked:
                break
        if stopped or blocked:  # hold the posture reached; the gait stands where it is
            for s in path.q:
                self.cmd_q[s] = np.asarray(self.state["q"][s], float)
            if H is not None:
                h = float(self.state["body"])
                self.cmd_body = None if h >= self.body.hi - 0.02 else h
            self._send([self._row()] * 2)
        else:
            if B is not None:
                self._base_settle(B[-1])
            if H is not None:
                self._body_settle(float(H[-1]))
            if path.q and not still:
                self._settle(list(path.q))
            moved = [s for s in path.hand if self.cmd_u[s] != u0[s]]
            if moved:
                self._send([self._row()] * max(0, HAND_STEPS - len(rows)))
        return Follow(self.steps - n0, self.steps >= self.step_limit, stopped, self.obs(cams=False))

    def _stuck(self, history, target, steps=CHUNK):
        """Short of `target` and the pelvis has not moved over the last STUCK steps."""
        dx, da, n = STUCK
        back = max(1, int(math.ceil(n / steps)))
        if len(history) <= back:
            return False
        now, then, target = history[-1], history[-1 - back], np.asarray(target, float)
        short = np.hypot(*(target[:2] - now[:2])) > 0.1 or abs(wrap(target[2] - now[2])) > math.radians(10)
        moved = np.hypot(*(now[:2] - then[:2])) > dx or abs(wrap(now[2] - then[2])) > da
        return short and not moved

    def _base_settle(self, goal):
        """End on the last waypoint: keep it as the walk target until the pelvis is within BASE_TOL and the gait stands,
        the walk stops or stalls, or BASE_SETTLE steps; then stand."""
        history, best = [self.state["base"].copy()], (np.inf, 0)  # (error, steps since it last improved by 1 cm)
        for _ in range(0, BASE_SETTLE, CHUNK):
            if self.ended or self.steps >= self.step_limit:
                return
            now = self.state["base"]
            err = np.hypot(*(goal[:2] - now[:2])) + 0.3 * abs(wrap(goal[2] - now[2]))  # (2 degrees count as 1 cm)
            best = (err, 0) if err < best[0] - 0.01 else (best[0], best[1] + CHUNK)
            near = np.hypot(*(goal[:2] - now[:2])) < BASE_TOL[0] and abs(wrap(goal[2] - now[2])) < BASE_TOL[1]
            if (near and not self.state["walking"]) or best[1] >= SETTLE_STALL:
                break
            self._send([self._row(goal)] * CHUNK)
            history.append(self.state["base"].copy())
            if self.state.get("stopped") or self._stuck(history, goal, CHUNK):
                break
        self._send([self._row()] * 2)

    def _body_settle(self, goal):
        """End at the pelvis height `goal`: the squat stops a few cm high, so command lower by what is left, a few
        times, waiting for it to settle each time."""
        cmd = goal if self.cmd_body is None else self.cmd_body
        for _ in range(BODY_ITERS):
            self._send([self._row()] * BODY_WAIT)
            if self.ended or self.steps >= self.step_limit:
                return
            err = goal - float(self.state["body"])
            if abs(err) < BODY_TOL:
                return
            cmd = float(np.clip(cmd + err, BODY_CMD_LO, self.body.hi))
            self.cmd_body = None if cmd >= self.body.hi - 0.02 else cmd

    def _settle(self, arms, tol=0.03, max_steps=40, every=5):
        """End on the last waypoint: hold it while a joint is still more than `tol` rad off and closing in (the
        simulator integrates the controller's sag away while a target rests)."""
        prev = None
        for _ in range(0, max_steps, every):
            if self.ended or self.steps >= self.step_limit:
                return
            err = max(np.abs(np.asarray(self.state["q"][s], float) - self.cmd_q[s]).max() for s in arms)
            if err < tol or (prev is not None and err > prev - 0.002):
                return
            prev = err
            self._send([self._row()] * every)

    def finish(self, out):
        """End the episode: hold still a few steps, then the success check (it held at some step), and save the front
        camera's video, the actions and native_result.json in `out`."""
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        hold = 0 if self.ended else min(HOLD, max(0, self.step_limit - self.steps))
        self.actions += [flat(self._row())] * hold
        try:
            result = self._req(op="finish", hold=hold)
            self.steps += hold
        except RuntimeError as e:
            result = {"success": False, "error": str(e), "frames": np.zeros((0, 1, 1, 3), np.uint8)}
        frames = np.asarray(result.pop("frames"))
        self.sim.close()
        try:
            import imageio

            if len(frames):
                imageio.mimwrite(out / "video.mp4", list(frames), fps=max(1, round(1 / (DT * max(1, self.frame_every)))),
                                 macro_block_size=1)  # fmt: skip
        except Exception as e:  # noqa: BLE001
            print(f"video not saved: {e!r}", flush=True)
        np.save(out / "actions.npy", np.asarray(self.actions, np.float32).reshape(-1, 20))
        (out / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in self.requests))
        native = dict(task=self.task, seed=self.seed, episode_seed=episode_seed(self.task, self.seed),
                      steps=min(self.steps, self.step_limit), step_limit=self.step_limit, success=bool(result.get("success")),
                      instruction=self.instruction, dt=DT, success_step=result.get("success_step"), fallen=result.get("fallen"),
                      env_steps=result.get("steps"), error=result.get("error"), infra=self.sim.lost,
                      plans=result.get("plans"), plan_ms=result.get("plan_ms"), tick_ms=result.get("tick_ms"),
                      actions="per step: left arm q (7), left hand closed, right arm q (7), right hand closed, walk target "
                      "x, y, yaw (NaN: standing), pelvis height (NaN: standing)")  # fmt: skip
        (out / "native_result.json").write_text(json.dumps(native, indent=2))
        return native


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def flat(row):
    """A row as actions.npy stores it: left arm q (7), left hand, right arm q (7), right hand, walk target x, y, yaw,
    pelvis height."""
    return np.r_[row["q"]["left"], row["u"]["left"], row["q"]["right"], row["u"]["right"], row.get("base", [np.nan] * 3),
                 row.get("body", np.nan)]  # fmt: skip


def unflat(a):
    """The row `flat` stored as `a`. Without its `drive` flag: the server begins a walk where the row before had no walk
    target, as every walk the robot sends begins."""
    a = np.asarray(a, float)
    row = {"q": {"left": a[:7].tolist(), "right": a[8:15].tolist()}, "u": {"left": int(a[7]), "right": int(a[15])}}
    if np.isfinite(a[16:19]).all():
        row["base"] = a[16:19].tolist()
    if np.isfinite(a[19]):
        row["body"] = float(a[19])
    return row
