"""BEHAVIOR-1K behind `roboskills.eval.interface.Robot`: the R1Pro, two 7-joint arms with parallel grippers on a torso and
a holonomic base, in one challenge task instance served by `sim/server.py` (launch.py starts it).

Per 30 Hz control step the simulator takes each arm's 7 joint targets, each gripper's opening and a world [x, y, yaw]
target for the base; the trunk stays at its reset pose, so the torso link the arms hang off (`torso_link4`) is fixed on
the base and is every arm model's root: `Arm.T_world_base` is its world pose, read from the simulator after every reply
(one array, shared by both arms, updated in place). What reaches the agent: rendered RGB-D with intrinsics and camera
poses (the head and both wrist cameras) and where each sees the robot (`robot_mask`), the measured arm joints, finger
travel and base pose, the robot's geometry, the step count and the instruction. `finish` alone reads the BDDL verdict.
"""

import json
import math
import sys
import threading
from pathlib import Path

import numpy as np

from roboskills.eval.behavior import INSTANCES, TASKS
from roboskills.eval.behavior.goal import describe
from roboskills.eval.behavior.hand import T_EE_TCP, WIDTH, finger_q, model, opening
from roboskills.eval.interface import Arm, Base, Camera, Follow, Hand, Obs
from roboskills.eval.wire import Sim


DT = 1.0 / 30  # one budget step: one control step of the challenge (4 physics steps at 120 Hz)
SIDES = ("left", "right")
VMAX = 1.5  # rad/s, arm joint speed for retiming
HAND_LEAD = 0.2  # s: gripper commands lead the path (the fingers take ~6 steps to cross their travel)
BLOCKED = 0.3  # rad: an arm this far behind its commanded path is held back by something (stop the motion)
BASE_LEAD = 0.1  # s: base targets lead the path by this much (the drive's lag behind a moving target)
BASE_TOL = (0.005, math.radians(0.5))  # m, rad: the base has settled on its last waypoint
BASE_STUCK = (0.002, math.radians(0.3), 10)  # m, rad, steps: moved less than this over that many steps
TILT_RISE = 4.0  # degrees a motion may lean the base past its tilt at the start (the simulator stops a drive there too)
BASE_LAG = (0.12, math.radians(10))  # m, rad: a base this far behind its commanded path is held back by something
BASE_SETTLE = 45  # steps at most spent settling the base on its last waypoint
HOLD = 10  # steps `finish` holds still before the verdict
GRASP = 15  # steps a closing hand holds from its command: assisted grasping attaches after 0.3 s of the pads on it
PRESS = np.array([0.018, 0.0, 0.0])  # the closed fingertips, in the TCP frame


class Behavior:
    def __init__(self, sim, task, seed, out, data, container_data="/data", frame_every=10):
        if task not in TASKS:
            raise ValueError(f"unknown task {task!r}; the cohort is {tuple(TASKS)}")
        self.task, self.seed, self.instance = task, str(seed), INSTANCES[str(seed)]
        self.frame_every = int(frame_every)  # control steps per video frame (the simulator records them)
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.sim, self.lock = Sim(sim), threading.Lock()  # the launcher calls from several threads, one at a time
        self.requests = []  # every act and observe sent, in order: what `replay` sends again (requests.jsonl)
        first = self._req(op="reset")
        facts = first["facts"]
        self.step_limit, self.steps, self.ended, self.dt = int(first["step_limit"]), 0, False, DT
        self.problem = first["problem"]
        self.instruction = TASKS[task][1] or first.get("instruction") or describe(self.problem)
        urdf = Path(facts["urdf"])
        if container_data and urdf.is_relative_to(
            container_data
        ):  # the dataset's path in the simulator's container -> on this host
            urdf = Path(data) / urdf.relative_to(container_data)
        self.T_mount = np.array(first["mount"], float)  # both arms' T_world_base: one array, updated in place
        path = model(str(urdf), self.out / "r1pro_arms.urdf")
        names, lo, hi = list(facts["joints"]), np.asarray(facts["lo"], float), np.asarray(facts["hi"], float)
        self.arms = {}
        for s in SIDES:
            joints = list(facts["arm_joints"][s])
            idx = [names.index(j) for j in joints]
            hand = Hand("jaw", T_EE_TCP.copy(), open=np.ones(1), closed=np.zeros(1), joints=list(facts["finger_joints"][s]),
                        model_q=finger_q, width=WIDTH, press="closed", press_point=PRESS.copy())  # fmt: skip
            self.arms[s] = Arm(s, path, joints, f"{s}_gripper_link", self.T_mount, lo[idx], hi[idx],
                               np.asarray(first["q"][s], float), hand, vmax=VMAX)  # fmt: skip
        self.base = Base(
            vmax=0.5, wmax=0.8, radius=0.52
        )  # the footprint: 0.24 m ahead of the origin, 0.39 behind, 0.34 aside
        self.frames, self.actions = [], []
        self._take(first)
        self.cmd_q = {s: np.asarray(self.state["q"][s], float) for s in SIDES}
        self.cmd_u = {s: opening(self.state["fingers"][s]) for s in SIDES}
        self.cmd_base = self.state["base"].copy()
        (self.out / "robot.json").write_text(json.dumps(dict(task=task, instance=self.instance, seed=self.seed,
                                                             instruction=self.instruction, step_limit=self.step_limit,
                                                             urdf=str(urdf)), indent=1))  # fmt: skip

    # ---- the simulator

    def _req(self, **msg):
        with self.lock:
            if msg["op"] in ("act", "observe"):  # OmniGibson's physics also depends on the renders in between
                self.requests.append(msg)
            return self.sim.request(**msg)

    def _take(self, rep):
        """Keep the latest state (and cameras); move the arms' mount with the base."""
        self.state = {k: rep[k] for k in ("q", "fingers", "base", "mount", "done")}
        self.state["tilt"], self.state["stopped"] = float(rep.get("tilt", 0.0)), rep.get("stopped")
        self.state["base"] = np.asarray(rep["base"], float)
        self.T_mount[:] = np.asarray(rep["mount"], float)
        if "cams" in rep:
            self.cams = rep["cams"]

    def obs(self, cams=True):
        cs = {}
        if cams and not self.ended:
            self._take(self._req(op="observe"))
        if cams:
            cs = {n: Camera(np.array(v["rgb"]), np.where(np.isfinite(v["depth"]) & (v["depth"] > 0), v["depth"], np.nan).astype(np.float32),
                            np.asarray(v["K"], float), np.asarray(v["T"], float)) for n, v in self.cams.items()}  # fmt: skip
            self.masks = {n: (cs[n].rgb, np.asarray(v["robot"], bool)) for n, v in self.cams.items() if "robot" in v}
        return Obs(cs, {s: np.asarray(self.state["q"][s], float) for s in SIDES},
                   {s: opening(self.state["fingers"][s]) for s in SIDES}, min(self.steps, self.step_limit),
                   self.step_limit, self.instruction, base=self.state["base"].copy())  # fmt: skip

    def robot_mask(self, cam, obs):
        """Where camera `cam` sees the robot (base, torso, head, arms, grippers) in `obs`: the simulator's instance
        segmentation of that very frame, grown 2 pixels; None for an observation other than the latest one."""
        rgb, mask = getattr(self, "masks", {}).get(cam, (None, None))
        return mask if cam in obs.cams and obs.cams[cam].rgb is rgb else None

    # ---- acting

    def _row(self, drive=False):
        """One control step's command: the arms' and hands' last commands, and the base's target while it drives
        (without one, the simulator holds the base where it is, pushing nothing back)."""
        row = {"q": {s: self.cmd_q[s].tolist() for s in SIDES}, "u": {s: float(self.cmd_u[s][0]) for s in SIDES}}
        if drive:
            row["base"] = self.cmd_base.tolist()
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

    def replay(self, requests):
        """Send a finished episode's requests (its `requests.jsonl`) again, in order: its rows and the renders between
        them, which OmniGibson's physics also depends on (its rows alone, `actions.npy`, do not reproduce an episode).
        The arms and hands then hold the last row's commands. Proprioception after."""
        for msg in requests:
            if self.ended:
                break
            if msg["op"] == "observe":
                self._take(self._req(op="observe"))
                continue
            self.actions += [flat(r) for r in msg["rows"]]
            self.steps += len(msg["rows"])
            self._take(self._req(**msg))
            self.ended = bool(self.state["done"]) or self.steps >= self.step_limit
            last = msg["rows"][-1]
            self.cmd_q = {s: np.asarray(last["q"][s], float) for s in SIDES}
            self.cmd_u = {s: np.array([last["u"][s]]) for s in SIDES}
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
        moved = [
            s
            for s in path.hand
            if abs(float(np.asarray(path.hand[s][-1], float).reshape(-1)[0] - self.cmd_u[s][0])) > 0.02
        ]
        still = all(np.ptp(np.asarray(Q, float), axis=0).max() < 1e-4 for Q in path.q.values())
        if stop is None and moved and still and path.base is None:  # the hand alone: command its end, wait for it
            u_old = {s: self.cmd_u[s].copy() for s in moved}
            for s in moved:
                self.cmd_u[s] = np.asarray(path.hand[s][-1], float).reshape(-1)[:1]
            self._send([self._row()] * 2)
            self._hand_settle(moved, u_old, sent=2)
            return Follow(self.steps - n0, self.steps >= self.step_limit, False, self.obs(cams=False))
        t = np.asarray(path.t, float)
        tt = np.arange(DT, t[-1] + 1e-9, DT)
        if len(tt) == 0 or tt[-1] < t[-1] - 1e-6:
            tt = np.r_[tt, t[-1]]
        B = None if path.base is None else np.asarray(path.base, float).reshape(len(t), 3)
        base0 = self.cmd_base.copy()
        rows, qs, us = [], [], []
        u_start = {s: self.cmd_u[s].copy() for s in path.hand}
        for ti in tt:
            k = min(int(np.searchsorted(t, ti + HAND_LEAD - 1e-9)), len(t) - 1)  # the gripper is commanded ahead
            for s in SIDES:
                if s in path.q:
                    Q = np.asarray(path.q[s], float)
                    self.cmd_q[s] = np.array(
                        [np.interp(ti, np.r_[0.0, t], np.r_[Q[0, j], Q[:, j]]) for j in range(Q.shape[1])]
                    )
                if s in path.hand:
                    self.cmd_u[s] = np.asarray(path.hand[s][k], float).reshape(-1)[:1]
            if B is not None:  # ahead of the path by the drive's lag
                self.cmd_base = self._base_at(base0, min(ti + BASE_LEAD, t[-1]), t, B)
            rows.append(self._row(drive=B is not None))
            qs.append({a: self.cmd_q[a].copy() for a in SIDES})
            us.append({a: self.cmd_u[a].copy() for a in SIDES})
        if B is not None:
            rows[0]["drive"] = True  # what the robot touches now is not in the way of this drive
        stopped = blocked = False
        tilt0 = self.state["tilt"]
        chunk = 3 if stop is not None else (5 if B is not None else 8)
        history = [self.state["base"].copy()]
        for i in range(0, len(rows), chunk):
            self._send(rows[i : i + chunk])
            if self.steps >= self.step_limit or self.ended:
                break
            last = min(i + chunk, len(rows)) - 1
            lag = max([np.abs(np.asarray(self.state["q"][a], float) - qs[last][a]).max() for a in path.q] or [0.0])
            if lag > BLOCKED and i + chunk < len(rows):  # the arm is held back (it hit something): stop there
                blocked = True
                print(f"adapter: arm blocked ({lag:.2f} rad behind its path); stopped the motion there", flush=True)
            history.append(self.state["base"].copy())
            if B is not None and self.state.get("stopped"):  # the simulator stopped the drive: it touched or tilted
                blocked = True
                why = "touched something" if self.state["stopped"] == "contact" else "began to tip"
                print(f"adapter: the base {why}; stopped the motion there", flush=True)
            elif self.state["tilt"] > tilt0 + TILT_RISE:
                blocked = True
                print(
                    f"adapter: the robot leans {self.state['tilt']:.0f} degrees; stopped the motion there", flush=True
                )
            elif B is not None and (
                self._base_behind(rows[last]["base"]) or self._base_stuck(history, rows[last]["base"], chunk)
            ):
                blocked = True
                print("adapter: base blocked short of its path; stopped the motion there", flush=True)
            if not blocked and stop is not None and i + chunk < len(rows) and stop(self.obs(cams=False)):
                stopped = True
            if stopped or blocked:
                self.cmd_u = us[last]
                break
        if stopped or blocked:  # hold the posture reached, and the base where it is
            for s in path.q:
                self.cmd_q[s] = np.asarray(self.state["q"][s], float)
            self.cmd_base = self.state["base"].copy()
        else:
            moved = [a for a in path.hand if abs(float(self.cmd_u[a][0] - u_start[a][0])) > 0.02]
            if moved:
                self._hand_settle(moved, u_start, sent=int(HAND_LEAD / DT))
            if path.q:
                self._settle(list(path.q))
            if B is not None:
                self._base_settle()
        return Follow(self.steps - n0, self.steps >= self.step_limit, stopped, self.obs(cams=False))

    def _base_behind(self, target):
        """The base lags its commanded path by more than free driving does (the server caps how far its target leads)."""
        now = self.state["base"]
        return (
            np.hypot(*(np.asarray(target[:2]) - now[:2])) > BASE_LAG[0] or abs(wrap(target[2] - now[2])) > BASE_LAG[1]
        )

    def _base_stuck(self, history, target, steps):
        """The base is short of where it is commanded and has not moved over the last BASE_STUCK steps."""
        dx, da, n = BASE_STUCK
        back = max(1, int(math.ceil(n / steps)))
        if len(history) <= back:
            return False
        now, then = history[-1], history[-1 - back]
        target = np.asarray(target, float)
        short = np.hypot(*(target[:2] - now[:2])) > 0.03 or abs(wrap(target[2] - now[2])) > math.radians(3)
        moved = np.hypot(*(now[:2] - then[:2])) > dx or abs(wrap(now[2] - then[2])) > da
        return short and not moved

    def _base_settle(self):
        """End on the last base waypoint: hold it while the base is off by more than BASE_TOL and still closing in."""
        history = [self.state["base"].copy()]
        for _ in range(BASE_SETTLE):
            if self.ended or self.steps >= self.step_limit:
                return
            now = self.state["base"]
            if (
                np.hypot(*(self.cmd_base[:2] - now[:2])) < BASE_TOL[0]
                and abs(wrap(self.cmd_base[2] - now[2])) < BASE_TOL[1]
            ):
                return
            self._send([self._row(drive=True)])
            history.append(self.state["base"].copy())
            if self.state.get("stopped") or self._base_stuck(history, self.cmd_base, 1):
                return

    def _settle(self, arms, tol=0.01, max_steps=6):
        """End on the last waypoint: hold its command a few steps while a joint is still more than `tol` rad off and
        still closing in."""
        prev = None
        for _ in range(max_steps):
            if self.ended or self.steps >= self.step_limit:
                return
            err = max(np.abs(np.asarray(self.state["q"][s], float) - self.cmd_q[s]).max() for s in arms)
            if err < tol or (prev is not None and err > 0.7 * prev):
                return
            prev = err
            self._send([self._row()])

    def _hand_settle(self, sides, u_old=None, sent=0, max_steps=15, tol=0.003, still=0.0005):
        """After a gripper command, hold still until the fingers reach it or stop on something between them; a closing
        hand holds at least GRASP steps from its command, so the simulator's assisted grasp attaches what it closed on."""
        closing = u_old is not None and any(self.cmd_u[s][0] < u_old[s][0] - 0.02 for s in sides)
        prev = None
        for _ in range(max_steps):
            if self.ended or self.steps >= self.step_limit:
                return
            now = {s: float(np.mean(self.state["fingers"][s])) for s in sides}
            reached = all(abs(now[s] - finger_q(self.cmd_u[s])[0]) < tol for s in sides)
            settled = reached or (prev is not None and all(abs(now[s] - prev[s]) < still for s in sides))
            if settled and (not closing or sent >= GRASP):
                return
            prev = now
            self._send([self._row()])
            sent += 1

    def finish(self, out):
        """End the episode: hold still a few steps, then the BDDL goal (the challenge's success: it held at some step),
        and save the head-camera video, the actions and native_result.json in `out`."""
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
                imageio.mimwrite(
                    out / "video.mp4",
                    list(frames),
                    fps=max(1, round(1 / (DT * max(1, self.frame_every)))),
                    macro_block_size=1,
                )
        except Exception as e:  # noqa: BLE001
            print(f"video not saved: {e!r}", flush=True)
        np.save(out / "actions.npy", np.asarray(self.actions, np.float32).reshape(-1, 19))
        (out / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in self.requests))
        native = dict(task=self.task, instance=self.instance, seed=self.seed, steps=min(self.steps, self.step_limit),
                      step_limit=self.step_limit, success=bool(result.get("success")), instruction=self.instruction,
                      dt=DT, success_step=result.get("success_step"), goal_at_end=result.get("goal_now"),
                      env_steps=result.get("steps"), broken=result.get("broken"), error=result.get("error"), infra=self.sim.lost,
                      actions="per step: left arm q (7), left opening, right arm q (7), right opening, base target x, y, yaw "
                      "(NaN: held where it is)")  # fmt: skip
        (out / "native_result.json").write_text(json.dumps(native, indent=2))
        return native


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def flat(row):
    """A row as actions.npy stores it: left arm q (7), left opening, right arm q (7), right opening, base x, y, yaw."""
    return np.r_[
        row["q"]["left"], row["u"]["left"], row["q"]["right"], row["u"]["right"], row.get("base", [np.nan] * 3)
    ]
