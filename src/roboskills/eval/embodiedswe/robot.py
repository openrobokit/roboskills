"""EmbodiedSWE-Bench behind `roboskills.eval.interface.Robot`: a Panda on a table, or the G1 humanoid (pelvis fixed, or
walking), in one RoboBench preset served by `sim/server.py` (launch.py starts it).

Per control step (1/60 s: RoboBench's 0.02 s joint-mode period rounded to two 120 Hz physics steps) the simulator
takes each arm's 7 joint targets and each hand's joint targets (the Panda's two fingers, the Dex3's seven); the walking
G1 also takes a world [x, y, yaw] target for its pelvis, with the path's velocity, which the server turns into its
walking policy's command. The arms hang off a link that never moves relative to the robot (the Panda's
`panda_link0`; the G1's `torso_link`, its waist held): `Arm.T_world_base` is that link's world pose, read from the
simulator after every reply (one array, shared by both arms, updated in place). What reaches the agent: rendered RGB-D
with intrinsics and camera poses and where each camera sees the robot (`robot_mask`), the measured joints, the pelvis
pose and height of the walking G1, the robot's geometry, the step count and the scene's own description. `finish` alone
reads the grader's verdict.
"""

import json
import math
import sys
import threading
from pathlib import Path

import numpy as np

from roboskills.eval.embodiedswe.hand import (
    CURL,
    PANDA_WIDTH,
    dex_pose,
    dex_tcp,
    index_tip,
    panda_opening,
    panda_q,
    panda_tcp,
)
from roboskills.eval.interface import Arm, Base, Camera, Follow, Hand, Obs
from roboskills.eval.wire import Sim


VMAX = {"franka": 1.5, "g1": 1.0}  # rad/s, arm joint speed for retiming
EE = {"franka": "panda_hand", "g1": "{}_hand_palm_link"}
PRESS_PANDA = np.array([0.009, 0.0, 0.0])  # the closed fingertips, in the TCP frame
HAND_LEAD = 0.15  # s: hand commands lead the path
# s: arm targets lead the path by the joint servos' lag (the Panda's 400/80 PD trails a target by 0.18 s; the G1's
# 3000/10 arms keep up)
ARM_LEAD = {"franka": 0.15, "g1": 0.0}
HAND_TOL = {"franka": (0.0005, 0.0002), "g1": (0.02, 0.002)}  # a hand reached its target / stopped moving: m or rad
BLOCKED = 0.3  # rad: an arm this far behind its commanded path is held back by something (stop the motion)
GRASP = 20  # steps a closing hand holds from its command before the motion goes on
BASE_TOL = (0.015, math.radians(2.0))  # m, rad: the pelvis has settled on its last waypoint
BASE_LAG = (0.3, math.radians(25))  # m, rad: a pelvis this far behind its path is held back (stop the motion)
BASE_SETTLE = (360, 90)  # steps at most spent settling the pelvis on its last waypoint, and without getting closer
TILT_RISE = 20.0  # degrees a motion may lean the pelvis past its lean at the start (walking leans it up to 17)
HOLD = 30  # steps `finish` holds still before the verdict


class EmbodiedSWE:
    def __init__(self, sim, task, seed, out, frame_every=20):
        self.task, self.seed, self.frame_every = task, str(seed), int(frame_every)
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.sim, self.lock = Sim(sim), threading.Lock()  # the launcher calls from several threads, one at a time
        first = self._req(op="reset")
        f = self.facts = first["facts"]
        self.kind, self.loco = f["robot"], bool(f["loco"])
        self.dt, self.step_limit, self.instruction = float(f["dt"]), int(f["step_limit"]), f["instruction"]
        self.steps, self.ended = int(first["steps"]), False
        self.T_mount = np.array(first["root"], float)  # every arm's T_world_base: one array, updated in place
        self.arms = {}
        for s in f["arm_joints"]:
            joints = list(f["hand_joints"][s])
            if self.kind == "franka":
                T = panda_tcp()
                hand = Hand("jaw", T, np.ones(1), np.zeros(1), joints=joints, model_q=panda_q, width=PANDA_WIDTH,
                            press="closed", press_point=PRESS_PANDA.copy())  # fmt: skip
            else:
                T, straight = dex_tcp(s), np.zeros(len(joints))
                hand = Hand("dex", T, straight, dex_pose(s, joints, CURL), joints=joints,
                            presets={"point": dex_pose(s, joints, CURL, keep=("index_0", "index_1")), "flat": straight},
                            press="point", press_point=index_tip(f["urdf"], s, self.out / "hand"))  # fmt: skip
            lo, hi = np.asarray(f["lo"][s], float), np.asarray(f["hi"][s], float)
            self.arms[s] = Arm(s, f["urdf"], list(f["arm_joints"][s]), EE[self.kind].format(s), self.T_mount,
                               lo + 0.01, hi - 0.01, np.asarray(first["q"][s], float), hand, vmax=VMAX[self.kind])  # fmt: skip
        self._take(first)
        self.cmd_q = {s: np.asarray(self.state["q"][s], float) for s in self.arms}
        self.cmd_h = {s: np.asarray(self.state["hand"][s], float) for s in self.arms}  # hand joint targets
        if self.loco:
            self.base = Base(vmax=0.1, wmax=0.3, radius=0.3, floor_z=float(f["floor"]), tol=0.05)
            self.cmd_base = self.state["base"].copy()  # (no `body`: the walking policy holds its pelvis at 0.72 m)
        (self.out / "robot.json").write_text(json.dumps(dict(task=task, seed=self.seed, instruction=self.instruction,
                                                             step_limit=self.step_limit, dt=self.dt, urdf=f["urdf"]), indent=1))  # fmt: skip

    # ---- the simulator

    def _req(self, **msg):
        with self.lock:
            return self.sim.request(**msg)

    def _take(self, rep):
        """Keep the latest state (and cameras); move the arms' mount with the robot."""
        self.state = {"q": rep["q"], "hand": rep["hand"], "done": rep["done"], "stopped": rep.get("stopped")}
        self.T_mount[:] = np.asarray(rep["root"], float)
        if self.loco:
            self.state.update(base=np.asarray(rep["base"], float), body=float(rep["body"]), tilt=float(rep["tilt"]))
        if "cams" in rep:
            self.cams = rep["cams"]

    def _hand_state(self, s):
        h = np.asarray(self.state["hand"][s], float)
        return panda_opening(h) if self.kind == "franka" else h

    def obs(self, cams=True):
        cs = {}
        if cams and not self.ended:
            self._take(self._req(op="observe"))
        if cams:
            cs = {n: Camera(np.array(v["rgb"]), np.where(np.isfinite(v["depth"]) & (v["depth"] > 0), v["depth"], np.nan).astype(np.float32),
                            np.asarray(v["K"], float), np.asarray(v["T"], float)) for n, v in self.cams.items()}  # fmt: skip
            self.masks = {n: (cs[n].rgb, np.asarray(v["robot"], bool)) for n, v in self.cams.items()}
        return Obs(cs, {s: np.asarray(self.state["q"][s], float) for s in self.arms}, {s: self._hand_state(s) for s in self.arms},
                   min(self.steps, self.step_limit), self.step_limit, self.instruction,
                   base=self.state["base"].copy() if self.loco else None)  # fmt: skip

    def robot_mask(self, cam, obs):
        """Where camera `cam` sees the robot in `obs`: the simulator's instance segmentation of that very frame (every
        mesh under the robot's prim), grown 2 pixels; None for an observation other than the latest one."""
        rgb, mask = getattr(self, "masks", {}).get(cam, (None, None))
        return mask if cam in obs.cams and obs.cams[cam].rgb is rgb else None

    # ---- acting

    def _h(self, s, u):
        """Hand joint targets of command u."""
        return panda_q(u) if self.kind == "franka" else np.asarray(u, float).reshape(-1)

    def _row(self, drive=None):
        """One control step's command: the arms' and hands' last commands; the walking G1's pelvis target and path
        velocity while it walks (`drive`; without one it stands still)."""
        row = {"q": {s: self.cmd_q[s].tolist() for s in self.arms}, "h": {s: self.cmd_h[s].tolist() for s in self.arms}}
        if self.loco and drive is not None:
            row["base"], row["vel"] = self.cmd_base.tolist(), list(drive)
        return row

    def _send(self, rows):
        """Send rows (one per control step). After the episode's end nothing runs, but the steps count."""
        rows = rows[: max(0, self.step_limit - self.steps)]
        self.steps += len(rows)
        if rows and not self.ended:
            self._take(self._req(op="act", rows=rows, render=False))
            self.ended = bool(self.state["done"])
        self.ended = self.ended or self.steps >= self.step_limit

    def replay(self, actions):
        """Step a finished episode's native actions (its `actions.npy`: every control step's action vector, the
        settling ones at boot first) as they were, from the step this episode is at; proprioception after."""
        A = np.asarray(actions, np.float32)[self.steps :]
        for i in range(0, len(A), 600):
            if self.ended:
                break
            self._send([{"a": a} for a in A[i : i + 600]])
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
        """The pelvis target at `t` s on the path from `start` (at 0 s) through rows B at times tt; the path's yaw is
        unwrapped and shifted by whole turns to start nearest `start`'s."""
        B = np.array(B, float)
        B[:, 2] = np.unwrap(B[:, 2])
        B[:, 2] += 2 * math.pi * round((start[2] - B[0, 2]) / (2 * math.pi))
        tt, B = np.r_[0.0, tt], np.vstack([np.asarray(start, float)[None], B])
        return np.array([np.interp(t, tt, B[:, j]) for j in range(3)])

    def _follow(self, path, stop=None):
        n0 = self.steps
        moved = [s for s in path.hand if np.abs(self._h(s, path.hand[s][-1]) - self.cmd_h[s]).max() > 0.02]
        still = all(np.ptp(np.asarray(Q, float), axis=0).max() < 1e-4 for Q in path.q.values())
        if stop is None and moved and still and path.base is None:  # the hand alone
            closing = self._closing(path.hand)
            for s in moved:
                self.cmd_h[s] = self._h(s, path.hand[s][-1])
            self._send([self._row()] * 2)
            self._hand_settle(moved, closing, sent=2)
            return Follow(self.steps - n0, self.steps >= self.step_limit, False, self.obs(cams=False))
        t = np.asarray(path.t, float)
        tt = np.arange(self.dt, t[-1] + 1e-9, self.dt)
        if len(tt) == 0 or tt[-1] < t[-1] - 1e-6:
            tt = np.r_[tt, t[-1]]
        B = None if path.base is None or not self.loco else np.asarray(path.base, float).reshape(len(t), 3)
        base0 = self.state["base"].copy() if self.loco else None
        closing = self._closing(path.hand)
        rows, qs, hs, lead = [], [], [], ARM_LEAD[self.kind]
        Q = {s: np.vstack([np.asarray(path.q[s], float)[:1], np.asarray(path.q[s], float)]) for s in path.q}
        at = lambda s, x: np.array([np.interp(x, np.r_[0.0, t], Q[s][:, j]) for j in range(Q[s].shape[1])])  # noqa: E731
        for ti in tt:
            k = min(int(np.searchsorted(t, ti + HAND_LEAD - 1e-9)), len(t) - 1)  # the hand is commanded ahead
            on_path = {}
            for s in self.arms:
                if s in path.q:  # commanded ahead by the servo's lag; the path itself is where the arm should be
                    self.cmd_q[s], on_path[s] = at(s, min(ti + lead, t[-1])), at(s, ti)
                if s in path.hand:
                    self.cmd_h[s] = self._h(s, path.hand[s][k])
            vel = None
            if B is not None:
                self.cmd_base = self._base_at(base0, ti, t, B)
                vel = (self._base_at(base0, min(ti + self.dt, t[-1]), t, B) - self.cmd_base) / self.dt
            rows.append(self._row(vel))
            qs.append(on_path)
            hs.append({a: self.cmd_h[a].copy() for a in self.arms})
        stopped = blocked = False
        tilt0 = self.state.get("tilt", 0.0)
        chunk = 3 if stop is not None else (6 if B is not None else 10)
        for i in range(0, len(rows), chunk):
            self._send(rows[i : i + chunk])
            if self.steps >= self.step_limit or self.ended:
                break
            last = min(i + chunk, len(rows)) - 1
            lag = max([np.abs(np.asarray(self.state["q"][a], float) - qs[last][a]).max() for a in path.q] or [0.0])
            if lag > BLOCKED and i + chunk < len(rows):  # the arm is held back (it hit something): stop there
                blocked = True
                print(f"adapter: arm blocked ({lag:.2f} rad behind its path); stopped the motion there", flush=True)
            elif self.loco and (self.state.get("stopped") or self.state["tilt"] > tilt0 + TILT_RISE):
                blocked = True
                print(
                    f"adapter: the robot leans {self.state['tilt']:.0f} degrees; stopped the motion there", flush=True
                )
            elif B is not None and self._base_behind(rows[last]["base"]):
                blocked = True
                print("adapter: the pelvis fell behind its path; stopped the motion there", flush=True)
            if not blocked and stop is not None and i + chunk < len(rows) and stop(self.obs(cams=False)):
                stopped = True
            if stopped or blocked:
                self.cmd_h = hs[last]
                break
        if stopped or blocked:  # hold the posture reached, and the pelvis where it is
            for s in path.q:
                self.cmd_q[s] = np.asarray(self.state["q"][s], float)
            if self.loco:
                self.cmd_base = self.state["base"].copy()
        else:
            moved = [a for a in path.hand if a in closing or np.abs(self.cmd_h[a] - hs[0][a]).max() > 0.02]
            if moved:
                self._hand_settle(moved, closing, sent=int(HAND_LEAD / self.dt))
            if path.q:
                self._settle(list(path.q))
            if B is not None:
                self._base_settle()
        return Follow(self.steps - n0, self.steps >= self.step_limit, stopped, self.obs(cams=False))

    def _closing(self, hand):
        """Arms whose hand the path closes (a smaller opening, or fingers curling)."""
        out = set()
        for s, U in hand.items():
            end = self._h(s, U[-1])
            if self.kind == "franka":
                if end.sum() < self.cmd_h[s].sum() - 0.002:
                    out.add(s)
            elif (np.abs(end) > np.abs(self.cmd_h[s]) + 0.05).any():
                out.add(s)
        return out

    def _base_behind(self, target):
        now = self.state["base"]
        return (np.hypot(*(np.asarray(target[:2]) - now[:2])) > BASE_LAG[0]
                or abs(wrap(target[2] - now[2])) > BASE_LAG[1])  # fmt: skip

    def _base_settle(self):
        """End on the last pelvis waypoint: hold it while the pelvis is off by more than BASE_TOL and closing in."""
        best, since = np.inf, 0
        for _ in range(0, BASE_SETTLE[0], 6):
            if self.ended or self.steps >= self.step_limit:
                return
            now = self.state["base"]
            d = np.hypot(*(self.cmd_base[:2] - now[:2]))
            a = abs(wrap(self.cmd_base[2] - now[2]))
            if d < BASE_TOL[0] and a < BASE_TOL[1]:
                return
            err = d + 0.3 * a
            best, since = (err, 0) if err < best - 0.005 else (best, since + 6)
            if since >= BASE_SETTLE[1] or self.state.get("stopped"):  # no longer closing in
                return
            self._send([self._row((0.0, 0.0, 0.0))] * 6)

    def _settle(self, arms, tol=0.01, max_steps=40):
        """End on the last waypoint: hold its command while a joint is still more than `tol` rad off and still closing
        in, two steps at a time."""
        prev = None
        for _ in range(0, max_steps, 2):
            if self.ended or self.steps >= self.step_limit:
                return
            err = max(np.abs(np.asarray(self.state["q"][s], float) - self.cmd_q[s]).max() for s in arms)
            if err < tol or (prev is not None and err > prev - 0.001):
                return
            prev = err
            self._send([self._row()] * 2)

    def _hand_settle(self, sides, closing=(), sent=0, max_steps=40):
        """After a hand command, hold still until the fingers reach it or stop on something; a closing hand holds at
        least GRASP steps from its command."""
        prev, (tol, still) = None, HAND_TOL[self.kind]
        for _ in range(max_steps):
            if self.ended or self.steps >= self.step_limit:
                return
            now = {s: np.asarray(self.state["hand"][s], float) for s in sides}
            reached = all(np.abs(now[s] - self.cmd_h[s]).max() < tol for s in sides)
            settled = reached or (prev is not None and all(np.abs(now[s] - prev[s]).max() < still for s in sides))
            if settled and (not set(sides) & set(closing) or sent >= GRASP):
                return
            prev = now
            self._send([self._row()])
            sent += 1

    def finish(self, out):
        """End the episode: hold still a few steps, then the grader's verdict on the final state, and save the front
        camera's video, the actions and native_result.json in `out`."""
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        hold = 0 if self.ended else min(HOLD, max(0, self.step_limit - self.steps))
        try:
            result = self._req(op="finish", hold=hold)
            self.steps += hold
        except RuntimeError as e:
            result = {
                "success": False,
                "error": str(e),
                "frames": np.zeros((0, 1, 1, 3), np.uint8),
                "actions": np.zeros((0, 0)),
            }
        frames, actions = np.asarray(result.pop("frames")), np.asarray(result.pop("actions"), np.float32)
        self.sim.close()
        try:
            import imageio

            if len(frames):
                imageio.mimwrite(out / "video.mp4", list(frames), fps=max(1, round(1 / (self.dt * max(1, self.frame_every)))),
                                 macro_block_size=1)  # fmt: skip
        except Exception as e:  # noqa: BLE001
            print(f"video not saved: {e!r}", flush=True)
        np.save(out / "actions.npy", actions)
        native = dict(task=self.task, seed=self.seed, steps=min(self.steps, self.step_limit), step_limit=self.step_limit,
                      success=bool(result.get("success")), score=result.get("score"), stages=result.get("stages"),
                      criteria=result.get("criteria"), instruction=self.instruction, dt=self.dt, broken=result.get("broken"),
                      error=result.get("error"), infra=self.sim.lost,
                      actions="per step: RoboBench's native action vector (the preset's composite controller order)")  # fmt: skip
        (out / "native_result.json").write_text(json.dumps(native, indent=2))
        return native


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi
