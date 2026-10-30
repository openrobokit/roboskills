"""RoboDojo behind `roboskills.eval.interface.Robot`: two ARX X5 arms with parallel grippers in Isaac Sim 5.1.

    make(task=T layout=L seed=S sim=SIM_PORT [mode=dev|eval] [suite=legacy|full5] [obs=full|official] [fix=robodojo_pr48] [out=RUN/server] [ws=/workspace])   # launch.py calls it, one per env

One episode = one RoboDojo layout (`arx_x5`, the layout file of eval_seed S), started here with the native reset of
the warm simulator listening on SIM_PORT (`sim/server.py`, in the Isaac Sim container on this host). `Clocked`
(clocked=True, from launch.py) runs one env of `sim/vector.py`'s continuously advancing batch instead. The benchmark's action,
per 25 Hz step: six joint targets per arm and one gripper opening per arm. Everything here is robot facts (the X5
URDF at the env's base poses, the gripper's travel and TCP) and the env's observation: camera RGB-D with intrinsics
and extrinsics, the measured arm joints and finger travel, the step count.
The default information mode is observations. Explicit information="privileged"
also enables scene(): read-only physical object poses and surface geometry,
without live task-target annotations or reward queries.

RoboDojo ends the episode as soon as its native check passes. This adapter hides that: motion after the end is
counted against the budget as if it ran, and only `finish` reads the outcome.

`obs=official`: RoboDojo's own eval observation (vendor env_cfg/arx_x5.yml): camera RGB only (no depth, intrinsics or
extrinsics), the joints, the gripper and the world EE pose. Each Camera's depth, K and T_world_cam then come from
`rgbd.py` (the cameras' self-calibration in `cams.json`, depth from RGB); a camera it does not cover has None.
"""

import copy
import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from roboskills.eval.interface import Arm, Camera, Follow, Hand, Obs
from roboskills.eval.robodojo.layouts import check_layout
from roboskills.eval.robodojo.rgbd import CAMS_JSON, RGBD
from roboskills.eval.wire import Sim


ROOT = Path.cwd()
DT = 1.0 / 25  # one budget step: one native action
SIDES = ("left", "right")
RECIPES = Path(__file__).with_name("recipes")  # the official wiki recipe of each task, given with its instruction
TRAVEL = 0.044  # each finger's travel, closed (pads touching) to open, m (URDF joint7/joint8 limits)
LO, HI = -0.01, 0.044  # the env's command range for a finger (robot_config gripper_scale): -0.01 squeezes past closed
BLOCKED = 0.3  # rad: an arm this far behind its commanded path is held back by something (stop the motion)
HAND_LEAD = 0.2  # s: gripper commands are sent this far ahead of the path (the fingers lag their target)
FINGER_STEP = 0.006  # m a finger travels per step, for the wait after a gripper command when the fingers are not
#                     measured (official observation): with them measured, a close or an open settles in 2-7 steps
TCP = 0.145  # the TCP (between the pads) along link6 +x (robot_config gripper_bias); the finger tips end at 0.158
ARM_LO = np.array([-np.pi, 0.0, 0.0, -2.05, -1.62, -np.pi])  # IK/planning windows: joints 2-5 on the above-the-table
ARM_HI = np.array([np.pi, 3.15, 3.15, 1.70, 1.62, np.pi])  # elbow branch


# The wrist cameras, self-calibrated from wrist RGB and the arms' joints (the epipolar fit of ORB matches between views
# the arm moved between; examples/robodojo_claude/VISUAL_GRASP.md): focal length (square pixels, centre at the image
# middle) and the mount's pitch, which is 10 deg more than the URDF's camera_joint says (30, not 20)
WRIST_F = 396.0  # px
WRIST_PITCH_FIX = 10.0  # deg, about the camera link's y axis
OPT_IN_LINK = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]], float)  # ROS camera link -> optical


def urdf_chain(src, tip):
    """[(joint type, origin 4x4, axis)] from the URDF's root to link `tip`."""
    root = ET.parse(src).getroot()
    by_child = {j.find("child").get("link"): j for j in root.findall("joint")}
    out, link = [], tip
    while link in by_child:
        j = by_child[link]
        o = j.find("origin")
        T = np.eye(4)
        if o is not None:
            T[:3, :3] = Rotation.from_euler("xyz", [float(v) for v in (o.get("rpy") or "0 0 0").split()]).as_matrix()
            T[:3, 3] = [float(v) for v in (o.get("xyz") or "0 0 0").split()]
        ax = j.find("axis")
        out.append((j.get("type"), T, np.array([float(v) for v in ax.get("xyz").split()]) if ax is not None else None))
        link = j.find("parent").get("link")
    return out[::-1]


def chain_fk(chain, q):
    """Pose of the chain's tip link in its root link for the revolute joints' angles `q` (in chain order)."""
    T, k = np.eye(4), 0
    for kind, T0, ax in chain:
        T = T @ T0
        if kind in ("revolute", "continuous"):
            R = np.eye(4)
            R[:3, :3] = Rotation.from_rotvec(ax * q[k]).as_matrix()
            T, k = T @ R, k + 1
    return T


def finger_target(u):
    """Gripper command u (0 closed .. 1 open, linear in the gap between the pads) -> the finger joint target.
    Closing goes past the pads' contact (to the env's -0.01 m) so a held object is squeezed, blended in below u=0.1."""
    u = float(np.clip(u, 0.0, 1.0))
    return TRAVEL * u + LO * max(0.0, 1.0 - u / 0.1)


def native_opening(u):
    """The env's normalized opening for command u (its finger target = LO + (HI - LO) * opening)."""
    return float(np.clip((finger_target(u) - LO) / (HI - LO), 0.0, 1.0))


class RoboDojo:
    def __init__(
        self,
        task,
        layout,
        seed,
        sim,
        mode="dev",
        fixes=(),
        out=None,
        frame_every=8,
        obs="full",
        ws="/workspace",
        information="observations",
        suite="legacy",
        video=None,
        vendor=None,
    ):
        check_layout(task, layout, seed, mode, suite)
        if obs not in ("full", "official"):
            raise ValueError("obs is full or official")
        self.task, self.layout, self.seed, self.mode, self.official = task, layout, seed, mode, obs == "official"
        self.out = Path(out or ROOT / "runs/robodojo_claude/tmp").resolve()
        self.frame_every, self.dt = frame_every, DT
        # the head-camera video.mp4 (MANIP_VIDEO=0: none, saves disk)
        self.video = os.environ.get("MANIP_VIDEO", "1") == "1" if video is None else bool(video)
        from robokit.assets.robots.assembly import arx_x5

        self.sim = Sim(sim)
        dev = (
            f"{ws}/{(self.out / 'dev').relative_to(ROOT)}" if mode == "dev" and self.out.is_relative_to(ROOT) else None
        )  # ws: ROOT in the sim's container
        if information not in ("observations", "privileged"):
            raise ValueError("Unknown information mode")
        self.information = information
        self._alias = {}  # native ids are renumbered by each reset; restores map them back
        self._reset = dict(
            op="reset",
            layout=layout,
            seed=seed,
            fixes=list(fixes),
            dev_dir=dev,
            official=self.official,
            information=information,
        )
        first = self._req(**self._reset)
        recipe = RECIPES / f"{task}.md"
        self.instruction = first["instruction"] + (f"\n\nTask notes:\n{recipe.read_text()}" if recipe.exists() else "")
        self.step_limit, self.steps = int(first["step_limit"]), 0
        self.ended, self.frames, self.actions = False, [], []
        self._take(first)
        self.out.mkdir(parents=True, exist_ok=True)  # the first observation, for checking resets against cold boots
        np.savez_compressed(self.out / "obs0.npz", **{f"{c}_{k}": v[k] for c, v in first["cams"].items() for k in ("rgb", "depth") if k in v},
                            **{k: np.asarray(v) for k, v in first["state"].items()})  # fmt: skip
        self.T_base = {}  # each arm's base_link in the world: where the env stands it (env_cfg/robot/dual_x5.yml)
        for s, r in zip(SIDES, first["robots"]):
            T = np.eye(4)
            T[:3, :3] = Rotation.from_quat(np.asarray(r["default_root_rot"], float)[[1, 2, 3, 0]]).as_matrix()
            T[:3, 3], self.T_base[s] = r["default_root_pos"], T
        self.table_z = float(self.T_base["left"][2, 3])  # the arms stand on the table top
        self.rgbd, self.est, self.est_step = None, {}, None
        if self.official and CAMS_JSON.exists():
            self.rgbd = RGBD(str(arx_x5.URDF_PATH), self.T_base, table_z=self.table_z)
        T_ee_tcp = np.eye(4)
        T_ee_tcp[0, 3] = TCP
        st, self.arms = first["state"], {}
        for s in SIDES:
            hand = Hand("jaw", T_ee_tcp, open=np.array([1.0]), closed=np.array([0.0]), joints=["joint7"],
                        model_q=lambda u: TRAVEL * np.clip(np.asarray(u, float).reshape(-1)[:1], 0, 1),  # joint8 mimics
                        width=2 * TRAVEL, press="closed", press_point=np.array([0.012, 0.0, 0.0]))  # fmt: skip
            self.arms[s] = Arm(s, arx_x5.URDF_PATH, [f"joint{i}" for i in range(1, 7)], "link6", self.T_base[s],
                               ARM_LO.copy(), ARM_HI.copy(), np.asarray(st[f"{s}_arm_joint_state"], float), hand,
                               vmax=4.0, collision_spheres=arx_x5.COLLISION_SPHERE_PATH)  # fmt: skip
        # the wrist camera: link6 -> camera_base -> camera (robot facts), from each arm's base
        self.cam_chain = urdf_chain(arx_x5.URDF_PATH, "camera")
        # the last command per arm (arms not moving hold it): joints and gripper command
        self.cmd_q = {s: np.asarray(st[f"{s}_arm_joint_state"], float) for s in SIDES}
        self.cmd_u = {s: self._u_measured(s) for s in SIDES}

    # ---- development rewinds

    def checkpoint(self):
        """Record the client state; native physics is recovered by replaying this episode's actions."""
        keep = ("steps", "ended", "state", "cams", "fresh", "cmd_q", "cmd_u", "est", "est_step")
        point = {key: copy.deepcopy(getattr(self, key)) for key in keep}
        point["frames"] = len(self.frames)
        if self.information == "privileged":
            point["objects"] = {o["id"]: o["pose"] for o in self.scene()["objects"]}
        return point

    def restore(self, point):
        """Repeat the seeded native reset, replay recorded actions and report the physical residual."""
        rows = self.actions[: point["steps"]]
        self._req(**self._reset)
        for k in range(0, len(rows), self.frame_every):
            if self._req(op="act", actions=rows[k : k + self.frame_every], render=False)["done"]:
                break
        replayed = self._req(op="observe")
        self.actions = list(rows)
        for key in ("steps", "ended", "state", "cams", "fresh", "cmd_q", "cmd_u", "est", "est_step"):
            setattr(self, key, copy.deepcopy(point[key]))
        del self.frames[point["frames"] :]
        residual = dict(
            joints_rad=max(
                float(np.max(np.abs(np.asarray(replayed["state"][k], float) - np.asarray(point["state"][k], float))))
                for k in point["state"]
                if k.endswith("_joint_state")
            )
        )
        self.state, self.cams, self.fresh = replayed["state"], replayed["cams"], True
        if "objects" in point:
            free, self._alias = dict(point["objects"]), {}
            for o in self._scene()["objects"]:
                stem = o["id"].rsplit("_", 1)[0]
                match = [i for i in free if i.rsplit("_", 1)[0] == stem]
                if match:
                    at = np.asarray(o["pose"], float)[:3, 3]
                    i = min(match, key=lambda i: np.linalg.norm(np.asarray(free[i], float)[:3, 3] - at))
                    self._alias[o["id"]] = i
                    del free[i]
            now = {o["id"]: np.asarray(o["pose"], float) for o in self.scene()["objects"]}
            residual["object_m"] = max(
                (
                    float(np.linalg.norm(now[i][:3, 3] - np.asarray(pose, float)[:3, 3]))
                    for i, pose in point["objects"].items()
                    if i in now
                ),
                default=0.0,
            )
            residual["unmatched_objects"] = sorted(free)
        return dict(replayed_actions=len(rows), residual=residual)

    # ---- the simulator

    def scene(self):
        """Read physical object geometry only in explicitly privileged episodes."""
        if self.information != "privileged":
            raise PermissionError("Simulator geometry is unavailable in observations mode")
        scene = self._scene()
        for o in scene["objects"]:
            o["id"] = self._alias.get(o["id"], o["id"])
        return scene

    def _scene(self):
        """The sim's raw object poses and local surface points (sim/state.py), made world poses and points here."""
        from robokit.xform.numpy import quaternion_to_matrix, transform_points

        scene, objects = self._req(op="scene"), []
        for o in scene["objects"]:
            try:
                T = np.eye(4)
                T[:3, :3] = quaternion_to_matrix(np.asarray(o.pop("quaternion_wxyz"), float))
                T[:3, 3] = o.pop("position")
                P = transform_points(np.asarray(o.pop("local_points"), float), T)
            except ValueError as error:
                scene["errors"].append(dict(object=o["id"], error=type(error).__name__, message=str(error)))
                continue
            objects.append(
                dict(o, pose=T.tolist(), points=P.tolist(), lower=P.min(0).tolist(), upper=P.max(0).tolist())
            )
        scene["objects"] = objects
        return scene

    def _req(self, **msg):
        return self.sim.request(**msg)

    def _take(self, rep):
        """Keep the latest state (and cameras, with the head camera as a video frame)."""
        self.state, self.fresh = rep["state"], "cams" in rep
        if self.fresh:
            self.cams = rep["cams"]
            if self.video:
                self.frames.append(np.asarray(rep["cams"]["cam_head"]["rgb"])[::2, ::2])

    def _u_measured(self, s):
        """Measured gripper opening in command units: the finger's travel over its full travel. Official observation:
        the gripper state is the last commanded opening (vendor obs_manager), not the measured travel."""
        if self.official and hasattr(self, "cmd_u"):
            return self.cmd_u[s].copy()
        return np.array([float(np.clip(self.state[f"{s}_gripper_joint_state"][0] / TRAVEL, 0.0, 1.0))])

    def obs(self, cams=True):
        cs = {}
        if cams:
            if self.official and self.est_step != self.steps:  # a fresh observe (a dev episode logs its ground truth)
                self._take(self._req(op="observe"))
                self.est, self.est_step = (
                    (self.rgbd({n: v["rgb"] for n, v in self.cams.items()}, self.state) if self.rgbd else {}),
                    self.steps,
                )
                if self.rgbd:  # the fits, for the record (not printed: a verb's printed text goes to the agent)
                    with open(self.out / "rgbd.jsonl", "a") as f:
                        f.write(json.dumps({"step": self.steps, **self.rgbd.log[-1]}) + "\n")
            elif not self.fresh:
                self._take(self._req(op="observe"))
            if self.official:  # RGB only: estimates (rgbd.py) where it has them; the wrist cameras' K and pose from the
                #                  robot model (`wrist_model`) where it does not
                cs = {n: Camera(np.array(v["rgb"]), *self.est[n]) if n in self.est else
                      Camera(np.array(v["rgb"]), None, *self.wrist_model(n.split("_")[1])) if n.endswith("_wrist") else
                      Camera(np.array(v["rgb"])) for n, v in self.cams.items()}  # fmt: skip
            else:
                cs = {n: Camera(np.array(v["rgb"]), np.where(np.isfinite(v["depth"]) & (v["depth"] > 0), v["depth"], np.nan).astype(np.float32),
                                np.asarray(v["K"], float), np.asarray(v["T"], float)) for n, v in self.cams.items()}  # fmt: skip
        q = {s: np.asarray(self.state[f"{s}_arm_joint_state"], float) for s in SIDES}
        return Obs(cs, q, {s: self._u_measured(s) for s in SIDES}, min(self.steps, self.step_limit), self.step_limit,
                   self.instruction)  # fmt: skip

    def wrist_model(self, s, f=None):
        """Arm `s`'s wrist camera from the robot model alone: its pose from the URDF at the measured joints, and K
        from the self-calibrated focal length (None if not calibrated)."""
        f = f or WRIST_F
        rgb = self.cams[f"cam_{s}_wrist"]["rgb"]
        h, w = np.asarray(rgb).shape[:2]
        K = np.array([[f, 0, (w - 1) / 2], [0, f, (h - 1) / 2], [0, 0, 1.0]]) if f else None
        fix = np.eye(4)
        fix[:3, :3] = Rotation.from_euler("y", WRIST_PITCH_FIX, degrees=True).as_matrix()
        T = (
            self.T_base[s]
            @ chain_fk(self.cam_chain, np.asarray(self.state[f"{s}_arm_joint_state"], float))
            @ fix
            @ OPT_IN_LINK
        )
        return K, T

    # ---- acting

    def _send(self, rows):
        """Send native actions (one per budget step). After the env's own end nothing runs, but the steps count."""
        rows = rows[: max(0, self.step_limit - self.steps)]
        self.actions.extend(rows)
        for k in range(0, len(rows), self.frame_every):
            chunk = rows[k : k + self.frame_every]
            before, self.steps = self.steps, self.steps + len(chunk)
            if not self.ended:
                rep = self._req(
                    op="act", actions=chunk, render=self.steps // self.frame_every > before // self.frame_every
                )
                self._take(rep)
                self.ended = bool(rep["done"])

    def _row(self):
        a = {f"{s}_arm_joint_state": [float(x) for x in self.cmd_q[s]] for s in SIDES}
        a.update({f"{s}_ee_joint_state": [native_opening(self.cmd_u[s][0])] for s in SIDES})
        return a

    def follow(self, path, stop=None):
        n0 = self.steps
        try:
            return self._follow(path, stop)
        finally:  # where the steps go: one line per motion, with the skill calls that asked for it
            f, who = sys._getframe(1), []
            while f is not None and len(who) < 4:
                file, name = f.f_code.co_filename, f.f_code.co_name
                if (
                    "/agent_as_policy/" in file
                    or file.endswith("/tools/robokit/bot.py")
                    or (file.endswith("/roboskills/core/agent.py") and name == "execute")
                ) and not name.startswith("_exec"):
                    who.append(name)
                f = f.f_back
            print(f"steps {self.steps - n0} at {n0} by {'<'.join(who)}", flush=True)

    def _follow(self, path, stop=None):
        n0 = self.steps
        u_end = {s: np.asarray(path.hand[s][-1], float).reshape(-1)[:1] for s in path.hand}
        moved = [s for s in path.hand if abs(float(u_end[s][0] - self.cmd_u[s][0])) > 0.02]
        if stop is None and moved and all(np.ptp(np.asarray(Q, float), axis=0).max() < 1e-4 for Q in path.q.values()):
            # the hand alone moves: command its end at once and hold until the fingers get there or stop on
            # something (as fast as the X5 fingers go; a timed ramp only adds steps)
            u_old = {s: self.cmd_u[s].copy() for s in moved}
            self.cmd_u.update(u_end)
            self._send([self._row()] * 2)
            self._hand_settle(moved, u_old, sent=2)
            return Follow(self.steps - n0, self.steps >= self.step_limit, False, self.obs(cams=False))
        t = np.asarray(path.t, float)
        tt = np.arange(DT, t[-1] + 1e-9, DT)
        if len(tt) == 0 or tt[-1] < t[-1] - 1e-6:
            tt = np.r_[tt, t[-1]]
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
            rows.append(self._row())
            qs.append({a: self.cmd_q[a].copy() for a in SIDES})
            us.append({a: self.cmd_u[a].copy() for a in SIDES})
        stopped = blocked = False
        chunk = 3 if stop is not None else self.frame_every
        for i in range(0, len(rows), chunk):
            self._send(rows[i : i + chunk])
            if self.steps >= self.step_limit or self.ended:
                break
            last = min(i + chunk, len(rows)) - 1
            lag = max(
                [np.abs(np.asarray(self.state[f"{a}_arm_joint_state"], float) - qs[last][a]).max() for a in path.q]
                or [0.0]
            )
            if lag > BLOCKED and i + chunk < len(rows):  # the arm is held back (it hit something): stop there
                blocked = True
                self.cmd_u = us[last]
                print(f"adapter: arm blocked ({lag:.2f} rad behind its path); stopped the motion there", flush=True)
                break
            if stop is not None and i + chunk < len(rows) and stop(self.obs(cams=False)):
                stopped = True
                self.cmd_u = us[last]
                break
        if stopped or blocked:  # hold the posture reached
            for s in path.q:
                self.cmd_q[s] = np.asarray(self.state[f"{s}_arm_joint_state"], float)
        else:
            moved = [a for a in path.hand if abs(float(self.cmd_u[a][0] - u_start[a][0])) > 0.02]
            if moved:
                self._hand_settle(moved, u_start, sent=int(HAND_LEAD / DT))
            if path.q:
                self._settle(list(path.q))
        return Follow(self.steps - n0, self.steps >= self.step_limit, stopped, self.obs(cams=False))

    def _settle(self, arms, tol=0.01, max_steps=4):
        """End on the last waypoint: hold its command a few steps while a joint is still more than `tol` rad off
        and still closing in (a hand pressed against something stops closing in: it stays where it is)."""
        prev = None
        for _ in range(max_steps):
            if self.ended or self.steps >= self.step_limit:
                return
            err = max(np.abs(np.asarray(self.state[f"{s}_arm_joint_state"], float) - self.cmd_q[s]).max() for s in arms)
            if err < tol or (prev is not None and err > 0.7 * prev):
                return
            prev = err
            self._send([self._row()])

    def _hand_settle(self, sides, u_old=None, sent=0, max_steps=15, tol=0.0015, still=0.0005):
        """After a gripper command, hold still until the fingers reach it
        or stop on something (an object between them), so what follows (a lift) starts with the object gripped.
        Official observation (no finger travel measured): hold as long as the fingers take to travel that far."""
        if self.official:
            far = (
                max(abs(finger_target(self.cmd_u[s][0]) - finger_target(u_old[s][0])) for s in sides)
                if u_old
                else TRAVEL
            )
            for _ in range(min(max_steps, max(0, int(np.ceil(far / FINGER_STEP)) - sent))):
                if self.ended or self.steps >= self.step_limit:
                    return
                self._send([self._row()])
            return
        prev = None
        for _ in range(max_steps):
            if self.ended or self.steps >= self.step_limit:
                return
            now = {s: float(np.mean(self.state[f"{s}_gripper_joint_state"])) for s in sides}
            print(f"fingers {sent} {' '.join(f'{s} {v:.4f}' for s, v in now.items())}", flush=True)
            sent += 1
            reached = all(abs(now[s] - np.clip(finger_target(self.cmd_u[s][0]), 0.0, TRAVEL)) < tol for s in sides)
            if reached or (prev is not None and all(abs(now[s] - prev[s]) < still for s in sides)):
                return
            prev = now
            self._send([self._row()])

    def robot_mask(self, cam, obs):
        return None  # the skills remove the robot with its model at the measured joints

    def finish(self, out):
        """End the episode: the env's native check with its final flag (RoboDojo scores success at any step, so
        holding still to the limit changes nothing), then save the head-camera frames and the actions."""
        out = Path(out)
        try:
            result = self._req(op="finish")
        except RuntimeError as e:
            result = {"success": False, "reason": "environment_error", "error": str(e)}
        self.sim.close()
        try:
            import imageio

            if self.frames:
                imageio.mimwrite(out / "video.mp4", self.frames, fps=round(1 / (DT * self.frame_every)))
        except Exception as e:  # noqa: BLE001
            print(f"video not saved: {e!r}", flush=True)
        if self.actions:
            keys = [f"{s}_arm_joint_state" for s in SIDES] + [f"{s}_ee_joint_state" for s in SIDES]
            np.save(out / "actions.npy", np.array([sum((a[k] for k in keys), []) for a in self.actions], np.float32))
        return dict(
            success=bool(result.get("success")),
            score=result.get("score"),
            reason=result.get("reason"),
            env_steps=result.get("steps"),
            steps=self.steps,
            step_limit=self.step_limit,
            task=self.task,
            layout=self.layout,
            seed=self.seed,
            mode=self.mode,
            obs="official" if self.official else "full",
            error=result.get("error"),
            infra=self.sim.lost,
            **{k: result[k] for k in ("truth",) if k in result},
        )  # fmt: skip  (truth: dev episodes only)


class Clocked(RoboDojo):
    """The RoboDojo adapter on a continuously advancing vector env: the simulator's step counter is the budget."""

    def _take(self, rep):
        super()._take(rep)
        if not getattr(self, "ended", False):  # after a native terminal, keep the outcome-hiding behaviour
            self.steps = int(rep["steps"])

    def _req(self, **msg):
        if msg["op"] == "observe" and msg.get("render", True) and getattr(self, "rendered", None) is not None:
            rep, self.rendered = self.rendered, None
            return rep
        rep = super()._req(**msg)
        if msg["op"] == "finish":
            self.clock_result = {k: rep[k] for k in ("physical_steps", "idle_steps", "commanded_steps")}
        return rep

    def finish(self, out):
        result = super().finish(out)
        result.update(getattr(self, "clock_result", {}), budget_basis="commanded_actions")
        return result

    def obs(self, cams=True):
        if not self.ended:
            rep = self._req(op="observe", render=bool(cams))
            self._take(rep)
            self.rendered = rep if cams and self.official else None  # obs=official observes again: give it this reply
        try:
            return super().obs(cams=cams)
        finally:
            self.rendered = None

    def _send(self, rows):
        if not self.ended:
            self._take(self._req(op="observe", render=False))
        return super()._send(rows)


def make(
    task,
    layout,
    seed,
    sim,
    mode="dev",
    fix="",
    out=None,
    frame_every="8",
    obs="full",
    ws="/workspace",
    information="observations",
    suite="legacy",
    clocked=False,
    video=None,
    vendor=None,
    **_,
):
    if mode not in ("dev", "eval"):
        raise ValueError("mode is dev or eval")
    return (Clocked if clocked else RoboDojo)(
        task,
        int(layout),
        int(seed),
        int(sim),
        mode,
        tuple(f for f in fix.split(",") if f),
        out,
        int(frame_every),
        obs,
        ws,
        information,
        suite,
        video,
        vendor,
    )
