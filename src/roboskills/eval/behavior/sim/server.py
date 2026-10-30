"""BEHAVIOR-1K's simulator: one OmniGibson app holding one challenge task instance, served to `robot.py` over a socket.

    /opt/conda/envs/behavior/bin/python sim/server.py TASK INSTANCE --port P   # in the official image (launch.py)

The scene, task instance and robot are the 2026 challenge's own: `omnigibson.eval.evaluator.Evaluator` builds them as
`omnigibson.eval.eval` does (partial scene load, the R1Pro of `eval/r1pro.yaml` with its 250 kg base and head aperture,
transition rules on, CPU PhysX, 30 Hz control over 120 Hz physics, the challenge's step limit) and loads the instance
with `load_task_instance` between two resets. One change, which the challenge allows: absolute joint-position
controllers for the arms, trunk and fingers and a position-mode holonomic base (OmniGibson's `r1pro_primitives.yaml`
controllers), so arm commands are joint targets and the base follows world waypoints. Assisted grasping stays on.

Each control step: the arms' joint targets, both grippers' finger targets, the trunk held at its reset pose, and a base
target [x, y, yaw] in the world turned into the controller's per-step displacement from the measured base pose (the
base tracks it with feedback; no target: it stays where it is). The BDDL goal is checked every step, as the challenge's
PredicateGoal does: the episode ends when it holds, or at the step limit; nothing moves after that.

Only robot state goes out: joints, finger travel, the base's and the arms' mounting link's world poses, rendered RGB-D
with intrinsics and camera poses, the step count, the end flag; and the task's BDDL definition (its objects, initial
facts and goal: no poses), which the instruction is made from. Object state, predicate values and scores never leave
the process, except `finish`'s verdict. A robot whose state turns nonfinite (the physics blew up) ends the episode.

Wire: `roboskills.eval.wire` (an 8-byte length, a JSON header, raw arrays), one client at a time.
Ops: ping; reset -> robot facts, the BDDL problem, the first state with cameras; act {rows, render} -> state;
observe -> state with cameras; links {names} -> world poses of robot links; finish {hold} -> verdict and video frames.
"""

import argparse
import faulthandler
import json
import math
import select
import signal
import socket
import sys
import time
import traceback
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[4]))  # the directory holding `roboskills` (mounted read-only)
from roboskills.eval.wire import recv, send  # noqa: E402


CAMERAS = {
    "head": "robot_r1:zed_link:Camera:0",
    "left_wrist": "robot_r1:left_realsense_link:Camera:0",
    "right_wrist": "robot_r1:right_realsense_link:Camera:0",
}  # fmt: skip  (eval/r1pro.yaml camera roles)
SIDES = ("left", "right")
GL_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0])  # USD camera axes (x right, y up, -z forward) -> OpenCV (y down, z forward)
SIZE = 720  # camera pixels per side: the challenge's head resolution (its wrist cameras are 480; here 720 too)
VIDEO = 4  # head frames downsampled by this for the video
GROW = 2  # pixels the robot's mask is grown by (its silhouette's antialiased edge)
OPEN_PAST = 0.01  # m past a finger's upper limit for "fully open": assisted grasping releases only at or above it
TILT_RISE = 4.0  # degrees a drive may lean the base past its tilt at the start (driving rocks it up to about 3)
GRACE = (0.1, 0.17)  # m, rad: how far a drive may move while still touching what it touched at its start
STALL = 3  # steps a drive may make no progress while the robot touches something
SOFT = ("rug", "carpet", "mat", "doormat", "bath_rug", "floor_mat", "bathmat")  # floor coverings: the base rolls over
BASE_STEP, TURN_STEP = 0.03, 0.05  # m, rad: the most the base target leads its pose (a wall is pushed gently)


def controllers():
    """`eval/r1pro.yaml`'s arm and trunk controllers; the fingers and base as absolute position controllers."""
    joint = dict(name="JointController", motor_type="position", command_input_limits=None,
                 command_output_limits=None, use_impedances=False, use_delta_commands=False)  # fmt: skip
    return dict(
        arm_left=dict(joint, pos_kp=150), arm_right=dict(joint, pos_kp=150), trunk=dict(joint, pos_kp=150),
        gripper_left=joint, gripper_right=joint, camera=dict(name="NullJointController"),
        base=dict(name="HolonomicBaseJointController", motor_type="position", command_input_limits=None,
                  use_impedances=False),
    )  # fmt: skip


def boot(task, instance):
    """The challenge's evaluator for `task`, with public-test instance `instance` (301..320) loaded. The cameras give
    RGBDFullResWrapper's RGB and linear depth, made at SIZE: resizing them after they exist (as the wrapper does, to
    head 720, wrists 480) fails in Isaac Sim 5.1's Replicator here ("Invalid NodeObj" detaching an annotator)."""
    from omegaconf import OmegaConf
    from omnigibson.macros import gm

    gm.HEADLESS = True
    gm.RENDER_VIEWER_CAMERA = False
    import omnigibson.eval.evaluator as ev
    from omnigibson.eval.utils.eval_utils import DEFAULT_EVAL_SEED, seed_everything

    robot = OmegaConf.to_container(OmegaConf.load(ev.DEFAULT_ROBOT_CONFIG_PATH), resolve=True)
    robot["controller_config"] = controllers()
    robot["obs_modalities"] = ["proprio", "rgb", "depth_linear"]
    robot["sensor_config"]["VisionSensor"]["sensor_kwargs"] = {"image_height": SIZE, "image_width": SIZE}
    cfg = OmegaConf.create(
        {
            "env_wrapper": {"_target_": "omnigibson.envs.EnvironmentWrapper"},  # (the cameras are set above)
            "policy_name": "local",
            "model": {"_target_": "omnigibson.eval.policies.LocalPolicy", "action_dim": None},
            "headless": True,
            "partial_scene_load": True,
            "max_steps": None,
            "write_video": False,
            "mode": "public_test",
            "seed": seed_everything(DEFAULT_EVAL_SEED),
            "task": {"name": task},
            "robot": robot,
        }
    )
    evaluator = ev.Evaluator(cfg)
    evaluator.reset()  # as eval.py: a clean robot before the instance loads
    evaluator.load_task_instance(int(instance))
    evaluator.reset()
    return evaluator


def pose(pos, quat):
    """4x4 from a position and an xyzw quaternion (torch or numpy)."""
    from scipy.spatial.transform import Rotation

    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(np.asarray(quat, float)).as_matrix()
    T[:3, 3] = np.asarray(pos, float)
    return T


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class Episode:
    """The loaded instance, stepped one control step at a time."""

    def __init__(self, evaluator, task, instance, frame_every):
        import omnigibson as og

        self.og, self.ev, self.task_name, self.instance = og, evaluator, task, int(instance)
        self.env = evaluator.env.env if hasattr(evaluator.env, "env") else evaluator.env  # the og.Environment
        self.task, self.robot = self.env.task, evaluator.robot
        r = self.robot
        self.names = list(r.joints)
        self.step_limit = int(self.task._termination_conditions["timeout"]._max_steps)
        self.idx = {k: np.asarray(v.cpu() if hasattr(v, "cpu") else v, int) for k, v in r.controller_action_idx.items()}
        self.arm_idx = {s: np.asarray(r.arm_control_idx[s].cpu(), int) for s in SIDES}
        self.finger_idx = {s: np.asarray(r.gripper_control_idx[s].cpu(), int) for s in SIDES}
        self.trunk_idx = np.asarray(r.trunk_control_idx.cpu(), int)
        lo, hi = (np.asarray(x.cpu(), float) for x in (r.joint_lower_limits, r.joint_upper_limits))
        self.lo, self.hi = lo, hi
        q = self.q()
        self.trunk = q[self.trunk_idx].copy()  # held here all episode: the arms' mount stays fixed on the base
        self.cmd = {"q": {s: q[self.arm_idx[s]].copy() for s in SIDES},
                    "u": {s: float(np.mean((q[self.finger_idx[s]] - lo[self.finger_idx[s]]) /
                                           (hi[self.finger_idx[s]] - lo[self.finger_idx[s]]))) for s in SIDES}}  # fmt: skip
        self.steps, self.done, self.success_step, self.broken = 0, False, None, None
        self.frame_every, self.frames = int(frame_every), []
        self.mount = r.trunk_link_names[-1]  # the trunk's last link: both arms hang off it
        self.sensors = {role: r.sensors[name] for role, name in CAMERAS.items()}  # robot.sensors keys
        self.seg = {}  # Replicator's instance-id segmentation on each camera's render product (the RGB-D's frame)
        from omnigibson import lazy

        with og.sim.editing_usd():
            for role, sensor in self.sensors.items():
                self.seg[role] = lazy.omni.replicator.core.AnnotatorRegistry.get_annotator(
                    "instance_id_segmentation_fast"
                )
                self.seg[role].attach([sensor.render_product])
        self.robot_prims = r.prim_path + "/"
        self.scene_prims = r.prim_path.rsplit("/", 1)[0] + "/"
        self.touching, self.log_contacts, self.drive_contacts = set(), set(), set()
        self.driving, self.stopped = False, None
        self.goal_s = []

    # ---- state

    def q(self):
        return np.asarray(self.robot.get_joint_positions().cpu(), float)

    def base(self):
        pos, quat = self.robot.get_position_orientation()
        T = pose(pos.cpu(), quat.cpu())
        return np.array([T[0, 3], T[1, 3], math.atan2(T[1, 0], T[0, 0])]), T

    def tilt(self):
        """Degrees the base leans from upright (its z axis from the world's)."""
        return math.degrees(math.acos(float(np.clip(self.base()[1][2, 2], -1.0, 1.0))))

    def contacts(self):
        """Names of what the robot touched over the last control step, besides the floor, floor coverings (SOFT) and what
        its hands hold."""
        from omnigibson.utils.usd_utils import RigidContactAPI

        scene = self.env.scene.idx
        if getattr(self, "rows", None) is None:
            self.rows = RigidContactAPI.get_contact_row_indices(scene, [self.robot])
        held = {o.name for o in (getattr(self.robot, "_ag_obj_in_hand", None) or {}).values() if o is not None}
        out = set()
        for _, other in RigidContactAPI.get_contact_pairs(scene, self.rows, None, current_only=False):
            if other.startswith(self.robot_prims):
                continue
            name = other[len(self.scene_prims) :].split("/")[0] if other.startswith(self.scene_prims) else other
            kind = name.rsplit("_", 2)[0] if name.count("_") >= 2 else name
            if not (name.startswith("floors") or "ground" in name.lower() or name in held or kind in SOFT):
                out.add(name)
        return out

    def link(self, name):
        pos, quat = self.robot.links[name].get_position_orientation()
        return pose(pos.cpu(), quat.cpu())

    def state(self):
        if self.broken:  # the physics blew up: the last sane state, and the episode over
            return dict(self.last, steps=self.steps, done=True)
        q = self.q()
        self.last = {
            "q": {s: q[self.arm_idx[s]] for s in SIDES},
            "fingers": {s: q[self.finger_idx[s]] for s in SIDES},
            "trunk": q[self.trunk_idx],
            "base": self.base()[0],
            "mount": self.link(self.mount),
            "tilt": self.tilt(),
            "stopped": self.stopped,
            "steps": self.steps,
            "step_limit": self.step_limit,
            "done": self.done,
        }
        return self.last

    def render(self, n=3):
        """Render the current state (the renderer needs a couple of passes to catch up with a moved stage)."""
        for _ in range(n):
            self.og.sim.render()

    def cameras(self):
        if self.broken and getattr(self, "last_cams", None):
            return self.last_cams
        self.render()
        out = {}
        for role, s in self.sensors.items():
            obs = s.get_obs()[0]
            rgb = np.asarray(obs["rgb"].cpu(), np.uint8)[..., :3]
            depth = np.asarray(obs["depth_linear"].cpu(), np.float32).squeeze()
            h, w = rgb.shape[:2]
            f = w * float(s.focal_length) / float(s.horizontal_aperture)  # square pixels
            K = np.array([[f, 0.0, (w - 1) / 2], [0.0, f, (h - 1) / 2], [0.0, 0.0, 1.0]])  # centres at integer pixels
            V = (
                np.asarray(s.camera_parameters["cameraViewTransform"], float).reshape(4, 4).T
            )  # world -> camera, as drawn
            T = (
                np.linalg.inv(V)
                if np.any(V)
                else pose(*(np.asarray(x.cpu(), float) for x in s.get_position_orientation()))
            )
            out[role] = {"rgb": rgb, "depth": depth, "K": K, "T": T @ GL_TO_CV, "robot": self.robot_mask(role)}
        self.last_cams = out
        return out

    def robot_mask(self, role):
        """(H, W) bool: where camera `role` sees the robot (any prim under the robot's), grown by GROW pixels."""
        import cv2

        seg = self.seg[role].get_data()
        ids, labels = np.asarray(seg["data"]).squeeze(), seg["info"]["idToLabels"]
        mine = [int(k) for k, v in labels.items() if str(v).startswith(self.robot_prims)]  # mesh prim paths
        mask = np.isin(ids, mine).astype(np.uint8)
        return cv2.dilate(mask, np.ones((2 * GROW + 1,) * 2, np.uint8)).astype(bool) if GROW else mask.astype(bool)

    # ---- acting

    def action(self, row):
        """One native 25-D action from a row {q: {arm: 7 targets}, u: {arm: opening 0..1}, base: [x, y, yaw] | None}."""
        import torch as th

        a = np.zeros(int(self.robot.action_dim), np.float32)
        for s in SIDES:
            if s in row.get("q", {}):
                self.cmd["q"][s] = np.clip(
                    np.asarray(row["q"][s], float), self.lo[self.arm_idx[s]], self.hi[self.arm_idx[s]]
                )
            if s in row.get("u", {}):
                self.cmd["u"][s] = float(np.clip(row["u"][s], 0.0, 1.0))
            a[self.idx[f"arm_{s}"]] = self.cmd["q"][s]
            lo, hi = self.lo[self.finger_idx[s]], self.hi[self.finger_idx[s]]
            u = self.cmd["u"][s]
            a[self.idx[f"gripper_{s}"]] = hi + OPEN_PAST if u >= 0.999 else lo + u * (hi - lo)
        a[self.idx["trunk"]] = self.trunk
        target = row.get("base")
        if (
            target is not None
        ):  # the displacement to the target in the base's own frame (the controller's command), capped
            (x, y, yaw), _ = self.base()
            dx, dy = target[0] - x, target[1] - y
            d = math.hypot(dx, dy)
            if d > BASE_STEP:
                dx, dy = dx * BASE_STEP / d, dy * BASE_STEP / d
            c, s_ = math.cos(yaw), math.sin(yaw)
            a[self.idx["base"]] = [
                c * dx + s_ * dy,
                -s_ * dx + c * dy,
                np.clip(wrap(target[2] - yaw), -TURN_STEP, TURN_STEP),
            ]
        return th.as_tensor(a)

    def tick(self, row):
        """One control step (no rendering), then the BDDL goal as the challenge's PredicateGoal checks it. A drive stops
        (its target becomes the base's pose) at the step after the robot touches something (what it touched when the
        drive began only counts once the base has left GRACE behind: driving away from it is fine, shoving it along is
        not), leans TILT_RISE degrees more than then, or has not moved for STALL steps while touching anything: the base
        never pushes into furniture."""
        if row.get("base") is not None:
            (x, y, yaw), _ = self.base()
            if row.get("drive") or not self.driving:  # a drive begins: what the robot touches now is not in its way
                self.driving, self.drive_contacts, self.drive_tilt = True, set(self.touching), self.tilt()
                self.track, self.drive_start = [], (x, y, yaw)
            x0, y0, yaw0 = self.drive_start
            if math.hypot(x - x0, y - y0) > GRACE[0] or abs(wrap(yaw - yaw0)) > GRACE[1]:
                self.drive_contacts = set()
            self.track = (self.track + [(x, y, yaw)])[-STALL - 1 :]
            new, tilt = self.touching - self.drive_contacts, self.tilt()
            moved = np.ptp(np.array(self.track), axis=0) if len(self.track) > STALL else np.full(3, np.inf)
            far = math.hypot(row["base"][0] - x, row["base"][1] - y) > 0.01 or abs(wrap(row["base"][2] - yaw)) > 0.02
            stalled = bool(self.touching) and far and moved[:2].max() < 0.002 and moved[2] < 0.004
            if self.stopped is None and (new or tilt > self.drive_tilt + TILT_RISE or stalled):
                self.stopped = "contact" if new or stalled else "tilt"
                print(
                    f"base stopped at step {self.steps}: new {sorted(new)} touching {sorted(self.touching)} "
                    f"tilt {tilt:.1f} stalled {stalled}",
                    flush=True,
                )  # fmt: skip  (diagnosis only; never sent)
            if self.stopped:
                row = dict(row, base=None)
        else:
            self.driving = False
        self.env._pre_step(self.action(row))
        with self.og.sim.render_on_step(False):
            self.og.sim.step()
        self.steps += 1
        pos, quat = self.robot.get_position_orientation()
        if not (np.isfinite(self.q()).all() and np.isfinite(pos.cpu()).all() and np.isfinite(quat.cpu()).all()):
            self.broken, self.done = f"the robot's state became nonfinite at step {self.steps}", True
            print(self.broken, flush=True)
            return
        self.touching = self.contacts()
        if self.touching - self.log_contacts:  # a log for diagnosis only; never sent
            print(f"contact step {self.steps}: {sorted(self.touching)} tilt {self.tilt():.1f}", flush=True)
        self.log_contacts = self.touching
        t = time.perf_counter()
        won, _ = self.task.compiled_task.check_goal(self.task._evaluate_predicate)
        self.goal_s.append(time.perf_counter() - t)
        if won and self.success_step is None:
            self.success_step = self.steps
        if won or self.steps >= self.step_limit:
            self.done = True
        if self.frame_every and self.steps % self.frame_every == 0:
            self.frame()

    def frame(self, cams=None):
        """A head-camera video frame: rendered now, unless an observation just did."""
        if cams is None:
            self.render(1)
            rgb = np.asarray(self.sensors["head"].get_obs()[0]["rgb"].cpu(), np.uint8)[..., :3]
        else:
            rgb = cams["head"]["rgb"]
        self.frames.append(np.ascontiguousarray(rgb[::VIDEO, ::VIDEO]))

    def act(self, rows, render):
        self.stopped = None
        for row in rows:
            if self.done:
                break  # the episode is over: nothing runs
            self.tick(row)
        out = self.state()
        if render:
            out["cams"] = self.cameras()
        return out

    def finish(self, hold):
        """Hold the last command `hold` steps (the base where it is), then the BDDL goal now and over the episode."""
        for _ in range(int(hold)):
            if self.done:
                break
            self.tick({})
        won, status = (False, {}) if self.broken else self.task.compiled_task.check_goal(self.task._evaluate_predicate)
        if won and self.success_step is None:
            self.success_step = self.steps
        return {
            "success": self.success_step is not None,
            "broken": self.broken,
            "success_step": self.success_step,
            "goal_now": bool(won),
            "goal_satisfied": len(status.get("satisfied", [])),
            "goal_total": len(status.get("satisfied", [])) + len(status.get("unsatisfied", [])),
            "steps": self.steps,
            "step_limit": self.step_limit,
            "goal_check_ms": round(1000 * float(np.mean(self.goal_s)), 2) if self.goal_s else None,
            "frames": np.stack(self.frames) if self.frames else np.zeros((0, 1, 1, 3), np.uint8),
        }

    def facts(self):
        """What the host builds the robot model from: names, limits, the URDF, the held trunk pose."""
        r = self.robot
        ji = lambda names: [self.names.index(n) for n in names]  # noqa: E731
        return {
            "urdf": str(r.urdf_path),
            "joints": self.names,
            "lo": self.lo,
            "hi": self.hi,
            "arm_joints": {s: list(r.arm_joint_names[s]) for s in SIDES},
            "finger_joints": {s: [self.names[i] for i in self.finger_idx[s]] for s in SIDES},
            "trunk_joints": [self.names[i] for i in self.trunk_idx],
            "eef_links": {s: r.eef_link_names[s] for s in SIDES},
            "finger_links": {s: list(r.finger_link_names[s]) for s in SIDES},
            "mount": self.mount,
            "base_link": r.base_footprint_link_name,
            "reset_q": np.asarray(
                r.reset_joint_pos.cpu() if hasattr(r.reset_joint_pos, "cpu") else r.reset_joint_pos, float
            ),
            "joint_index": {n: ji([n])[0] for n in self.names},
        }

    def problem(self):
        """The task's BDDL problem (its objects, initial facts and goal), as the challenge defines it."""
        import bddl

        name, i = self.task.activity_name, self.task.activity_definition_id
        return (Path(bddl.__file__).parent / "activity_definitions" / name / f"problem{i}.bddl").read_text()


def handle(ep, req):
    op = req.get("op")
    if op == "ping":
        return {"pong": True, "task": ep.task_name, "instance": ep.instance}
    if op == "reset":  # the instance as loaded at boot; one episode per process
        if ep.steps:
            raise RuntimeError("one episode per simulator: start a new one for another")
        cams = ep.cameras()
        ep.frame(cams)
        return {**ep.state(), "cams": cams, "facts": ep.facts(), "problem": ep.problem()}
    if op == "act":
        return ep.act(req["rows"], bool(req.get("render")))
    if op == "observe":
        cams = ep.cameras()
        return {**ep.state(), "cams": cams}
    if op == "links":  # robot links only (for checking the robot model against the simulator)
        return {"poses": {n: ep.link(n) for n in req["names"]}}
    if op == "finish":
        return ep.finish(req.get("hold", 0))
    raise ValueError(f"unknown op {op!r}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("task")
    p.add_argument("instance", type=int, help="the challenge's task instance id (public test: 301..320)")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--frame-every", type=int, default=10, help="control steps per head-camera video frame")
    a = p.parse_args()
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    t0 = time.time()
    lst = socket.create_server(("127.0.0.1", a.port))  # bound first: a taken port fails fast, before the boot
    ep = Episode(boot(a.task, a.instance), a.task, a.instance, a.frame_every)
    print(f"ready {a.task} {a.instance} port {a.port} boot {time.time() - t0:.0f}s", flush=True)
    conn = None
    while True:
        ready, _, _ = select.select([lst] + ([conn] if conn else []), [], [], 60)
        if lst in ready:
            if conn:
                conn.close()
            conn, _ = lst.accept()
            continue
        if conn is None or conn not in ready:
            continue
        try:
            req = recv(conn)
        except (EOFError, OSError):
            conn.close()
            conn = None
            continue
        t = time.time()
        try:
            rep = handle(ep, req)
        except Exception as e:  # noqa: BLE001  (the reply carries it)
            traceback.print_exc()
            rep = {"error": f"{type(e).__name__}: {e}"}
        print(f"{req.get('op')} {len(req.get('rows', ()))} {time.time() - t:.2f}s", flush=True)
        try:
            send(conn, rep)
        except OSError:
            conn.close()
            conn = None
        if req.get("op") == "finish":
            print(json.dumps({k: v for k, v in rep.items() if k != "frames"}), flush=True)


if __name__ == "__main__":
    main()
