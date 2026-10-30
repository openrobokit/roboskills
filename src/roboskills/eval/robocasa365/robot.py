"""RoboCasa365 kitchens (robocasa/robocasa 4f8a298 = RoboCasa 1.0.1 on robosuite 1.5.2) behind the shared Robot contract.

RoboCasa builds the kitchen, the task, the robot's start and the success check; the robot is its PandaOmron (a Panda arm
on a torso lift on an Omron holonomic base). Control is RoboCasa's composite controller with EFE's standard-mode layout:
absolute arm joint targets, the gripper, base joint velocities, the torso, the hybrid-mode flag (13 channels). Only
rendered RGB-D, camera calibration, proprioception and robot geometry reach the agent.

Frames. The base's three joints (forward and side slides, then a yaw hinge 0.21 m behind the arm) move it in a fixed
joint frame J, the robot root's pose with the joints at zero. `Obs.base` is the yaw hinge's axis on the floor (the middle
of the footprint, where the base turns in place) with the base's heading; the arm's first joint stands 0.21 m ahead of it.
"""

import json
import os
import random
import threading
import xml.etree.ElementTree as ET
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from roboskills.eval.interface import Arm, Base, Camera, Follow, Hand, Obs
from roboskills.eval.libero.hand import AXES, WIDTH, finger_q, opening, radians_xml
from roboskills.eval.robocasa365 import TASKS


os.environ.setdefault("MUJOCO_GL", "egl")  # robosuite picks its renderer when imported; this runtime renders with EGL
SIZE = 512  # camera pixels per side
HOME = np.array([0.0, -1.5, 0.0, -3.0, 0.0, 1.9, 0.785])  # folded up over the shoulder: the least of the cameras' view
CAMERAS = ("robot0_agentview_left", "robot0_agentview_center", "robot0_agentview_right", "robot0_eye_in_hand")
VIDEO_CAMERA = "robot0_agentview_center"
VIDEO = 10  # control steps per video frame
TOL, FINGER_TOL = 0.005, 0.001  # rad, m/s: a motion has settled on its last waypoint and the fingers have stopped
BASE_TOL = (0.004, 0.005)  # m, rad: the base has settled on its last waypoint
SETTLE = 30  # control steps at most spent settling there
BLOCKED = (0.10, np.radians(10.0), 10)  # m, rad, steps: a base this far off its path this long is blocked
GAIN = 6.0  # 1/s: base feedback on its measured joints, on top of the path's velocity
EPS = np.array([0.02, 0.02, 0.03])  # m/s, m/s, rad/s: below these the base's friction feed-forward fades out
ROOT, BASE, YAW = "robot0_link0", "mobilebase0_base", "mobilebase0_joint_mobile_yaw"
BASE_JOINTS = ("mobilebase0_joint_mobile_forward", "mobilebase0_joint_mobile_side", YAW)
TORSO = "mobilebase0_joint_torso_height"
ROBOT = "robot0_base"  # the robot's root body: everything under it is the robot
# The arm: absolute joint targets (EFE's standard mode); delta while RoboCasa settles the scene with zero actions at
# reset, so that its settling holds the start posture instead of driving to the zero posture.
ARM = dict(type="JOINT_POSITION", input_max=10, input_min=-10, output_max=1.0, output_min=-1.0, kp=1500,
           damping_ratio=1, impedance_mode="fixed", kp_limits=[0, 2000], input_type="delta", interpolation=None,
           ramp_ratio=0.2, gripper={"type": "GRIP"})  # fmt: skip
TORSO_CTRL = dict(type="JOINT_POSITION", input_max=1, input_min=-1, output_max=1.0, output_min=-1.0, kp=2000,
                  damping_ratio=1, impedance_mode="fixed", input_type="delta", interpolation=None)  # fmt: skip


def pose(xpos, xmat):
    """A 4x4 pose from MuJoCo's position and (flat) rotation matrix."""
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = np.reshape(xmat, (3, 3)), xpos
    return T


def rot(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def heading(R):
    """The yaw of an upright rotation matrix (flat or 3x3)."""
    R = np.reshape(R, (3, 3))
    return np.arctan2(R[1, 0], R[0, 0])


def command(t, Q, start, now, lead, dt):
    """The target at `now` s on the path from `start` (at 0 s) through rows Q at times t, plus `lead` s of its
    velocity: the native controller's steady lag (kd/kp) behind a moving target, fed forward."""
    t, Q = np.r_[0.0, t], np.vstack([start, Q])
    at = lambda s: np.array([np.interp(s, t, Q[:, j]) for j in range(Q.shape[1])])
    return at(now) + lead * (at(now + dt / 2) - at(now - dt / 2)) / dt


class Kinematics:
    """The base's joints [forward, side, yaw] <-> its pose [x, y, yaw] in the world (`Obs.base`). The slides move along
    the joint frame J's axes (origin c, heading psi); the yaw hinge turns about its anchor p (in J, at zero yaw) moved
    by the slides, so the base point is c + R(psi) (slides + p) and its heading psi + yaw."""

    def __init__(self, c, psi, p):
        self.c, self.psi, self.p = np.asarray(c, float), float(psi), np.asarray(p, float)

    def pose(self, q):
        return np.r_[self.c + rot(self.psi) @ (q[:2] + self.p), self.psi + q[2]]

    def joints(self, b, near=None):
        """Joints for world poses b (..., 3); headings taken by whole turns to start nearest the joint yaw `near`."""
        b = np.asarray(b, float)
        q = np.empty_like(b)
        q[..., :2] = (b[..., :2] - self.c) @ rot(self.psi) - self.p
        q[..., 2] = b[..., 2] - self.psi
        if near is not None:
            first = q.reshape(-1, 3)[0, 2]
            q[..., 2] -= 2 * np.pi * np.round((first - near) / (2 * np.pi))
        return q


def base_action(v, theta, lo, hi, friction):
    """RoboCasa's base action for joint velocities v [forward, side, yaw]: its legacy base controller turns the action's
    x, y by theta (the base's heading now less at reset) into the joint frame and scales [-1, 1] onto the velocity
    actuators' ranges [lo, hi]. The joints' dry friction stalls a velocity servo below friction (m/s, rad/s): fed
    forward, faded out near zero velocity so that the base rests still."""
    v = np.asarray(v, float)
    v = v + friction * np.clip(v / EPS, -1.0, 1.0)
    v = np.clip(v, lo, hi)
    a = (v - (hi + lo) / 2) / ((hi - lo) / 2)
    a[:2] = rot(-theta) @ a[:2]
    return a


def arm_xml(source, root):
    """The robot's MJCF cut to the subtree under body `root`, placed at the origin: RoboKit's arm model. Actuators and
    sensors of what was cut away go too, and so do meshes no geom uses."""
    xml = radians_xml(source)
    body = next(b for b in xml.iter("body") if b.get("name") == root)
    world = xml.find("worldbody")
    for child in list(world):
        world.remove(child)
    body.set("pos", "0 0 0")
    for key in ("quat", "euler", "axisangle", "xyaxes", "zaxis"):
        body.attrib.pop(key, None)
    world.append(body)
    joints = {j.get("name") for j in body.iter("joint")}
    sites = {s.get("name") for s in body.iter("site")}
    meshes = {g.get("mesh") for g in body.iter("geom") if g.get("mesh")}
    for tag, keep in (("actuator", lambda e: e.get("joint") in joints), ("sensor", lambda e: e.get("site") in sites)):
        for parent in xml.findall(tag):
            for e in list(parent):
                if not keep(e):
                    parent.remove(e)
    for asset in xml.findall("asset"):
        for e in list(asset):
            if e.tag == "mesh" and e.get("name") not in meshes:
                asset.remove(e)
    for tag in ("contact", "equality", "tendon"):
        for e in xml.findall(tag):
            xml.remove(e)
    return xml


def scene(task, seed):
    """RoboCasa's `task` kitchen with its PandaOmron, reset with `seed` (layout, style, robot start, objects)."""
    import robocasa  # noqa: F401  (registers the kitchen environments)
    import robosuite
    from robocasa.models.fixtures import counter
    from robosuite.controllers import load_composite_controller_config

    # RoboCasa drops duplicate counter-top regions through a set of XML elements, which iterates in memory-address
    # order: a seed then gives one of several scenes from process to process. Keep their order instead (one dict).
    counter.set = lambda items=(): dict.fromkeys(items)
    random.seed(seed)  # some RoboCasa placement code draws from the global generators
    np.random.seed(seed)
    config = load_composite_controller_config(controller=None, robot="PandaOmron")
    config["body_parts"]["right"] = dict(ARM)
    config["body_parts"]["torso"] = dict(TORSO_CTRL)
    env = robosuite.make(task, robots="PandaOmron", controller_configs=config, has_renderer=False,
                         has_offscreen_renderer=True, use_camera_obs=False, ignore_done=True, seed=seed,
                         translucent_robot=False)  # fmt: skip
    env.reset()
    parts = env.robots[0].composite_controller.part_controllers
    for part in ("right", "torso"):
        parts[part].input_type = "absolute"
    return env


class RoboCasa:
    REMOTE = ("replay",)  # served beside the Robot contract (eval/serve.py)

    def __init__(self, task: str, seed: int, out: str):
        import mujoco
        from robosuite.renderers.context.egl_context import EGL

        self.task, self.seed, self.out = task, int(seed), Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.step_limit = TASKS[task][0]
        self._lock, self._depth = threading.RLock(), 0
        try:
            self.env = env = scene(task, self.seed)
        finally:
            EGL.eglReleaseThread()  # the reset made a new EGL context current in this thread: let it go
        sim, robot = env.sim, env.robots[0]
        m, d = sim.model, sim.data
        self.instruction = env.get_ep_meta()["lang"]
        self.layout, self.style = env.layout_id, env.style_id
        self.dt, self.steps, self.ended = float(env.control_timestep), 0, False
        self.actions, self.frames, self._masks = [], [], {}
        self.robot = robot
        self.split = robot.composite_controller._action_split_indexes
        control = robot.part_controllers
        self.lead = np.asarray(control["right"].kd, float) / np.asarray(control["right"].kp, float)
        self.qpos = np.asarray(robot._ref_joint_pos_indexes)
        self.fingers = (robot._ref_gripper_joint_pos_indexes["right"], robot._ref_gripper_joint_vel_indexes["right"])
        ids = [m.joint_name2id(j) for j in BASE_JOINTS]
        self.bq, self.bv = m.jnt_qposadr[ids], m.jnt_dofadr[ids]
        self.torso = d.qpos[m.jnt_qposadr[m.joint_name2id(TORSO)]].copy()  # held at its reset height
        actuators = [m.actuator_name2id(f"mobilebase0_actuator_mobile_{n}") for n in ("forward", "side", "yaw")]
        self.vlo, self.vhi = m.actuator_ctrlrange[actuators].T
        self.friction = m.dof_frictionloss[self.bv] / m.actuator_gainprm[actuators, 0]  # stall velocity of each servo
        self.base_ctrl = control["base"]
        self.center = m.site_name2id(f"{self.base_ctrl.naming_prefix}center")
        body = m.body_name2id(BASE)
        parent, local = m.body_parentid[body], np.zeros(9)
        mujoco.mju_quat2Mat(local, m.body_quat[body])
        R = d.xmat[parent].reshape(3, 3)  # the joint frame J: the base body's frame with its joints at zero
        self.kin = Kinematics((d.xpos[parent] + R @ m.body_pos[body])[:2], heading(R @ local.reshape(3, 3)),
                              m.jnt_pos[m.joint_name2id(YAW)][:2])  # fmt: skip
        self.base_cmd = d.qpos[self.bq].copy()  # the joint targets the base holds
        self.robot_geoms = self._robot_geoms()

        gripper = robot.gripper["right"]
        ee = m.body_name2id(gripper.root_body)
        site = robot.eef_site_id["right"]
        sim.forward()
        tcp = np.eye(4)
        tcp[:3, :3] = AXES  # +x out of the hand, +y the jaws' squeeze axis
        tcp[:3, 3] = d.xmat[ee].reshape(3, 3).T @ (d.site_xpos[site] - d.xpos[ee])  # the grip site: between pads
        hand = Hand("jaw", tcp, np.ones(1), np.zeros(1), joints=list(gripper.joints), model_q=finger_q,
                    width=WIDTH)  # fmt: skip
        model = self.out / "robot0.xml"
        model.write_text(ET.tostring(arm_xml(robot.robot_model.get_xml(), ROOT), encoding="unicode"))
        joints = list(robot.robot_model.joints)
        lo, hi = np.array([m.jnt_range[m.joint_name2id(j)] for j in joints]).T
        q = d.qpos[self.qpos].copy()
        self.root = m.body_name2id(ROOT)
        self.arms = {"right": Arm("right", str(model), joints, gripper.root_body, pose(d.xpos[self.root],
                                  d.xmat[self.root]), lo, hi, HOME.copy(), hand)}  # fmt: skip
        self.cmd = {"right": (q, hand.open.copy())}
        self.base = Base(vmax=0.5, wmax=1.0, radius=0.44)
        self.frames.append(self._frame())  # rendering also leaves the EGL context released for the serving thread

    def _robot_geoms(self):
        """Ids of every geom on the robot: base, torso, arm and gripper (the bodies under its root)."""
        m = self.env.sim.model
        root = m.body_name2id(ROBOT)
        bodies = [b for b in range(m.nbody) if self._under(b, root)]
        return np.flatnonzero(np.isin(m.geom_bodyid, bodies))

    def _under(self, b, root):
        parent = self.env.sim.model.body_parentid
        while b > 0 and b != root:
            b = parent[b]
        return b == root

    @contextmanager
    def _gl(self):
        """Robosuite's EGL context, current in this thread for the block and released after: the launcher observes from
        its own threads, and an EGL context is current in one thread at a time. Making it current and releasing it cost
        tens of ms, so an observation or a whole motion holds it once (nested blocks reuse it)."""
        from robosuite.renderers.context.egl_context import EGL

        with self._lock:
            self._depth += 1
            try:
                if self._depth == 1:
                    self.env.sim._render_context_offscreen.gl_ctx.make_current()
                yield
            finally:
                self._depth -= 1
                if self._depth == 0:
                    EGL.eglReleaseThread()

    def _render(self, **kw):
        with self._gl():
            return self.env.sim.render(**kw)

    def _frame(self):
        self.env.sim.forward()
        return self._render(width=256, height=256, camera_name=VIDEO_CAMERA)[::-1].copy()

    def _camera(self, name):
        from robosuite.utils.camera_utils import get_real_depth_map
        from scipy.ndimage import binary_dilation

        sim = self.env.sim
        rgb, depth = self._render(width=SIZE, height=SIZE, camera_name=name, depth=True)
        seg = self._render(width=SIZE, height=SIZE, camera_name=name, segmentation=True)[::-1]
        # the robot's pixels, grown by 2 px over the anti-aliased edges where segment colours blend
        self._masks[name] = binary_dilation((seg[..., 0] == 5) & np.isin(seg[..., 1], self.robot_geoms), iterations=2)
        cid = sim.model.camera_name2id(name)
        f = 0.5 * SIZE / np.tan(np.radians(sim.model.cam_fovy[cid]) / 2)
        K = np.array([[f, 0.0, (SIZE - 1) / 2], [0.0, f, (SIZE - 1) / 2], [0.0, 0.0, 1.0]])
        T = pose(sim.data.cam_xpos[cid], sim.data.cam_xmat[cid]) @ np.diag([1.0, -1.0, -1.0, 1.0])
        return Camera(rgb[::-1].copy(), get_real_depth_map(sim, depth[::-1]).astype(np.float32), K, T)

    def _sync(self, full=False):
        """Bodies (and with `full`, cameras) where the joints are now; the arm model's root follows the base and torso."""
        import mujoco

        sim = self.env.sim
        if full:
            sim.forward()
        else:
            mujoco.mj_kinematics(sim.model._model, sim.data._data)
        d = sim.data
        self.arms["right"].T_world_base[...] = pose(d.xpos[self.root], d.xmat[self.root])

    def _base_pose(self):
        b = self.kin.pose(self.env.sim.data.qpos[self.bq])
        b[2] = wrap(b[2])
        return b

    def obs(self, cams: bool = True) -> Obs:
        self._sync(full=cams)
        cameras = {}
        if cams:
            with self._gl():
                cameras = {name: self._camera(name) for name in CAMERAS}
        d = self.env.sim.data
        return Obs(cameras, {"right": d.qpos[self.qpos].copy()}, {"right": opening(d.qpos[self.fingers[0]])},
                   self.steps, self.step_limit, self.instruction, self._base_pose())  # fmt: skip

    def robot_mask(self, cam: str, obs: Obs):
        """The robot's pixels (base, torso, arm, gripper) in `cam` at the last observation with cameras."""
        return self._masks[cam].copy() if obs.cams and cam in self._masks else None

    def _action(self, q, u, v):
        """The 13-channel native action: arm joint targets, gripper (-1 open .. 1 close for jaw command u 0 closed ..
        1 open, by its sign), base joint velocities v, the torso's reset height, arm mode (-1)."""
        d = self.env.sim.data
        theta = heading(d.site_xmat[self.center]) - heading(self.base_ctrl.init_ori)  # as the base controller turns it
        a = np.zeros(self.split["torso"][1] + 1)
        a[slice(*self.split["right"])] = q
        a[slice(*self.split["right_gripper"])] = 1.0 - 2.0 * np.clip(u, 0.0, 1.0)
        a[slice(*self.split["base"])] = base_action(v, theta, self.vlo, self.vhi, self.friction)
        a[slice(*self.split["torso"])] = self.torso
        a[-1] = -1.0
        return a

    def _step(self, cmd, v):
        if self.ended:
            return
        q, u = cmd["right"]
        self._native(self._action(q, u, v))

    def _native(self, a):
        """One control step of the native action `a`, counted, recorded and filmed as every step."""
        self.env.step(a)
        self.steps += 1
        self.actions.append(a)
        if self.steps % VIDEO == 0:
            self.frames.append(self._frame())
        self.ended = self.steps >= self.step_limit

    def replay(self, actions):
        """Step a finished episode's native actions (its `actions.npy`) as they were sent; proprioception after. The
        arm and hand then hold the last action's targets, and the base where it stands."""
        actions = np.asarray(actions, float)
        with self._gl():  # for the video frames on the way
            for a in actions:
                if self.ended:
                    break
                self._native(a.copy())
        if self.actions:
            a = self.actions[-1]
            u = (1.0 - a[slice(*self.split["right_gripper"])]) / 2  # the jaw command `_action` sent as 1 - 2u
            self.cmd["right"] = (a[slice(*self.split["right"])].copy(), u)
            self.base_cmd = self.env.sim.data.qpos[self.bq].copy()
        return self.obs(cams=False)

    def _settled(self, path, cmd, qb):
        d = self.env.sim.data
        arm = all(np.abs(d.qpos[self.qpos] - cmd[s][0]).max() < TOL for s in path.q)
        hand = all(np.abs(d.qvel[self.fingers[1]]).max(initial=0.0) < FINGER_TOL for s in path.hand)
        e = qb - d.qpos[self.bq]
        base = path.base is None or (np.linalg.norm(e[:2]) < BASE_TOL[0] and abs(e[2]) < BASE_TOL[1])
        return arm and hand and base

    def follow(self, path, stop=None) -> Follow:
        with self._gl():  # for the video frames on the way
            return self._follow(path, stop)

    def _follow(self, path, stop):
        t = np.asarray(path.t, float)
        end = int(np.ceil(t[-1] / self.dt - 1e-9))
        before, stopped, blocked, off, cmd = self.steps, False, False, 0, dict(self.cmd)
        d = self.env.sim.data
        Qb = None if path.base is None else self.kin.joints(path.base, near=self.base_cmd[2])  # its path in its joints
        qb = self.base_cmd
        for k in range(1, end + SETTLE + 1):
            if self.ended:
                break
            now = k * self.dt
            for s, (q, u) in self.cmd.items():
                if s in path.q:
                    q = command(t, path.q[s], q, now, self.lead, self.dt)
                if s in path.hand:
                    u = command(t, path.hand[s], u, now, 0.0, self.dt)
                cmd[s] = (q, u)
            v = np.zeros(3)
            if Qb is not None:  # where the base should be as this step starts, and its velocity over the step
                qb = command(t, Qb, self.base_cmd, now - self.dt, 0.0, self.dt)
                v = (command(t, Qb, self.base_cmd, now, 0.0, self.dt) - qb) / self.dt
            if k > end and self._settled(path, cmd, qb):
                break
            e = qb - d.qpos[self.bq]
            off = off + 1 if np.linalg.norm(e[:2]) > BLOCKED[0] or abs(e[2]) > BLOCKED[1] else 0
            if blocked := off >= BLOCKED[2]:  # a blocked base ends the motion where it stands
                break
            self._step(cmd, v + GAIN * e)
            if stop is not None and stop(self.obs(False)):
                stopped = True
                break
        measured = self.obs(False)
        here = stopped or blocked
        for s, (q, u) in cmd.items():  # hold the last waypoint, or where a stopped arm is
            self.cmd[s] = (measured.q[s] if here else np.asarray(path.q[s][-1], float), u) if s in path.q else (q, u)
        if Qb is not None:
            self.base_cmd = d.qpos[self.bq].copy() if here else Qb[-1].copy()
        return Follow(self.steps - before, self.ended, stopped, measured)

    def _hold(self):
        """One step holding the arm, hand and base where they were last sent."""
        e = self.base_cmd - self.env.sim.data.qpos[self.bq]
        self._step(self.cmd, GAIN * e)

    def finish(self, out) -> dict:
        import imageio.v2 as imageio

        with self._gl():
            for _ in range(10):  # RoboCasa's check on the state the agent leaves, after 10 still steps
                if self.ended:
                    break
                self._hold()
            success = bool(self.env._check_success())
            self.frames.append(self._frame())
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(out / "video.mp4", self.frames, fps=10, macro_block_size=1)
        np.save(out / "actions.npy", np.asarray(self.actions))
        result = dict(task=self.task, seed=self.seed, layout=int(self.layout), style=int(self.style), steps=self.steps,
                      step_limit=self.step_limit, success=success, instruction=self.instruction, dt=self.dt)  # fmt: skip
        (out / "native_result.json").write_text(json.dumps(result, indent=2))
        self.env.sim._render_context_offscreen.gl_ctx.make_current()  # closing frees GL objects in this thread
        self.env.close()
        return result
