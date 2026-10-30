"""EmbodiedSWE-Bench's simulator: one RoboBench env (Isaac Lab 2.3.2 on Isaac Sim 5.1) holding one preset at one seed,
served to `robot.py` over a socket.

    /isaac-sim/python.sh sim/server.py PRESET SEED --port P --out DIR   # in the simulator's image (launch.py)

The env is the preset's own (`ENVS.get(preset)().build(seed=...)`, then `reset(seed=...)`, as EmbodiedSWE's
`eval/grader/grade.py` builds it) wrapped in `GradedEnv` with the suite's grader, which records its rubric after every
step and gives the verdict on the final state. One change before the build, as EmbodiedSWE's own VLA evaluation does
(`vla/eval/sim.py`): our RGB-D cameras (`TASKS`' fixed views and the robot's `EGO` ones) join the scene's and the
robot's assets, with depth and instance segmentation. Nothing is added after the build (a second `sim.reset` breaks the
coffee task's grasp welds).

Each control step takes the arms' joint targets, the hands' joint targets and, for the walking G1, a world [x, y, yaw]
target for its pelvis (with the path's velocity): the server turns it into the frozen walking policy's [vx, vy, wz]
command with feedback from the measured pelvis pose (`walk`), at the control rate; no target: it stands still. Its
hip-height input stays at 0.72 m (the policy holds 0.723 m whatever it is given). The G1's waist holds its reset
posture all episode, so its torso is the arms' fixed mount.

Only robot state goes out: joints, the arms' mounting link's and the pelvis's world poses, rendered RGB-D with
intrinsics and camera poses, where each camera sees the robot, the step count, the end flag; the robot's own geometry
(a URDF with meshes, written from the robot's USD into `out`); and the scene's own `describe()` text. Object state,
rubric values and scores never leave the process, except `finish`'s verdict.

Wire: `roboskills.eval.wire` (an 8-byte length, a JSON header, raw arrays), one client at a time.
Ops: ping; reset -> robot facts and the first state with cameras; act {rows, render} -> state (a row {"a": ...}: a
recorded step's native action vector, sent as it was: a replay); observe -> state with cameras; links {names} -> world
poses of robot bodies (the check's only); finish {hold} -> verdict, video, actions.
"""

import argparse
import faulthandler
import importlib
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
from roboskills.eval.embodiedswe import EGO, TASKS, step_limit  # noqa: E402
from roboskills.eval.wire import recv, send  # noqa: E402


SIZE = (640, 480)  # camera pixels (width, height)
RENDER_PASSES = 2  # renders per observation: the RTX renderer catches up with a moved stage on the second
GROW = 2  # pixels the robot's mask is grown by (its silhouette's antialiased edge)
VIDEO = 2  # the video camera's frames downsampled by this
SETTLE = 30  # control steps the reset posture is held before the first observation: dropped objects land
ROBOT = "/World/envs/env_0/Robot"
ROOT = {"franka": "panda_link0", "g1": "torso_link"}  # the arms' mount: the root of the robot model the host gets
SIDES = {"franka": ("right",), "g1": ("left", "right")}
G1_ARM = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")
WALK_GAIN = (1.5, 1.5)  # 1/s: pelvis position and heading error -> velocity command, on top of the path's own velocity
WALK_CLIP = (0.25, 0.3, 0.6)  # |vx|, |vy| m/s, |wz| rad/s: within the policy's band, where its gait stays upright
WALK_MIN = 0.15  # m/s: a smaller translation command barely steps (0.1 forward walks at 0.05, sideways at 0.02)
WALK_TOL = (0.01, math.radians(1.5))  # m, rad: errors the pelvis is not stepped for
STEP = (0.15, 45)  # m/s sideways, steps: a turn without translation steps side to side (the policy cannot pivot)
TILT_STOP = 30.0  # degrees the pelvis may lean before a walk is stopped (it leans up to about 17 walking)
GL = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])  # world-convention camera axes -> OpenCV's


def matrix(q):
    """3x3 from a (w, x, y, z) quaternion."""
    w, x, y, z = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])  # fmt: skip


def quat(R):
    """(w, x, y, z) of a rotation matrix."""
    w = math.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = math.copysign(math.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    return (w, x, y, z)


def pose(p, q):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = matrix(q), np.asarray(p, float)
    return T


def look(eye, target):
    """4x4 of a world-convention camera (+x forward, +z up) at `eye` looking at `target`."""
    f = np.asarray(target, float) - eye
    f /= np.linalg.norm(f)
    left = np.cross([0.0, 0.0, 1.0], f)
    left = left / np.linalg.norm(left) if np.linalg.norm(left) > 1e-6 else np.array([0.0, 1.0, 0.0])
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = np.stack([f, left, np.cross(f, left)], 1), eye
    return T


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def walk(now, target, vel, step=0):
    """The walking policy's [vx, vy, wz] (the pelvis's own frame) that brings the pelvis at `now` [x, y, yaw] to the
    world `target` [x, y, yaw] moving at world `vel` [vx, vy, wz] (at control step `step`): the path's velocity plus
    WALK_GAIN times the error, a translation of at least WALK_MIN while the pelvis is off by more than WALK_TOL, and
    side steps (STEP) while it turns with no translation to make; within WALK_CLIP."""
    kp, kw = WALK_GAIN
    ex, ey, ea = target[0] - now[0], target[1] - now[1], wrap(target[2] - now[2])
    c, s = math.cos(now[2]), math.sin(now[2])
    vx, vy = vel[0] + kp * ex, vel[1] + kp * ey
    out = np.array([c * vx + s * vy, -s * vx + c * vy, vel[2] + kw * ea])
    speed = math.hypot(out[0], out[1])
    if math.hypot(ex, ey) > WALK_TOL[0] and 1e-9 < speed < WALK_MIN:
        out[:2] *= WALK_MIN / speed
    elif speed < WALK_MIN and (abs(ea) > WALK_TOL[1] or abs(vel[2]) > 0.05):
        out[1] += STEP[0] * (1 if (step // STEP[1]) % 2 else -1)
    return np.clip(out, -np.array(WALK_CLIP), WALK_CLIP)


def rpy(R):
    """URDF roll, pitch, yaw (R = Rz(yaw) Ry(pitch) Rx(roll))."""
    if abs(R[2, 0]) < 1 - 1e-9:
        return math.atan2(R[2, 1], R[2, 2]), math.asin(-R[2, 0]), math.atan2(R[1, 0], R[0, 0])
    return 0.0, math.copysign(math.pi / 2, -R[2, 0]), math.atan2(-R[0, 1], R[1, 1])


def urdf(name, joints, root, meshes, visual_only=()):
    """URDF text of `root` and every link below it. `joints`: [{name, type (revolute|prismatic|fixed), parent, child,
    T0, T1 (4x4: the joint frame in the parent's and in the child's frame, PhysX's localPose0/1), axis (0..2 in that
    frame), lo, hi}], a revolute joint's frame at its child's origin; `meshes`: {link: mesh file}, each also its
    collision geometry unless the link is `visual_only`."""
    below, order, names, todo = {}, [], set(), [root]
    for j in joints:
        below.setdefault(j["parent"], []).append(j)
    while todo:
        for j in below.get(todo.pop(0), []):
            order.append(j)
            todo.append(j["child"])
    out = [f'<robot name="{name}">']
    for link in [root] + [j["child"] for j in order]:
        geo = f'<geometry><mesh filename="{meshes[link]}"/></geometry>' if link in meshes else ""
        body = f"<visual>{geo}</visual>" if geo else ""
        if geo and link not in visual_only:
            body += f'<collision name="{link}_collision">{geo}</collision>'
        out.append(f'<link name="{link}">{body}</link>')
    for j in order:
        T0, T1 = np.asarray(j["T0"], float), np.asarray(j["T1"], float)
        T = T0 @ np.linalg.inv(T1)  # the child's frame in the parent's at q = 0
        if j["type"] == "revolute" and np.abs(T1[:3, 3]).max() > 1e-6:  # URDF turns a child about its own origin
            raise ValueError(
                f"joint {j['name']}: its frame is off {j['child']}'s origin; no URDF joint turns like that"
            )
        name = j["name"] if j["name"] not in names else f"{j['name']}_{j['child']}"
        names.add(name)
        xml = (f'<joint name="{name}" type="{j["type"]}"><parent link="{j["parent"]}"/><child link="{j["child"]}"/>'
               f'<origin xyz="{" ".join(f"{v:.9g}" for v in T[:3, 3])}" rpy="{" ".join(f"{v:.9g}" for v in rpy(T[:3, :3]))}"/>')  # fmt: skip
        if j["type"] != "fixed":
            axis = T1[:3, :3][:, j["axis"]]  # the joint's axis, in the child's frame
            xml += (f'<axis xyz="{" ".join(f"{v:.9g}" for v in axis)}"/>'
                    f'<limit lower="{j["lo"]:.9g}" upper="{j["hi"]:.9g}" effort="1000" velocity="10"/>')  # fmt: skip
        out.append(xml + "</joint>")
    return "".join(out) + "</robot>"


def export_model(stage, kind, out):
    """The robot's RoboKit model from its USD: the joints below ROOT[kind] (PhysX's frames and limits) and each body's
    visual meshes in its frame, as OBJ files; the root link visual only. Returns the URDF's path."""
    from pxr import Usd, UsdGeom, UsdPhysics

    robot, xf = stage.GetPrimAtPath(ROBOT), UsdGeom.XformCache()
    bodies = {str(p.GetPath()): p for p in Usd.PrimRange(robot) if p.HasAPI(UsdPhysics.RigidBodyAPI)}
    joints = []
    for p in Usd.PrimRange(robot):
        if not p.IsA(UsdPhysics.Joint):
            continue
        j = UsdPhysics.Joint(p)
        b0, b1 = j.GetBody0Rel().GetTargets(), j.GetBody1Rel().GetTargets()
        if not b0 or not b1 or str(b0[0]) not in bodies or str(b1[0]) not in bodies:
            continue  # the root's joint to the world
        kind_ = (
            "revolute"
            if p.IsA(UsdPhysics.RevoluteJoint)
            else "prismatic"
            if p.IsA(UsdPhysics.PrismaticJoint)
            else "fixed"
        )
        rec = dict(
            name=p.GetName(), type=kind_, parent=bodies[str(b0[0])].GetName(), child=bodies[str(b1[0])].GetName()
        )
        for k, (pos, rot) in enumerate(
            ((j.GetLocalPos0Attr(), j.GetLocalRot0Attr()), (j.GetLocalPos1Attr(), j.GetLocalRot1Attr()))
        ):
            q = rot.Get()
            rec[f"T{k}"] = pose(pos.Get(), (q.GetReal(), *q.GetImaginary())).tolist()
        if kind_ != "fixed":
            rec["axis"] = "XYZ".index(str(p.GetAttribute("physics:axis").Get()))
            lo, hi = (float(p.GetAttribute(f"physics:{a}Limit").Get()) for a in ("lower", "upper"))
            rec["lo"], rec["hi"] = (math.radians(lo), math.radians(hi)) if kind_ == "revolute" else (lo, hi)
        joints.append(rec)
    (out / "meshes").mkdir(parents=True, exist_ok=True)
    meshes = {}
    for path, body in bodies.items():
        inv = xf.GetLocalToWorldTransform(body).GetInverse()
        V, F, n = [], [], 0
        it = iter(Usd.PrimRange(body, Usd.TraverseInstanceProxies()))
        for p in it:
            if p != body and p.HasAPI(UsdPhysics.RigidBodyAPI):
                it.PruneChildren()
                continue
            rel = str(p.GetPath())[len(path) :]
            if not p.IsA(UsdGeom.Mesh) or "/collisions" in rel:
                continue
            img = UsdGeom.Imageable(p)
            if img.ComputeVisibility() == UsdGeom.Tokens.invisible or img.ComputePurpose() == UsdGeom.Tokens.guide:
                continue
            m = UsdGeom.Mesh(p)
            P, counts, idx = (np.asarray(a.Get(), float) for a in (m.GetPointsAttr(), m.GetFaceVertexCountsAttr(),
                                                                      m.GetFaceVertexIndicesAttr()))  # fmt: skip
            if not len(P) or not len(counts):
                continue
            M = np.array(xf.GetLocalToWorldTransform(p) * inv)  # row vectors: p_body = [p, 1] @ M
            V.append(np.c_[P, np.ones(len(P))] @ M)
            starts = np.r_[0, np.cumsum(counts)[:-1]].astype(int)
            for k in range(3, int(counts.max()) + 1):  # fan-triangulate the k-gons
                s = starts[counts == k]
                for i in range(1, k - 1):
                    F.append(np.stack([idx[s], idx[s + i], idx[s + i + 1]], 1).astype(int) + n)
            n += len(P)
        if V:
            name = body.GetName()
            meshes[name] = out / "meshes" / f"{name}.obj"
            V = np.concatenate(V)[:, :3]
            lines = [f"v {a:.6f} {b:.6f} {c:.6f}" for a, b, c in V] + [
                f"f {a + 1} {b + 1} {c + 1}" for a, b, c in np.concatenate(F)
            ]
            meshes[name].write_text("\n".join(lines) + "\n")
    root = ROOT[kind]
    path = out / f"{kind}.urdf"
    path.write_text(urdf(kind, joints, root, {k: str(v) for k, v in meshes.items()}, visual_only=(root,)))
    (out / "joints.json").write_text(json.dumps(joints))
    return path


def camera_cfgs(task, kind):
    """TiledCamera configs for the task's fixed views (under the env) and the robot's own (under its link), and each
    camera's pose in its parent's frame (OpenCV axes)."""
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import TiledCameraCfg

    views = {n: (None, eye, target, focal) for n, (eye, target, focal) in TASKS[task]["views"].items()}
    views.update(EGO[kind])
    cfgs, poses = {}, {}
    for name, (link, eye, target, focal) in views.items():
        T = look(np.asarray(eye, float), target)
        cfgs[name] = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/" + (f"Robot/{link}/" if link else "") + f"Cam_{name}", width=SIZE[0], height=SIZE[1],
            data_types=["rgb", "distance_to_image_plane", "instance_id_segmentation_fast"], update_period=0.0,
            colorize_instance_id_segmentation=False,
            offset=TiledCameraCfg.OffsetCfg(pos=tuple(T[:3, 3]), rot=quat(T[:3, :3]), convention="world"),
            spawn=sim_utils.PinholeCameraCfg(focal_length=focal, clipping_range=(0.02, 40.0)))  # fmt: skip
        T[:3, :3] = T[:3, :3] @ GL
        poses[name] = (link, T)
    return cfgs, poses


def boot(task, seed):
    """The preset's env with our cameras, reset at `seed`, wrapped in GradedEnv with its suite's grader."""
    from isaaclab.app import AppLauncher

    AppLauncher(headless=True, enable_cameras=True)
    import robobench

    robobench.discover()
    from robobench.core import GradedEnv
    from robobench.core.registries import ENVS, ROBOTS, SCENES

    cfg = ENVS.get(task)()
    kind = cfg.robot
    scene_cls, robot_cls = SCENES.get(cfg.scene), ROBOTS.get(cfg.robot)
    cams, poses = camera_cfgs(task, kind)
    fixed = {n: c for n, c in cams.items() if poses[n][0] is None}
    ego = {n: c for n, c in cams.items() if poses[n][0] is not None}
    scene_assets, robot_assets = scene_cls.assets, robot_cls.assets
    scene_cls.assets = lambda self: {**scene_assets(self), **{f"cam_{n}": c for n, c in fixed.items()}}
    robot_cls.assets = lambda self: {**robot_assets(self), **{f"cam_{n}": c for n, c in ego.items()}}
    try:
        env = cfg.build(num_envs=1, seed=int(seed))
    finally:
        scene_cls.assets, robot_cls.assets = scene_assets, robot_assets
    env.reset(seed=int(seed))
    suite, scene = task.split(".")[:2]
    graders = importlib.import_module(f"robobench.suites.{suite}.grader").GRADERS  # loaded here only, never sent
    grader = (graders[scene] if scene in graders else graders[cfg.scene])(env)
    return env, GradedEnv(env, grader), grader, kind, poses


class Episode:
    """The env, stepped one control step at a time."""

    def __init__(self, env, genv, grader, kind, poses, task, seed, out, frame_every):
        import torch

        self.torch, self.env, self.genv, self.grader, self.kind = torch, env, genv, grader, kind
        self.task, self.seed, self.out = task, int(seed), out
        self.loco = task.endswith(".loco_joint")
        self.art = art = env.robot.articulation
        self.names, self.bodies = list(art.joint_names), list(art.body_names)
        self.dt = float(env.dt * env.robot.control_period)
        self.step_limit = step_limit(task, self.dt)
        leaves = env.robot.controller.controllers  # composite([arms(+waist), hands(, legs)])
        slot, at = {}, 0
        for leaf in leaves[:2]:
            for k, j in enumerate(leaf.joint_ids):
                slot[self.names[int(j)]] = at + k
            at += leaf.action_dim
        self.dim = sum(leaf.action_dim for leaf in leaves)
        hand_names = [self.names[int(j)] for j in leaves[1].joint_ids]
        sides = SIDES[kind]
        if kind == "franka":
            self.arm_joints = {"right": [f"panda_joint{i}" for i in range(1, 8)]}
        else:
            self.arm_joints = {s: [f"{s}_{n}_joint" for n in G1_ARM] for s in sides}
        self.hand_joints = {s: [n for n in hand_names if kind == "franka" or n.startswith(s + "_")] for s in sides}
        ji = {n: i for i, n in enumerate(self.names)}
        self.arm_idx = {s: [ji[n] for n in v] for s, v in self.arm_joints.items()}
        self.hand_idx = {s: [ji[n] for n in v] for s, v in self.hand_joints.items()}
        self.arm_slot = {s: [slot[n] for n in v] for s, v in self.arm_joints.items()}
        self.hand_slot = {s: [slot[n] for n in v] for s, v in self.hand_joints.items()}
        limits = art.data.joint_pos_limits[0].cpu().numpy()
        self.lo, self.hi = limits[:, 0], limits[:, 1]
        q = self.q()
        self.act = np.zeros(self.dim, np.float32)  # the command held: every driven joint at its reset posture
        for n, k in slot.items():
            self.act[k] = q[ji[n]]
        if self.loco:
            self.act[-4:] = (0.0, 0.0, 0.0, 0.72)  # standing still at the policy's own hip height (it holds 0.72 m
            #                                         whatever height it is given: no posture to command)
        self.root = self.bodies.index(ROOT[kind])
        ground = env.stage.GetPrimAtPath("/World/ground")
        self.floor = 0.0
        if ground:
            from pxr import UsdGeom

            self.floor = float(UsdGeom.XformCache().GetLocalToWorldTransform(ground).ExtractTranslation()[2])
        self.cams = {n: env.iscene[f"cam_{n}"] for n in poses}
        self.cam_pose = poses
        self.video = "front"
        self.urdf = export_model(env.stage, kind, out / "robot")
        self.steps, self.done, self.broken, self.stopped = 0, False, None, None
        self.frame_every, self.frames, self.actions = int(frame_every), [], []
        for _ in range(SETTLE):
            self.tick({})

    # ---- state

    def q(self):
        return self.art.data.joint_pos[0].cpu().numpy().astype(float)

    def link(self, k):
        d = self.art.data
        return pose(d.body_pos_w[0, k].cpu().numpy(), d.body_quat_w[0, k].cpu().numpy())

    def pelvis(self):
        """[x, y, yaw] of the pelvis (the articulation's root), its height over the floor and its tilt in degrees."""
        d = self.art.data
        R = matrix(d.root_quat_w[0].cpu().numpy())
        p = d.root_pos_w[0].cpu().numpy()
        tilt = math.degrees(math.acos(float(np.clip(R[2, 2], -1.0, 1.0))))
        return np.array([p[0], p[1], math.atan2(R[1, 0], R[0, 0])]), float(p[2] - self.floor), tilt

    def state(self):
        q = self.q()
        out = {"q": {s: q[i] for s, i in self.arm_idx.items()}, "hand": {s: q[i] for s, i in self.hand_idx.items()},
               "root": self.link(self.root), "steps": self.steps, "done": self.done, "stopped": self.stopped}  # fmt: skip
        if self.loco:
            out["base"], out["body"], out["tilt"] = self.pelvis()
        return out

    def render(self, n=RENDER_PASSES):
        for _ in range(n):
            self.env.sim.render()

    def cameras(self, names=None):
        """RGB, depth (metres along the optical axis), K, the OpenCV pose and the robot's mask, per camera."""
        import cv2

        self.render()
        out = {}
        for name in names or self.cams:
            cam = self.cams[name]
            cam.update(0.0, force_recompute=True)
            d = cam.data
            seg = d.output["instance_id_segmentation_fast"][0, ..., 0].cpu().numpy()
            info = (
                d.info["instance_id_segmentation_fast"]
                if isinstance(d.info, dict)
                else d.info[0]["instance_id_segmentation_fast"]
            )
            mine = [int(k) for k, v in (info or {}).get("idToLabels", {}).items() if str(v).startswith(ROBOT + "/")]
            mask = cv2.dilate(np.isin(seg, mine).astype(np.uint8), np.ones((2 * GROW + 1,) * 2, np.uint8))
            link, T = self.cam_pose[name]
            if link is not None:
                T = self.link(self.bodies.index(link)) @ T
            out[name] = {"rgb": d.output["rgb"][0].cpu().numpy().astype(np.uint8),
                         "depth": d.output["distance_to_image_plane"][0, ..., 0].cpu().numpy().astype(np.float32),
                         "K": d.intrinsic_matrices[0].cpu().numpy().astype(float), "T": T, "robot": mask.astype(bool)}  # fmt: skip
        return out

    # ---- acting

    def tick(self, row):
        """One control step: the row's arm and hand targets (the rest hold), and for the walking G1 the pelvis's
        target pose and velocity (none: stand still) and height; or the native action vector `a` itself."""
        if row.get("a") is not None:  # a recorded step's action, as it was (a replay)
            self.act[:] = row["a"]
        else:
            self.command(row)
        action = self.torch.as_tensor(self.act[None], dtype=self.torch.float32, device=self.env.device)
        self.genv.step(action)
        self.steps += 1
        self.actions.append(self.act.copy())
        if not np.isfinite(self.q()).all():
            self.broken = f"the robot's state became nonfinite at step {self.steps}"
            print(self.broken, flush=True)
        self.done = self.done or self.broken is not None or self.steps >= self.step_limit
        if self.frame_every and self.steps % self.frame_every == 0:
            self.frame()

    def command(self, row):
        """Set the command held from a row (see `tick`)."""
        for s, v in row.get("q", {}).items():
            i = self.arm_idx[s]
            self.act[self.arm_slot[s]] = np.clip(v, self.lo[i], self.hi[i])
        for s, v in row.get("h", {}).items():
            i = self.hand_idx[s]
            self.act[self.hand_slot[s]] = np.clip(v, self.lo[i], self.hi[i])
        if self.loco:
            now, _, tilt = self.pelvis()
            if row.get("base") is not None and tilt > TILT_STOP and self.stopped is None:
                self.stopped = "tilt"
                print(f"walk stopped at step {self.steps}: the pelvis leans {tilt:.0f} degrees", flush=True)
            go = row.get("base") is not None and self.stopped is None
            self.act[-4:-1] = walk(now, row["base"], row.get("vel", (0.0, 0.0, 0.0)), self.steps) if go else 0.0
            if row.get("cmd") is not None:  # the policy's raw [vx, vy, wz, hip height] (calibration only)
                self.act[-4:] = row["cmd"]

    def frame(self, cams=None):
        """A video frame of the fixed front view: rendered now, unless an observation just did."""
        if cams is None:
            self.render(1)
            cam = self.cams[self.video]
            cam.update(0.0, force_recompute=True)
            rgb = cam.data.output["rgb"][0].cpu().numpy()
        else:
            rgb = cams[self.video]["rgb"]
        self.frames.append(np.ascontiguousarray(np.asarray(rgb, np.uint8)[::VIDEO, ::VIDEO]))

    def act_rows(self, rows, render):
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
        """Hold the last command `hold` steps (a walking G1 stands), then the grader's verdict on the final state."""
        for _ in range(int(hold)):
            if self.done:
                break
            self.tick({})
        verdict = self.genv.verdict()[0]
        stages = {k: round(float(v[0]), 4) for k, v in self.grader.measure().items()}
        return {
            "success": bool(verdict["success"]) and self.broken is None,
            "score": float(verdict["score"]),
            "stages": stages,
            "criteria": self.grader.describe(),
            "broken": self.broken,
            "steps": self.steps,
            "step_limit": self.step_limit,
            "frames": np.stack(self.frames) if self.frames else np.zeros((0, 1, 1, 3), np.uint8),
            "actions": np.asarray(self.actions, np.float32),
        }

    def facts(self):
        """What the host builds the robot from: the model, joints and limits, the cameras, the clock and the text."""
        f = {"robot": self.kind, "loco": self.loco, "task": self.task, "seed": self.seed, "dt": self.dt,
             "step_limit": self.step_limit, "settle": SETTLE, "instruction": self.env.scene.describe(),
             "urdf": str(self.urdf), "root": ROOT[self.kind], "arm_joints": self.arm_joints,
             "hand_joints": self.hand_joints, "cams": {n: p[0] for n, p in self.cam_pose.items()},
             "lo": {s: self.lo[i] for s, i in {**self.arm_idx}.items()}, "hi": {s: self.hi[i] for s, i in self.arm_idx.items()},
             "hand_lo": {s: self.lo[i] for s, i in self.hand_idx.items()},
             "hand_hi": {s: self.hi[i] for s, i in self.hand_idx.items()}, "floor": self.floor}  # fmt: skip
        return f


def handle(ep, req):
    op = req.get("op")
    if op == "ping":
        return {"pong": True, "task": ep.task, "seed": ep.seed}
    if op == "reset":  # the env as reset at boot; one episode per process
        if ep.steps > SETTLE:
            raise RuntimeError("one episode per simulator: start a new one for another")
        cams = ep.cameras()
        ep.frame(cams)
        return {**ep.state(), "cams": cams, "facts": ep.facts()}
    if op == "act":
        return ep.act_rows(req["rows"], bool(req.get("render")))
    if op == "observe":
        return {**ep.state(), "cams": ep.cameras()}
    if op == "links":  # robot bodies only (for checking the robot model against the simulator)
        return {"poses": {n: ep.link(ep.bodies.index(n)) for n in req["names"]}}
    if op == "finish":
        return ep.finish(req.get("hold", 0))
    raise ValueError(f"unknown op {op!r}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("task")
    p.add_argument("seed", type=int)
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--out", type=Path, required=True, help="where the robot model is written (shared with the host)")
    p.add_argument("--frame-every", type=int, default=20, help="control steps per video frame")
    a = p.parse_args()
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    t0 = time.time()
    lst = socket.create_server(("127.0.0.1", a.port))  # bound first: a taken port fails fast, before the boot
    env, genv, grader, kind, poses = boot(a.task, a.seed)
    ep = Episode(env, genv, grader, kind, poses, a.task, a.seed, a.out, a.frame_every)
    print(f"ready {a.task} {a.seed} port {a.port} boot {time.time() - t0:.0f}s", flush=True)
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
            print(json.dumps({k: v for k, v in rep.items() if k not in ("frames", "actions")}), flush=True)


if __name__ == "__main__":
    main()
