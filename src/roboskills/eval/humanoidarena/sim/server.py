"""HumanoidArena's simulator: one Isaac Sim app holding one official test episode, served to `robot.py` over a socket.

    /isaac-sim/python.sh sim/server.py TASK EPISODE_SEED --port P --arena ARENA --sonic SONIC   # in the image (launch.py)

The scene, its seeded layout, the 50 Hz control (200 Hz physics), the controller and the success check are upstream's
(HumanoidArena 68479287, `isaaclab_twist2_g1`): the task's official test config, the episode seed set before the reset as
its evaluator does, the frozen GEAR-SONIC encoder/decoder under the SONIC action provider with its semantic40 input
(HumanoidArena's VLA action: pelvis orientation relative to the episode's initial heading, the 29 joint angles, binary
hands). Isaac Sim 6.1 fixes (`fixes.py`) and the session layout follow RAX's HumanoidArena backend (Apache-2.0).

Each control step: each arm's 7 joint targets and each hand's binary command from the row; the legs, waist and pelvis
orientation from the gait (`gait.py`): GEAR-SONIC's kinematic planner walking toward the row's world [x, y, yaw] base
target with feedback from the measured pelvis, squatting to its pelvis height `body`, or standing. A walk stops (the gait
stands where it is) at the step after a robot body other than a foot touches something it was not touching when the walk
began, or when the pelvis leans TILT_RISE degrees more than then. After the step: upstream's reward sync, the success
check (Football: a positive reward, as RAX reads it) and upstream's fall rule; success, a fall or the step limit ends the
episode, and nothing moves after that.

Only robot state goes out: the arm and hand joints, the pelvis [x, y, yaw] and height, the torso link's pose (the arms'
mount), the lean, the camera's RGB-D with intrinsics and pose, where it sees the robot, the step count and the end flag.
Object state, rewards and success never leave the process, except `finish`'s verdict.

Wire: `roboskills.eval.wire` (an 8-byte length, a JSON header, raw arrays), one client at a time.
Ops: ping; reset -> robot facts, the first state with the camera; act {rows, render} -> state; observe -> state with the
camera; links {names} -> world poses of robot bodies; finish {hold} -> verdict and video frames.
"""

import argparse
import faulthandler
import json
import math
import os
import select
import signal
import socket
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[4]))  # the directory holding `roboskills` (mounted read-only)
from roboskills.eval.humanoidarena import TASKS  # noqa: E402
from roboskills.eval.humanoidarena.sim.gait import (  # noqa: E402
    CANONICAL,
    Gait,
    Planner,
    command,
    qconj,
    qmul,
    qyaw,
    rot6d,
    yaw_of,
)
from roboskills.eval.wire import recv, send  # noqa: E402


SIDES = ("left", "right")
ARM = {s: [f"{s}_{j}_joint" for j in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll",
                                       "wrist_pitch", "wrist_yaw")] for s in SIDES}  # fmt: skip
ARM_IDX = {s: [CANONICAL.index(n) for n in ARM[s]] for s in SIDES}
CAMERA = "agent_front_camera"  # upstream's front camera, rendered from under the environment root (as RAX does)
SUBFRAMES = 4  # RTX subframes per synchronized camera render
VIDEO = 2  # video frames downsampled by this
GROW = 2  # pixels the robot's mask is grown by
TILT_RISE = 20.0  # degrees a walk may lean the pelvis past its lean at the start (walking sways it up to 15)
TOUCH_N = 30.0  # N on a non-foot body that counts as touching
REST, SAG_GAIN, SAG_MAX = 0.002, 0.06, 0.3  # rad per step an arm target may move and count as resting; the share of
#   the joint error added to its offset each resting step (time constant about 0.3 s); the largest offset, rad
FALL = (60.0, 75.0, 50.0, 5)  # upstream's fall rule: soft and hard tilt (deg), contact force (N), confirming steps
CRITICAL = ("pelvis", "torso", "waist", "hip", "thigh", "knee", "calf", "shin", "head")
WORLD_TO_CV = np.array([[0.0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]])  # camera axes: Isaac Lab's "world"
#   convention (x forward, z up) -> OpenCV (x right, y down, z forward)
ASSET_ROOT = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1"


def pose(pos, quat_wxyz):
    from scipy.spatial.transform import Rotation

    T = np.eye(4)
    w, x, y, z = (float(v) for v in quat_wxyz)
    T[:3, :3] = Rotation.from_quat([x, y, z, w]).as_matrix()
    T[:3, 3] = np.asarray(pos, float)
    return T


def boot(task, seed, arena, sonic):
    """The task's env with the official test config and episode seed, and the SONIC provider taking semantic40 rows."""
    gym_id, package, test, _, _ = TASKS[task]
    sys.path.insert(0, str(arena))  # tasks, action_provider, common_env_objects, tools, ...
    os.environ.update(PROJECT_ROOT=str(arena), PYTHONDONTWRITEBYTECODE="1", OPEN_DOOR_STRICT_SUCCESS="1",
                      OPEN_DOOR_STRICT_REQUIRE_LATCH_UNLOCKED="1", OPEN_DOOR_STRICT_REQUIRE_LEAF_OPEN="1",
                      OPEN_DOOR_STRICT_REQUIRE_GEOMETRY="0", OPEN_DOOR_SUCCESS_LEAF_ANGLE_DEG="60.0")  # fmt: skip
    args = SimpleNamespace(
        task=gym_id,
        input_source="vla",
        gmt_backend="sonic",
        action_source="sonic_wholebody",
        robot_type="unitree_g1_refpose_v3_1",
        device="cuda:0",
        replay_file="",
        enable_dex3_dds=True,
        task_runtime_profile="inference" if task in ("opendoor", "warehouse_navigation") else "auto",
        sonic_encoder_path=str(Path(sonic) / "model_encoder.onnx"),
        sonic_decoder_path=str(Path(sonic) / "model_decoder.onnx"),
        recording_save_dir="/tmp/ha-recording",
        sonic_effort_control=False,
        sonic_vla_root_rot6d_layout="row",
        sonic_vla_root_max_delta_deg=26.0,
        enable_world_camera=False,
        enable_perspective_camera=False,
    )  # fmt: skip  (RAX's, as upstream's SONIC evaluation sets them)
    from task_runtime_profiles import apply_task_runtime_profile

    apply_task_runtime_profile(args, context="live_inference")
    import carb

    settings = carb.settings.get_settings()
    settings.set_bool("/isaaclab/cameras_enabled", True)
    for key in ("cloud", "default", "nvidia"):  # a few authored references resolve against the 5.1 asset root
        settings.set("/persistent/isaac/asset_root/" + key, ASSET_ROOT)
    import onnxruntime
    import torch

    torch.inference_mode().__enter__()
    onnxruntime.preload_dlls(cuda=True, cudnn=True, msvc=False)  # CUDA 12 / cuDNN 9 from torch's wheels, for ONNX
    from roboskills.eval.humanoidarena.sim import fixes

    fixes.install_compat()
    import copy

    import gymnasium as gym
    import tasks  # noqa: F401  (upstream's Gym registrations)
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
    from tasks.common_env_config import apply_env_config_yaml

    cfg = parse_env_cfg(gym_id, device="cuda:0", num_envs=1)
    cfg.env_name = gym_id
    apply_env_config_yaml(cfg, str(Path(arena) / f"tasks/common_test_config/base_test/{test}_sonic_test.yaml"),
                          task_name=gym_id, route_name="sonic")  # fmt: skip
    cfg.scene.front_camera.data_types = ["rgb"]
    cam = copy.deepcopy(cfg.scene.front_camera)
    cam.prim_path = "{ENV_REGEX_NS}/AgentFrontCamera"
    cam.data_types = ["rgb", "distance_to_image_plane", "instance_id_segmentation_fast"]
    cam.colorize_instance_id_segmentation = False
    setattr(cfg.scene, CAMERA, cam)
    cfg.sim.enable_scene_query_support = True
    cfg.seed, cfg.object_reset_seed_source = int(seed), "env_seed"  # as upstream's _configure_episode_seed_state
    cfg._episode_runtime_seed, cfg._episode_object_seed_counter = int(seed), 0
    cfg._current_episode_object_seed, cfg._current_episode_object_seed_source = None, ""
    env = gym.make(gym_id, cfg=cfg).unwrapped
    if callable(getattr(cfg, "initialize_task_scene", None)):
        cfg.initialize_task_scene(env, args)
    if task == "football_single":
        import omni.usd

        fixes.exact_goal_mesh(omni.usd.get_context().get_stage(), cfg)
    env.sim.reset()
    env.reset()
    return env, cfg, args, make_provider(env, args)


def make_provider(env, args):
    """Upstream's SonicActionProvider with the agent's semantic40 rows in place of its VLA model (RAX's provider)."""
    import onnxruntime as ort
    from action_provider.action_provider_sonic import SonicActionProvider

    class Provider(SonicActionProvider):
        def _make_session(self, path):
            so = ort.SessionOptions()
            so.intra_op_num_threads, so.inter_op_num_threads = 2, 1
            return ort.InferenceSession(path, sess_options=so, providers=[("CUDAExecutionProvider", {"device_id": 0}),
                                                                         "CPUExecutionProvider"])  # fmt: skip

        def _setup_lerobot_vla(self, args_cli):
            pass

        def _setup_hand_dds(self, args_cli):
            self._dex3_dds = None

        def _recording_enabled_for_current_mode(self):
            return False

        def _should_refresh_lerobot_visuals_next_step(self):
            return False

        def _infer_lerobot_semantic_action(self):
            return np.asarray(self.row, np.float32)

    p = Provider(env, args)
    p.row, p.decoder_calls = None, 0
    decode = p._decoder.run

    def counted(*a, **k):  # every control step must run the decoder once, without error
        out = decode(*a, **k)
        if all(np.isfinite(v).all() for v in out):
            p.decoder_calls += 1
        return out

    p._decoder = SimpleNamespace(run=counted, get_inputs=p._decoder.get_inputs)
    p._should_start_recording_on_first_call = False
    if p._encoder.get_inputs()[0].shape[-1] != 1762 or p._decoder.get_inputs()[0].shape[-1] != 994:
        raise ValueError("the runtime needs the released SONIC encoder (1762) and decoder (994)")
    return p


class Episode:
    def __init__(self, env, cfg, args, provider, task, seed, sonic, frame_every):
        import torch

        from roboskills.eval.humanoidarena.sim import fixes

        self.torch, self.env, self.cfg, self.p, self.task, self.seed = torch, env, cfg, provider, task, int(seed)
        self.robot = env.scene["robot"]
        self.step_limit, self.frame_every = int(TASKS[task][3]), int(frame_every)
        self.door = fixes.DoorReset(env, cfg) if task == "opendoor" else None
        data = self.robot.data
        names, bodies = list(data.joint_names), list(data.body_names)
        self.joint_idx = [names.index(n) for n in CANONICAL]
        self.hand_names = {s: [names[int(i)] for i in getattr(provider, f"_{s}_hand_idx")] for s in SIDES}
        self.hand_idx = {s: [names.index(n) for n in self.hand_names[s]] for s in SIDES}
        self.bodies = bodies
        self.torso = bodies.index("torso_link")
        self.critical = [i for i, n in enumerate(bodies) if any(t in n.lower() for t in CRITICAL)
                         and not any(t in n.lower() for t in ("ankle", "foot"))]  # fmt: skip
        contacts = env.scene["contact_forces"]
        self.touch_bodies = [i for i, n in enumerate(contacts.body_names) if not any(t in n for t in ("ankle", "foot"))]
        self.contact_names = list(contacts.body_names)
        self.gait = Gait(Planner(str(Path(sonic) / "planner_sonic.onnx")))
        self.reset_episode(fixes)
        self.cam = env.scene[CAMERA]
        front = env.scene["front_camera"]
        front.reset()
        data = self.robot.data  # (a physics restart made it anew)
        bp, bq = data.body_pos_w[0, self.torso].cpu().numpy(), data.body_quat_w[0, self.torso].cpu().numpy()
        qi = qconj(bq)
        self.mount = (
            qmul(qmul(qi, np.r_[0.0, front.data.pos_w[0].cpu().numpy() - bp]), bq)[1:],
            qmul(qi, front.data.quat_w_world[0].cpu().numpy()),
        )  # the camera in the torso link
        self.robot_prefix = self.robot.cfg.prim_path.replace("env_.*", "env_0") + "/"

    def reset_episode(self, fixes):
        """The official episode: the seeded reset as upstream's evaluator does it; physics restarted and the reset
        replayed when it wrote a static collider; the door restored; the gait standing where the robot stands."""
        env, cfg = self.env, self.cfg

        def seeded():  # upstream's _reset_environment_for_episode, with env.reset seeding Isaac Lab's RNGs (RAX's)
            import random

            random.seed(self.seed)
            np.random.seed(self.seed)
            self.torch.manual_seed(self.seed)
            cfg.seed, cfg._episode_runtime_seed, cfg._episode_object_seed_counter = self.seed, self.seed, 0
            env.reset(seed=self.seed)
            cfg._episode_object_seed_counter = 0
            if getattr(cfg, "event_manager", None) is not None and getattr(cfg, "startup_task_reset_enabled", True):
                cfg.event_manager.trigger("reset_all_self", env)
            if self.door is not None:
                self.door.restore()
                cfg.reset_open_door_latch(env, reason="episode_reset")

        seeded()
        self.restarted = fixes.needs_physics_restart(cfg)
        if self.restarted:
            env.sim.reset()
            seeded()
        self.p.on_env_reset()
        self.p._should_start_recording_on_first_call = False
        env.scene.write_data_to_sim()
        env.sim.forward()
        self.render(1)
        d = self.robot.data
        root = d.root_state_w[0, :7].cpu().numpy()
        self.gait.reset(root[:3], root[3:7], d.joint_pos[0, self.joint_idx].cpu().numpy())
        h0 = self.p._lerobot_vla_runtime._episode_ref_to_world_heading_quat_wxyz
        self.heading0 = qconj(h0 if h0 is not None else qyaw(yaw_of(root[3:7])))
        self.cmd = {"q": {s: d.joint_pos[0, [self.joint_idx[i] for i in ARM_IDX[s]]].cpu().numpy() for s in SIDES},
                    "u": {s: 0 for s in SIDES}}  # fmt: skip
        self.steps, self.done, self.success_step, self.fallen, self.fall_streak = 0, False, None, False, 0
        self.frames, self.walk, self.stopped, self.tick_s = [], None, None, []
        self.want = {s: self.cmd["q"][s].copy() for s in SIDES}
        self.sag = {s: np.zeros(7) for s in SIDES}  # offsets added to the arms' targets (integral action)

    # ---- state

    def base(self):
        root = self.robot.data.root_state_w[0, :7].cpu().numpy()
        return np.array([root[0], root[1], yaw_of(root[3:7])]), float(root[2]), root

    def tilt(self):
        g = self.robot.data.projected_gravity_b[0].cpu().numpy()
        return math.degrees(math.acos(float(np.clip(-g[2], -1.0, 1.0))))

    def link(self, i):
        d = self.robot.data
        return pose(d.body_pos_w[0, i].cpu().numpy(), d.body_quat_w[0, i].cpu().numpy())

    def state(self):
        d = self.robot.data
        q = d.joint_pos[0].cpu().numpy()
        base, z, _ = self.base()
        return {"q": {s: q[[self.joint_idx[i] for i in ARM_IDX[s]]] for s in SIDES},
                "hand": {s: q[self.hand_idx[s]] for s in SIDES}, "base": base, "body": z, "mount": self.link(self.torso),
                "tilt": self.tilt(), "stopped": self.stopped, "walking": not self.gait.settled(), "steps": self.steps,
                "step_limit": self.step_limit, "done": self.done}  # fmt: skip

    def render(self, frames, sync=False):
        """Render without stepping physics; `sync` steps Replicator so the camera's buffers hold this frame."""
        if sync:
            import omni.replicator.core as rep

            rep.orchestrator.step(rt_subframes=frames, delta_time=0.0, pause_timeline=False)
        else:
            for _ in range(frames):
                self.env.sim.render()

    def cameras(self):
        """cam_front: RGB, depth along the optical axis (0: nothing), K, the OpenCV camera pose in the world, the robot's
        mask; the camera placed on the measured torso link first, as upstream mounts it."""
        torch, d = self.torch, self.robot.data
        self.render(1)
        bp, bq = d.body_pos_w[0, self.torso].cpu().numpy(), d.body_quat_w[0, self.torso].cpu().numpy()
        p = bp + qmul(qmul(bq, np.r_[0.0, self.mount[0]]), qconj(bq))[1:]
        q = qmul(bq, self.mount[1])
        dev = self.env.device
        self.cam.set_world_poses(torch.tensor(p[None], device=dev, dtype=torch.float32),
                                 torch.tensor(q[None], device=dev, dtype=torch.float32), convention="world")  # fmt: skip
        self.render(SUBFRAMES, sync=True)
        self.cam.update(self.env.physics_dt, force_recompute=True)
        out, info = self.cam.data.output, self.cam.data.info[0]
        rgb = out["rgb"][0, :, :, :3].cpu().numpy().astype(np.uint8)
        depth = out["distance_to_image_plane"][0].cpu().numpy().reshape(rgb.shape[:2])
        depth = np.where(np.isfinite(depth) & (depth > 0), depth, 0).astype(np.float32)
        T = pose(p, q) @ WORLD_TO_CV  # the pose it was just placed at (its own pose buffers lag a frame behind)
        K = self.cam.data.intrinsic_matrices[0].cpu().numpy().astype(float)
        return {"cam_front": {"rgb": rgb, "depth": depth, "K": K, "T": T,
                              "robot": self.mask(out["instance_id_segmentation_fast"][0], info)}}  # fmt: skip

    def mask(self, seg, info):
        import cv2

        ids = seg.cpu().numpy().reshape(seg.shape[0], seg.shape[1])
        labels = (info.get("instance_id_segmentation_fast") or {}).get("idToLabels", {})
        mine = [int(k) for k, v in labels.items() if str(v).startswith(self.robot_prefix)]
        m = np.isin(ids, mine).astype(np.uint8)
        return cv2.dilate(m, np.ones((2 * GROW + 1,) * 2, np.uint8)).astype(bool)

    # ---- acting

    def touching(self):
        f = self.env.scene["contact_forces"].data.net_forces_w[0].cpu().numpy()
        return {self.contact_names[i] for i in self.touch_bodies if np.linalg.norm(f[i]) > TOUCH_N}

    def tick(self, row):
        """One 50 Hz control step from a row {q: {arm: 7}, u: {arm: 0/1}, base: [x, y, yaw] | None, body: h | None}."""
        from tools.get_reward import sync_reward_after_physics_step

        t0 = time.perf_counter()
        for s in SIDES:
            if s in row.get("q", {}):
                self.cmd["q"][s] = np.asarray(row["q"][s], float)
            if s in row.get("u", {}):
                self.cmd["u"][s] = int(bool(row["u"][s]))
        base, _, _ = self.base()
        target = row.get("base")
        if target is not None:
            if row.get("drive") or self.walk is None:  # a walk begins: what touches now is not in its way
                self.walk = {"touch": self.touching(), "tilt": self.tilt()}
            new, tilt = self.touching() - self.walk["touch"], self.tilt()
            if self.stopped is None and (new or tilt > self.walk["tilt"] + TILT_RISE):
                self.stopped = "contact" if new else "tilt"
                print(f"walk stopped at step {self.steps}: new {sorted(new)} tilt {tilt:.1f}", flush=True)
            if self.stopped:
                target = None
        else:
            self.walk = None
        hold = self.gait.cmd[3] if self.gait.cmd is not None else yaw_of(self.gait.frame()[0])  # (a new facing replans)
        cmd = command(base, target, row.get("body"), hold_yaw=hold, walking=self.gait.moving())
        quat, joints, z = self.gait.step(cmd)
        q = self.robot.data.joint_pos[0].cpu().numpy()
        for s in SIDES:  # SONIC holds an arm a few degrees off its targets: integrate the error while the target rests
            want = self.cmd["q"][s]
            if np.abs(want - self.want[s]).max() < REST:
                err = want - q[[self.joint_idx[i] for i in ARM_IDX[s]]]
                self.sag[s] = np.clip(self.sag[s] + SAG_GAIN * err, -SAG_MAX, SAG_MAX)
            self.want[s] = want.copy()
            joints[ARM_IDX[s]] = want + self.sag[s]
        self.p.row = np.r_[
            0.0, 0.0, z, rot6d(qmul(self.heading0, quat)), joints, self.cmd["u"]["left"], self.cmd["u"]["right"]
        ]
        t1 = time.perf_counter()
        calls = self.p.decoder_calls
        if self.p.get_action(self.env) is None or self.p.decoder_calls != calls + 1:
            raise RuntimeError("the SONIC control step failed")  # (upstream falls back to a default pose; not here)
        t2 = time.perf_counter()
        sync_reward_after_physics_step(self.env)
        self.steps += 1
        if self.success() and self.success_step is None:
            self.success_step = self.steps
        self.fallen = self.fall()
        self.done = self.success_step is not None or self.fallen or self.steps >= self.step_limit
        t3 = time.perf_counter()
        if self.frame_every and self.steps % self.frame_every == 0:
            self.frames.append(self.cameras()["cam_front"]["rgb"][::VIDEO, ::VIDEO].copy())
        self.tick_s.append(
            (t1 - t0, t2 - t1, t3 - t2, time.perf_counter() - t3)
        )  # gait, control+physics, checks, video

    def success(self):
        if self.task == "football_single":  # the reward the manager computed after this step (RAX's reading)
            buf = getattr(self.env, "reward_buf", None)
            buf = self.env.reward_manager._reward_buf if buf is None else buf
            return bool(buf[0].item() > 1e-6)
        import importlib

        rewards = importlib.import_module(f"tasks.g1_tasks.{TASKS[self.task][1]}.mdp.rewards")
        return bool(rewards.compute_success_mask(self.env)[0].item())

    def fall(self):
        """Upstream's rule: the pelvis tilts past 75 degrees, or past 60 with 50 N on a critical body, 5 steps running."""
        soft, hard, force, confirm = FALL
        up = float(-self.robot.data.projected_gravity_b[0, 2].item())
        forces = ()
        contacts = getattr(self.robot.data, "body_net_contact_force_w", None)
        if contacts is not None and self.critical:
            forces = self.torch.linalg.vector_norm(contacts[0, self.critical], dim=-1).tolist()
        cand = up < math.cos(math.radians(hard)) or (
            up < math.cos(math.radians(soft)) and any(f >= force for f in forces)
        )
        self.fall_streak = self.fall_streak + 1 if cand else 0
        return self.fall_streak >= confirm

    def act(self, rows, render):
        self.stopped = None
        for row in rows:
            if self.done:
                break
            self.tick(row)
        out = self.state()
        if render:
            out["cams"] = self.cameras()
        return out

    def finish(self, hold):
        for _ in range(int(hold)):
            if self.done:
                break
            self.tick({})
        return {"success": self.success_step is not None, "success_step": self.success_step, "fallen": self.fallen,
                "steps": self.steps, "step_limit": self.step_limit, "restarted_physics": self.restarted,
                "plans": self.gait.plans, "plan_ms": round(1000 * float(np.mean(self.gait.plan_s)), 1) if self.gait.plan_s else None,
                "tick_ms": dict(zip(("gait", "control", "checks", "video"), np.round(1000 * np.mean(self.tick_s, axis=0), 1).tolist()))
                if self.tick_s else None,
                "frames": np.stack(self.frames) if self.frames else np.zeros((0, 1, 1, 3), np.uint8)}  # fmt: skip

    def facts(self):
        d = self.robot.data
        lim = d.soft_joint_pos_limits[0].cpu().numpy()
        from pico_server.data_utils.params import DEFAULT_HAND_POSE

        poses = DEFAULT_HAND_POSE["unitree_g1_with_hands"]
        return {"arm_joints": ARM, "hand_joints": self.hand_names,
                "lo": {n: float(lim[i, 0]) for i, n in enumerate(d.joint_names)},
                "hi": {n: float(lim[i, 1]) for i, n in enumerate(d.joint_names)},
                "hand_open": {s: list(map(float, poses[s]["open"])) for s in SIDES},
                "hand_closed": {s: list(map(float, poses[s]["close"])) for s in SIDES},
                "bodies": self.bodies, "seed": self.seed, "restarted_physics": self.restarted}  # fmt: skip


def handle(ep, req):
    op = req.get("op")
    if op == "ping":
        return {"pong": True, "task": ep.task, "seed": ep.seed}
    if op == "reset":  # the episode as loaded at boot; one per process
        if ep.steps:
            raise RuntimeError("one episode per simulator: start a new one for another")
        cams = ep.cameras()
        ep.frames.append(cams["cam_front"]["rgb"][::VIDEO, ::VIDEO].copy())
        return {**ep.state(), "cams": cams, "facts": ep.facts()}
    if op == "act":
        return ep.act(req["rows"], bool(req.get("render")))
    if op == "observe":
        return {**ep.state(), "cams": ep.cameras()}
    if op == "links":  # robot bodies only (for checking the robot model against the simulator)
        return {"poses": {n: ep.link(ep.bodies.index(n)) for n in req["names"]}}
    if op == "finish":
        return ep.finish(req.get("hold", 0))
    raise ValueError(f"unknown op {op!r}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("task", choices=sorted(TASKS))
    p.add_argument(
        "seed", type=int, help="the episode's environment seed (official: roboskills.eval.humanoidarena.episode_seed)"
    )
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--arena", required=True, help="the pinned HumanoidArena checkout's isaaclab_twist2_g1")
    p.add_argument("--sonic", required=True, help="GEAR-SONIC's encoder, decoder and planner")
    p.add_argument("--frame-every", type=int, default=25, help="control steps per video frame")
    a = p.parse_args()
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    t0 = time.time()
    lst = socket.create_server(("127.0.0.1", a.port))  # bound first: a taken port fails fast, before the boot
    from isaaclab.app import AppLauncher

    exp = "/isaac-sim/apps/isaacsim.exp.base.python.kit"
    app = AppLauncher(headless=True, enable_cameras=True, **({"experience": exp} if Path(exp).is_file() else {})).app
    try:
        ep = Episode(*boot(a.task, a.seed, Path(a.arena), a.sonic), a.task, a.seed, a.sonic, a.frame_every)
    except BaseException:  # Kit may hang closing a half-built stage: leave at once
        traceback.print_exc()
        sys.stdout.flush()
        os._exit(1)
    print(f"ready {a.task} {a.seed} port {a.port} boot {time.time() - t0:.0f}s", flush=True)
    conn = None
    try:
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
    finally:
        app.close()


if __name__ == "__main__":
    main()
