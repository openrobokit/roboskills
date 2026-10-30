"""RoboTwin 2.0 behind `roboskills.eval.interface.Robot`: the aloha-agilex robot (two 6-DoF arms with parallel grippers) in SAPIEN.

`RoboTwin(task)` holds RoboTwin's task env for many episodes; `reset(seed)` starts one exactly as RoboTwin's eval
starts each test episode on the same env object (scripts/eval_policy_xpolicylab.py): `close_env` (render caches
cleared every `clear_cache_freq` episodes), then `setup_demo(now_ep_num, seed, is_test=True)` on the `demo_clean`
config, which seeds numpy/torch and rebuilds the scene. The benchmark's action is a 14-D joint target (left 6 joints,
left gripper 0..1, right 6, right gripper) that RoboTwin time-parameterizes (TOPP) and executes in one call; each
call is one step of the task's step limit. A joint path is sent as the fewest waypoints that keep it within a small
tolerance of the planned one, so a straight approach stays straight and a free motion costs a few steps.

Everything here is robot facts (the URDF, base pose, gripper travel, the TCP 12 cm along link6) and observations:
camera RGB-D (RoboTwin's depth is in mm; here m) with intrinsics/extrinsics, the robot's own pixels (the renderer's
per-pixel link ids), measured joints and gripper openings. RoboTwin ends an episode at the first physics step where
the task's check passes; here the check is latched instead (the env never learns it, so the scene goes on as it
would) and only `finish` reads it.
Observation mode is the default. Explicit information="privileged" also enables
scene(): read-only native physical object poses and collision-surface geometry,
without live task-target annotations or reward queries.
"""

import contextlib
import functools
import importlib
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[4]
RT = Path(os.environ.get("ROBOTWIN") or ROOT / "runs/vendor/RoboTwin").resolve()  # the RoboTwin checkout (setup.sh)
os.chdir(RT)  # RoboTwin resolves ./assets and ./env_cfg from its root
sys.path.insert(0, str(RT))
import transforms3d as t3d  # noqa: E402
import yourdfpy  # noqa: E402

from roboskills.eval.interface import Arm, Camera, Follow, Hand, Obs  # noqa: E402
from roboskills.eval.robotwin.hand import WIDTH, finger, model_q  # noqa: E402


# Joint limits for planning: the URDF's, pulled in where the arm folds into itself or the table.
LIMITS = np.array([(-2.6, 2.6), (0.0, 3.4), (0.0, 3.4), (-1.9, 1.9), (-1.8, 1.8), (-3.2, 3.2)])
TCP = 0.12  # the TCP (between the finger pads) along link6's +x (RoboTwin's gripper_bias)
TOL = 0.03  # rad: how far a sent joint path may stray from the planned one between waypoints
SIDES = (("left", "fl", 0), ("right", "fr", 7))  # arm, joint prefix, offset in the 14-D action


def task_args(task, config="demo_clean"):
    """The args RoboTwin's eval (load_task_args) builds for a test episode, plus depth."""
    cfg = RT / "env_cfg/task_config"
    args = yaml.safe_load((cfg / f"{config}.yml").read_text())
    args.update(task_name=task, task_config=config, ckpt_setting="default", policy_name="agent", eval_mode=True,
                eval_video_log=False, render_freq=0, dual_arm_embodied=True)  # fmt: skip
    args["data_type"].update(rgb=True, qpos=True, endpose=True, depth=True)
    cameras = yaml.safe_load((cfg / "_camera_config.yml").read_text())[args["camera"]["head_camera_type"]]
    args["head_camera_h"], args["head_camera_w"] = cameras["h"], cameras["w"]
    (name,) = args["embodiment"]
    robot_file = yaml.safe_load((cfg / "_embodiment_config.yml").read_text())[name]["file_path"]
    args.update(left_robot_file=robot_file, right_robot_file=robot_file)
    args["left_embodiment_config"] = args["right_embodiment_config"] = yaml.safe_load(
        (Path(robot_file) / "config.yml").read_text()
    )
    return args


def make_env(task):
    return importlib.import_module(f"envs.{task}").__dict__[task]()


def _frame(p=None, x=0.0):
    T = np.eye(4)
    if p is None:
        T[0, 3] = x
    else:
        T[:3, :3], T[:3, 3] = t3d.quaternions.quat2mat(p[3:]), p[:3]
    return T


def simplify(Q, tol=TOL):
    """Indices of rows of Q to keep so every sample stays within `tol` of the piecewise-linear path through them."""
    keep, stack = {0, len(Q) - 1}, [(0, len(Q) - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        s = np.linspace(0, 1, j - i + 1)[1:-1, None]
        err = np.abs(Q[i + 1 : j] - (Q[i] + s * (Q[j] - Q[i]))).max(1)
        k = int(np.argmax(err))
        if err[k] > tol:
            keep.add(i + 1 + k)
            stack += [(i, i + 1 + k), (i + 1 + k, j)]
    return sorted(keep)


class RoboTwin:
    REMOTE = ("scene", "replay")  # served beside the Robot contract (eval/serve.py)

    def __init__(self, task, *, information="observations"):
        if information not in {"observations", "privileged"}:
            raise ValueError(f"Unknown information mode {information!r}")
        self.information = information
        self.task, self.args, self.env = task, task_args(task), make_env(task)
        self.instruction = json.loads((RT / "description/task_instruction" / f"{task}.json").read_text())[
            "full_description"
        ]
        self.n, self.arms, self.dt, self.success = 0, None, 0.1, False
        check = self.env.check_success

        def latched():  # RoboTwin checks at every physics step; remember a pass, never end the episode on it
            self.success = self.success or bool(check())
            return False

        self.env.check_success = latched
        g_lo, g_hi = self.args["left_embodiment_config"][
            "gripper_scale"
        ]  # RoboTwin's gripper 0..1 -> finger g_lo..g_hi
        self.g_lo, self.g_span = g_lo, g_hi - g_lo

    def reset(self, seed):
        """Start an episode on `seed`, as RoboTwin's eval does between test episodes."""
        env = self.env
        if self.n:
            env.close_env(clear_cache=self.n % self.args["clear_cache_freq"] == 0)
        self.n += 1  # a failed setup (UnStableError) is closed at the next reset too, as the eval does
        env.setup_demo(now_ep_num=self.n - 1, seed=int(seed), is_test=True, **self.args)
        self.seed, self.success = int(seed), False
        self._ent = {"left": env.robot.left_entity, "right": env.robot.right_entity}
        self._idx = {}
        for s, p, _ in SIDES:
            names = [j.get_name() for j in self._ent[s].get_active_joints()]
            self._idx[s] = [names.index(f"{p}_joint{j}") for j in range(1, 8)]  # 6 arm joints + the finger joint
        ents = {id(e): e for e in self._ent.values()}.values()
        self.robot_ids = np.array(sorted({ln.entity.per_scene_id for e in ents for ln in e.get_links()}))
        cams = env.cameras
        self._cams = {"left_camera": cams.left_camera, "right_camera": cams.right_camera,
                      **dict(zip(cams.static_camera_name, cams.static_camera_list))}  # fmt: skip
        self._seg = {}
        self.cmd = np.asarray(env.get_obs()["joint_action"]["vector"], float)  # the last joint target sent
        if self.arms is None:
            self._build_arms()
        for s, _, off in SIDES:  # the same Arm objects every episode (the verbs' kinematics hold them)
            self.arms[s].home[:] = self.cmd[off : off + 6]
        self.frames, self.actions = [self._head()], []

    def _build_arms(self):
        cfg = self.args["left_embodiment_config"]
        urdf = yourdfpy.URDF.load(str(RT / self.args["left_robot_file"] / cfg["urdf_path"]), build_scene_graph=False,
                                  load_meshes=True)  # fmt: skip
        for _, p, _ in SIDES:
            for j, (lo, hi) in enumerate(LIMITS, 1):
                lim = urdf.joint_map[f"{p}_joint{j}"].limit
                lim.lower, lim.upper = lo, hi
        Tb, g_lo = _frame(np.asarray(cfg["robot_pose"][0], float)), self.g_lo
        self.arms = {}
        for s, p, off in SIDES:
            hand = Hand("jaw", _frame(x=TCP), np.ones(1), np.zeros(1), joints=[f"{p}_joint7", f"{p}_joint8"],
                        model_q=functools.partial(model_q, g_lo=g_lo), width=WIDTH,
                        press_point=np.array([0.015, 0.0, 0.0]))  # fmt: skip
            self.arms[s] = Arm(s, urdf, [f"{p}_joint{j}" for j in range(1, 7)], f"{p}_link6", Tb, LIMITS[:, 0].copy(),
                               LIMITS[:, 1].copy(), self.cmd[off:off + 6].copy(), hand, vmax=1.5)  # fmt: skip

    # ---- observations

    def _measured(self):
        """{arm: (6 joints, gripper opening 0 closed .. 1 open)}, measured."""
        out = {}
        for s, e in self._ent.items():
            q = np.asarray(e.get_qpos(), float)[self._idx[s]]
            out[s] = (q[:6], np.clip([q[6] / (WIDTH / 2)], 0.0, 1.0))  # the pads' gap over WIDTH
        return out

    def _head(self):
        self.env._update_render()
        cam = self._cams["head_camera"]
        cam.take_picture()
        return (np.asarray(cam.get_picture("Color"))[..., :3] * 255).clip(0, 255).astype(np.uint8)

    def obs(self, cams=True):
        m, cs = self._measured(), {}
        if cams:
            for name, c in self.env.get_obs()["observation"].items():
                ext = np.eye(4)
                ext[:3] = np.asarray(c["extrinsic_cv"])  # world -> camera
                depth = np.asarray(c["depth"], np.float32) / 1000.0  # RoboTwin reports mm
                cs[name] = Camera(np.asarray(c["rgb"]), depth, np.asarray(c["intrinsic_cv"], float), np.linalg.inv(ext))
                if name in self._cams:  # the picture get_obs just took: the robot's links by their render ids
                    self._seg[name] = np.isin(self._cams[name].get_picture("Segmentation")[..., 1], self.robot_ids)
        return Obs(cs, {s: m[s][0] for s in m}, {s: m[s][1] for s in m}, int(self.env.take_action_cnt),
                   int(self.env.step_lim), self.instruction)  # fmt: skip

    def robot_mask(self, cam, obs):
        return self._seg.get(cam) if obs.cams else None

    def scene(self):
        """Read physical native geometry only in an explicitly privileged episode."""
        if self.information != "privileged":
            raise PermissionError("Simulator geometry is unavailable in observations mode")
        from roboskills.eval.robotwin.state import snapshot

        return snapshot(self)

    # ---- actions

    @property
    def ended(self):
        return self.env.take_action_cnt >= self.env.step_lim

    def act(self, row, frame=True):
        with contextlib.redirect_stdout(io.StringIO()):  # RoboTwin prints a progress line per step
            self.env.take_action(np.asarray(row, float), action_type="qpos")
        self.cmd = np.asarray(row, float)
        self.actions.append(self.cmd.copy())
        if frame:
            self.frames.append(self._head())

    def replay(self, actions):
        """Step a finished episode's native actions (its `actions.npy`) as they were sent; proprioception after."""
        for row in np.asarray(actions, float):
            if self.ended:
                break
            self.act(row)
        return self.obs(cams=False)

    def follow(self, path, stop=None):
        n0 = self.env.take_action_cnt
        rows = np.repeat(self.cmd[None], len(path.t), 0)
        for s, _, off in SIDES:
            if s in path.q:
                rows[:, off : off + 6] = path.q[s]
            if s in path.hand:  # RoboTwin's gripper command for the jaw commands
                rows[:, off + 6] = np.clip((finger(path.hand[s][:, 0], self.g_lo) - self.g_lo) / self.g_span, 0.0, 1.0)
        # the grippers weigh half as much as the joints in the tolerance (0..1 is ~5.5 cm of finger travel)
        idx = simplify(np.c_[rows[:, :6], rows[:, 7:13], rows[:, [6, 13]] * 0.5])
        stopped = False
        for k in idx[1:] if len(idx) > 1 else idx:
            if self.ended:
                break
            self.act(rows[k])
            if stop is not None and stop(self.obs(cams=False)):
                stopped = True
                break
        return Follow(int(self.env.take_action_cnt - n0), self.ended, stopped, self.obs(cams=False))

    def finish(self, out):
        import imageio

        n = 0
        while not self.ended and not self.success:  # hold still to the step limit (RoboTwin would end on success)
            self.act(self.cmd, frame=n % 10 == 0)
            n += 1
        if self.frames:
            imageio.mimwrite(Path(out) / "video.mp4", self.frames, fps=4)
        np.save(Path(out) / "actions.npy", np.array(self.actions))
        return dict(success=bool(self.success), steps=int(self.env.take_action_cnt), step_lim=int(self.env.step_lim),
                    hold_steps=n, seed=self.seed)  # fmt: skip

    def close(self):
        if self.n:
            self.env.close_env(clear_cache=True)
