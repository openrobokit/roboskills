"""DexJoCo camera/proprioception adapter for the shared agent-as-policy launcher."""

import json
import os
import sys
import time
import types
from pathlib import Path


os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import mujoco
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from roboskills.eval.dexjoco import LIMITS, TASKS
from roboskills.eval.dexjoco import vendor as vendor_dir
from roboskills.eval.interface import Arm, Camera, Follow, Hand, Obs


SITE_PALM = np.array([[0.0, 0.0, 1.0, 0.05], [0.0, -1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.03], [0.0, 0.0, 0.0, 1.0]])
# TCP among the fingers; its x approaches out of the palm, y squeezes along the fingers.
PALM_TCP = np.array([[1.0, 0.0, 0.0, 0.04], [0.0, 0.0, -1.0, 0.0], [0.0, 1.0, 0.0, 0.07], [0.0, 0.0, 0.0, 1.0]])
OPEN = np.array([0.0] * 12 + [0.263, 0.0, 0.0, 0.0])
CLOSED = np.array([0.0, 1.2, 1.3, 1.2] * 3 + [1.3, 0.5, 0.5, 1.1])


class DexJoCo:
    REMOTE = ("replay",)  # served beside the Robot contract (eval/serve.py)

    def __init__(self, task: str, seed: int, out: str, vendor=None):
        from dexjoco.tasks import CONFIG_MAPPING

        assets = vendor_dir(vendor)
        xmls = assets / "dexjoco/dexjoco/sim/envs/xmls"
        assert task in TASKS
        self.task, self.seed, self.out = task, int(seed), Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.dual = task.startswith("bimanual")
        self.sides = ("right", "left") if self.dual else ("right",)
        self.dt, self.steps, self.step_limit = 0.02, 0, LIMITS[task]
        self._success, self._native_ended, self.ended = False, False, False
        self.env = CONFIG_MAPPING[task]().get_environment(
            policy_mode=True, render_mode="rgb_array", randomize=False, seed=self.seed, randomize_dynamics=False
        )
        self.raw = self.env.unwrapped
        module = sys.modules[type(self.raw).__module__]
        module.time = types.SimpleNamespace(time=time.time, sleep=lambda _: None)
        initial, _ = self.env.reset(seed=self.seed)
        self.m, self.d = self.raw._model, self.raw._data
        self._native_render = self.raw.render
        self._native_renderer = self.raw._mj_viewer if task == "water_plant" else self.raw._viewer
        self._cached_rgb = self._render()
        # Native steps render observations even when the caller needs proprioception only.
        self.raw.render = lambda: self._cached_rgb
        samples = self.m.vis.quality.offsamples
        self.m.vis.quality.offsamples = 0  # MSAA blends integer segmentation colours at object boundaries.
        self._depth = mujoco.Renderer(self.m, 640, 640)
        self._depth.enable_depth_rendering()
        self._segment = mujoco.Renderer(self.m, 640, 640)
        self._segment.enable_segmentation_rendering()
        self.m.vis.quality.offsamples = samples
        self._vopt = mujoco.MjvOption()
        self._vopt.geomgroup[5] = 0  # Native policy cameras hide task target-region annotations.
        for renderer in (self._depth, self._segment):
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
        if task == "water_plant":
            self._cams = dict(wrist=self.raw._wrist_camera_id, front=self.raw._front_camera_id)
        else:
            names = ["ego", "wrist_left", "wrist_right"] if self.dual else ["front", "ego_left", "ego_right", "wrist"]
            self._cams = dict(zip(names, self.raw.camera_id))
        self._site, self._arm, self._hand, self.arms = {}, {}, {}, {}
        for side in self.sides:
            suffix = f"_{side}" if self.dual else ""
            anames = [f"joint{i}{suffix}" for i in range(1, 8)]
            fingers = ("rf", "mf", "ff", "th") if side == "left" else ("ff", "mf", "rf", "th")
            hnames = [f"{f}j{i}{suffix}" for f in fingers for i in range(4)]
            self._site[side] = self.m.site(f"attachment_site{suffix}").id
            self._arm[side] = [int(self.m.joint(n).qposadr[0]) for n in anames]
            self._hand[side] = [int(self.m.joint(n).qposadr[0]) for n in hnames]
            h = Hand(
                "dex",
                PALM_TCP.copy(),
                OPEN.copy(),
                CLOSED.copy(),
                joints=hnames,
                presets={"flat": OPEN.copy(), "point": np.r_[OPEN[:4], CLOSED[4:]]},
            )
            if side == "left":
                h.presets["point"] = CLOSED.copy()
                h.presets["point"][8:12] = OPEN[8:12]
            lo, hi = np.array([self.m.joint(n).range for n in anames]).T
            robot_xml = xmls / (f"panda_allegro_{side}.xml" if self.dual else "panda_allegro_copy.xml")
            base_id = self.m.body(f"base{suffix}").id
            base = np.eye(4)
            base[:3, :3] = self.d.xmat[base_id].reshape(3, 3)
            base[:3, 3] = self.d.xpos[base_id]
            self.arms[side] = Arm(
                side,
                str(robot_xml),
                anames,
                f"allegro_palm{suffix}",
                base,
                lo,
                hi,
                self.d.qpos[self._arm[side]].copy(),
                h,
                vmax=1.0,
            )
            contact = mujoco.MjData(self.m)
            contact.qpos[:] = self.d.qpos
            contact.qpos[self._hand[side]] = h.closed
            mujoco.mj_kinematics(self.m, contact)
            palm = self.site(side) @ SITE_PALM
            tip_body = self.m.body(f"ff_tip{suffix}").id
            tip = next(
                g
                for g in range(self.m.ngeom)
                if self.m.geom_bodyid[g] == tip_body and self.m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE
            )
            cap_axis = contact.geom_xmat[tip].reshape(3, 3)[:, 2]
            surface = (
                contact.geom_xpos[tip]
                + palm[:3, 0] * self.m.geom_size[tip, 0]
                + cap_axis * np.sign(cap_axis @ palm[:3, 0]) * self.m.geom_size[tip, 1]
            )
            h.press_point = (np.linalg.inv(palm @ PALM_TCP) @ np.r_[surface, 1.0])[:3]
        robot_bodies = {self.m.body(f"base_{s}" if self.dual else "base").id for s in self.sides}
        for i in range(1, self.m.nbody):
            if int(self.m.body_parentid[i]) in robot_bodies:
                robot_bodies.add(i)
        self._robot_bodies = np.array(sorted(robot_bodies))
        self._mask = {}
        cfg = yaml.safe_load((assets / f"configs/rand_obj/{task}.yaml").read_text())
        self.instruction = cfg["prompt"]
        self.cmd = {s: (self.site(s), self.d.qpos[self._hand[s]].copy()) for s in self.sides}
        self.frames, self.actions = [initial["ego" if self.dual else "front"][::2, ::2]], []
        self.record = True
        self._video_index = list(self._cams).index("ego" if self.dual else "front")
        from OpenGL import EGL

        EGL.eglMakeCurrent(EGL.eglGetCurrentDisplay(), EGL.EGL_NO_SURFACE, EGL.EGL_NO_SURFACE, EGL.EGL_NO_CONTEXT)

    def site(self, side: str) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3] = self.d.site_xmat[self._site[side]].reshape(3, 3)
        T[:3, 3] = self.d.site_xpos[self._site[side]]
        return T

    def _render(self):
        # Gymnasium switches contexts only among its own viewers, not our RGB-D renderers.
        self._native_renderer._viewers["rgb_array"].make_context_current()
        return self._native_render()

    def obs(self, cams: bool = True) -> Obs:
        cameras = {}
        if cams:
            self._cached_rgb = self._render()
            for (name, cid), rgb in zip(self._cams.items(), self._cached_rgb):
                self._depth.update_scene(self.d, camera=cid, scene_option=self._vopt)
                depth = self._depth.render().copy()
                h, w = depth.shape
                f = 0.5 * h / np.tan(np.radians(self.m.cam_fovy[cid]) / 2)
                K = np.array([[f, 0.0, (w - 1) / 2], [0.0, f, (h - 1) / 2], [0.0, 0.0, 1.0]])
                T = np.eye(4)
                T[:3, :3] = self.d.cam_xmat[cid].reshape(3, 3) @ np.diag([1.0, -1.0, -1.0])
                T[:3, 3] = self.d.cam_xpos[cid]
                cameras[name] = Camera(rgb, depth, K, T)
                self._segment.update_scene(self.d, camera=cid, scene_option=self._vopt)
                seg = self._segment.render()
                gid = np.maximum(seg[..., 0], 0)
                self._mask[name] = (seg[..., 0] >= 0) & np.isin(self.m.geom_bodyid[gid], self._robot_bodies)
        return Obs(
            cameras,
            {s: self.d.qpos[self._arm[s]].copy() for s in self.sides},
            {s: self.d.qpos[self._hand[s]].copy() for s in self.sides},
            self.steps,
            self.step_limit,
            self.instruction,
        )

    def robot_mask(self, cam: str, obs: Obs) -> np.ndarray:
        return self._mask[cam].copy()

    def act(self, targets: dict) -> None:
        if self.ended:
            return
        self.cmd.update(targets)
        parts = {}
        for s, (T, u) in self.cmd.items():
            q = Rotation.from_matrix(T[:3, :3]).as_quat()[[3, 0, 1, 2]]
            parts[s] = np.r_[T[:3, 3], q, u]
        a = (
            np.r_[parts["right"][:7], parts["left"][:7], parts["right"][7:], parts["left"][7:]]
            if self.dual
            else parts["right"]
        )
        self._step(a)

    def _step(self, a) -> None:
        _, _, terminated, truncated, info = self.env.step(a)
        if not self._native_ended:
            self._success = bool(info.get("succeed", False))
            self._native_ended = bool(terminated or truncated)
        self.steps += 1
        self.actions.append(a.copy())
        if self.record and self.steps % 10 == 0:
            self.frames.append(self._render()[self._video_index][::2, ::2].copy())
        self.ended = self.steps >= self.step_limit

    def replay(self, actions) -> Obs:
        """Step a finished episode's native actions (its `actions.npy`) as they were sent; proprioception after."""
        for a in np.asarray(actions, float):
            if self.ended:
                break
            self._step(a)
        return self.obs(cams=False)

    def follow(self, path, stop=None) -> Follow:
        before = self.steps
        stopped = False
        previous = {s: self.cmd[s][0].copy() for s in self.sides}
        final = {}
        for t in np.arange(self.dt, float(path.t[-1]) + self.dt / 2, self.dt):
            targets = {}
            for s in self.sides:
                T, u = (x.copy() for x in self.cmd[s])
                if s in path.T:
                    k = min(int(np.searchsorted(path.t, t)), len(path.t) - 1)
                    T = path.T[s][k] @ np.linalg.inv(SITE_PALM)
                    final[s] = T.copy()
                    # Feed forward the native overdamped operational-space controller's lag.
                    lead = 0.39 * (T[:3, 3] - previous[s][:3, 3]) / self.dt
                    lead *= min(1.0, 0.12 / max(np.linalg.norm(lead), 1e-9))
                    turn = Rotation.from_matrix(T[:3, :3] @ previous[s][:3, :3].T).as_rotvec() * (0.57 / self.dt)
                    turn *= min(1.0, 0.3 / max(np.linalg.norm(turn), 1e-9))
                    previous[s] = T.copy()
                    T[:3, 3] += lead
                    T[:3, :3] = Rotation.from_rotvec(turn).as_matrix() @ T[:3, :3]
                if s in path.hand:
                    u = np.array([np.interp(t, path.t, path.hand[s][:, j]) for j in range(16)])
                targets[s] = (T, u)
            self.act(targets)
            if self.ended or (stop is not None and (stopped := bool(stop(self.obs(False))))):
                break
        if not stopped:
            for s, T in final.items():
                self.cmd[s] = (T, self.cmd[s][1])
            for _ in range(40):
                residual = max(
                    [
                        np.linalg.norm(self.site(s)[:3, 3] - T[:3, 3]) / 0.006
                        + Rotation.from_matrix(self.site(s)[:3, :3] @ T[:3, :3].T).magnitude() / 0.06
                        for s, T in final.items()
                    ]
                    or [0.0]
                )
                if residual <= 1.0 or self.ended:
                    break
                self.act(self.cmd)
                if stop is not None and (stopped := bool(stop(self.obs(False)))):
                    break
        if stopped:
            self.cmd = {s: (self.site(s), self.d.qpos[self._hand[s]].copy()) for s in self.sides}
        return Follow(self.steps - before, self.ended, stopped, self.obs(False))

    def finish(self, out) -> dict:
        while not self.ended and not self._native_ended:
            self.act(self.cmd)
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(out / "video.mp4", self.frames, fps=5)
        np.save(out / "actions.npy", np.asarray(self.actions))
        result = dict(
            task=self.task, seed=self.seed, steps=self.steps, success=self._success, native_ended=self._native_ended
        )
        (out / "native_result.json").write_text(json.dumps(result, indent=2))
        self.close()
        return result

    def close(self) -> None:
        """Release the renderers and environment, once (`finish` and the server's exit both call it)."""
        if getattr(self, "_closed", False):
            return
        self._closed = True
        self._depth.close()
        self._segment.close()
        self.env.close()
