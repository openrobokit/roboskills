"""CaP-X's Robosuite tasks (capgym/cap-x 53e9966, envs/simulators/robosuite_*.py) behind the shared Robot contract.

The scenes, resets, controller, step limits and success checks are CaP-X's, rebuilt on its robosuite fork; no CaP-X
code runs. By default only rendered RGB-D, calibration, proprioception and robot geometry reach the agent.
Explicit privileged episodes additionally expose read-only physical geometry through scene().
"""

import importlib
import json
import types
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from roboskills.eval.interface import Arm, Camera, Follow, Hand, Obs
from roboskills.eval.libero.hand import AXES, WIDTH, finger_q, opening, radians_xml
from roboskills.eval.robosuite import TASKS


SIZE = 512  # camera pixels per side, as CaP-X renders them
HOME = np.array([0.0, -1.585, 0.0, -2.645, 0.0, 1.0, 0.785])  # CaP-X's nut-assembly start: folded out of view
TOL, FINGER_TOL = 0.005, 0.001  # rad, m/s: a motion has settled on its last waypoint and the fingers have stopped
SETTLE = 30  # control steps at most spent settling there (fingers creep the last millimetres)
VIDEO = 10  # control steps per video frame
ENVS = {  # task: robosuite environment, CaP-X's arguments
    "cube_lifting": ("lift.Lift", {}),
    "cube_stack": ("stack.Stack", {}),
    "cube_restack": ("stack.Stack", dict(cube_A_length=0.02, cube_B_length=0.02)),
    "spill_wipe": ("wipe.Wipe", {}),
    "nut_assembly": ("nut_assembly.NutAssemblySquare", {}),
    "two_arm_lift": ("two_arm_lift.TwoArmLift", dict(env_configuration="opposed")),
    "two_arm_handover": ("two_arm_handover.TwoArmHandover", dict(env_configuration="opposed", prehensile=True)),
}
# CaP-X's absolute joint-position controller (panda_joint_ctrl.json; two_arm_lift: panda_joint_ctrl2.json).
JOINT = dict(type="JOINT_POSITION", input_max=10, input_min=-10, output_max=1.0, output_min=-1.0, kp=1500, kd=400,
             kv=200, ramp_ratio=0.2, input_type="absolute", gripper={"type": "GRIP"})  # fmt: skip
CTRL = dict(JOINT, interpolation="linear", kp_limits=[0, 1400])
CTRL2 = dict(JOINT, interpolation=None, kp_limits=[0, 1000], velocity_limits=[-5, 5])


def action(arms, q, u):
    """Robosuite's action, robot by robot: absolute joint targets q, then the gripper's -1 (open) .. 1 (close) for
    jaw commands u (0 closed .. 1 open; none for the sponge). The native gripper moves by the sign alone."""
    return np.concatenate([np.r_[q[a], 1.0 - 2.0 * np.clip(u[a], 0.0, 1.0)] for a in arms])


def command(t, Q, start, now, lead, dt):
    """The target at `now` s on the path from `start` (at 0 s) through rows Q at times t, plus `lead` s of its
    velocity: the native controller's steady lag (kd/kp) behind a moving target, fed forward."""
    t, Q = np.r_[0.0, t], np.vstack([start, Q])
    at = lambda s: np.array([np.interp(s, t, Q[:, j]) for j in range(Q.shape[1])])
    return at(now) + lead * (at(now + dt / 2) - at(now - dt / 2)) / dt


def sides(T_world_cam, bases):
    """Arm names in robot order: as the camera (OpenCV pose in the world) sees their bases, "left" and "right"; one
    arm is "right"."""
    if len(bases) == 1:
        return ["right"]
    c = (np.asarray(bases, float) - T_world_cam[:3, 3]) @ T_world_cam[:3, :3]
    return ["left", "right"] if c[0, 0] / c[0, 2] < c[1, 0] / c[1, 2] else ["right", "left"]


def stacked(sampler, fixtures=None, reference=None, on_top=True):
    """CaP-X's restack layout (StackedObjectRandomSampler) as a sampler's `sample`: the first cube anywhere in range,
    the second on it 1 cm off centre, each turned at random about z."""
    a, b = sampler.mujoco_objects
    x = sampler._sample_x(a.horizontal_radius) + sampler.reference_pos[0]
    y = sampler._sample_y(a.horizontal_radius) + sampler.reference_pos[1]
    z = sampler.z_offset + sampler.reference_pos[2] - a.bottom_offset[-1]
    first = ((x, y, z), sampler._sample_quat(), a)
    z += a.top_offset[-1] - b.bottom_offset[-1] + max(float(sampler.z_offset), 0.01)
    return {a.name: first, b.name: ((x - 0.01, y, z), sampler._sample_quat(), b)}


def scene(task, seed, env=None):
    """CaP-X's environment for `task`, reset to the layout of `seed` as Robo Use seeds CaP-X's tasks."""
    from robosuite.utils.placement_samplers import UniformRandomSampler

    path, kw = ENVS[task]
    module, name = path.split(".")
    cls = getattr(importlib.import_module(f"robosuite.environments.manipulation.{module}"), name)
    two = task.startswith("two_arm")
    ctrl = CTRL2 if task == "two_arm_lift" else CTRL
    if env is None:
        env = cls(
            robots=["Panda", "Panda"] if two else ["Panda"],
            controller_configs=[{"type": "BASIC", "body_parts": {"right": dict(ctrl)}} for _ in range(1 + two)],
            has_renderer=False,
            has_offscreen_renderer=True,
            use_camera_obs=False,
            reward_shaping=True,
            horizon=TASKS[task][0],
            ignore_done=True,  # the step limit is enforced here
            **kw,
        )
    if task in ("cube_stack", "cube_restack"):
        env.placement_initializer = sampler = UniformRandomSampler(
            name="ObjectSampler",
            mujoco_objects=[env.cubeA, env.cubeB],
            x_range=[-0.12, 0.16] if task == "cube_restack" else [-0.18, 0.18],
            y_range=[-0.12, 0.12],
            rotation=None,
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=True,
            reference_pos=env.table_offset,
            z_offset=0.01,
            rng=env.rng,
        )
        if task == "cube_restack":
            sampler.sample = types.MethodType(stacked, sampler)
    # One generator for the robot's start noise and every object sampler, seeded after construction.
    env.rng = rng = np.random.default_rng(seed)
    placer = getattr(env, "placement_initializer", None)
    for s in [placer, *getattr(placer, "samplers", {}).values()] if placer is not None else []:
        s.rng = rng
    env.reset()
    sim, robot = env.sim, env.robots[0]
    if two:  # CaP-X's scene camera, above and behind the table
        cid = sim.model.camera_name2id("agentview")
        quat = np.array([0.653, 0.271, 0.271, 0.653])
        sim.model.cam_pos[cid], sim.model.cam_quat[cid] = [1.5, 0.0, 2.5], quat / np.linalg.norm(quat)
        sim.forward()
    elif task == "nut_assembly":  # the arm folded out of the top view, then 100 steps holding it there, gripper open
        sim.data.qpos[robot._ref_joint_pos_indexes] = HOME
        sim.forward()
        for _ in range(100):
            env.step(np.r_[HOME, -1.0])
    else:  # the wrist turned by pi, then 50 free physics steps
        sim.data.qpos[robot._ref_joint_pos_indexes[6]] -= np.pi
        for _ in range(50):
            sim.forward()
            sim.step()
    return env


def pose(xpos, xmat):
    """A 4x4 pose from MuJoCo's position and (flat) rotation matrix."""
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = np.reshape(xmat, (3, 3)), xpos
    return T


class Robosuite:
    REMOTE = ("scene", "replay")  # served beside the Robot contract (eval/serve.py)

    def __init__(self, task: str, seed: int, out: str, *, information="observations"):
        if information not in {"observations", "privileged"}:
            raise ValueError(f"Unknown information mode {information!r}")
        self.information = information
        self.task, self.seed, self.out = task, int(seed), Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.step_limit, self.instruction = TASKS[task]
        self.env = env = scene(task, self.seed)
        sim = env.sim
        m, d = sim.model, sim.data
        self.dt, self.steps, self.ended = float(env.control_timestep), 0, False
        self.actions, self.frames = [], []
        bases = [d.body_xpos[m.body_name2id(r.robot_model.root_body)] for r in env.robots]
        cid = m.camera_name2id("agentview")
        self.order = sides(pose(d.cam_xpos[cid], d.cam_xmat[cid]) @ np.diag([1.0, -1.0, -1.0, 1.0]), bases)
        two = len(self.order) == 2
        self.cams = {"agentview": "agentview"}
        self.arms, self.qpos, self.fingers, self.lead, self.cmd = {}, {}, {}, {}, {}
        for i, (robot, side) in enumerate(zip(env.robots, self.order)):
            gripper = robot.gripper["right"]
            self.cams[f"wrist_{side}" if two else "wrist"] = f"{robot.robot_model.naming_prefix}eye_in_hand"
            self.qpos[side] = np.asarray(robot._ref_joint_pos_indexes)
            self.fingers[side] = (
                robot._ref_gripper_joint_pos_indexes["right"],
                robot._ref_gripper_joint_vel_indexes["right"],
            )
            control = robot.part_controllers["right"]
            self.lead[side] = np.asarray(control.kd, float) / np.asarray(control.kp, float)
            ee = m.body_name2id(gripper.root_body)
            site = robot.eef_site_id["right"]
            tcp = np.eye(4)
            tcp[:3, :3] = AXES  # +x out of the hand, +y the jaws' squeeze axis
            tcp[:3, 3] = d.xmat[ee].reshape(3, 3).T @ (d.site_xpos[site] - d.xpos[ee])  # the grip site: between pads
            if gripper.dof:
                hand = Hand(
                    "jaw",
                    tcp,
                    np.ones(1),
                    np.zeros(1),
                    joints=list(gripper.joints),
                    model_q=finger_q,
                    width=WIDTH,
                )
            else:  # CaP-X's sponge: no joints, nothing to command; the TCP is its wiping face
                hand = Hand("dex", tcp, np.zeros(0), np.zeros(0))
            model = self.out / f"robot{i}.xml"
            xml = radians_xml(robot.robot_model.get_xml())
            root = xml.find("worldbody/body")  # its pose in the scene is T_world_base
            root.set("pos", "0 0 0")
            root.attrib.pop("quat", None)
            model.write_text(ET.tostring(xml, encoding="unicode"))
            joints = list(robot.robot_model.joints)
            lo, hi = np.array([m.jnt_range[m.joint_name2id(j)] for j in joints]).T
            q = d.qpos[self.qpos[side]].copy()
            base = m.body_name2id(robot.robot_model.root_body)
            T_world_base = pose(d.xpos[base], d.xmat[base])
            self.arms[side] = Arm(
                side, str(model), joints, gripper.root_body, T_world_base, lo, hi, np.r_[HOME[:6], q[6]], hand
            )
            self.cmd[side] = (q, hand.open.copy())
        self.frames.append(self._frame())  # rendering last also releases the EGL context for the serving thread

    def reset(self, seed):
        """Retain the server; rebuild native randomized assets so each seed matches a cold episode."""
        self.seed = int(seed)
        self.env.sim._render_context_offscreen.gl_ctx.make_current()
        self.env._destroy_sim()  # release the old EGL context before native reset rebuilds randomized object sizes
        scene(self.task, self.seed, self.env)
        self.__dict__.pop("_collision_scene", None)
        self.steps, self.ended, self.actions, self.frames = 0, False, [], []
        data = self.env.sim.data
        for side in self.order:
            q = data.qpos[self.qpos[side]].copy()
            self.cmd[side] = (q, self.arms[side].hand.open.copy())
            self.arms[side].home[-1] = q[-1]
        self.frames.append(self._frame())

    def _render(self, **kw):
        """Robosuite's offscreen render with its EGL context current for this call only: the launcher observes from
        its own threads, and an EGL context is current in one thread at a time."""
        from robosuite.renderers.context.egl_context import EGL

        sim = self.env.sim
        sim._render_context_offscreen.gl_ctx.make_current()
        try:
            return sim.render(**kw)
        finally:
            EGL.eglReleaseThread()

    def _frame(self):
        return self._render(width=256, height=256, camera_name="agentview")[::-1].copy()

    def _hand(self, side):
        q = self.env.sim.data.qpos[self.fingers[side][0]]
        return opening(q) if len(q) else np.zeros(0)

    def _camera(self, name):
        from robosuite.utils.camera_utils import get_real_depth_map

        sim = self.env.sim
        rgb, depth = self._render(width=SIZE, height=SIZE, camera_name=name, depth=True)
        cid = sim.model.camera_name2id(name)
        f = 0.5 * SIZE / np.tan(np.radians(sim.model.cam_fovy[cid]) / 2)
        K = np.array([[f, 0.0, (SIZE - 1) / 2], [0.0, f, (SIZE - 1) / 2], [0.0, 0.0, 1.0]])
        T = pose(sim.data.cam_xpos[cid], sim.data.cam_xmat[cid]) @ np.diag([1.0, -1.0, -1.0, 1.0])
        return Camera(rgb[::-1].copy(), get_real_depth_map(sim, depth[::-1]).astype(np.float32), K, T)

    def obs(self, cams: bool = True) -> Obs:
        cameras = {}
        if cams:
            self.env.sim.forward()  # cameras and links where the joints are now
            cameras = {name: self._camera(native) for name, native in self.cams.items()}
        qpos = self.env.sim.data.qpos
        return Obs(
            cameras,
            {s: qpos[self.qpos[s]].copy() for s in self.order},
            {s: self._hand(s) for s in self.order},
            self.steps,
            self.step_limit,
            self.instruction,
        )

    def robot_mask(self, cam: str, obs: Obs):
        return None

    def scene(self):
        """Read physical collision surfaces only in explicitly privileged episodes."""
        if self.information != "privileged":
            raise PermissionError("Simulator geometry is unavailable in observations mode")
        from roboskills.eval.physical import CollisionScene

        if not hasattr(self, "_collision_scene"):
            model = self.env.sim.model
            roots = [model.body_name2id(name) for robot in self.env.robots
                     for name in (robot.robot_model.root_body, robot.gripper["right"].root_body)]  # fmt: skip
            self._collision_scene = CollisionScene(model, roots)
        return self._collision_scene.snapshot(self.env.sim.data)

    def _step(self, cmd):
        if self.ended:
            return
        self._act(action(self.order, {s: q for s, (q, _) in cmd.items()}, {s: u for s, (_, u) in cmd.items()}))

    def _act(self, a):
        self.env.step(a)
        self.steps += 1
        self.actions.append(a)
        if self.steps % VIDEO == 0:
            self.frames.append(self._frame())
        self.ended = self.steps >= self.step_limit

    def replay(self, actions):
        """Step a finished episode's native actions (its `actions.npy`) as they were sent, then hold the last of them
        as the command (so `finish` holds it still, as it held the episode's last command); proprioception after."""
        last = None
        for a in np.asarray(actions, float):
            if self.ended:
                break
            self._act(a)
            last = a
        if last is not None:
            i = 0
            for s in self.order:  # robot order, as `action` lays them out: joint targets, then the gripper (or none)
                q, u = self.cmd[s]
                self.cmd[s] = (last[i : i + len(q)].copy(), (1.0 - last[i + len(q) : i + len(q) + len(u)]) / 2.0)
                i += len(q) + len(u)
        return self.obs(cams=False)

    def _settled(self, path, cmd):
        d = self.env.sim.data
        arms = all(np.abs(d.qpos[self.qpos[s]] - cmd[s][0]).max() < TOL for s in path.q)
        return arms and all(np.abs(d.qvel[self.fingers[s][1]]).max(initial=0.0) < FINGER_TOL for s in path.hand)

    def follow(self, path, stop=None) -> Follow:
        t = np.asarray(path.t, float)
        end = int(np.ceil(t[-1] / self.dt - 1e-9))
        before, stopped, cmd = self.steps, False, dict(self.cmd)
        for k in range(1, end + SETTLE + 1):
            if self.ended:
                break
            now = k * self.dt
            for s, (q, u) in self.cmd.items():
                if s in path.q:
                    q = command(t, path.q[s], q, now, self.lead[s], self.dt)
                if s in path.hand:
                    u = command(t, path.hand[s], u, now, 0.0, self.dt)
                cmd[s] = (q, u)
            if k > end and self._settled(path, cmd):
                break
            self._step(cmd)
            if stop is not None and stop(self.obs(False)):
                stopped = True
                break
        measured = self.obs(False)
        for s, (q, u) in cmd.items():  # hold the last waypoint, or where a stopped arm is
            self.cmd[s] = (measured.q[s] if stopped else np.asarray(path.q[s][-1], float), u) if s in path.q else (q, u)
        return Follow(self.steps - before, self.ended, stopped, measured)

    def _success(self):
        """CaP-X's task_completed(): robosuite's check; restack also fails with both cubes 4 cm over the robot base."""
        env = self.env
        done = bool(env._check_success())
        if done and self.task == "cube_restack":
            d, base = env.sim.data, env.sim.model.body_name2id("fixed_mount0_base")
            z = [
                (d.xmat[base].reshape(3, 3).T @ (d.xpos[i] - d.xpos[base]))[2]
                for i in (env.cubeA_body_id, env.cubeB_body_id)
            ]
            done = not min(z) > 0.04
        return done

    def finish(self, out) -> dict:
        import imageio.v2 as imageio

        for _ in range(10):  # CaP-X judges the state its program leaves; Robo Use after 10 still steps
            if self.ended:
                break
            self._step(self.cmd)
        success = self._success()
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        self.frames.append(self._frame())
        imageio.mimsave(out / "video.mp4", self.frames, fps=10, macro_block_size=1)
        np.save(out / "actions.npy", np.asarray(self.actions))
        result = dict(
            task=self.task,
            seed=self.seed,
            steps=self.steps,
            step_limit=self.step_limit,
            success=bool(success),
            instruction=self.instruction,
            arms=self.order,
            dt=self.dt,
        )
        (out / "native_result.json").write_text(json.dumps(result, indent=2))
        return result

    def close(self):
        self.env.sim._render_context_offscreen.gl_ctx.make_current()
        self.env.close()
