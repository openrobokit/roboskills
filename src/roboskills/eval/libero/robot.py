"""LIBERO and LIBERO-Pro behind `roboskills.eval.interface.Robot`: one Panda arm with a parallel gripper in MuJoCo.

Only the host owns the environment and native checker. Any LIBERO-Pro benchmark suite (e.g. `libero_goal`,
`libero_goal_swap` for position and `libero_goal_task` for task perturbations) and task index can be loaded.
Seeded reset never restores saved state; only development sessions may rewind to a checkpoint taken earlier
in the same episode.
"""

import copy
import hashlib
import importlib.metadata
import os
import random
import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from roboskills.eval.interface import Arm, Camera, Follow, Hand, Obs
from roboskills.eval.libero.hand import AXES, WIDTH, finger_q, opening, radians_xml
from roboskills.eval.records import save


SUITES = ("libero_goal_task", "libero_object_task", "libero_spatial_task", "libero_10_task")  # development suites
RUNTIME = "native-v10"
JOINT_DELTA = 0.2
JOINT_KP = 150.0
TRACKING_GATE = 0.2
JOINT_TOLERANCE = 0.002
MATCHED_PROFILES = ("matched-v1", "matched-v2", "matched-v3")


def profile_settings(name="native-v10"):
    """Return explicit native settings; historical defaults keep their identity."""
    profiles = {
        "native-v10": dict(
            runtime=RUNTIME,
            policy_ticks=1000,
            native_horizon=1000,
            settle_ticks=10,
            joint_delta=JOINT_DELTA,
            joint_kp=JOINT_KP,
            tracking_gate=TRACKING_GATE,
            joint_tolerance=JOINT_TOLERANCE,
            hard_reset=False,
            ignore_done=True,
            seeded_rng=False,
            measured_settle_gripper=False,
            reject_initial_success=True,
            mujoco="3.3.2",
        ),
        "matched-v1": dict(
            runtime="native-matched-v1",
            policy_ticks=990,
            native_horizon=1000,
            settle_ticks=10,
            joint_delta=0.05,
            joint_kp=50.0,
            tracking_gate=TRACKING_GATE,
            joint_tolerance=0.01,
            hard_reset=True,
            ignore_done=False,
            seeded_rng=True,
            measured_settle_gripper=True,
            reject_initial_success=False,
            mujoco="3.3.0",
        ),
    }
    profiles["matched-v2"] = dict(
        profiles["matched-v1"], runtime="native-matched-v2", camera_height=512, camera_width=800
    )
    profiles["matched-v3"] = dict(profiles["matched-v2"], runtime="native-matched-v3", respect_native_done=True)
    if name not in profiles:
        raise ValueError(f"Unknown native protocol profile: {name!r}")
    return profiles[name]


class Libero:
    REMOTE = ("scene", "replay")  # served beside the Robot contract (eval/serve.py)
    tracking_gate = TRACKING_GATE
    joint_tolerance = JOINT_TOLERANCE
    profile = "native-v10"
    runtime = RUNTIME

    def __init__(
        self,
        suite,
        directory,
        *,
        task=0,
        profile="native-v10",
        steps=None,
        resolution=None,
        joint_delta=None,
        joint_kp=None,
        tracking_gate=None,
    ):
        settings = profile_settings(profile)
        expected_size = (settings.get("camera_height", 320), settings.get("camera_width", 320))
        resolution = expected_size if resolution is None else resolution
        height, width = (resolution, resolution) if isinstance(resolution, int) else resolution
        steps = settings["policy_ticks"] if steps is None else steps
        joint_delta = settings["joint_delta"] if joint_delta is None else joint_delta
        joint_kp = settings["joint_kp"] if joint_kp is None else joint_kp
        tracking_gate = settings["tracking_gate"] if tracking_gate is None else tracking_gate
        if profile in MATCHED_PROFILES:
            actual = importlib.metadata.version("mujoco")
            if actual != settings["mujoco"]:
                raise ValueError(f"Matched native profile requires MuJoCo {settings['mujoco']}, found {actual}")
            if (steps, height, width, joint_delta, joint_kp, tracking_gate) != (
                settings["policy_ticks"],
                *expected_size,
                settings["joint_delta"],
                settings["joint_kp"],
                settings["tracking_gate"],
            ):
                raise ValueError("Matched native profile settings cannot be overridden")
        from libero import get_libero_path
        from libero.benchmark import get_benchmark
        from libero.envs import OffScreenRenderEnv

        try:
            benchmark = get_benchmark(suite)(task_order_index=0)
        except KeyError:
            raise ValueError(f"Unknown LIBERO benchmark suite: {suite}") from None
        if not 0 <= int(task) < benchmark.n_tasks:
            raise ValueError(f"{suite} has tasks 0-{benchmark.n_tasks - 1}, not {task}")
        if not (0 < joint_delta <= 0.5 and 0 < tracking_gate <= 0.2 and 0 < joint_kp <= 300):
            raise ValueError("Robot joint action range/tracking gate must be positive bounded radians")
        self.tracking_gate = float(tracking_gate)
        self.joint_tolerance = settings["joint_tolerance"]
        self.profile, self.runtime = profile, settings["runtime"]
        self.respect_native_done = settings.get("respect_native_done", False)
        self.suite, self.task, self.directory = suite, int(task), Path(directory)
        self.camera_size = (width, height)
        self.directory.mkdir(parents=True, exist_ok=True)
        task = benchmark.get_task(self.task)
        bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        self.wrapper = OffScreenRenderEnv(
            bddl_file_name=str(bddl),
            camera_heights=height,
            camera_widths=width,
            camera_depths=True,
            controller="JOINT_POSITION",
            control_freq=20,
            horizon=settings["native_horizon"] if profile in MATCHED_PROFILES else steps,
            ignore_done=settings["ignore_done"],
            hard_reset=settings["hard_reset"],
            render_gpu_device_id=int(os.environ.get("MUJOCO_EGL_DEVICE_ID", "0")),
        )
        self.env, self.step_limit, self.dt = self.wrapper.env, steps, 0.05
        # The native reset rebuilds the robot controller from this configuration.
        # Torque limits and scene physics remain SDK defaults; the trajectory
        # controller never assigns simulator state.
        if profile == "native-v10":
            self.env.robots[0].controller_config.update(
                kp=float(joint_kp), damping_ratio=1.0, output_max=float(joint_delta), output_min=-float(joint_delta)
            )
        self.instruction = self.wrapper.language_instruction
        self.information, self.arms, self.n = "observations", None, 0
        self.cameras = ("agentview", "robot0_eye_in_hand")

    def reset(self, seed, *, deadline=None, evaluation=False):
        reserved_check = evaluation and self.profile in {"matched-v2", "matched-v3"} and 1 <= int(seed)
        if int(seed) < 51 and not reserved_check:
            raise ValueError("Official test seeds are outside this development campaign")
        self.deadline = deadline or time.monotonic() + 900
        self.seed, self.steps, self.success = int(seed), 0, False
        self.frames, self.actions, self.ended = [], [], False
        self.command = 1.0
        settings = profile_settings(self.profile)
        if settings["seeded_rng"]:
            random.seed(int(seed))
            np.random.seed(int(seed))
        self.wrapper.seed(int(seed))
        self._gl()
        self.raw = self.wrapper.reset()
        # Procedural resets spawn some objects above their supports. Let physics
        # settle before exposing geometry, as the benchmark's reference protocol
        # does. These reset ticks are recorded separately from the policy budget.
        self.reset_actions = []
        for _ in range(settings["settle_ticks"]):
            if time.monotonic() > self.deadline:
                raise TimeoutError("Native reset deadline exceeded")
            gripper = -1.0
            if settings["measured_settle_gripper"]:
                measured = float(np.clip(np.asarray(self.raw["robot0_gripper_qpos"])[0] / 0.04, 0.0, 1.0))
                gripper = 1.0 - 2.0 * measured
            action = np.r_[np.zeros(7), gripper]
            self.raw, _, _, _ = self.wrapper.step(action)
            self.reset_actions.append(action.tolist())
        if settings["measured_settle_gripper"]:
            self.command = float(np.clip(np.asarray(self.raw["robot0_gripper_qpos"])[0] / 0.04, 0.0, 1.0))
        self.initial_success = bool(self.wrapper.check_success())
        if self.initial_success and settings["reject_initial_success"]:
            raise ValueError("Development scene already satisfies its goal immediately after reset")
        self.n += 1
        state = np.asarray(self.env.sim.get_state().flatten())
        self.reset_fingerprint = hashlib.sha256(state.tobytes()).hexdigest()
        from roboskills.eval.physical import layout_fingerprint

        self.layout_fingerprint = layout_fingerprint(self.env.sim.model, self.env.sim.data, self._robot_roots())
        if settings["hard_reset"]:
            self.arms = None
        self._robot_facts()
        self._capture()
        self._gl(release=True)

    def _robot_facts(self):
        if self.arms is not None:
            return
        robot, sim = self.env.robots[0], self.env.sim
        description = self.directory / "panda.xml"
        # Robot-only native description: excludes fixtures, object meshes and goals.
        xml = radians_xml(robot.robot_model.get_xml())
        description.write_text(ET.tostring(xml, encoding="unicode"))
        joints = list(robot.robot_model.joints)
        fingers = list(robot.gripper.joints)
        limits = np.asarray([sim.model.jnt_range[sim.model.joint_name2id(j)] for j in joints])
        tcp = np.eye(4)
        tcp[:3, :3], tcp[:3, 3] = AXES, [0, 0, 0.097]
        hand = Hand(
            kind="jaw",
            T_ee_tcp=tcp,
            open=np.array([1.0]),
            closed=np.array([0.0]),
            joints=fingers,
            model_q=finger_q,
            width=WIDTH,
        )
        self.arms = {
            "right": Arm(
                "right",
                str(description),
                joints,
                robot.gripper.root_body,
                np.eye(4),
                limits[:, 0],
                limits[:, 1],
                np.array(robot.init_qpos, float),
                hand,
                vmax=1.0,
            )
        }
        self.qpos = np.asarray(robot._ref_joint_pos_indexes)
        self.finger_qpos = np.asarray(robot._ref_gripper_joint_pos_indexes)
        self.site = robot.eef_site_id
        controller = robot.controller
        self.input_lo, self.input_hi = np.asarray(controller.input_min), np.asarray(controller.input_max)
        self.output_lo, self.output_hi = np.asarray(controller.output_min), np.asarray(controller.output_max)
        if any(value.shape != (7,) for value in (self.input_lo, self.input_hi, self.output_lo, self.output_hi)):
            raise ValueError("Native joint controller must expose seven fixed-impedance joint commands")
        if np.any(self.output_hi <= self.output_lo):
            raise ValueError("Native joint action scaling must have positive range")
        self.controller_settings = dict(
            output_max_rad=self.output_hi.tolist(),
            output_min_rad=self.output_lo.tolist(),
            kp=np.asarray(controller.kp).tolist(),
            kd=np.asarray(controller.kd).tolist(),
            tracking_gate_rad=self.tracking_gate,
            endpoint_tolerance_rad=self.joint_tolerance,
        )
        if self.profile in MATCHED_PROFILES:
            if not (
                np.allclose(controller.kp, 50.0)
                and np.allclose(controller.kd, 2 * np.sqrt(50.0))
                and np.allclose(self.output_hi, 0.05)
                and np.allclose(self.output_lo, -0.05)
            ):
                raise ValueError(
                    f"Matched controller defaults disagree with measured settings: {self.controller_settings}"
                )
        self._camera = __import__("robosuite.utils.camera_utils", fromlist=["camera_utils"])
        from roboskills.tools.robokit.robot_model import RobotModel

        # Place the robot-only description by measured FK, so every user of `arms` starts aligned.
        model = RobotModel(self)
        self.calibrate(model)

    def _capture(self):
        self.frames.append(np.asarray(self.raw["agentview_image"])[::-1].copy())

    def calibrate(self, model):
        """Align the robot-only description to measured robot FK, never scene state."""
        arm = self.arms["right"]
        data = self.env.sim.data
        measured = np.eye(4)
        measured[:3, :3] = data.get_body_xmat(arm.ee_link).reshape(3, 3)
        measured[:3, 3] = data.get_body_xpos(arm.ee_link)
        i = model.info["right"]
        limits = i.M.rk.spec.actuated_joint_limits[i.idx]
        if not (np.allclose(limits[:, 0], arm.lo, atol=1e-6) and np.allclose(limits[:, 1], arm.hi, atol=1e-6)):
            raise ValueError("Robot description joint limits disagree with native robot facts")
        predicted = model.links("right", model.fq("right"))[i.ee]
        correction = measured @ np.linalg.inv(predicted)
        arm.T_world_base = correction @ i.M.root.T_world_base  # the model reads its root pose from the arm
        tcp = model.tcp("right")
        site = data.site_xpos[self.site]
        error = float(np.linalg.norm(tcp[:3, 3] - site))
        if error > 0.001:
            raise ValueError(f"Robot TCP and native proprioception disagree by {error:.6f} m")
        save(self.directory / "calibration.json", dict(T_world_model=i.M.root.T_world_base.tolist(), tcp_error_m=error))

    def obs(self, cams=True):
        sim = self.env.sim
        cameras = {}
        if cams:
            height, width = self.raw["agentview_image"].shape[:2]
            for name in self.cameras:
                depth = self._camera.get_real_depth_map(sim, self.raw[name + "_depth"])[::-1].squeeze()
                cameras[name] = Camera(
                    self.raw[name + "_image"][::-1].copy(),
                    depth.copy(),
                    self._camera.get_camera_intrinsic_matrix(sim, name, height, width),
                    self._camera.get_camera_extrinsic_matrix(sim, name),
                )
        fingers = np.asarray(sim.data.qpos[self.finger_qpos])
        return Obs(
            cameras,
            {"right": sim.data.qpos[self.qpos].copy()},
            {"right": opening(fingers)},
            self.steps,
            self.step_limit,
            self.instruction,
        )

    def scene(self):
        """Read physical collision surfaces and joints in privileged episodes only."""
        if self.information != "privileged":
            raise PermissionError("Simulator geometry is unavailable in observations mode")
        from roboskills.eval.physical import CollisionScene

        if not hasattr(self, "_collision_scene"):
            model = self.env.sim.model
            self._collision_scene = CollisionScene(model, self._robot_roots())
        return self._collision_scene.snapshot(self.env.sim.data)

    def checkpoint(self):
        """Capture the full MuJoCo integration state and episode counters for development rewinds."""
        import mujoco

        model, data = self.env.sim.model._model, self.env.sim.data._data
        kind = mujoco.mjtState.mjSTATE_INTEGRATION
        state = np.empty(mujoco.mj_stateSize(model, kind))
        mujoco.mj_getState(model, data, state, kind)
        return dict(
            state=state,
            raw=copy.deepcopy(self.raw),
            counters=dict(steps=self.steps, success=self.success, ended=self.ended, command=self.command),
            frames=len(self.frames),
            actions=len(self.actions),
            timestep=self.env.timestep,
            done=self.env.done,
        )

    def restore(self, point):
        """Return the simulator and episode counters to a development checkpoint."""
        import mujoco

        model, data = self.env.sim.model._model, self.env.sim.data._data
        mujoco.mj_setState(model, data, point["state"], mujoco.mjtState.mjSTATE_INTEGRATION)
        mujoco.mj_forward(model, data)
        self.raw = copy.deepcopy(point["raw"])
        for key, value in point["counters"].items():
            setattr(self, key, value)
        del self.frames[point["frames"] :], self.actions[point["actions"] :]
        self.env.timestep, self.env.done = point["timestep"], point["done"]

    def _gl(self, release=False):
        """Make the offscreen render context current in the calling thread, or release it from this one. The launcher
        makes an env in one thread and serves it from another, and an EGL context is current in one thread at a time;
        robosuite renders without making its context current."""
        context = getattr(getattr(getattr(self, "env", None), "sim", None), "_render_context_offscreen", None)
        if context is None:
            return
        if release:
            from OpenGL import EGL

            EGL.eglMakeCurrent(EGL.eglGetCurrentDisplay(), EGL.EGL_NO_SURFACE, EGL.EGL_NO_SURFACE, EGL.EGL_NO_CONTEXT)
            self._gl_thread = None
        elif getattr(self, "_gl_thread", None) != threading.get_ident():
            context.gl_ctx.make_current()
            self._gl_thread = threading.get_ident()

    def _robot_roots(self):
        robot, model = self.env.robots[0], self.env.sim.model
        return [model.body_name2id(robot.robot_model.root_body), model.body_name2id(robot.gripper.root_body)]

    def _tick(self, target=None, hand=None):
        if time.monotonic() > self.deadline:
            raise TimeoutError("Native episode wall-clock deadline exceeded")
        if self.steps >= self.step_limit or (getattr(self, "respect_native_done", False) and self.ended):
            self.ended = True
            return
        if hand is not None:
            self.command = float(np.clip(np.asarray(hand).reshape(-1)[0], 0, 1))
        action = np.zeros(8)
        if target is not None:
            sim = self.env.sim
            delta = np.asarray(target, float) - sim.data.qpos[self.qpos]
            command = self.input_lo + (delta - self.output_lo) * (self.input_hi - self.input_lo) / (
                self.output_hi - self.output_lo
            )
            action[:7] = np.clip(command, self.input_lo, self.input_hi)
        action[-1] = 1 - 2 * self.command
        self._step(action)

    def _step(self, action):
        self._gl()
        self.raw, _, native_done, _ = self.wrapper.step(action)
        self.actions.append(action.tolist())
        self.steps += 1
        self.success = bool(self.success or self.wrapper.check_success())
        self.ended = self.steps >= self.step_limit or (
            getattr(self, "respect_native_done", False) and bool(native_done)
        )
        if self.steps % 5 == 0:
            self._capture()

    def replay(self, actions):
        """Step a finished episode's native actions (its `actions.npy`) as they were sent; proprioception after."""
        for action in np.asarray(actions, float):
            if self.ended:
                break
            self._step(action)
        return self.obs(cams=False)

    def follow(self, path, stop=None):
        if self.ended:
            return Follow(0, True, False, self.obs(cams=False))
        t = np.asarray(path.t, float)
        if t.ndim != 1 or not len(t) or not np.isfinite(t).all() or np.any(np.diff(t) <= 0) or t[-1] < 0:
            raise ValueError("Trajectory times must be finite and increasing")
        joints = path.q.get("right")
        hold = self.env.sim.data.qpos[self.qpos].copy()
        if joints is not None:
            joints = np.asarray(joints, float)
            if joints.shape != (len(t), 7) or not np.isfinite(joints).all():
                raise ValueError("Joint trajectory must contain one finite seven-joint target per time")
        before, stopped = self.steps, False
        at, settling, errors = min(self.dt, t[-1]), 0, []
        # Advance only as the physical robot follows the planned joint path.
        # Native delta-action limits can stretch nominal planner timing; contact
        # must not begin while its approach pose is still unreached.
        for _ in range(self.step_limit - before):
            if self.ended:
                break
            target = hold
            if joints is not None:
                target = np.array([np.interp(at, t, joints[:, j]) for j in range(7)])
            hand = None
            if "right" in path.hand:
                hand = [np.interp(at, t, np.asarray(path.hand["right"])[:, 0])]
            self._tick(target, hand)
            if stop is not None and stop(self.obs(cams=False)):
                stopped = True
                break
            error = float(np.max(np.abs(self.env.sim.data.qpos[self.qpos] - target)))
            if at >= t[-1]:
                settling += 1
                if settling > 8 and error <= self.joint_tolerance:
                    break
            elif error <= self.tracking_gate:
                at = min(t[-1], at + self.dt)
                errors = []
                continue
            errors.append(error)
            if len(errors) >= 20 and error > self.joint_tolerance and min(errors[-20:-10]) - min(errors[-10:]) < 0.0001:
                # Return a physical shortfall instead of spending the entire
                # episode on a blocked, unchanged target.
                break
        return Follow(self.steps - before, self.ended, stopped, self.obs(cams=False))

    def finish(self, directory):
        import imageio.v2 as imageio

        directory = Path(directory)
        self._capture()
        final = bool(self.wrapper.check_success())
        receipt = dict(
            seed=self.seed,
            runtime=self.runtime,
            profile=self.profile,
            camera_width=self.camera_size[0],
            camera_height=self.camera_size[1],
            control_frequency=20,
            mujoco_version=importlib.metadata.version("mujoco"),
            controller="JOINT_POSITION",
            controller_settings=self.controller_settings,
            suite=self.suite,
            task_id=self.task,
            instruction=self.instruction,
            init_mode="seeded",
            reset_fingerprint=self.reset_fingerprint,
            layout_fingerprint=self.layout_fingerprint,
            initial_success=self.initial_success,
            reset_steps=len(self.reset_actions),
            native_horizon=int(self.env.horizon),
            respect_native_done=getattr(self, "respect_native_done", False),
            success=bool(self.success or final),
            final_success=final,
            steps=self.steps,
            step_limit=self.step_limit,
            information=self.information,
        )
        np.save(directory / "actions.npy", np.asarray(self.actions))
        np.save(directory / "reset-actions.npy", np.asarray(self.reset_actions))
        imageio.mimsave(directory / "video.mp4", self.frames, fps=4, macro_block_size=1)
        save(directory / "native.json", receipt)
        return receipt

    def close(self):
        self.wrapper.close()
