"""RoboDojo's simulator, warm: one Isaac Sim app for one task, serving many episodes through the benchmark's own reset.

    /isaac-sim/python.sh sim/server.py TASK --port P --vendor VENDOR/RoboDojo --warmup LAYOUT --seed S   # in the container

Each episode starts as a round of RoboDojo's own eval loop (`src/eval_client/main.py`, `eval_env.py`): `env.close()`
(a new stage; the app stays up), `set_saved_layout` from the eval_seed's layout file, `reset(seed=[layout])` (which
relaunches the sim and spawns that layout), `setup_scene` with its stability check, the robot's and reward's init
state. So a warm reset builds the scene as a cold boot's first reset does. The boot includes one such round
(`--warmup`, on the first episode's own layout): the first sim launch after Kit starts is where Isaac Sim sometimes
crashes or hangs, and the runner's boot deadline then catches it before an episode depends on the sim. Actions are the native 25 Hz joint
targets, interpolated over the physics substeps as the evaluator does; success is only the native check.
Native expert planners are blocked (curobo/mplib imports fail). Object state is
available only through the read-only scene operation after a privileged reset;
partial rewards and task-target annotations never enter that snapshot.

Wire: `roboskills.eval.wire` on 127.0.0.1:P, one client at a time; a new connection replaces the old one.
Ops: ping; reset {layout, seed, fixes, dev_dir, official} -> first obs with instruction and robot bases; act {actions, render}
-> obs; observe -> obs with cameras; finish -> {success, steps, step_limit, reason}. Prints `ready` once warm.
Reset's information flag defaults to observations; scene rejects that mode before
reading physical geometry. Privileged scene returns physical objects only.
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
import types
from pathlib import Path

import numpy as np


sys.path.append(str(Path(__file__).resolve().parents[4]))  # the directory holding `roboskills` (mounted)
from roboskills.eval.wire import recv, send  # noqa: E402


ACTION = {"left_arm_joint_state": 6, "right_arm_joint_state": 6, "left_ee_joint_state": 1, "right_ee_joint_state": 1}


class Unstable(RuntimeError):
    """The layout failed RoboDojo's own stability check (as native eval's UnStableError)."""


# ---- inside Isaac Sim


def boot(task, vendor, device):
    """Isaac Sim app + RoboDojo config for `task` (native eval config `arx_x5`, one env, native planners disabled)."""
    sys.path.insert(0, str(vendor))
    os.chdir(vendor)

    class NoPlanners:  # the benchmark's own motion planners are off limits
        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] in ("curobo", "mplib"):
                raise ImportError(f"native planner {name} is disabled")

    sys.meta_path.insert(0, NoPlanners())
    stub = types.ModuleType("env.planner_manager.curobo_planner")
    stub.CuroboPlanner = None
    sys.modules[stub.__name__] = stub
    bind_namespaces(vendor, ("env",))
    threads = str(max(1, min(8, int(os.environ.get("OMP_NUM_THREADS", "4")))))
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "PXR_WORK_THREAD_LIMIT"):
        os.environ[k] = threads  # OMP alone does not cap Kit's workers
    from env.camera_manager.capture.render_sync import add_zero_delay_kit_args
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    cli = parser.parse_args(["--headless", "--enable_cameras", "--device", device])
    add_zero_delay_kit_args(cli)  # camera frames must not lag physics
    if os.getuid() == 0:
        cli.kit_args += " --allow-root"
    gpu = int(device.split(":")[1]) if device.startswith("cuda:") else 0
    cli.kit_args += (
        f" --/plugins/carb.tasking.plugin/threadCount={threads} --/renderer/multiGpu/enabled=false"
        f" --/renderer/activeGpu={gpu}"
    )
    app = AppLauncher(cli).app
    bind_namespaces(vendor, ("env", "utils", "task"))
    from isaaclab.sim import SimulationContext

    aim = SimulationContext.set_camera_view

    def set_camera_view(self, *args, **kwargs):  # aims the UI viewport's camera, not a task camera; on a fresh stage
        try:  # that camera is sometimes not made yet ("Accessed invalid null prim"), which would fail the sim's launch
            aim(self, *args, **kwargs)
        except RuntimeError as e:
            print(f"viewport camera not aimed: {e}", flush=True)

    SimulationContext.set_camera_view = set_camera_view

    from omegaconf import OmegaConf
    from utils.pipeline_utils import process_config, process_randomization

    cfg = vendor / "env_cfg"
    eval_cfg = OmegaConf.load(cfg / "arx_x5.yml")
    eval_cfg.update(task_name=task, num_envs=1, seed=0)
    config = OmegaConf.create(
        {k: OmegaConf.load(cfg / k / f"{eval_cfg.config[k]}.yml") for k in ("sim", "scene", "robot", "camera")}
    )
    config.task_env = OmegaConf.load(vendor / "task/RoboDojo/config" / f"{task}.yml")
    config.eval_cfg = eval_cfg
    config, _ = process_config(process_randomization(config), task)
    config.sim.scene.num_envs = 1
    config.sim.seed = [0]
    for robot in config.robot.robots:
        robot.need_planner = False
    config.camera.default_frequency = 25
    for name in ("common", "cam_head", "cam_left_wrist", "cam_right_wrist"):
        config.camera.annotator[name] = {
            "enabled": True,
            "rgb_capture": {"type": "rgb", "device": "cpu"},
            "depth_capture": {"type": "distance_to_image_plane", "device": "cpu"},
        }
    return app, config


def bind_namespaces(vendor, names):
    """RoboDojo's `env`/`utils`/`task` are namespace dirs; Kit loads its own `utils` package first. Put RoboDojo's first."""
    import importlib.machinery

    for name in names:
        path = str(vendor / name)
        pkg = sys.modules.get(name)
        if pkg is None:
            pkg = sys.modules[name] = types.ModuleType(name)
            pkg.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
        pkg.__path__ = [path] + [p for p in getattr(pkg, "__path__", []) if p != path]
        if pkg.__spec__ is not None:
            pkg.__spec__.submodule_search_locations = pkg.__path__


def build(config, app):
    """The task's env, built once; `start` begins an episode with the evaluator's reset."""
    from copy import deepcopy

    from env.seed_manager.seed_manager import SeedManager
    from task.RoboDojo.task_registry import load_task_class

    _, task_class = load_task_class(config.eval_cfg.task_name)

    class Sim(task_class):
        def __init__(self):
            super().__init__(config, app)
            self.take_action_cnt, self.end_flag, self.success = [0], [True], [False]  # no live episode
            self.scene_manager.layout_manager.replay = True
            self.rounds = 0
            self.action_interval = int(round(1.0 / (self.dt * 25)))
            self.dev, self.official = None, False

        def start(self, layout, seed):
            """One round of the native eval loop; a reset that fails (not the layout's own stability check) runs once
            more on a fresh stage. Isaac Sim now and then reads the closed stage's physics scene while the new one is
            made ("Accessed schema on invalid prim" in SimulationManager.initialize_physics, a stage event not yet
            delivered), which fails that one reset only."""
            try:
                return self.round(layout, seed)
            except Unstable:
                raise
            except Exception:  # noqa: BLE001
                traceback.print_exc()
            print("reset failed: once more on a fresh stage", flush=True)
            self.forget()
            self.round(layout, seed)

        def forget(self):
            """What a failed reset leaves: an env that failed in gym.make (not self.sim) whose destructor would close it
            whenever it is collected, and its close() clears whatever simulation context is current then (a later
            episode's); that env's simulation context; physics scenes of the closed stage; articulations whose physics
            never started (no initial joint positions: the retry's close() fails moving them offscreen; the stage goes
            anyway); the scene objects it spawned, still waiting to be initialized (on the closed stage)."""
            import gc

            from isaaclab.envs import DirectRLEnv
            from isaaclab.sim import SimulationContext
            from isaacsim.core.simulation_manager import SimulationManager

            gc.collect()  # (outside the except block, whose traceback holds the half-made env)
            for (
                obj
            ) in gc.get_objects():  # (type(), not isinstance: a weakref.proxy of a dead object raises ReferenceError)
                if issubclass(type(obj), DirectRLEnv) and obj is not self.sim and not getattr(obj, "_is_closed", True):
                    obj._is_closed = True  # (its close() now does nothing)
            if self.sim is None and SimulationContext.instance() is not None:
                SimulationContext.clear_instance()
            for path, api in list(SimulationManager._physics_scene_apis.items()):
                if not api.GetPrim().IsValid():
                    del SimulationManager._physics_scene_apis[path]
            for objs in self.scene_manager._articulation_objects:
                for key in [k for k, o in objs.items() if not hasattr(o, "initial_joint_positions")]:
                    del objs[key]
            self.scene_manager.pending_initialization.clear()

        def round(self, layout, seed):
            """One round of the native eval loop for one env: close, then EvalEnv.reset + setup_scene."""
            if self.rounds:
                self.close()  # as main.py after every round: the next reset relaunches the sim on a new stage
            self.rounds += 1
            cfg = deepcopy(config.eval_cfg)
            cfg.seed = seed  # the eval_seed picks the layout directory (Assets/Eval_Layout/RoboDojo/arx_x5/<seed>)
            seeds = SeedManager(cfg)
            seeds.init_eval()
            self.env_seeds = [layout]
            self.take_action_cnt, self.end_flag, self.success = [0], [False], [True]
            layouts = self.scene_manager.layout_manager
            layouts.set_saved_layout(0, seeds.get_seed_scene_info(layout))
            super().reset(seed=[layout])
            self.scene_manager.apply_saved_poses(env_idx_list=[0])
            stable, unstable = layouts.check_layout_stability(self)
            if not stable or unstable or not layouts.layout_valid[0]:
                raise Unstable("native layout stability check failed")
            for _ in range(10):
                self.render()
            for i in range(200):
                self.sim_step()
                if i % 5 == 0:
                    self.render()
            self.robot_manager.set_origin_endpose()
            self.robot_manager.set_robot_init_state()
            self.reward_manager.init_state()
            self.run_reward()

        def instruction(self):
            """As native eval (ObsManager -> DescManager): the task template filled with the scene's descriptions."""
            from env.description_manager.desc_manager import DescManager

            desc = DescManager(num_envs=1, description_cfg=config.eval_cfg.get("description", {}) or {}, desc_type="seen",
                               seeds_per_env=list(self.env_seeds))  # fmt: skip
            desc.initialize(self)
            return desc.get_one_description()[0] or self.gen_instruction(0)[0]

        def terminal(self, final=False):
            if not self.end_flag[0]:
                limit = self.take_action_cnt[0] >= self.step_lim
                won = self.reward_manager.get_reward(final_check=final or limit or not self.success[0])[0] > 1 - 1e-3
                if won or final or limit or not self.success[0]:
                    self.end_flag[0], self.success[0] = True, bool(won)
            return self.end_flag[0]

        def joint_action(self, action):
            """One native action: arm joint targets and gripper openings, interpolated over the physics substeps."""
            if set(action) != set(ACTION) or any(len(action[k]) != n or not all(math.isfinite(v) for v in action[k])
                                                 for k, n in ACTION.items()):  # fmt: skip
                raise ValueError(f"an action is {ACTION} finite numbers")
            self.take_action_cnt[0] += 1
            targets, currents, grips = {}, {}, {}
            for robot in self.robot_manager.robot_list:
                arm, grip = (
                    self.robot_manager.process_name(robot.arm_name),
                    self.robot_manager.process_name(robot.gripper_name),
                )
                targets[arm] = {"position": action[arm]}
                currents[arm] = self.robot_manager.get_joint(robot, [0])[0]
                lo, hi = robot.gripper_scale
                u = min(max(float(action[grip][0]), 0.0), 1.0)
                value = lo + (hi - lo) * (u if robot.gripper_move["sign"] == 1 else 1 - u)
                mimic = robot.gripper_move["mimic"]
                grips[grip] = (lo, hi, mimic)
                targets[grip] = {"position": [value, value * mimic[1] + mimic[2]]}
                actual = self.robot_manager.get_end_effector_real_val(robot, [0])[0][0]
                currents[grip] = np.array([actual, actual * mimic[1] + mimic[2]])
            seq = [deepcopy(targets) for _ in range(self.action_interval)]
            n = int(np.floor(self.action_interval * 0.8))
            for i in range(n):
                a = (i + 1) / (n + 1)
                for key, target in targets.items():
                    pos = ((1 - a) * currents[key] + a * np.array(target["position"])).tolist()
                    if key in grips:
                        lo, hi, mimic = grips[key]
                        pos = [float(np.clip(pos[0], lo, hi))] * 2
                        pos[1] = pos[0] * mimic[1] + mimic[2]
                    seq[i][key]["position"] = pos
            control = self.robot_manager.control_manager
            control.push([0], [seq])
            while not control.get_empty([0]):
                super().step(meta_control_list=control.pop([0]))
                self.sim_step(render=False)
            self.reward_manager.step(env_idx_list=[0])
            self.terminal()
            if self.dev:
                self.dev_log()

        def dev_log(self):
            """Dev episodes only, never sent to the agent: native check progress (when it changes) and object poses."""
            try:
                m = self.reward_manager
                row = {
                    "checks_left": len(m.check_list[0]),
                    "queries": [q[2] for q in m.query_list[0]],
                    "success": self.end_flag[0] and self.success[0],
                }
                if m.check_list[0]:
                    row["stage"] = [int(m.check_once(c, 0)) for c in m.check_list[0][0]]
                if row != getattr(self, "_dev_last", None):
                    self._dev_last = row
                    with open(self.dev / "reward.jsonl", "a") as f:
                        f.write(json.dumps({"t": self.take_action_cnt[0], **row}) + "\n")
                if self.take_action_cnt[0] % 25 == 0:
                    lm = m.func_parser.layout_manager
                    poses = {f"{r.get('label')}:{r['inst_name']}": [round(float(v), 4) for p in lm.get_instance_pose(inst_name=r["inst_name"], env_idx=0) for v in p]
                             for recs in lm.object_records_by_type.values() for r in recs.layout_records_by_env[0]}  # fmt: skip
                    (self.dev / "poses.json").write_text(json.dumps({"t": self.take_action_cnt[0], "poses": poses}))
                    with open(
                        self.dev / "poses.jsonl", "a"
                    ) as f:  # every snapshot (dev only), for offline grasp checks
                        f.write(json.dumps({"t": self.take_action_cnt[0], "poses": poses}) + "\n")
            except Exception as e:  # noqa: BLE001  (a log must never end an episode)
                with open(self.dev / "reward.jsonl", "a") as f:
                    f.write(json.dumps({"error": repr(e)[:300]}) + "\n")

    return Sim()


def fix_pr48(env, settle=20):
    """Upstream RoboDojo PR #48 (open; not in the pinned commit): push articulations' authored drive targets to PhysX
    on reset. Without it a SpringButton rests at ratio ~0.79 and checks that it springs back above 0.9 never pass."""
    import torch
    from isaacsim.core.utils.prims import get_prim_at_path
    from isaacsim.core.utils.types import ArticulationAction

    layout = env.scene_manager.layout_manager
    for record in layout.object_records_by_type["Articulation"].layout_records_by_env[0]:
        obj = layout.get_scene_object(0, record["inst_name"])
        if obj is None or not obj.physics_cfg.get("reset_drive", False):
            continue
        by_name = {}
        for path, value in obj.initial_drive_targets.items():
            prim = get_prim_at_path(path)
            if value is not None and prim is not None and prim.IsValid():
                by_name[path.rsplit("/", 1)[-1]] = float(
                    np.deg2rad(value) if obj._get_joint_drive_type(prim) == "angular" else value
                )
        if any(n not in by_name for n in obj.dof_names):
            continue
        targets = np.asarray([by_name[n] for n in obj.dof_names], np.float32)
        lo, hi = (np.asarray(x, np.float32).reshape(-1) for x in (obj.lower_joint_positions, obj.upper_joint_positions))
        start = torch.tensor(np.clip(targets, lo, hi), dtype=torch.float32)
        obj.initial_joint_positions = start.clone()
        obj.set_current_joint_positions(start.clone())
        obj.apply_action(ArticulationAction(joint_positions=torch.tensor(targets)))
    for _ in range(settle):
        env.sim_step()


def observe(env, render, gt=False):
    """Robot state (measured joints, gripper openings, world EE poses), step count; with `render`, every camera's
    RGB-D, K and pose. An `official` episode (RoboDojo's eval observation, env_cfg/arx_x5.yml: no depth, intrinsics
    or extrinsics) gets RGB only; in a dev episode, `gt` then writes the rest to the dev directory (never sent)."""
    state = {}
    rm = env.robot_manager
    for robot in rm.robot_list:
        arm, grip = rm.process_name(robot.arm_name), rm.process_name(robot.gripper_name)
        state[arm] = np.asarray(rm.get_joint(robot, [0])[0], float)
        raw = np.asarray(rm.get_end_effector_real_val(robot, [0])[0], float)
        state[grip.replace("ee_joint_state", "gripper_joint_state")] = raw  # finger travel, m
        state[arm.split("_")[0] + "_ee_pose"] = np.asarray(rm.get_real_endpose(robot, [0])[0], float)  # xyz, wxyz
    if not all(np.isfinite(v).all() for v in state.values()):
        raise RuntimeError("robot joints became nonfinite")
    obs = {"state": state, "steps": env.take_action_cnt[0], "step_limit": env.step_lim, "done": env.end_flag[0]}
    if render:
        import transforms3d as t3d
        from env.camera_manager.capture.render_sync import wait_for_latest_cameras

        wait_for_latest_cameras(env, capture_manager=env.capture_manager, min_passes=1, max_passes=16)
        captures = env.capture_manager.step(env_ids=[0])
        obs["cams"] = {}
        for i in range(env.camera_manager.num_cams):
            cam = env.camera_manager.cameras[0][i]
            K = np.array(cam.get_intrinsics_matrix(), float)
            K[1, 1] = K[
                0, 0
            ]  # RTX renders square pixels, centres at integer indices (the API's fy tilts depth by 2.6 cm)
            K[:2, 2] -= 0.5
            pos, quat = (
                np.asarray(x.cpu() if hasattr(x, "cpu") else x, float) for x in cam.get_world_pose(camera_axes="ros")
            )
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = t3d.quaternions.quat2mat(quat), pos
            ch = captures[i]
            obs["cams"][env.camera_manager.camera_names[0][i]] = {
                "rgb": np.asarray(ch["rgb"][0]["data"])[..., :3].astype(np.uint8),
                "depth": np.asarray(ch["distance_to_image_plane"][0]["data"], np.float32).squeeze(),
                "K": K,
                "T": T,
            }
        if env.official:
            if gt and env.dev:  # with the observation itself (RGB, state), for offline metrics
                np.savez_compressed(env.dev / f"gt_{obs['steps']:05d}.npz", **state, **{f"{c}_{k}": v[k].astype(np.float16) if k == "depth" else v[k]
                                    for c, v in obs["cams"].items() for k in ("rgb", "depth", "K", "T")})  # fmt: skip
            obs["cams"] = {c: {"rgb": v["rgb"]} for c, v in obs["cams"].items()}
    return obs


def handle(env, config, req):
    op = req.get("op")
    if op == "ping":
        return {"pong": True, "task": config.eval_cfg.task_name}
    if op == "reset":
        information = req.get("information", "observations")
        if information not in ("observations", "privileged"):
            raise ValueError("Unknown information mode")
        env.allow_scene = information == "privileged"
        env.public_geometry_cache = {}
        env.dev = Path(req["dev_dir"]) if req.get("dev_dir") else None
        env.official = bool(req.get("official"))
        if env.dev:
            env.dev.mkdir(parents=True, exist_ok=True)
        try:
            env.start(int(req["layout"]), int(req["seed"]))
        except BaseException:
            env.end_flag, env.take_action_cnt = [True], [0]  # no live episode on a failed reset
            raise
        if "robodojo_pr48" in req.get("fixes", ()):
            fix_pr48(env)
        robots = [{k: list(r[k]) for k in ("default_root_pos", "default_root_rot")} for r in config.robot.robots]
        return {**observe(env, True, gt=True), "instruction": env.instruction(), "robots": robots}
    if env.end_flag[0] and env.take_action_cnt[0] == 0:
        raise RuntimeError("no live episode: reset first")
    if op == "act":
        for action in req["actions"]:
            if env.end_flag[0]:
                break  # the env's own end: nothing more runs
            env.joint_action(action)
        return observe(env, bool(req.get("render")))
    if op == "observe":
        return observe(env, True, gt=True)
    if op == "scene":
        if not getattr(env, "allow_scene", False):
            raise PermissionError("Simulator geometry is unavailable in observations mode")
        from roboskills.eval.robodojo.sim.state import snapshot

        return snapshot(env)
    if op == "finish":
        env.terminal(final=True)
        return {
            "success": bool(env.success[0]),
            "steps": env.take_action_cnt[0],
            "step_limit": env.step_lim,
            "reason": "native_success" if env.success[0] else "terminal_failure",
        }
    raise ValueError(f"unknown op {op!r}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("task")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--vendor", type=Path, required=True, help="the RoboDojo checkout (read-only)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--warmup", type=int, required=True, help="layout of the warm-up round (the first episode's)")
    p.add_argument("--seed", type=int, required=True, help="its eval_seed")
    a = p.parse_args()
    faulthandler.register(signal.SIGUSR1, all_threads=True)  # the runner asks a hung boot for its stacks
    t0 = time.time()
    lst = socket.create_server(("127.0.0.1", a.port))  # bound first: a taken port fails fast, before the boot
    app, config = boot(a.task, a.vendor.resolve(), a.device)
    env = build(config, app)
    try:
        env.start(a.warmup, a.seed)
    except Unstable:  # the layout's own failure: it shows again in its episode (any other failure ends the boot)
        traceback.print_exc()
    print(f"ready {a.task} port {a.port} boot {time.time() - t0:.0f}s", flush=True)
    conn = None
    while app.is_running():
        ready, _, _ = select.select([lst] + ([conn] if conn else []), [], [], 60)
        if lst in ready:
            if conn:
                conn.close()
            conn, _ = lst.accept()
            continue
        if conn not in ready or conn is None:
            continue
        try:
            req = recv(conn)
        except (EOFError, OSError):
            conn.close()
            conn = None
            continue
        t = time.time()
        try:
            rep = handle(env, config, req)
        except Exception as e:  # noqa: BLE001  (the reply carries it; a fatal sim error ends the process instead)
            traceback.print_exc()
            rep = {"error": f"{type(e).__name__}: {e}"}
        print(f"{req.get('op')} {len(req.get('actions', ()))} {time.time() - t:.2f}s", flush=True)
        try:
            send(conn, rep)
        except OSError:
            conn.close()
            conn = None


if __name__ == "__main__":
    main()
