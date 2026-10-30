"""One native RoboDojo scene batch, with an independent episode socket per env.

Once every env has its first action, all live envs share one physics clock that
advances without waiting for any agent. Only commanded actions spend an env's
step budget; an env with no queued action holds its targets, and those idle
physical steps are recorded separately. One wave of layouts is reset together
before clients connect. Each client claims its preassigned layout once.

--official: every env gets RoboDojo's eval observation (camera RGB only: no depth, K or camera pose), as server.py's
`official` reset; the episode servers' adapters then run with obs=official.

--events PATH: the batch's event log for `replay.py`: the batch, then every physics tick's commands, every rendered
observation and every episode's end (its outcome, final joints and object poses), in the order they happened. Idle
envs keep moving while others act, so an env's own actions alone do not reproduce its episode.
"""

import argparse
import inspect
import json
import math
import os
import queue
import socket
import sys
import threading
import time
import traceback
from collections import deque
from copy import deepcopy
from pathlib import Path

import numpy as np


sys.path.append(str(Path(__file__).resolve().parents[4]))  # the directory holding `roboskills` (mounted)
from roboskills.eval.robodojo.layouts import check_layout  # noqa: E402
from roboskills.eval.robodojo.sim.server import ACTION, boot  # noqa: E402
from roboskills.eval.wire import recv, send  # noqa: E402


KIT_THREADS = 4  # carb.tasking workers at least (see main)
CONVEYOR_TASKS = ("pick_from_conveyor_by_image", "match_and_pick_from_conveyor")
BLACK = 10  # a first frame's mean RGB below this is a failed render: 0-5 when one fails, 59-114 when lit


def clock_mode(task):
    """RD_CLOCK: `shared` (physics runs on while agents think) or `frozen` (physics advances only when every live env
    has an action queued, as RoboDojo's own eval and the serial runner). Default: frozen for the conveyor tasks (a
    belt otherwise carries objects off while the planner thinks), shared for the rest."""
    return os.environ.get("RD_CLOCK") or ("frozen" if task in CONVEYOR_TASKS else "shared")


def fix_particle_material_api():
    """Accept upstream's zero-valued drag/lift on Isaac Sim 6's PBD API (the fluid tasks)."""
    from omni.physx.scripts import particleUtils

    native = particleUtils.add_pbd_particle_material
    parameters = inspect.signature(native).parameters
    unsupported = ("drag", "lift")
    if all(name in parameters for name in unsupported):
        return

    def compatible(*args, **kwargs):
        for name in unsupported:
            if name not in parameters:
                value = kwargs.pop(name, None)
                if value not in (None, 0, 0.0):
                    raise ValueError(f"unsupported nonzero PBD material {name}: {value}")
        return native(*args, **kwargs)

    particleUtils.add_pbd_particle_material = compatible
    print("enabled zero-valued PBD drag/lift compatibility", flush=True)


def count_step(env, commanded):
    """Keep action budget distinct from elapsed physical control intervals."""
    live = [i for i, done in enumerate(env.end_flag) if not done]
    for e in live:
        env.physical_steps[e] += 1
        if e in commanded:
            env.commanded_steps[e] += 1
            env.take_action_cnt[e] += 1
        else:
            env.idle_steps[e] += 1
    return live


def fix_buttons(env, settle=20):
    """Upstream RoboDojo PR #48 (open; not in the pinned commit): push articulations' authored drive targets to PhysX
    on reset. Without it a SpringButton rests at ratio ~0.79 and checks that it springs back above 0.9 never pass."""
    import torch
    from isaacsim.core.utils.prims import get_prim_at_path
    from isaacsim.core.utils.types import ArticulationAction

    layout = env.scene_manager.layout_manager
    for e in range(env.num_envs):
        for record in layout.object_records_by_type["Articulation"].layout_records_by_env[e]:
            obj = layout.get_scene_object(e, record["inst_name"])
            if obj is None or not obj.physics_cfg.get("reset_drive", False):
                continue
            by_name = {}
            for path, value in obj.initial_drive_targets.items():
                prim = get_prim_at_path(path)
                if value is not None and prim is not None and prim.IsValid():
                    angular = obj._get_joint_drive_type(prim) == "angular"
                    by_name[path.rsplit("/", 1)[-1]] = float(np.deg2rad(value) if angular else value)
            if any(n not in by_name for n in obj.dof_names):
                continue
            targets = np.asarray([by_name[n] for n in obj.dof_names], np.float32)
            lo, hi = (
                np.asarray(x, np.float32).reshape(-1) for x in (obj.lower_joint_positions, obj.upper_joint_positions)
            )
            start = torch.tensor(np.clip(targets, lo, hi), dtype=torch.float32)
            obj.initial_joint_positions = start.clone()
            obj.set_current_joint_positions(start.clone())
            obj.apply_action(ArticulationAction(joint_positions=torch.tensor(targets)))
    for _ in range(settle):
        env.sim_step()


def build(config, app, layouts, seed, official=False):
    from env.description_manager.desc_manager import DescManager
    from env.seed_manager.seed_manager import SeedManager
    from task.RoboDojo.task_registry import load_task_class

    _, task_class = load_task_class(config.eval_cfg.task_name)

    class VectorEnv(task_class):
        def __init__(self):
            def phase(name):
                print(json.dumps(dict(event="startup_phase", phase=name, time=time.time())), flush=True)

            phase("native_constructor")
            super().__init__(config, app)
            self.eval_seed = seed  # as EvalEnv sets it: tasks read it (imitate_sorting_sequence's trajectories)
            phase("load_saved_layouts")
            self.take_action_cnt = [0] * self.num_envs
            self.idle_steps, self.physical_steps, self.commanded_steps = ([0] * self.num_envs for _ in range(3))
            self.end_flag = [False] * self.num_envs
            self.success = [True] * self.num_envs
            self.action_interval = round(1 / (self.dt * 25))
            self.scene_manager.layout_manager.replay = True
            cfg = deepcopy(config.eval_cfg)
            cfg.seed = seed
            seeds = SeedManager(cfg)
            seeds.init_eval()
            for e, layout in enumerate(layouts):
                self.scene_manager.layout_manager.set_saved_layout(e, seeds.get_seed_scene_info(layout))
            self.env_seeds = layouts
            phase("native_reset")
            super().reset(seed=layouts)
            phase("apply_saved_poses")
            self.scene_manager.apply_saved_poses(env_idx_list=list(range(self.num_envs)))
            phase("layout_stability")
            stable, unstable = self.scene_manager.layout_manager.check_layout_stability(self)
            if not stable or unstable or not all(self.scene_manager.layout_manager.layout_valid):
                raise RuntimeError(f"native layout stability check failed: {unstable}")
            phase("initial_render")
            for _ in range(10):
                self.render()
            phase("settle_physics")
            for i in range(200):
                self.sim_step()
                if i % 5 == 0:
                    self.render()
            phase("robot_and_reward_reset")
            self.robot_manager.set_origin_endpose()
            self.robot_manager.set_robot_init_state()
            self.reward_manager.init_state()
            self.run_reward()
            if hasattr(self, "get_score"):  # the task's process score, registered as EvalEnv.run_eval does
                self.get_score()
            self.support_arm()
            desc = DescManager(
                num_envs=self.num_envs,
                description_cfg=config.eval_cfg.get("description", {}) or {},
                desc_type="seen",
                seeds_per_env=layouts,
            )
            desc.initialize(self)
            self.instructions = [v or self.gen_instruction(e)[0] for e, v in enumerate(desc.get_one_description())]
            self.fixes = ["robodojo_pr48"] if config.eval_cfg.task_name == "swap_blocks" else []
            if self.fixes:
                fix_buttons(self)
            self.captures = None
            self.frozen_obs = {}
            self.official = official
            if os.environ.get("RD_STEP_LIMIT"):  # e.g. 10**9: the native step limit lifted (not the official budget)
                self.step_lim = int(os.environ["RD_STEP_LIMIT"])
            print(
                json.dumps(
                    dict(
                        event="scene",
                        num_envs=self.num_envs,
                        layouts=layouts,
                        env_spacing=float(config.sim.scene.env_spacing),
                        dt=self.dt,
                        action_substeps=self.action_interval,
                        cameras=[
                            dict(name=name, resolution=list(cam._resolution))
                            for name, cam in zip(self.camera_manager.camera_names[0], self.camera_manager.cameras[0])
                        ],
                    )
                ),
                flush=True,
            )

        def truth(self, e):
            """Dev episodes only, never sent to the agent: the native check stage still open (each condition and
            whether it holds now) and every object's pose [x y z qw qx qy qz], for a script's developer."""
            try:
                m = self.reward_manager
                return dict(stages_left=len(m.check_list[e]),
                            checks=[[repr(c), bool(m.check_once(c, e))] for c in (m.check_list[e][0] if m.check_list[e] else [])],
                            poses={k: [round(v, 4) for v in p] for k, p in self.poses(e).items()})  # fmt: skip
            except Exception as exc:  # noqa: BLE001  (a log must never end an episode)
                return dict(error=repr(exc)[:300])

        def poses(self, e):
            """Every object's pose [x y z qw qx qy qz] in env e, by `label:instance`."""
            lm = self.reward_manager.func_parser.layout_manager
            return {f"{r.get('label')}:{r['inst_name']}": [float(v) for p in lm.get_instance_pose(inst_name=r["inst_name"], env_idx=e) for v in p]
                    for recs in lm.object_records_by_type.values() for r in recs.layout_records_by_env[e]}  # fmt: skip

        def terminal(self, e, final=False):
            if self.end_flag[e]:
                return
            final = final or self.take_action_cnt[e] >= self.step_lim or not self.success[e]
            # Native final checks mutate per-env lists. Finalizing one episode
            # must not consume the other episodes' final checks/scores.
            saved = {}
            if final:
                for name in ("final_check_list", "final_score_list"):
                    saved[name] = getattr(self.reward_manager, name)
                    setattr(self.reward_manager, name, [v if i == e else [] for i, v in enumerate(saved[name])])
            try:
                won = self.reward_manager.get_reward(final_check=final)[e] > 1 - 1e-3
            finally:
                for name, values in saved.items():
                    setattr(self.reward_manager, name, values)
            if won or final:
                self.end_flag[e], self.success[e] = True, bool(won)

        def score(self, e):
            """EvalEnv's episode score: 1 on success, else the process score reached (0-1; 0 without one)."""
            if self.success[e]:
                return 1.0
            return float(self.reward_manager.get_score(reward_lst=[0.0] * self.num_envs)[e]) / 100

        def sequence(self, e, action):
            if set(action) != set(ACTION) or any(
                len(action[k]) != n or not all(math.isfinite(v) for v in action[k]) for k, n in ACTION.items()
            ):
                raise ValueError(f"an action is {ACTION} finite numbers")
            targets, currents, grips = {}, {}, {}
            rm = self.robot_manager
            for robot in (
                r for r in rm.robot_list if r.type == "target"
            ):  # a task's support arm replays its own trajectory
                arm, grip = rm.process_name(robot.arm_name), rm.process_name(robot.gripper_name)
                targets[arm] = {"position": action[arm]}
                currents[arm] = rm.get_joint(robot, [e])[e]
                lo, hi = robot.gripper_scale
                u = float(np.clip(action[grip][0], 0, 1))
                value = lo + (hi - lo) * (u if robot.gripper_move["sign"] == 1 else 1 - u)
                mimic = robot.gripper_move["mimic"]
                grips[grip] = (lo, hi, mimic)
                targets[grip] = {"position": [value, value * mimic[1] + mimic[2]]}
                actual = rm.get_end_effector_real_val(robot, [e])[e][0]
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
            queue = getattr(self, "support_arm_action", None)
            for step in seq if queue else []:  # as EvalEnv.process_control_info: one queued command per sub-step
                if not queue[e]:
                    break
                step.update(queue[e].pop(0))
            return seq

        def support_arm(self, envs=None):
            """A task with a support arm (make_kong, play_tic_tac_toe, imitate_sorting_sequence): queue its trajectory
            and check it, as EvalEnv does after each reward step."""
            if not getattr(self, "interact", False):
                return
            for e in range(self.num_envs) if envs is None else envs:
                if hasattr(self, "query_support_arm_traj"):
                    self.query_support_arm_traj(env_idx=e)
                if hasattr(self, "check_support_arm_stable"):
                    self.check_support_arm_stable(env_idx=e)

        def joint_actions(self, actions):
            t0 = time.time()
            ids = sorted(actions)
            sequences = [self.sequence(e, actions[e]) for e in ids]
            control = self.robot_manager.control_manager
            if ids:
                control.push(ids, sequences)
            # Waiting does not spend the action budget, but native task checks
            # still see physical changes in every live environment.
            live = count_step(self, ids)
            timings = dict(sequence_seconds=time.time() - t0, control_seconds=0.0, simulation_seconds=0.0)
            for _ in range(self.action_interval):
                t0 = time.time()
                if ids:
                    super().step(meta_control_list=control.pop(ids))
                timings["control_seconds"] += time.time() - t0
                t0 = time.time()
                self.sim_step(render=False)
                timings["simulation_seconds"] += time.time() - t0
            self.captures = None
            t0 = time.time()
            self.reward_manager.step(env_idx_list=live)
            self.support_arm(live)
            for e in live:
                self.terminal(e)
                if self.end_flag[e]:
                    # Preserve the terminal observation while other envs keep
                    # stepping, matching the original single-env adapter.
                    self.frozen_obs[e] = deepcopy(self.observe(e, True))
            timings["reward_seconds"] = time.time() - t0
            return timings

        def observe(self, e, render):
            if e in self.frozen_obs:
                obs = dict(self.frozen_obs[e])
                if not render:
                    obs.pop("cams", None)
                return obs
            rm, state = self.robot_manager, {}
            for robot in (r for r in rm.robot_list if r.type == "target"):
                arm, grip = rm.process_name(robot.arm_name), rm.process_name(robot.gripper_name)
                state[arm] = np.asarray(rm.get_joint(robot, [e])[e], float)
                gripper = rm.get_end_effector_real_val(robot, [e])[e]
                state[grip.replace("ee_joint_state", "gripper_joint_state")] = np.asarray(gripper, float)
                pose = rm.get_real_endpose(robot, [e], is_relative=True)[e]
                state[arm.split("_")[0] + "_ee_pose"] = np.asarray(pose, float)
            if not all(np.isfinite(v).all() for v in state.values()):
                raise RuntimeError(f"env {e}: nonfinite robot state")
            obs = dict(
                state=state,
                steps=self.take_action_cnt[e],
                step_limit=self.step_lim,
                done=self.end_flag[e],
                physical_steps=self.physical_steps[e],
                idle_steps=self.idle_steps[e],
                commanded_steps=self.commanded_steps[e],
            )
            if render:
                import transforms3d as t3d
                from env.camera_manager.capture.render_sync import wait_for_latest_cameras

                if self.captures is None:
                    t0 = time.time()
                    wait_for_latest_cameras(self, capture_manager=self.capture_manager, min_passes=1, max_passes=16)
                    self.captures = self.capture_manager.step()
                    print(
                        json.dumps(dict(event="capture", seconds=time.time() - t0, steps=list(self.take_action_cnt))),
                        flush=True,
                    )
                obs["cams"] = {}
                origin = np.asarray(self.env_origins[e].cpu(), float)
                for i, cam in enumerate(self.camera_manager.cameras[e]):
                    K = np.asarray(cam.get_intrinsics_matrix(), float)
                    K[1, 1] = K[0, 0]
                    K[:2, 2] -= 0.5
                    pos, quat = [
                        np.asarray(x.cpu() if hasattr(x, "cpu") else x, float)
                        for x in cam.get_world_pose(camera_axes="ros")
                    ]
                    T = np.eye(4)
                    T[:3, :3], T[:3, 3] = t3d.quaternions.quat2mat(quat), pos - origin
                    ch = self.captures[i]
                    obs["cams"][self.camera_manager.camera_names[e][i]] = dict(
                        rgb=np.asarray(ch["rgb"][e]["data"])[..., :3].astype(np.uint8),
                        depth=np.asarray(ch["distance_to_image_plane"][e]["data"], np.float32).squeeze(),
                        K=K,
                        T=T,
                    )
                if self.official:
                    obs["cams"] = {c: {"rgb": v["rgb"]} for c, v in obs["cams"].items()}
            return obs

    return VectorEnv()


def check_cameras(env):
    """Refuse a batch whose cameras render black. Without NVIDIA's NGX library RoboDojo's DLAA leaves some cameras'
    tiles unlit for the sim's whole life while depth stays right, and the agents would act blind."""
    for e in range(env.num_envs):
        for name, cam in env.observe(e, True)["cams"].items():
            if (mean := float(cam["rgb"].mean())) < BLACK:
                raise RuntimeError(f"camera {name} renders black in env {e} (mean RGB {mean:.1f}); is NGX mounted?")


def events(path, **batch):
    """The event log's writer (`--events`), its first line the batch; without a path, a writer that keeps nothing."""
    if not path:
        return lambda **event: None
    f = open(path, "w", buffering=1)  # one line per event, flushed: a crashed batch keeps what it ran

    def log(**event):
        f.write(json.dumps(event, default=lambda x: x.tolist()) + "\n")

    log(op="batch", **batch)
    return log


def ending(env, e):
    """An ended episode as the log records it: its outcome, final joints and object poses."""
    try:
        poses = env.poses(e)
    except Exception as exc:  # noqa: BLE001  (a log must never end an episode)
        poses = dict(error=repr(exc)[:300])
    return dict(success=bool(env.success[e]), score=env.score(e), state=env.observe(e, False)["state"], poses=poses)


def serve(env, config, ports, layouts, seed, clock="shared", log=None):
    n = len(ports)
    log = log or events(None)
    jobs = queue.Queue()
    owners, finished, pending = {}, set(), {}
    started = False

    def talk(e, conn):
        with conn:
            try:
                while True:
                    req = recv(conn)
                    box = queue.Queue()
                    jobs.put((e, req, box, id(conn)))
                    send(conn, box.get())
            except (OSError, EOFError):
                jobs.put((e, {"op": "disconnect"}, queue.Queue(), id(conn)))

    def listen(e, listener):
        while True:
            conn, _ = listener.accept()
            threading.Thread(target=talk, args=(e, conn), daemon=True).start()

    def look(e, render):
        obs = env.observe(e, render)
        if render:  # a capture renders the whole scene: the replay renders where the run did
            log(op="observe", env=e)
        return obs

    listeners = [socket.create_server(("127.0.0.1", p)) for p in ports]
    for e, listener in enumerate(listeners):
        threading.Thread(target=listen, args=(e, listener), daemon=True).start()
    print(f"ready vector_sim num_envs={n} ports={ports} clock={clock}", flush=True)
    while env.app.is_running():
        # Bound RPC work so neither physics nor pending observations can starve.
        for request_index in range(64):
            try:
                e, req, box, owner = jobs.get(timeout=0.01 if request_index == 0 else 0)
            except queue.Empty:
                break
            op, reply = req["op"], None
            try:
                if op == "ping":
                    reply = dict(pong=True, env_id=e, num_envs=n)
                elif op == "reset":
                    if e in owners or e in finished:
                        raise RuntimeError("each env can be claimed once per scene batch")
                    if bool(req.get("official")) != env.official:
                        raise ValueError(
                            f"this scene batch serves {'official' if env.official else 'full'} observations"
                        )
                    if sorted(req.get("fixes", [])) != sorted(getattr(env, "fixes", [])):
                        raise ValueError("the configured task fixes are required")
                    if (int(req["layout"]), int(req["seed"])) != (layouts[e], seed):
                        raise ValueError("reset does not match this env's preassigned layout")
                    owners[e] = owner
                    reply = dict(
                        **look(e, True),
                        instruction=env.instructions[e],
                        robots=[
                            {k: list(r[k]) for k in ("default_root_pos", "default_root_rot")}
                            for r in config.robot.robots
                        ],
                    )
                elif op == "disconnect":
                    if owners.get(e) == owner and e not in finished:
                        finished.add(e)
                        env.terminal(e, final=True)
                        log(op="disconnect", env=e)
                        print(json.dumps(dict(event="disconnect", env=e, steps=env.take_action_cnt[e])), flush=True)
                elif owners.get(e) != owner:
                    raise RuntimeError("reset must claim this environment before acting")
                elif op == "observe":
                    reply = look(e, bool(req.get("render", True)))
                elif op == "finish":
                    truth = env.truth(e) if env.dev else None  # before the final check, which consumes stages
                    env.terminal(e, final=True)
                    finished.add(e)
                    log(op="finish", env=e, **ending(env, e))
                    reply = dict(
                        success=bool(env.success[e]),
                        score=env.score(e),
                        steps=env.take_action_cnt[e],
                        step_limit=env.step_lim,
                        physical_steps=env.physical_steps[e],
                        idle_steps=env.idle_steps[e],
                        commanded_steps=env.commanded_steps[e],
                        reason="native_success" if env.success[e] else "terminal_failure",
                    )
                    print(json.dumps(dict(event="finish", env=e, layout=layouts[e], **reply)), flush=True)
                    if truth is not None:
                        reply["truth"] = truth
                elif op == "act":
                    if e in pending:
                        raise RuntimeError("one outstanding action request per environment")
                    if not req["actions"] or env.end_flag[e]:
                        reply = look(e, bool(req.get("render")))
                    else:
                        pending[e] = dict(actions=deque(req["actions"]), render=bool(req.get("render")), box=box)
                else:
                    raise ValueError(f"unknown op {op}")
            except Exception as exc:
                traceback.print_exc()
                reply = dict(error=f"{type(exc).__name__}: {exc}")
            if reply is not None:
                box.put(reply)

        live = [i for i in range(n) if i not in finished and not env.end_flag[i]]
        ready = [i for i in live if i in pending]
        if live and len(owners) == n and not started and len(ready) == len(live):
            # One initial start gate, then all live environments share a clock.
            # A slow agent never stalls another env after this gate opens.
            started = True
            print(
                json.dumps(
                    dict(event="clock_started", time=time.time(), simulation_hz=25, steps=list(env.take_action_cnt))
                ),
                flush=True,
            )
        # frozen clock: lockstep, so no env's world moves while its agent thinks (best with one env per simulator)
        if started and live and (clock != "frozen" or len(ready) == len(live)):
            t0 = time.time()
            try:
                actions = {i: pending[i]["actions"].popleft() for i in ready}
                log(op="act", envs=actions)
                timings = env.joint_actions(actions) or {}
            except Exception as exc:
                traceback.print_exc()
                for i in ready:
                    pending.pop(i)["box"].put(dict(error=f"batched physics: {exc!r}"))
                    finished.add(i)
                continue
            print(
                json.dumps(
                    dict(
                        event="step",
                        time=time.time(),
                        envs=ready,
                        steps=list(env.take_action_cnt),
                        physical_steps=list(env.physical_steps),
                        commanded_steps=list(env.commanded_steps),
                        idle_envs=[i for i in live if i not in ready],
                        idle_steps=list(env.idle_steps),
                        seconds=time.time() - t0,
                        **timings,
                    )
                ),
                flush=True,
            )
            for i in ready:
                if not pending[i]["actions"] or env.end_flag[i]:
                    job = pending.pop(i)
                    job["box"].put(look(i, job["render"]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", default="general_pickup")
    p.add_argument("--mode", choices=("dev", "eval"), default="dev")
    p.add_argument("--suite", choices=("legacy", "full5"), default="legacy")
    p.add_argument("--ports", required=True)
    p.add_argument("--layouts", required=True)
    p.add_argument("--seed", type=int, default=2)
    p.add_argument("--vendor", type=Path, required=True)
    p.add_argument("--official", action="store_true", help="camera RGB only (RoboDojo's eval observation)")
    p.add_argument("--events", type=Path, help="the batch's event log, for replay.py")
    a = p.parse_args()
    ports, layouts = [int(x) for x in a.ports.split(",")], [int(x) for x in a.layouts.split(",")]
    if len(ports) != len(layouts):
        p.error("one port per layout")
    env, config = start(a.task, a.mode, a.suite, layouts, a.seed, a.vendor, a.official)
    log = events(a.events, task=a.task, mode=a.mode, suite=a.suite, layouts=layouts, seed=a.seed, official=a.official,
                 fixes=env.fixes, step_limit=env.step_lim)  # fmt: skip
    serve(env, config, ports, layouts, a.seed, clock_mode(a.task), log)


def start(task, mode, suite, layouts, seed, vendor, official=False):
    """Boot Isaac Sim and build the batch: one env per layout."""
    for layout in layouts:
        check_layout(task, layout, seed, mode, suite)
    # boot() gives Kit's carb.tasking min(8, OMP_NUM_THREADS) workers. With one, the first RTX render of a
    # general_pickup scene sometimes never returns: the main thread spins in librtx.raytracing waiting on a task while
    # every worker sleeps. Kit gets at least 4 workers (server.py's default), whatever the batch's CPU threads are.
    os.environ["OMP_NUM_THREADS"] = str(max(KIT_THREADS, int(os.environ.get("OMP_NUM_THREADS") or KIT_THREADS)))
    # Record Python stacks during a startup hang without changing the scene or
    # terminating it. The timer is disabled before any evaluation starts.
    import faulthandler

    faulthandler.dump_traceback_later(120, repeat=True)
    try:
        app, config = boot(task, Path(vendor).resolve(), "cuda:0")
        fix_particle_material_api()
        config.eval_cfg.num_envs = config.sim.scene.num_envs = len(layouts)
        config.sim.seed = layouts
        env = build(config, app, layouts, seed, official)
        env.dev = mode == "dev"
        check_cameras(env)
    except Exception:  # Kit's exit then waits on a crash dialog, holding the failed start until the launcher gives up
        traceback.print_exc()
        sys.stderr.flush()
        os._exit(1)
    finally:
        faulthandler.cancel_dump_traceback_later()
    return env, config


if __name__ == "__main__":
    main()
