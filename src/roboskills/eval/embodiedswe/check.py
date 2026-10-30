"""Check one EmbodiedSWE-Bench episode end to end on one GPU of this host (SKILL.md's smoke test); writes `check.json`
and each camera's first RGB, depth and robot mask (`cam_<name>.png`) to `out`.

    python -m roboskills.eval.embodiedswe.check task=puzzle.coffee.franka.joint seed=1 gpu=0 out=runs/eswe-check/coffee-1

Startup time; per-observation latency and GPU memory; each camera's RGB-D, intrinsics and pose, checked against the
robot model: the depth of every pixel the robot's mask covers, back-projected through K and the pose, against the
model's surface there (camera, depth, mask and model agree when it lies on it); every model link and each TCP against
the simulator's bodies; both arms to joint targets; each hand closed and opened; for the walking G1 its pelvis along
world waypoints (forward, back, sideways, turns), with `Obs.base` and the arms' mount against the simulator, and how
far it drifts standing while its arms move; `finish` and its files. Nothing here reaches the agent: it reads robot
bodies only.
"""

import json
import math
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from roboskills.eval.embodiedswe import launch
from roboskills.eval.embodiedswe.robot import wrap
from roboskills.eval.interface import Trajectory


def gpu_mib(gpu):
    out = subprocess.run(["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=False).stdout  # fmt: skip
    return int(out.strip() or 0)


def angle(R):
    return math.degrees(math.acos(float(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def cameras(robot, model, o, out):
    """Per camera: size, valid depth, K, pose orthonormality, the robot mask's share, how far the back-projected masked
    pixels lie from the robot model's surface (median, mm) and the share within 1 cm (the G1's model is its torso and
    arms: its pelvis and legs, masked too, lie off it)."""
    from PIL import Image
    from scipy.spatial import cKDTree

    surface = cKDTree(model.robot_points(o))
    res = {}
    for name, c in o.cams.items():
        mask = robot.robot_mask(name, o)
        d = c.depth
        ok = np.isfinite(d) & (d > 0)
        v, u = np.nonzero(ok & (mask if mask is not None else False))
        z = d[v, u]
        P = (
            np.c_[(u - c.K[0, 2]) * z / c.K[0, 0], (v - c.K[1, 2]) * z / c.K[1, 1], z] @ c.T_world_cam[:3, :3].T
            + c.T_world_cam[:3, 3]
        )
        dist = surface.query(P)[0] if len(P) else np.zeros(0)
        R = c.T_world_cam[:3, :3]
        res[name] = dict(size=list(c.rgb.shape[1::-1]), depth_valid=round(float(ok.mean()), 3), fx=round(float(c.K[0, 0]), 2),
                         fy=round(float(c.K[1, 1]), 2), cx=round(float(c.K[0, 2]), 2), cy=round(float(c.K[1, 2]), 2),
                         rgb_mean=round(float(c.rgb.mean()), 1), pose_orthonormal_err=float(np.abs(R.T @ R - np.eye(3)).max()),
                         camera_xyz=np.round(c.T_world_cam[:3, 3], 3).tolist(),
                         robot_px=None if mask is None else round(float(np.mean(mask)), 3),
                         robot_to_model_mm=round(1000 * float(np.median(dist)), 1) if len(dist) else None,
                         robot_on_model=round(float(np.mean(dist < 0.01)), 3) if len(dist) else None)  # fmt: skip
        dz = np.nan_to_num(d, nan=0.0)
        dz = (255 * np.clip(dz / max(1e-6, np.percentile(dz[ok], 99) if ok.any() else 1.0), 0, 1)).astype(np.uint8)
        m = (mask if mask is not None else np.zeros(d.shape, bool)).astype(np.uint8) * 255
        Image.fromarray(np.concatenate([c.rgb, np.repeat(dz[..., None], 3, 2), np.repeat(m[..., None], 3, 2)], 1)).save(
            out / f"cam_{name}.png"
        )
    return res


def links(robot, model):
    """Every model link (each arm's model, at the measured joints) and each TCP against the simulator's bodies."""
    out = {}
    for a, info in model.info.items():
        names = list(info.M.rk.link_names)
        sim = robot._req(op="links", names=[n for n in names if n != "palm"])["poses"]
        T = model.links(a, model.fq(a))
        pos = [
            1000 * float(np.linalg.norm(T[k][:3, 3] - np.asarray(sim[n])[:3, 3]))
            for k, n in enumerate(names)
            if n in sim
        ]
        rot = [angle(T[k][:3, :3].T @ np.asarray(sim[n])[:3, :3]) for k, n in enumerate(names) if n in sim]
        ee = np.asarray(sim[robot.arms[a].ee_link]) @ robot.arms[a].hand.T_ee_tcp
        tcp = model.tcp(a)
        out[a] = dict(links=len(pos), link_pos_mm_max=round(max(pos), 3), link_rot_deg_max=round(max(rot), 4),
                      tcp_pos_mm=round(1000 * float(np.linalg.norm(tcp[:3, 3] - ee[:3, 3])), 3),
                      tcp_rot_deg=round(angle(tcp[:3, :3].T @ ee[:3, :3]), 4),
                      mount_mm=round(1000 * float(np.abs(robot.arms[a].T_world_base[:3, 3] - np.asarray(sim[robot.facts["root"]])[:3, 3]).max()), 3))  # fmt: skip
    return out


def arms(robot):
    """Every arm from where it is to a few joints moved (within limits), together, over 40 steps."""
    o = robot.obs(cams=False)
    b0 = o.base
    goal = {}
    for s, arm in robot.arms.items():
        dq = np.array([0.15, -0.15, 0.1, 0.2, 0.1, -0.15, 0.2]) * (1 if s == "left" else -1)
        if robot.kind == "franka":  # up and back, clear of what stands under the hand at reset
            dq = np.array([0.2, -0.2, 0.1, -0.2, 0.1, 0.15, 0.2])
        goal[s] = np.clip(o.q[s] + dq, arm.lo + 0.05, arm.hi - 0.05)
    n = 40
    a = np.linspace(0, 1, n + 1)[1:, None]
    f = robot.follow(Trajectory(t=np.arange(1, n + 1) * robot.dt, q={s: o.q[s] + a * (goal[s] - o.q[s]) for s in robot.arms},
                                hand={}, T={}))  # fmt: skip
    o = robot.obs(cams=False)
    back = robot.follow(Trajectory(t=np.arange(1, n + 1) * robot.dt, q={s: goal[s] + a * (robot.arms[s].home - goal[s]) for s in robot.arms},
                                   hand={}, T={}))  # fmt: skip
    out = dict(steps=f.steps, err_rad={s: round(float(np.abs(o.q[s] - goal[s]).max()), 4) for s in robot.arms},
               back_steps=back.steps)  # fmt: skip
    if b0 is not None:  # a standing robot's pelvis while its arms move
        b = robot.obs(cams=False).base
        out.update(base_drift_mm=round(1000 * float(np.hypot(*(b[:2] - b0[:2]))), 1),
                   base_drift_deg=round(math.degrees(abs(wrap(b[2] - b0[2]))), 2))  # fmt: skip
    return out


def hands(robot):
    """Each hand closed (in the air), then opened: steps and the measured state against the command."""
    res = {}
    for preset in ("closed", "open"):
        for s, arm in robot.arms.items():
            u = getattr(arm.hand, preset)
            f = robot.follow(Trajectory(t=np.arange(1, 16) * robot.dt, q={}, hand={s: np.repeat(u[None], 15, 0)}, T={}))
            got = robot.obs(cams=False).hand[s]
            res[f"{s}_{preset}"] = dict(steps=f.steps, err=round(float(np.abs(got - u).max()), 4))
    return res


def walk(robot, dx, dy, dyaw):
    """The pelvis by (dx, dy) in its own frame and turned dyaw, along waypoints within the base's speeds (a smooth
    start and stop peaking at them, as the skills' planner drives)."""
    b0 = robot.obs(cams=False).base
    c, s = math.cos(b0[2]), math.sin(b0[2])
    goal = b0 + [c * dx - s * dy, s * dx + c * dy, dyaw]
    n = max(2, int(math.ceil(1.5 * max(math.hypot(dx, dy) / robot.base.vmax, abs(dyaw) / robot.base.wmax) / robot.dt)))
    a = np.linspace(0, 1, n + 1)[1:, None]
    a = 3 * a**2 - 2 * a**3
    t0 = time.monotonic()
    f = robot.follow(Trajectory(t=np.arange(1, n + 1) * robot.dt, q={}, hand={}, T={}, base=b0 + a * (goal - b0)))
    o = robot.obs(cams=False)
    sim = np.asarray(robot._req(op="observe")["base"], float)
    return dict(move=[dx, dy, round(math.degrees(dyaw), 1)], path_steps=n, steps=f.steps, wall_s=round(time.monotonic() - t0, 1),
                err_mm=round(1000 * float(np.hypot(*(o.base[:2] - goal[:2]))), 1),
                err_deg=round(math.degrees(abs(wrap(o.base[2] - goal[2]))), 2), tilt=round(robot.state["tilt"], 2),
                obs_vs_sim_mm=round(1000 * float(np.hypot(*(o.base[:2] - sim[:2]))), 3))  # fmt: skip


def main(task, seed="1", gpu="0", out=None, **kw):
    out = Path(out or f"runs/embodiedswe-check/{task}-s{seed}").resolve()
    out.mkdir(parents=True, exist_ok=True)
    report = dict(task=task, seed=seed, gpu=str(gpu))
    mib0, t0 = gpu_mib(gpu), time.monotonic()
    with ThreadPoolExecutor(1) as maker:  # made in one thread, used from others, as the launcher does
        robot = maker.submit(launch.make, task=task, seed=seed, out=str(out / "server"), gpu=gpu, **kw).result()
    pool = ThreadPoolExecutor(2)

    def run(key, fn, *args):  # one part of the check, in another thread; a failure is recorded and the check goes on
        try:
            report[key] = pool.submit(fn, *args).result()
        except Exception as e:  # noqa: BLE001
            report[key] = dict(error=f"{type(e).__name__}: {e}"[:500])
        return report[key]

    try:
        report["startup_s"] = round(time.monotonic() - t0, 1)
        report.update(
            instruction=robot.instruction, step_limit=robot.step_limit, dt=robot.dt, steps_at_start=robot.steps
        )
        lat = []
        for _ in range(3):
            t = time.monotonic()
            o = pool.submit(robot.obs).result()
            lat.append(time.monotonic() - t)
        report["obs_s"] = [round(x, 3) for x in lat]
        t = time.monotonic()
        pool.submit(robot.obs, False).result()
        report["obs_proprio_ms"] = round(1000 * (time.monotonic() - t), 2)
        report["gpu_mib"] = gpu_mib(gpu) - mib0
        from roboskills.tools.robokit.robot_model import RobotModel

        t = time.monotonic()
        model = RobotModel(robot)
        report["model_s"] = round(time.monotonic() - t, 1)
        run("cameras", cameras, robot, model, o, out)
        run("links_start", links, robot, model)
        t = time.monotonic()
        run("arms", arms, robot)
        report["arm_follow_s"] = round(time.monotonic() - t, 2)
        report["step_ms"] = round(
            1000
            * report["arm_follow_s"]
            / max(1, report["arms"].get("steps", 1) + report["arms"].get("back_steps", 0)),
            1,
        )
        run("hands", hands, robot)
        run("links_after", links, robot, model)
        if robot.loco:
            moves = ((0.4, 0.0, 0.0), (-0.4, 0.0, 0.0), (0.0, 0.3, 0.0), (0.0, -0.3, 0.0), (0.0, 0.0, math.radians(45)),
                     (0.0, 0.0, math.radians(-45)), (0.3, 0.0, math.radians(30)), (0.0, 0.0, math.radians(-30)))  # fmt: skip
            report["walk"] = [run("_", walk, robot, *m) for m in moves]
            run("links_after_walk", links, robot, model)
            o = robot.obs()
            run("cameras_after_walk", cameras, robot, model, o, out)
        report.pop("_", None)
        report["steps_used"] = robot.steps
        report["finish"] = robot.finish(out)
        report["files"] = sorted(p.name for p in out.iterdir() if p.is_file())
    finally:
        pool.shutdown()
        launch.close()
        (out / "check.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    return report


if __name__ == "__main__":
    main(**dict(kv.split("=", 1) for kv in sys.argv[1:]))
