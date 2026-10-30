"""Check one HumanoidArena episode end to end on one GPU of this host (SKILL.md's smoke test); writes `check.json` to out.

    python -m roboskills.eval.humanoidarena.check task=sit_sofa seed=1 gpu=0 out=runs/humanoidarena-check/sit_sofa-1

Startup time; per-observation latency and GPU memory; the camera's RGB-D, intrinsics and pose (depth back-projected
through K and the pose: the floor at z = 0) and its robot mask; the arm model's palm links and TCPs against the
simulator's, and its root against the torso link, after squatting and walking; both arms to joint targets and back
(sag-corrected); both hands closed and opened; the pelvis lowered to 0.6 m and raised (`Obs.body`); walking to world
waypoints (forward, sideways, a turn, and a diagonal back with a turn, while the task's budget lasts) with the error
at the end and the steps taken, `Obs.base` against the simulator's pelvis; `obs`/`follow`
from threads other than the one that made the robot; `finish` and its files. Nothing here reaches the agent: it reads
robot links only.
"""

import json
import math
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from roboskills.eval.humanoidarena import launch
from roboskills.eval.humanoidarena.hand import SIDES
from roboskills.eval.humanoidarena.robot import DT, wrap
from roboskills.eval.interface import Trajectory


def gpu_mib(gpu):
    out = subprocess.run(["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=False).stdout  # fmt: skip
    return int(out.strip() or 0)


def angle(R):
    return math.degrees(math.acos(float(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def camera(robot, o):
    """Image size, valid depth, K, pose orthonormality, the camera's height, the lowest back-projected points' height
    and the share of pixels the robot's mask covers."""
    out = {}
    for name, c in o.cams.items():
        mask = robot.robot_mask(name, o)
        ok = np.isfinite(c.depth) & (c.depth > 0)
        v, u = np.nonzero(ok)
        z = c.depth[ok]
        P = np.c_[(u - c.K[0, 2]) * z / c.K[0, 0], (v - c.K[1, 2]) * z / c.K[1, 1], z]
        W = P @ c.T_world_cam[:3, :3].T + c.T_world_cam[:3, 3]
        R = c.T_world_cam[:3, :3]
        out[name] = dict(size=[c.rgb.shape[1], c.rgb.shape[0]], depth_valid=round(float(ok.mean()), 3),
                         depth_median_m=round(float(np.median(z)), 3) if len(z) else None, fx=round(float(c.K[0, 0]), 2),
                         cx=round(float(c.K[0, 2]), 2), rgb_mean=round(float(c.rgb.mean()), 1),
                         pose_orthonormal_err=float(np.abs(R.T @ R - np.eye(3)).max()), camera_z=round(float(c.T_world_cam[2, 3]), 3),
                         pitch_down_deg=round(math.degrees(math.asin(float(-R[2, 2]))), 1),
                         points_z_p1=round(float(np.percentile(W[:, 2], 1)), 4) if len(W) else None,
                         robot_px=None if mask is None else round(float(np.mean(mask)), 3))  # fmt: skip
    return out


def tcp(robot, model):
    """The model's palm links and torso against the simulator's bodies (position mm, rotation deg)."""
    names = [f"{s}_hand_palm_link" for s in SIDES] + ["torso_link"]
    P = robot._req(op="links", names=names)["poses"]
    out = {}
    for s in SIDES:
        T = model.tcp(s) @ np.linalg.inv(robot.arms[s].hand.T_ee_tcp)  # the model's palm link
        sim = np.asarray(P[f"{s}_hand_palm_link"], float)
        out[s] = dict(palm_mm=round(1000 * float(np.linalg.norm(T[:3, 3] - sim[:3, 3])), 2),
                      palm_deg=round(angle(T[:3, :3].T @ sim[:3, :3]), 3))  # fmt: skip
    torso = np.asarray(P["torso_link"], float)
    out["mount_mm"] = round(1000 * float(np.linalg.norm(torso[:3, 3] - robot.arms["left"].T_world_base[:3, 3])), 3)
    return out


def walk(robot, dx, dy, dyaw):
    """Walk by (dx, dy) in the pelvis's own frame and turn dyaw, along waypoints within the base's speeds (a smooth start
    and stop peaking at them, as the skills' planner drives)."""
    b0 = robot.obs(cams=False).base
    c, s = math.cos(b0[2]), math.sin(b0[2])
    goal = b0 + [c * dx - s * dy, s * dx + c * dy, dyaw]
    n = max(2, int(math.ceil(1.5 * max(math.hypot(dx, dy) / robot.base.vmax, abs(dyaw) / robot.base.wmax) / DT)))
    a = np.linspace(0, 1, n + 1)[1:, None]
    a = 3 * a**2 - 2 * a**3
    t = time.monotonic()
    f = robot.follow(Trajectory(t=np.arange(1, n + 1) * DT, q={}, hand={}, T={}, base=b0 + a * (goal - b0)))
    o = robot.obs(cams=False)
    sim = robot._req(op="links", names=["pelvis"])["poses"]["pelvis"]
    return dict(move=[dx, dy, round(math.degrees(dyaw), 1)], path_steps=n, steps=f.steps, stopped=robot.state.get("stopped"),
                tilt=round(robot.state["tilt"], 2), err_mm=round(1000 * float(np.hypot(*(o.base[:2] - goal[:2]))), 1),
                err_deg=round(math.degrees(abs(wrap(o.base[2] - goal[2]))), 2), body=round(o.body, 3),
                obs_vs_sim_mm=round(1000 * float(np.hypot(*(o.base[:2] - np.asarray(sim)[:2, 3]))), 3),
                wall_s=round(time.monotonic() - t, 1), ended=robot.ended)  # fmt: skip


def posture(robot, height):
    """Lower (or raise) the pelvis to `height` at the body's speed, as the skills' planner does."""
    o = robot.obs(cams=False)
    goal = float(np.clip(height, robot.body.lo, robot.body.hi))
    n = max(2, int(math.ceil(1.5 * abs(goal - o.body) / robot.body.vmax / DT)))
    a = np.linspace(0, 1, n + 1)[1:]
    a = 3 * a**2 - 2 * a**3
    f = robot.follow(Trajectory(t=np.arange(1, n + 1) * DT, q={}, hand={}, T={}, body=o.body + a * (goal - o.body)))
    o = robot.obs(cams=False)
    return dict(target=round(goal, 3), steps=f.steps, body=round(o.body, 3), err_mm=round(1000 * abs(o.body - goal), 1),
                tilt=round(robot.state["tilt"], 2), ended=robot.ended)  # fmt: skip


def arms(robot):
    """Both arms from where they are to a few joints moved (within limits), together, and back."""
    o = robot.obs(cams=False)
    start, goal = {}, {}
    for s in SIDES:
        arm = robot.arms[s]
        start[s] = o.q[s]
        goal[s] = np.clip(o.q[s] + np.array([-0.4, 0.0, 0.0, -0.3, 0.0, 0.0, 0.0]), arm.lo + 0.05, arm.hi - 0.05)
    out = []
    for a_, b_ in ((start, goal), (goal, start)):
        n = 40
        w = np.linspace(0, 1, n + 1)[1:, None]
        f = robot.follow(
            Trajectory(t=np.arange(1, n + 1) * DT, q={s: a_[s] + w * (b_[s] - a_[s]) for s in SIDES}, hand={}, T={})
        )
        o = robot.obs(cams=False)
        out.append(dict(steps=f.steps, err_rad={s: round(float(np.abs(o.q[s] - b_[s]).max()), 4) for s in SIDES}))
    return out


def hands(robot):
    """Close both hands, then open them: the measured joints against each command."""
    out = []
    for key in ("closed", "open"):
        u = {s: getattr(robot.arms[s].hand, key) for s in SIDES}
        o0 = robot.obs(cams=False)
        f = robot.follow(
            Trajectory(t=np.array([DT]), q={s: o0.q[s][None] for s in SIDES}, hand={s: u[s][None] for s in SIDES}, T={})
        )
        o = robot.obs(cams=False)
        out.append(
            dict(
                command=key, steps=f.steps, err_rad={s: round(float(np.abs(o.hand[s] - u[s]).max()), 3) for s in SIDES}
            )
        )
    return out


def main(task, seed="1", gpu="0", out=None, **kw):
    out = Path(out or f"runs/humanoidarena-check/{task}-s{seed}").resolve()
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
        report["instruction"], report["step_limit"] = robot.instruction, robot.step_limit
        report["stand_body"], report["press_point"] = (
            robot.stand,
            {s: robot.arms[s].hand.press_point.round(4).tolist() for s in SIDES},
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
        report["camera"] = camera(robot, o)
        from roboskills.tools.robokit.robot_model import RobotModel

        model = RobotModel(robot)
        run("tcp_start", tcp, robot, model)
        n0, t = robot.steps, time.monotonic()
        run("arms", arms, robot)
        report["sim_step_ms"] = round(1000 * (time.monotonic() - t) / max(1, robot.steps - n0), 1)
        run("hands", hands, robot)
        report["posture"] = [run("_", posture, robot, h) for h in (0.6, robot.body.hi)]
        run("tcp_after_posture", tcp, robot, model)
        report["walk"] = []
        for m in ((0.5, 0.0, 0.0), (0.0, 0.3, 0.0), (0.0, 0.0, math.radians(60)), (-0.5, -0.3, math.radians(-60))):
            if robot.step_limit - robot.steps < 400:  # (the budget is the task's: Boxing has 900 steps)
                break
            report["walk"].append(run("_", walk, robot, *m))
        run("tcp_after_walk", tcp, robot, model)
        report["steps_used"] = robot.steps
        report.pop("_", None)
        report["after"] = camera(robot, pool.submit(robot.obs).result())
        report["finish"] = robot.finish(out)
        report["files"] = sorted(p.name for p in out.iterdir() if p.is_file())
    finally:
        pool.shutdown()
        launch.close()
        (out / "check.json").write_text(json.dumps(report, indent=1, default=float))
    print(json.dumps(report, indent=1, default=float))
    return report


if __name__ == "__main__":
    main(**dict(kv.split("=", 1) for kv in sys.argv[1:]))
