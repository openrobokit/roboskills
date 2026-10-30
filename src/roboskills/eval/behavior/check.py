"""Check one BEHAVIOR episode end to end on one GPU of this host (SKILL.md's smoke test); writes `check.json` to `out`.

    python -m roboskills.eval.behavior.check task=turning_on_radio seed=1 gpu=0 out=runs/behavior-check/radio-1

Startup time; per-observation latency and GPU memory; each camera's RGB-D, intrinsics and pose (depth back-projected
through K and the pose: the floor at z = 0); the robot model's TCP against the simulator's (its own grasp frame, the
`eef_link` 6 cm down the gripper, and the fingers' midpoint) for both arms; the base driven along world waypoints
(forward, back, sideways, a turn and back) to within 3 cm / 3 deg, with `Obs.base` and every arm's `T_world_base` against
the simulator after each; both arms to joint targets; each camera's robot mask; 4 m straight ahead, into whatever stands there (the drive stops at
the first touch, the robot upright); `obs`/`follow` from threads other than the one that made the
robot; `finish` and its files. Nothing here reaches the agent: it reads robot links only.
"""

import json
import math
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from roboskills.eval.behavior import launch
from roboskills.eval.behavior.hand import T_EE_TCP
from roboskills.eval.behavior.robot import DT, SIDES, wrap
from roboskills.eval.interface import Trajectory


def gpu_mib(gpu):
    out = subprocess.run(["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=False).stdout  # fmt: skip
    return int(out.strip() or 0)


def angle(R):
    return math.degrees(math.acos(float(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def cameras(robot, o):
    """Per camera: image size, valid depth, K, pose orthonormality, the lowest back-projected points' height, and the
    share of its pixels the robot's mask covers."""
    out = {}
    for name, c in o.cams.items():
        mask = robot.robot_mask(name, o)
        h, w = c.rgb.shape[:2]
        d = c.depth
        ok = np.isfinite(d) & (d > 0)
        v, u = np.nonzero(ok)
        z = d[ok]
        P = np.c_[(u - c.K[0, 2]) * z / c.K[0, 0], (v - c.K[1, 2]) * z / c.K[1, 1], z]
        W = P @ c.T_world_cam[:3, :3].T + c.T_world_cam[:3, 3]
        R = c.T_world_cam[:3, :3]
        out[name] = dict(size=[w, h], depth_valid=round(float(ok.mean()), 3), depth_median_m=round(float(np.median(z)), 3),
                         fx=round(float(c.K[0, 0]), 2), cx=round(float(c.K[0, 2]), 2), rgb_mean=round(float(c.rgb.mean()), 1),
                         pose_orthonormal_err=float(np.abs(R.T @ R - np.eye(3)).max()),
                         camera_z=round(float(c.T_world_cam[2, 3]), 3),
                         points_z_p1=round(float(np.percentile(W[:, 2], 1)), 4) if len(W) else None,
                         robot_px=None if mask is None else round(float(np.mean(mask)), 3))  # fmt: skip
    return out


def tcp(robot, model):
    """Model TCP vs the simulator's: its eef_link (origin and axes) and the finger links' midpoint off the TCP's axis."""
    names = [n for s in SIDES for n in (f"{s}_gripper_link", f"{s}_eef_link", f"{s}_gripper_finger_link1",
                                         f"{s}_gripper_finger_link2", "torso_link4")]  # fmt: skip
    P = robot._req(op="links", names=sorted(set(names)))["poses"]
    out = {}
    for s in SIDES:
        T = model.tcp(s)
        eef = np.asarray(P[f"{s}_eef_link"], float)  # OmniGibson's grasp frame: its z along the fingers
        R_sim = eef[:3, :3] @ np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])  # eef axes -> TCP axes
        mid = (
            np.asarray(P[f"{s}_gripper_finger_link1"], float)[:3, 3]
            + np.asarray(P[f"{s}_gripper_finger_link2"], float)[:3, 3]
        ) / 2
        off = mid - T[:3, 3]
        g = np.asarray(P[f"{s}_gripper_link"], float) @ T_EE_TCP
        out[s] = dict(pos_mm=round(1000 * float(np.linalg.norm(T[:3, 3] - eef[:3, 3])), 2),
                      rot_deg=round(angle(T[:3, :3].T @ R_sim), 3),
                      gripper_link_pos_mm=round(1000 * float(np.linalg.norm(T[:3, 3] - g[:3, 3])), 2),
                      fingers_off_axis_mm=round(1000 * float(np.linalg.norm(off - (off @ T[:3, 0]) * T[:3, 0])), 2))  # fmt: skip
    mount = np.asarray(P["torso_link4"], float)
    out["mount_mm"] = round(1000 * float(np.abs(mount[:3, 3] - robot.arms["left"].T_world_base[:3, 3]).max()), 3)
    return out


def drive(robot, dx, dy, dyaw):
    """Drive the base by (dx, dy) in its own frame and turn dyaw, along waypoints within the base's speeds (a smooth
    start and stop peaking at them, as the skills' planner drives)."""
    b0 = robot.obs(cams=False).base
    c, s = math.cos(b0[2]), math.sin(b0[2])
    goal = b0 + [c * dx - s * dy, s * dx + c * dy, dyaw]
    n = max(2, int(math.ceil(1.5 * max(math.hypot(dx, dy) / robot.base.vmax, abs(dyaw) / robot.base.wmax) / DT)))
    a = np.linspace(0, 1, n + 1)[1:, None]
    a = 3 * a**2 - 2 * a**3
    f = robot.follow(Trajectory(t=np.arange(1, n + 1) * DT, q={}, hand={}, T={}, base=b0 + a * (goal - b0)))
    o = robot.obs(cams=False)
    sim = np.asarray(robot._req(op="observe")["base"], float)  # (a fresh read: the same simulator, rendered)
    return dict(move=[dx, dy, round(math.degrees(dyaw), 1)], steps=f.steps, stopped=f.stopped,
                blocked=robot.state.get("stopped"), tilt=round(robot.state["tilt"], 2),
                err_mm=round(1000 * float(np.hypot(*(o.base[:2] - goal[:2]))), 2),
                err_deg=round(math.degrees(abs(wrap(o.base[2] - goal[2]))), 3),
                obs_vs_sim_mm=round(1000 * float(np.hypot(*(o.base[:2] - sim[:2]))), 3))  # fmt: skip


def arms(robot):
    """Both arms from where they are to a few joints moved (within limits), together."""
    o = robot.obs(cams=False)
    goal = {}
    for s in SIDES:
        arm = robot.arms[s]
        q = o.q[s] + np.array([0.0, 0.0, 0.0, -0.3, 0.0, 0.2, 0.0]) * (1 if s == "left" else -1)
        goal[s] = np.clip(q, arm.lo + 0.05, arm.hi - 0.05)
    n = 30
    a = np.linspace(0, 1, n + 1)[1:, None]
    f = robot.follow(
        Trajectory(t=np.arange(1, n + 1) * DT, q={s: o.q[s] + a * (goal[s] - o.q[s]) for s in SIDES}, hand={}, T={})
    )
    o = robot.obs(cams=False)
    return dict(steps=f.steps, err_rad={s: round(float(np.abs(o.q[s] - goal[s]).max()), 4) for s in SIDES})


def blocked(robot, far=4.0):
    """Drive `far` m straight ahead (into whatever is in the way, in a house), starting and stopping smoothly as the
    skills' planner does: where the base stops, why, and how far the robot leans."""
    b0 = robot.obs(cams=False).base
    n = int(math.ceil(1.5 * far / robot.base.vmax / DT))
    a = np.linspace(0, 1, n + 1)[1:, None]
    a = 3 * a**2 - 2 * a**3
    goal = b0 + [far * math.cos(b0[2]), far * math.sin(b0[2]), 0.0]
    f = robot.follow(Trajectory(t=np.arange(1, n + 1) * DT, q={}, hand={}, T={}, base=b0 + a * (goal - b0)))
    o = robot.obs(cams=False)
    return dict(commanded_m=far, path_steps=n, steps=f.steps, driven_m=round(float(np.hypot(*(o.base[:2] - b0[:2]))), 3),
                blocked=robot.state.get("stopped"), tilt=round(robot.state["tilt"], 2),
                held_at_mm=round(1000 * float(np.hypot(*(robot.cmd_base[:2] - o.base[:2]))), 1))  # fmt: skip


def main(task, seed="1", gpu="0", out=None, **kw):
    out = Path(out or f"runs/behavior-check/{task}-s{seed}").resolve()
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
        report["cameras"] = cameras(robot, o)
        from roboskills.tools.robokit.robot_model import RobotModel

        model = RobotModel(robot)
        run("tcp_start", tcp, robot, model)
        moves = ((0.3, 0.0, 0.0), (-0.3, 0.0, 0.0), (0.0, 0.2, 0.0), (0.0, -0.2, 0.0), (0.0, 0.0, math.radians(60)),
                 (0.0, 0.0, math.radians(-60)), (0.2, 0.1, math.radians(30)))  # fmt: skip
        report["base"] = [run("_", drive, robot, *m) for m in moves]
        run("tcp_after_base", tcp, robot, model)
        t = time.monotonic()
        run("arms", arms, robot)
        report["arm_follow_s"] = round(time.monotonic() - t, 2)
        report["sim_step_ms"] = round(1000 * report["arm_follow_s"] / max(1, report["arms"].get("steps", 1)), 1)
        run("tcp_after_arms", tcp, robot, model)
        run("blocked", blocked, robot)
        report["steps_used"] = robot.steps
        report.pop("_", None)
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
