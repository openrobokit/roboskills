"""RoboCasa365 runtime in the simulator: reset, cameras, base and arm tracking, the arm model riding on the base,
rendering from other threads, and the terminal files. It builds the episode's Robot in this process, as its server does
(`launch.episode`). Runs only in the runtime's environment on a GPU host (SKILL.md):

    MUJOCO_GL=egl uv run --no-sync --project src/roboskills/eval/robocasa365 --with pytest \
        python -m pytest -s tests/eval/test_robocasa365_sim.py

ROBOCASA365_SEEDS (default "1") picks the seeds; every cohort task runs with each. Each case prints one JSON line of
measurements."""

import json
import os
import threading
import time

import numpy as np
import pytest


pytest.importorskip("robocasa")

from roboskills.eval.interface import Trajectory  # noqa: E402
from roboskills.eval.robocasa365 import TASKS, launch  # noqa: E402
from roboskills.tools.robokit.robot_model import RobotModel  # noqa: E402


SEEDS = os.environ.get("ROBOCASA365_SEEDS", "1").split(",")


def threaded(fn, *args):
    """fn(*args) in a thread of its own, as the harness calls the robot."""
    out = {}

    def run():
        try:
            out["value"] = fn(*args)
        except BaseException as e:  # raised again in the caller's thread
            out["error"] = e

    t = threading.Thread(target=run)
    t.start()
    t.join()
    if "error" in out:
        raise out["error"]
    return out["value"]


def sim_frames(r):
    """The simulator's own base [x, y, yaw] (the yaw hinge's axis), arm root and grip-site poses."""
    sim = r.env.sim
    sim.forward()
    m, d = sim.model, sim.data
    R = d.xmat[m.body_name2id("mobilebase0_base")].reshape(3, 3)
    base = np.r_[d.xanchor[m.joint_name2id("mobilebase0_joint_mobile_yaw")][:2], np.arctan2(R[1, 0], R[0, 0])]
    root, site = m.body_name2id("robot0_link0"), r.robot.eef_site_id["right"]
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = d.xmat[root].reshape(3, 3), d.xpos[root]
    return base, T, d.site_xpos[site].copy(), d.site_xmat[site].reshape(3, 3).copy()


def check_frames(r, model):
    """Obs.base, Arm.T_world_base and the model's TCP against the simulator: errors (m, rad)."""
    o = r.obs(cams=False)
    base, T, p, R = sim_frames(r)
    tcp = model.tcp("right")
    # the TCP's approach (+x) is the grip site's z, its squeeze axis (+y) the site's x
    Rt = R @ np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    turn = np.arccos(np.clip((np.trace(tcp[:3, :3].T @ Rt) - 1) / 2, -1, 1))
    yaw = abs((o.base[2] - base[2] + np.pi) % (2 * np.pi) - np.pi)
    return dict(base=float(np.linalg.norm(o.base[:2] - base[:2])), base_yaw=float(yaw),
                root=float(np.abs(r.arms["right"].T_world_base - T).max()),
                tcp=float(np.linalg.norm(tcp[:3, 3] - p)), tcp_turn=float(turn))  # fmt: skip


def drive_path(r, delta):
    """The shared planner's drive: from the measured base by `delta` [forward, left, turn] in its own frame, on a
    3s^2 - 2s^3 profile over 1.5 x distance / speed limit."""
    b = r.obs(cams=False).base
    c, s = np.cos(b[2]), np.sin(b[2])
    goal = b + np.r_[c * delta[0] - s * delta[1], s * delta[0] + c * delta[1], delta[2]]
    dur = max(1.5 * np.hypot(*delta[:2]) / r.base.vmax, 1.5 * abs(delta[2]) / r.base.wmax, 0.2)
    t = np.linspace(0, dur, max(2, int(np.ceil(dur / 0.02))) + 1)[1:]
    a = 3 * (t / dur) ** 2 - 2 * (t / dur) ** 3
    return Trajectory(t, {}, {}, {}, base=b + a[:, None] * (goal - b)), goal


def drive(r, options):
    """Drive by the first of `options` that no fixture blocks: (delta, position error m, heading error rad, steps)."""
    for delta in options:
        path, goal = drive_path(r, np.asarray(delta, float))
        f = threaded(r.follow, path)
        b = f.obs.base
        err = float(np.linalg.norm(b[:2] - goal[:2])), float(abs((b[2] - goal[2] + np.pi) % (2 * np.pi) - np.pi))
        if err[0] < 0.02 and err[1] < np.radians(2):
            break
    return delta, *err, f.steps


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("task", list(TASKS))
def test_episode(task, seed, tmp_path):
    t0 = time.time()
    r = threaded(launch.episode, task, seed, str(tmp_path))  # made in one thread, observed and moved from others
    made = time.time() - t0
    model = RobotModel(r)
    report = dict(task=task, seed=int(seed), layout=int(r.layout), style=int(r.style), make_s=round(made, 1))
    times = []
    for _ in range(2):
        t0 = time.time()
        o = threaded(r.obs)
        times.append(time.time() - t0)
        assert set(o.cams) == {
            f"robot0_{c}" for c in ("agentview_left", "agentview_center", "agentview_right", "eye_in_hand")
        }
        for name, cam in o.cams.items():
            assert cam.rgb.shape == (512, 512, 3) and cam.depth.shape == (512, 512)
            ok = np.isfinite(cam.depth) & (cam.depth > 0)
            assert ok.mean() > 0.95 and 0.03 < np.median(cam.depth[ok]) < 6.0, name
            assert 0.0 < r.robot_mask(name, o).mean() < 0.5, name
    report["obs_s"] = round(float(np.mean(times)), 3)
    frames = [check_frames(r, model)]
    report["drives"] = {}
    q = np.pi / 2
    for name, options in (("straight", [[-0.45, 0, 0], [0.3, 0, 0]]), ("strafe", [[0, 0.2, 0], [0, -0.2, 0]]),
                          ("turn", [[0, 0, q], [0, 0, -q], [0, 0, q / 2], [0, 0, -q / 2]])):  # fmt: skip
        delta, pos, yaw, steps = drive(r, options)
        report["drives"][name] = dict(
            by=delta, mm=round(1000 * pos, 2), deg=round(float(np.degrees(yaw)), 3), steps=steps
        )
        assert pos < 0.02 and yaw < np.radians(2), (name, pos, yaw)
        frames.append(check_frames(r, model))
    q0 = r.obs(cams=False).q["right"]
    goal = q0 + np.array([0.3, 0.2, -0.2, 0.3, 0.2, -0.3, 0.4])
    t = np.arange(1, 41) * r.dt
    path = Trajectory(t, {"right": q0 + (goal - q0) * ((1 - np.cos(np.pi * t / t[-1])) / 2)[:, None]}, {}, {})
    f = threaded(r.follow, path)
    report["arm_rad"] = round(float(np.abs(f.obs.q["right"] - goal).max()), 5)
    assert report["arm_rad"] < 0.01
    frames.append(check_frames(r, model))
    worst = {k: max(fr[k] for fr in frames) for k in frames[0]}
    report.update(base_mm=round(1000 * worst["base"], 3), root_mm=round(1000 * worst["root"], 3),
                  tcp_mm=round(1000 * worst["tcp"], 3), tcp_deg=round(float(np.degrees(worst["tcp_turn"])), 3))  # fmt: skip
    assert worst["base"] < 1e-4 and worst["base_yaw"] < 1e-4 and worst["root"] < 1e-4
    assert worst["tcp"] < 0.005 and worst["tcp_turn"] < np.radians(2)
    result = threaded(r.finish, str(tmp_path))
    assert {"video.mp4", "actions.npy", "native_result.json"} <= set(os.listdir(tmp_path))
    native = json.loads((tmp_path / "native_result.json").read_text())
    assert native["task"] == task and native["layout"] == r.layout and native["steps"] == r.steps
    assert native["success"] == result["success"] and native["step_limit"] == TASKS[task][0] and native["instruction"]
    assert np.load(tmp_path / "actions.npy").shape == (r.steps, 13)
    report.update(steps=r.steps, success=result["success"], instruction=native["instruction"])
    print(json.dumps(report))
