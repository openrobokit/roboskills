"""HumanoidArena runtime without the simulator: the official episode seeds, the gait's commands and resampling, the Dex3
TCP, the arm model, `follow` over a fake simulator (arms, hands, walking with feedback, a blocked walk, the pelvis
height, the step limit, the moving mount), `replay`, and where `launch` runs the simulator."""

import json
import math
import xml.etree.ElementTree as ET
from types import SimpleNamespace

import numpy as np
import pytest

from roboskills.eval import runtime
from roboskills.eval.humanoidarena import GROUPS, TASKS, episode_seed, launch
from roboskills.eval.humanoidarena.hand import T_ee_tcp, closed, model
from roboskills.eval.humanoidarena.robot import DT, HumanoidArena, flat, unflat
from roboskills.eval.humanoidarena.sim import gait
from roboskills.eval.interface import Body, Trajectory


def test_seeds_are_the_official_test_episodes_and_limits_upstreams():
    assert GROUPS == {"1": 0, "2": 1}
    # RAX's official_seeds(task)[0] and [20]: group 0 and group 1, repeat 0
    assert episode_seed("opendoor", "1") == 1165506222 and episode_seed("opendoor", "2") == 559722726
    assert episode_seed("sit_sofa", "1") == 1737204259 and episode_seed("warehouse_navigation", "2") == 1867786343
    assert {k: v[3] for k, v in TASKS.items()} == {"warehouse_navigation": 1800, "boxing_bag": 900, "opendoor": 1800,
                                                    "football_single": 2000, "sit_sofa": 2000}  # fmt: skip
    with pytest.raises(ValueError):
        episode_seed("opendoor", "3")


def test_joint_orders_round_trip_and_rot6d_is_semantic40s_row_layout():
    q = np.arange(29.0)
    np.testing.assert_array_equal(q[gait.TO_MUJOCO][gait.TO_CANONICAL], q)
    assert gait.MUJOCO[:6] == ("left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
                               "left_ankle_pitch_joint", "left_ankle_roll_joint")  # fmt: skip
    np.testing.assert_allclose(gait.rot6d([1, 0, 0, 0]), [1, 0, 0, 1, 0, 0])
    a = 0.3
    np.testing.assert_allclose(gait.rot6d(gait.qyaw(a)), [math.cos(a), -math.sin(a), math.sin(a), math.cos(a), 0, 0])
    assert gait.yaw_of(gait.qmul(gait.qyaw(0.2), gait.qyaw(0.5))) == pytest.approx(0.7)


def test_commands_walk_toward_the_target_turn_in_place_stand_and_squat():
    base = np.array([0.0, 0.0, 0.0])
    mode, speed, move, face, h = gait.command(base, [1.0, 0.0, 0.5])
    assert mode == gait.SLOW_WALK and speed == gait.SPEED[1] and np.allclose(move, [1, 0]) and face == 0.5
    mode, speed, move, _, _ = gait.command(base, [0.05, 0.0, 0.0])  # standing: starts only past 1.5 POS_TOL
    assert mode == gait.IDLE
    assert gait.command(base, [0.05, 0.0, 0.0], walking=True)[0] == gait.SLOW_WALK
    mode, _, move, face, _ = gait.command(base, [0.0, 0.0, 1.0])
    assert mode == gait.SLOW_WALK and not move.any() and face == 1.0  # turn in place
    assert gait.command(base, None, body=0.5)[0::4] == (gait.SQUAT, 0.5)
    assert gait.command(base, None, body=0.78, hold_yaw=0.3)[0::3] == (gait.IDLE, 0.3)


def test_resampling_keeps_ends_and_slerps_quaternions():
    f = np.zeros((4, 36))
    f[:, 0] = [0.0, 0.1, 0.2, 0.3]
    f[:, 3:7] = [gait.qyaw(a) for a in (0.0, 0.3, 0.6, 0.9)]
    r = gait.resample(f, 30, 50)
    assert len(r) == 6 and r[0, 0] == 0.0 and r[-1, 0] == pytest.approx(0.3)
    assert gait.yaw_of(r[1, 3:7]) == pytest.approx(0.3 * 30 / 50 * 1.0, rel=1e-6)


class FakePlanner:
    """Moves the pelvis along `move` at `speed` (30 Hz frames), keeps the joints, turns to `face`."""

    def __call__(self, ctx, mode, speed, move, face, height):
        out = np.repeat(ctx[-1:], 40, 0)
        v = max(speed, 0.0) if mode == gait.SLOW_WALK else 0.0
        out[:, 0] += v * np.arange(40) / 30 * move[0]
        out[:, 1] += v * np.arange(40) / 30 * move[1]
        out[:, 3:7] = gait.qyaw(face)
        if mode == gait.SQUAT:
            out[:, 2] = height
        return out


def test_gait_plans_on_command_changes_and_holds_a_static_plan():
    g = gait.Gait(FakePlanner())
    g.reset([0, 0, 0.79], [1, 0, 0, 0], np.zeros(29))
    cmd = gait.command(np.zeros(3), None)
    for _ in range(100):
        g.step(cmd)
    assert g.plans == 1  # standing: one plan, held at its end
    cmd = gait.command(np.zeros(3), [1.0, 0.0, 0.0])
    for _ in range(60):
        g.step(cmd)
    assert g.plans == 2 + 60 // (gait.REPLAN + 1)  # walking: planned again every REPLAN steps
    q, j, z = g.frame()
    assert z == pytest.approx(0.79) and gait.yaw_of(q) == pytest.approx(0.0, abs=1e-6)
    for _ in range(100):
        q, j, z = g.step(gait.command(np.zeros(3), None, body=0.5))
    assert z == pytest.approx(0.5)


def test_dex3_tcp_is_right_handed_out_of_the_palm_and_along_the_fingers():
    for s, face in (("left", -1), ("right", 1)):
        T = T_ee_tcp(s)
        assert np.linalg.det(T[:3, :3]) == pytest.approx(1.0)
        np.testing.assert_allclose(T[:3, 0], [0, face, 0])
        np.testing.assert_allclose(T[:3, 1], [1, 0, 0])
        assert T[1, 3] * face > 0 and T[0, 3] == pytest.approx(0.091)
    o, c = np.zeros(7), np.ones(7)
    assert closed(c * 0.8, o, c) == 1 and closed(c * 0.2, o, c) == 0


def test_model_keeps_the_torso_and_below_with_absolute_meshes(tmp_path):
    src = tmp_path / "g1.urdf"
    links = ["pelvis", "torso_link", "left_shoulder_pitch_link", "head_link"]
    xml = ['<robot name="g1">']
    for name in links:
        xml.append(f'<link name="{name}"><visual><geometry><mesh filename="meshes/{name}.STL"/></geometry></visual>'
                   f'<collision><geometry><mesh filename="meshes/{name}.STL"/></geometry></collision></link>')  # fmt: skip
    for parent, child, kind in [("pelvis", "torso_link", "revolute"), ("torso_link", "left_shoulder_pitch_link", "revolute"),
                                ("torso_link", "head_link", "fixed")]:  # fmt: skip
        xml.append(f'<joint name="{child}_j" type="{kind}"><parent link="{parent}"/><child link="{child}"/>'
                   '<axis xyz="0 1 0"/><limit lower="-1" upper="1" effort="1" velocity="1"/></joint>')  # fmt: skip
    src.write_text("".join(xml) + "</robot>")
    root = ET.parse(model(src, tmp_path / "out.urdf")).getroot()
    assert sorted(link.get("name") for link in root.findall("link")) == [
        "head_link",
        "left_shoulder_pitch_link",
        "torso_link",
    ]
    arm = root.find("link[@name='left_shoulder_pitch_link']")
    assert arm.find("collision/geometry/mesh").get("filename") == str(tmp_path / "meshes/left_shoulder_pitch_link.STL")
    assert not root.find("link[@name='torso_link']").findall("collision")


class FakeSim:
    """The server's wire: arms move `lag` of the way to their targets per step; the pelvis walks toward its target at
    up to `v` m/s (and turns), unless a wall at x = `wall` stops it; the hands go straight to their binary pose."""

    def __init__(self, lag=0.5, v=0.4, wall=None, limit=3000, sag=0.0):
        self.q = {s: np.zeros(7) for s in ("left", "right")}
        self.hand = {s: np.zeros(7) for s in ("left", "right")}
        self.base, self.body, self.steps, self.limit, self.lag, self.v, self.wall = (
            np.zeros(3),
            0.79,
            0,
            limit,
            lag,
            v,
            wall,
        )
        self.rows, self.lost, self.walking, self.stopped, self.sag = [], None, False, None, sag
        self.observed = []

    def state(self):
        c, s = math.cos(self.base[2]), math.sin(self.base[2])
        mount = np.eye(4)
        mount[:2, :2] = [[c, -s], [s, c]]
        mount[:3, 3] = [self.base[0], self.base[1], self.body + 0.3]
        return dict(q={k: v.copy() for k, v in self.q.items()}, hand={k: v.copy() for k, v in self.hand.items()},
                    base=self.base.copy(), body=self.body, mount=mount, tilt=1.0, stopped=self.stopped,
                    walking=self.walking, steps=self.steps, step_limit=self.limit, done=self.steps >= self.limit)  # fmt: skip

    def request(self, op, rows=(), **_):
        if op == "observe":
            self.observed.append(self.steps)
            return self.state()
        assert op == "act"
        self.stopped = None
        for row in rows:
            self.rows.append(row)
            for s, q in row["q"].items():
                self.q[s] += self.lag * (np.asarray(q) - self.sag - self.q[s])  # (a controller that sags)
            for s, u in row["u"].items():
                self.hand[s] = np.full(7, float(u))
            target = row.get("base")
            self.walking = target is not None and np.hypot(*(np.asarray(target[:2]) - self.base[:2])) > 0.04
            if target is not None:
                d = np.asarray(target, float) - self.base
                step = d[:2] * min(1.0, self.v * DT / max(np.hypot(*d[:2]), 1e-9))
                self.base[:2] += step
                self.base[2] += float(np.clip(d[2], -0.6 * DT, 0.6 * DT))
                if self.wall is not None and self.base[0] > self.wall:
                    self.base[0], self.stopped = self.wall, "contact"
            h = row.get("body", 0.79)
            self.body += float(np.clip(h - self.body, -0.15 * DT, 0.15 * DT))
            self.steps += 1
        return self.state()

    def close(self):
        pass


def fake(**kw):
    r = object.__new__(HumanoidArena)
    r.sim, r.lock = FakeSim(**kw), __import__("threading").Lock()
    r.step_limit, r.steps, r.ended, r.dt, r.instruction = r.sim.limit, 0, False, DT, "test"
    r.T_mount = np.eye(4)
    hand = SimpleNamespace(open=np.zeros(7), closed=np.ones(7))
    r.arms = {s: SimpleNamespace(T_world_base=r.T_mount, hand=hand) for s in ("left", "right")}
    r.body, r.stand = Body(lo=0.45, hi=0.79), 0.79
    for s in ("left", "right"):
        r.arms[s].lo, r.arms[s].hi = np.full(7, -3.0), np.full(7, 3.0)
    r.frames, r.actions, r.requests = [], [], []
    r._take(r.sim.state())
    r.cmd_q = {s: np.zeros(7) for s in ("left", "right")}
    r.cmd_u = {s: 0 for s in ("left", "right")}
    r.cmd_body = None
    return r


def path(n=15, q=None, base=None, hand=None, body=None):
    t = np.arange(1, n + 1) * DT
    return Trajectory(t=t, q=q or {}, hand=hand or {}, T={}, base=None if base is None else np.asarray(base, float),
                      body=None if body is None else np.asarray(body, float))  # fmt: skip


def test_follow_moves_one_arm_holds_the_other_and_stands():
    r = fake()
    r.cmd_q["right"] = np.full(7, 0.3)
    Q = np.linspace(0, 0.5, 16)[1:, None] * np.ones(7)
    f = r.follow(path(q={"left": Q}))
    np.testing.assert_allclose(r.obs(cams=False).q["left"], 0.5, atol=0.02)
    assert all(row["q"]["right"] == [0.3] * 7 and "base" not in row and "body" not in row for row in r.sim.rows)
    assert f.steps == len(r.sim.rows) >= 15 and not f.stopped


def test_follow_walks_to_world_waypoints_and_moves_the_mount():
    r = fake()
    mount = r.arms["left"].T_world_base
    n = 150
    B = np.c_[np.linspace(0, 1.0, n + 1)[1:], np.zeros(n), np.linspace(0, math.pi / 2, n + 1)[1:]]
    r.follow(path(n=n, base=B))
    o = r.obs(cams=False)
    assert np.hypot(*(o.base[:2] - [1.0, 0.0])) < 0.05 and abs(o.base[2] - math.pi / 2) < math.radians(5)
    assert r.arms["right"].T_world_base is mount
    np.testing.assert_allclose(mount[:2, 3], o.base[:2])
    assert r.sim.rows[0].get("drive") and "base" not in r.sim.rows[-1]  # it stands once it has arrived


def test_a_blocked_walk_ends_the_motion_and_stands_there():
    r = fake(wall=0.3)
    n = 150
    B = np.c_[np.linspace(0, 2.0, n + 1)[1:], np.zeros(n), np.zeros(n)]
    f = r.follow(path(n=n, base=B))
    assert f.steps < 60 and r.obs(cams=False).base[0] == pytest.approx(0.3)
    assert "base" not in r.sim.rows[-1]


def test_the_pelvis_height_moves_and_holds_until_raised():
    r = fake()
    n = 100
    r.follow(path(n=n, body=np.linspace(0.79, 0.6, n + 1)[1:]))
    assert r.obs(cams=False).body == pytest.approx(0.6, abs=0.04) and r.cmd_body == pytest.approx(0.6)
    r.follow(path(n=10, q={"left": np.zeros((10, 7))}))
    assert r.sim.rows[-1]["body"] == pytest.approx(0.6)  # held through an arm motion
    r.follow(path(n=n, body=np.linspace(0.6, 0.79, n + 1)[1:]))
    assert r.cmd_body is None and r.obs(cams=False).body == pytest.approx(0.79, abs=0.04)
    r.follow(path(n=10, q={"left": np.zeros((10, 7))}))
    assert "body" not in r.sim.rows[-1]  # standing again


def test_hands_are_binary_and_held_and_the_step_limit_ends_motion():
    r = fake()
    f = r.follow(path(n=1, hand={"left": np.ones((1, 7)) * 0.9}))
    assert r.cmd_u["left"] == 1 and f.steps >= 20 and r.obs(cams=False).hand["left"][0] == 1.0
    r = fake(limit=40)
    f = r.follow(path(n=60, q={"left": np.linspace(0, 1, 61)[1:, None] * np.ones(7)}))
    assert f.done and r.ended and r.steps == 40 and len(r.sim.rows) == 40


def test_a_replay_sends_the_recorded_requests_again_and_holds_the_last_command():
    """A request replay sends the renders between the rows too; an `actions.npy` replay sends the rows alone."""
    r = fake()
    n = 60
    r.follow(path(n=n, base=np.c_[np.linspace(0, 0.5, n + 1)[1:], np.zeros(n), np.zeros(n)]))
    r._take(r._req(op="observe"))  # (as `obs()` sends it)
    r.follow(
        path(n=40, q={"left": np.linspace(0, 0.4, 41)[1:, None] * np.ones(7)}, body=np.linspace(0.79, 0.6, 41)[1:])
    )
    r.follow(path(n=1, hand={"right": np.ones((1, 7))}))
    again = fake()
    obs = again.replay([json.loads(json.dumps(m)) for m in r.requests])
    assert again.sim.rows == r.sim.rows and again.sim.observed == r.sim.observed and obs.steps == r.steps
    assert again.cmd_q["left"] == pytest.approx(r.cmd_q["left"]) and again.cmd_u == {"left": 0, "right": 1}
    assert again.cmd_body == pytest.approx(r.cmd_body)
    rows = np.array([flat(row) for row in r.sim.rows], np.float32)  # as `finish` saves them
    natively = fake()
    natively.replay(rows)
    assert natively.sim.rows == [unflat(a) for a in rows] and not natively.sim.observed
    assert sum("drive" in row for row in r.sim.rows) == 1 and not any("drive" in row for row in natively.sim.rows)
    assert natively.steps == r.steps and np.abs(natively.state["base"] - r.state["base"]).max() < 1e-5
    limited = fake(limit=50)
    limited.replay(rows)
    assert limited.steps == 50 and limited.ended and len(limited.sim.rows) == 50


class SimProcess:
    """The simulator's process: records its command and reports ready at once."""

    started = []

    def __init__(self, cmd, stdout, env=None, **kw):
        SimProcess.started.append((cmd, env))
        stdout.write("ready sit_sofa 1 port 1 boot 1s\n")
        stdout.flush()
        self.pid = 4242

    def poll(self):
        return None

    def terminate(self):
        pass

    def wait(self, timeout=None):
        return 0


def start(monkeypatch, tmp_path, **kw):
    SimProcess.started = []
    monkeypatch.setenv("HOME", str(tmp_path))
    for name in ("HA_RUNTIME", "HA_IMAGE", "HA_SANDBOX", "HA_DOCKER_ARGS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(launch.os, "getuid", lambda: 4242)
    monkeypatch.setattr(launch.tempfile, "tempdir", str(tmp_path / "tmp"))
    monkeypatch.setattr(runtime.subprocess, "Popen", SimProcess)
    monkeypatch.setattr(runtime.os, "killpg", lambda pid, sig: None)
    port, out = launch.start("sit_sofa", "1", tmp_path / "out", "3", vendor=tmp_path / "vendor", **kw)
    assert out == (tmp_path / "out").resolve() and isinstance(port, int)
    return SimProcess.started[0]


def test_docker_runs_the_sim_as_root_with_the_slots_caches(monkeypatch, tmp_path):
    """The image runs as root; its caches and home live in the cache slot's volume, one container per slot."""
    docker = []
    monkeypatch.setattr(
        runtime.subprocess, "run", lambda cmd, **kw: docker.append(cmd) or SimpleNamespace(returncode=0)
    )
    monkeypatch.setattr(runtime.glob, "glob", lambda pattern: [])
    sim, _ = start(monkeypatch, tmp_path, runtime="docker", docker_args="--cpus 8")
    run = docker[0]
    assert run[-3:] == ["ghcr.io/openrobokit/roboskills-humanoidarena:0.2.0", "-c", "sleep infinity"]
    assert "--gpus" in run and run[run.index("--gpus") + 1] == "device=3"
    assert "type=volume,source=ha-4242-cache-0,target=/cache" in run and "HOME=/cache/home" in run
    assert " ".join(run).count("--ulimit") == 2 and run[run.index("--cpus") + 1] == "8"
    assert sim[:2] == ["docker", "exec"] and "-u" not in sim[:4]  # as root
    assert sim[sim.index("-c") + 2 : sim.index("-c") + 4] == ["_", "/isaac-sim/python.sh"]  # the Vulkan wrapper first
    assert sim[sim.index("--port") - 1] == str(episode_seed("sit_sofa", "1"))
    start(monkeypatch, tmp_path, runtime="docker")  # a second simulator while the first runs: the next slot
    assert "type=volume,source=ha-4242-cache-1,target=/cache" in docker[-1]
    launch.close()
    assert docker[-1][:3] == ["docker", "rm", "-f"]


def test_apptainer_runs_the_sim_as_the_caller_with_persistent_slot_caches(monkeypatch, tmp_path):
    """Kit's caches and the home persist in the slot on the host's disk; its data, logs, /tmp and /dev/shm are private."""
    monkeypatch.setattr(runtime.subprocess, "check_output", lambda cmd, **kw: "GPU-6d1f\n")
    monkeypatch.setattr(runtime.glob, "glob", lambda pattern: [])
    (sandbox := tmp_path / ".roboskills/images/roboskills-humanoidarena-0.2.0").mkdir(parents=True)  # built before
    sim, env = start(monkeypatch, tmp_path, runtime="apptainer")
    cache, run = tmp_path / "tmp/sk-ha-4242/cache-0", tmp_path / "tmp/sk-4242" / f"ha-4242-{launch.os.getpid()}-0"
    assert sim[:2] == ["apptainer", "exec"] and env["NVIDIA_VISIBLE_DEVICES"] == "GPU-6d1f"
    assert sim[sim.index("--home") + 1] == f"{cache}/home:{tmp_path}"
    assert str(sandbox) in sim
    for bind in (f"{cache}:/cache:rw", f"{cache}/kit-cache:/isaac-sim/kit/cache:rw", f"{cache}/var-tmp:/var/tmp:rw",
                 f"{run}/isaac-sim-kit-data:/isaac-sim/kit/data:rw", f"{run}/dev-shm:/dev/shm:rw", f"{run}/tmp:/tmp:rw"):  # fmt: skip
        assert bind in sim
    assert "HOME=/cache/home" not in sim and "OMNI_KIT_ACCEPT_EULA=YES" in sim
    launch.close()
    assert not run.exists() and (cache / "kit-cache").is_dir()
    start(monkeypatch, tmp_path, runtime="apptainer")  # the slot was released: reused, warm
    assert f"{cache}:/cache:rw" in SimProcess.started[0][0]
    launch.close()
