"""EmbodiedSWE-Bench runtime without the simulator: the cohort and its budgets, the robot model written from PhysX's
joint frames, the camera and walking math of the server, the Panda and Dex3 hand conventions, `follow` and `replay`
over a fake simulator (arms, the hand alone, the step limit, the walking G1's pelvis and height, a blocked walk), and
where `make` runs the simulator (docker, apptainer, natively inside the image)."""

import math
import threading
import xml.etree.ElementTree as ET
from types import SimpleNamespace

import numpy as np
import pytest

from roboskills.eval import runtime
from roboskills.eval.embodiedswe import EGO, SEEDS, TASKS, launch, step_limit
from roboskills.eval.embodiedswe.hand import (
    CURL,
    PANDA_TCP_Z,
    R_palm,
    dex_pose,
    dex_tcp,
    panda_opening,
    panda_q,
    panda_tcp,
    part,
)
from roboskills.eval.embodiedswe.robot import GRASP, EmbodiedSWE
from roboskills.eval.embodiedswe.sim.server import STEP, WALK_CLIP, WALK_MIN, look, matrix, pose, quat, urdf, walk
from roboskills.eval.interface import Base, Trajectory


DT = 1 / 60


def test_cohort_presets_budgets_and_cameras():
    assert len(TASKS) == 6 and SEEDS == ("1", "2")
    for task, spec in TASKS.items():
        suite, scene, robot, mode = task.split(".")
        assert mode in ("joint", "loco_joint") and robot in EGO
        assert step_limit(task, DT) == round((45 * spec["goals"] + 30) * 60)
        for eye, target, focal in spec["views"].values():
            assert np.linalg.norm(np.subtract(target, eye)) > 0.5 and 8 < focal < 30
    assert step_limit("puzzle.classify_objects.g1.joint", DT) == 9900


def rot(axis, a):
    k = np.eye(3)[axis]
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K


def T_of(R=np.eye(3), p=(0, 0, 0)):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, p
    return T


def fk(joint, q):
    """URDF semantics: origin, then a turn about (or a slide along) the axis in the child's frame."""
    o = joint.find("origin")
    r, p, y = map(float, o.get("rpy").split())
    T = T_of(rot(2, y) @ rot(1, p) @ rot(0, r), [float(v) for v in o.get("xyz").split()])
    if joint.get("type") == "fixed":
        return T
    a = np.array([float(v) for v in joint.find("axis").get("xyz").split()])
    M = np.eye(4)
    if joint.get("type") == "revolute":
        k = a / np.linalg.norm(a)
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        M[:3, :3] = np.eye(3) + math.sin(q) * K + (1 - math.cos(q)) * K @ K
    else:
        M[:3, 3] = q * a
    return T @ M


def test_urdf_from_physx_frames_moves_like_physx():
    """A joint whose frames are turned in both bodies (as Isaac's Panda's are): the URDF child moves as PhysX moves it,
    T0 * motion(axis of the joint frame) * T1^-1."""
    T0 = T_of(rot(1, -math.pi / 2), (0, 0, 0.333))
    T1 = T_of(rot(1, -math.pi / 2) @ rot(0, 0.3))
    joints = [dict(name="j1", type="revolute", parent="a", child="b", T0=T0.tolist(), T1=T1.tolist(), axis=0, lo=-1, hi=1),
              dict(name="slide", type="prismatic", parent="b", child="c", T0=T_of(p=(0, 0, 0.1)).tolist(),
                   T1=T_of(rot(2, 0.5)).tolist(), axis=0, lo=0, hi=0.04),
              dict(name="fix", type="fixed", parent="c", child="d", T0=np.eye(4).tolist(), T1=np.eye(4).tolist()),
              dict(name="fix", type="fixed", parent="b", child="e", T0=np.eye(4).tolist(), T1=np.eye(4).tolist()),
              dict(name="other", type="fixed", parent="x", child="y", T0=np.eye(4).tolist(), T1=np.eye(4).tolist())]  # fmt: skip
    root = ET.fromstring(urdf("r", joints, "a", {"a": "/m/a.obj", "b": "/m/b.obj"}, visual_only=("a",)))
    assert [link.get("name") for link in root.findall("link")] == ["a", "b", "c", "e", "d"]  # x, y: not below a
    assert [j.get("name") for j in root.findall("joint")] == ["j1", "slide", "fix", "fix_d"]  # unique names
    assert root.find("link[@name='a']/collision") is None and root.find("link[@name='b']/collision") is not None
    j1, slide = root.find("joint[@name='j1']"), root.find("joint[@name='slide']")
    for q in (0.0, 0.7, -0.4):
        physx = T0 @ T_of(rot(0, q)) @ np.linalg.inv(T1)
        np.testing.assert_allclose(fk(j1, q), physx, atol=1e-7)
        physx = T_of(p=(0, 0, 0.1)) @ T_of(p=(q / 10, 0, 0)) @ np.linalg.inv(T_of(rot(2, 0.5)))
        np.testing.assert_allclose(fk(slide, q / 10), physx, atol=1e-7)
    assert float(j1.find("limit").get("upper")) == 1.0
    joints[0]["T1"] = T_of(p=(0.01, 0, 0)).tolist()  # a turn about a point off the child's origin: no URDF for it
    with pytest.raises(ValueError, match="off b's origin"):
        urdf("r", joints, "a", {})


def test_camera_frames_and_quaternions():
    T = look(np.array([1.0, -1.0, 1.5]), (0.0, 0.0, 0.5))
    R = T[:3, :3]
    assert np.linalg.det(R) == pytest.approx(1.0) and np.allclose(R.T @ R, np.eye(3))
    np.testing.assert_allclose(R[:, 0], np.array([-1.0, 1.0, -1.0]) / math.sqrt(3))  # +x looks at the target
    assert R[2, 1] == pytest.approx(0.0) and R[2, 2] > 0  # level, upright
    np.testing.assert_allclose(matrix(quat(R)), R, atol=1e-9)
    down = look(np.zeros(3), (0.0, 0.0, -1.0))[:3, :3]  # straight down: still a rotation
    assert np.linalg.det(down) == pytest.approx(1.0)
    np.testing.assert_allclose(pose([1, 2, 3], quat(R))[:3, 3], [1, 2, 3])


def test_walk_commands_the_path_velocity_in_the_pelvis_frame_plus_feedback():
    now = np.array([1.0, 2.0, math.pi / 2])  # facing +y
    np.testing.assert_allclose(walk(now, now, (0.0, 0.2, 0.0)), [0.2, 0.0, 0.0], atol=1e-12)  # ahead: +vx
    np.testing.assert_allclose(walk(now, now, (0.2, 0.0, 0.0)), [0.0, -0.2, 0.0], atol=1e-12)  # world +x: its right
    v = walk(now, now + [0.0, 0.02, 0.0], (0.0, 0.0, 0.0))
    assert v[0] == pytest.approx(WALK_MIN) and abs(v[1]) < 1e-12  # 2 cm behind: forward, at a speed that steps
    assert np.allclose(walk(now, now + [0.0, 0.005, 0.0], (0.0, 0.0, 0.0)), 0.0, atol=0.01)  # within tolerance
    turn = [walk(now, now + [0.0, 0.0, -0.5], (0.0, 0.0, 0.0), k) for k in (0, STEP[1])]
    assert all(v[2] < 0 for v in turn) and turn[0][1] == -turn[1][1] == -STEP[0]  # turning: side steps, alternating
    v = walk(now, np.array([1.0, 2.0, -math.pi + 0.1 - math.pi / 2]), (0.0, 0.0, 0.0))  # turn the short way
    assert v[2] > 0
    np.testing.assert_allclose(np.abs(walk(now, now + [5.0, 5.0, 3.0], (1.0, 1.0, 1.0))), WALK_CLIP)


def test_panda_jaw_and_tcp():
    for u in (0.0, 0.3, 1.0):
        np.testing.assert_allclose(panda_opening(panda_q([u])), [u])
    assert panda_q([1.0]) == pytest.approx([0.04, 0.04])
    T = panda_tcp()
    assert np.linalg.det(T[:3, :3]) == pytest.approx(1.0)
    np.testing.assert_allclose(T[:3, 0], [0, 0, 1])  # approach: out of the hand
    np.testing.assert_allclose(T[:3, 1], [0, 1, 0])  # squeeze: the fingers' travel
    assert T[2, 3] == PANDA_TCP_Z


def test_dex3_tcp_and_postures_mirror_between_hands():
    names = {s: [f"{s}_hand_{p}_joint" for p in ("index_0", "index_1", "middle_0", "middle_1", "thumb_0", "thumb_1",
                                                    "thumb_2")] for s in ("left", "right")}  # fmt: skip
    assert part("left_hand_thumb_2_joint") == "thumb_2"
    for s, m in (("left", -1), ("right", 1)):
        T, R = dex_tcp(s), R_palm(s)
        assert np.linalg.det(T[:3, :3]) == pytest.approx(1.0) and np.linalg.det(R) == pytest.approx(1.0)
        np.testing.assert_allclose(T[:3, 0], [0, m, 0])  # out of the palm (it faces m * y)
        np.testing.assert_allclose(T[:3, 1], [1, 0, 0])  # along the fingers: thumb -> fingers
        assert T[1, 3] * m > 0.03  # in front of the palm
        np.testing.assert_allclose(R[:, 0], T[:3, 0]) and np.testing.assert_allclose(R[:, 2], [1, 0, 0])
    left, right = dex_pose("left", names["left"], CURL), dex_pose("right", names["right"], CURL)
    assert left[0] < 0 < right[0] and left[6] > 0 > right[6]  # fingers and thumb curl the other way per hand
    np.testing.assert_allclose(left, -right)  # (thumb_0 is 0 in a fist)
    point = dex_pose("left", names["left"], CURL, keep=("index_0", "index_1"))
    assert point[0] == point[1] == 0 and point[2] < 0


class FakeSim:
    """The server's wire: arms move `lag` of the way to their targets per step, fingers go straight to theirs; a
    walking pelvis moves `base_lag` of the way to its target unless a wall at x = `wall` stops it."""

    def __init__(self, loco=False, lag=1.0, base_lag=0.5, wall=None, limit=1000):
        self.sides = ("left", "right") if loco else ("right",)
        self.q = {s: np.zeros(7) for s in self.sides}
        self.h = {s: np.zeros(7 if loco else 2) for s in self.sides}
        self.loco, self.lag, self.base_lag, self.wall, self.limit = loco, lag, base_lag, wall, limit
        self.base, self.body, self.steps, self.rows, self.lost = np.zeros(3), 0.72, 0, [], None

    def state(self):
        root = np.eye(4)
        root[:3, 3] = [self.base[0], self.base[1], 1.0]
        out = dict(q={k: v.copy() for k, v in self.q.items()}, hand={k: v.copy() for k, v in self.h.items()}, root=root,
                   steps=self.steps, done=self.steps >= self.limit, stopped=None)  # fmt: skip
        if self.loco:
            out.update(base=self.base.copy(), body=self.body, tilt=0.0)
        return out

    def request(self, op, rows=(), **_):
        assert op == "act"
        for row in rows:
            self.rows.append(row)
            for s, q in row.get("q", {}).items():
                self.q[s] += self.lag * (np.asarray(q) - self.q[s])
            for s, h in row.get("h", {}).items():
                self.h[s] = np.asarray(h, float)
            if row.get("base") is not None:
                self.base += self.base_lag * (np.asarray(row["base"]) + np.asarray(row["vel"]) * DT - self.base)
                if self.wall is not None:
                    self.base[0] = min(self.base[0], self.wall)
            self.steps += 1
        return self.state()

    def close(self):
        pass


def fake(loco=False, **kw):
    r = object.__new__(EmbodiedSWE)
    r.sim, r.lock = FakeSim(loco, **kw), threading.Lock()
    r.kind, r.loco = ("g1", True) if loco else ("franka", False)
    r.step_limit, r.steps, r.ended, r.dt, r.instruction = r.sim.limit, 0, False, DT, "test"
    r.T_mount = np.eye(4)
    r.arms = {s: SimpleNamespace(T_world_base=r.T_mount) for s in r.sim.sides}
    r.frames, r.actions = [], []
    r._take(r.sim.state())
    r.cmd_q = {s: np.zeros(7) for s in r.arms}
    r.cmd_h = {s: np.zeros(7 if loco else 2) for s in r.arms}
    if loco:
        r.base, r.cmd_base = Base(vmax=0.3, wmax=0.3), np.zeros(3)
    return r


def path(n=15, **kw):
    return Trajectory(t=np.arange(1, n + 1) * DT, q=kw.pop("q", {}), hand=kw.pop("hand", {}), T={}, **kw)


def test_follow_streams_an_arm_and_holds_the_rest():
    r = fake()
    Q = np.linspace(0, 0.5, 16)[1:, None] * np.ones(7)
    f = r.follow(path(q={"right": Q}))
    np.testing.assert_allclose(r.obs(cams=False).q["right"], 0.5, atol=1e-6)
    assert all(row["h"]["right"] == [0.0, 0.0] and "base" not in row for row in r.sim.rows)
    assert f.steps == len(r.sim.rows) >= 15 and not f.stopped


def test_the_hand_alone_holds_a_close_and_opens_quickly_and_the_step_limit_ends_motion():
    r = fake()
    r.cmd_h["right"] = np.full(2, 0.04)
    r.sim.h["right"] = np.full(2, 0.04)
    f = r.follow(path(hand={"right": np.zeros((15, 1))}))
    assert f.steps == GRASP and r.obs(cams=False).hand["right"][0] == pytest.approx(0.0)
    f = r.follow(path(hand={"right": np.ones((15, 1))}))
    assert f.steps <= 4 and r.obs(cams=False).hand["right"][0] == pytest.approx(1.0)
    r = fake(limit=20)
    f = r.follow(path(n=60, q={"right": np.linspace(0, 1, 61)[1:, None] * np.ones(7)}))
    assert f.done and r.ended and r.steps == 20 and len(r.sim.rows) == 20


def test_a_walking_pelvis_follows_world_waypoints_and_stands_still_after():
    r = fake(loco=True)
    B = np.c_[np.linspace(0, 0.6, 61)[1:], np.zeros(60), np.linspace(0, math.pi / 2, 61)[1:]]
    r.follow(path(n=60, base=B))
    o = r.obs(cams=False)
    assert np.hypot(*(o.base[:2] - [0.6, 0.0])) < 0.015 and abs(o.base[2] - math.pi / 2) < math.radians(2)
    assert all("base" in row and len(row["vel"]) == 3 for row in r.sim.rows)
    np.testing.assert_allclose(r.arms["left"].T_world_base[:2, 3], o.base[:2])  # the mount moves with it
    assert o.body is None  # (the walking policy holds its own height)
    n = len(r.sim.rows)
    r.follow(path(n=20, q={"left": np.full((20, 7), 0.1)}))
    assert all("base" not in row for row in r.sim.rows[n:])  # an arm motion: the robot stands still


def test_a_blocked_walk_ends_the_motion_where_the_pelvis_stopped():
    r = fake(loco=True, wall=0.2)
    B = np.c_[np.linspace(0, 1.5, 91)[1:], np.zeros(90), np.zeros(90)]
    f = r.follow(path(n=90, base=B))
    assert f.steps < 60  # the pelvis fell behind its path at the wall
    np.testing.assert_allclose(r.cmd_base[:2], [0.2, 0.0], atol=1e-6)


def test_a_replay_steps_the_recorded_actions_from_the_step_the_episode_is_at():
    r = fake(limit=1000)
    r.steps = 30  # the settling steps ran at boot
    A = np.arange(1300 * 9, dtype=np.float32).reshape(1300, 9)
    obs = r.replay(A)
    assert [row["a"].tolist() for row in r.sim.rows] == A[30:1000].tolist()  # as sent, to the step limit
    assert r.ended and obs.steps == 1000


class SimProcess:
    """The simulator's process: records how it was started and reports ready at once."""

    started = []

    def __init__(self, cmd, stdout, env=None, cwd=None, **kw):
        SimProcess.started.append((cmd, env, cwd))
        stdout.write("ready puzzle.coffee.franka.joint 2\n")
        stdout.flush()
        self.pid = 4242

    def poll(self):
        return None

    def terminate(self):
        pass

    def wait(self, timeout=None):
        return 0


def launch_one(monkeypatch, tmp_path, env=(), **kw):
    SimProcess.started = []
    monkeypatch.setenv("HOME", str(tmp_path))
    for name in ("ESWE_RUNTIME", "ESWE_IMAGE", "ESWE_SANDBOX", "ESWE_CACHE_PREFIX", "ESWE_DOCKER_ARGS"):
        monkeypatch.delenv(name, raising=False)
    for name, value in dict(env).items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(runtime.os, "getuid", lambda: 4242)
    monkeypatch.setattr(runtime.os, "getgid", lambda: 4343)
    monkeypatch.setattr(runtime.os, "killpg", lambda pid, sig: None)
    monkeypatch.setattr(runtime.subprocess, "Popen", SimProcess)
    monkeypatch.setattr(launch, "EmbodiedSWE", lambda port, task, seed, out, every: (task, seed))
    vendor = tmp_path / "EmbodiedSWE"
    vendor.mkdir(exist_ok=True)
    assert launch.make("puzzle.coffee.franka.joint", "2", tmp_path / "out", "3", vendor, **kw) == (
        "puzzle.coffee.franka.joint", "2")  # fmt: skip
    return (*SimProcess.started[0], vendor)


def test_docker_runs_the_sim_as_the_caller_in_the_read_only_checkout_with_robodojos_caches(monkeypatch, tmp_path):
    docker = []
    monkeypatch.setattr(runtime.subprocess, "run", lambda cmd, **kw: docker.append(cmd) or SimpleNamespace(returncode=0))  # fmt: skip
    sim, _, _, vendor = launch_one(monkeypatch, tmp_path, runtime="docker")
    run = docker[0]
    assert run[-3:] == [launch.IMAGE, "-c", "sleep infinity"] and "device=3" in run
    assert f"type=bind,source={vendor},target=/bench,readonly" in run  # the asset fetcher locks in .cache
    assert f"type=bind,source={vendor}/.cache,target=/bench/.cache" in run and (vendor / ".cache").is_dir()
    assert "type=volume,source=sk-rd-4242-kitcache,target=/isaac-sim/kit/cache" in run  # RoboDojo's: the same Kit
    assert {"PYTHONPATH=/bench", "HOME=/tmp", "HF_HUB_OFFLINE=1"} <= set(run)
    assert sim[:6] == ["docker", "exec", "-u", "4242:4343", "-w", "/bench"]
    assert sim[7:9] == ["/isaac-sim/python.sh", str(launch.HERE / "sim/server.py")]
    launch.close()
    assert docker[-1][:3] == ["docker", "rm", "-f"]
    docker.clear()
    env = dict(ESWE_IMAGE="eswe:mine", ESWE_CACHE_PREFIX="mine", ESWE_DOCKER_ARGS="--cpus 8")
    launch_one(monkeypatch, tmp_path, env, runtime="docker")
    assert docker[0][-5:] == ["--cpus", "8", "eswe:mine", "-c", "sleep infinity"]
    assert "type=volume,source=mine-optixcache,target=/var/tmp/OptixCache_robodojo" in docker[0]
    launch.close()


def test_inside_the_image_the_sim_runs_natively_in_the_checkout(monkeypatch, tmp_path):
    python = tmp_path / "python.sh"
    python.touch()
    monkeypatch.setattr(launch, "PYTHON", str(python))
    monkeypatch.setattr(runtime.subprocess, "run", lambda *a, **kw: pytest.fail("a container was started"))
    sim, env, cwd, vendor = launch_one(monkeypatch, tmp_path)
    assert sim[0] == str(python) and cwd == str(vendor)
    assert env["PYTHONPATH"] == str(vendor) and env["CUDA_VISIBLE_DEVICES"] == "3" and env["HF_HUB_OFFLINE"] == "1"
    launch.close()


def test_without_docker_the_sim_runs_in_an_apptainer_sandbox_of_the_image(monkeypatch, tmp_path):
    monkeypatch.setattr(runtime.shutil, "which", lambda name: None)
    monkeypatch.setattr(runtime.subprocess, "check_output", lambda cmd, **kw: "GPU-3\n")
    monkeypatch.setattr(runtime.tempfile, "tempdir", str(tmp_path / "tmp"))
    (sandbox := tmp_path / ".roboskills/images/roboskills-embodiedswe-0.2.0").mkdir(parents=True)  # built before
    sim, env, _, vendor = launch_one(monkeypatch, tmp_path)
    assert sim[:2] == ["apptainer", "exec"] and env["NVIDIA_VISIBLE_DEVICES"] == "GPU-3"
    assert sim[sim.index("/isaac-sim/python.sh") - 1] == str(sandbox)
    assert sim[sim.index("--pwd") + 1] == "/bench" and "--home" in sim
    assert f"{vendor}:/bench:ro" in sim and f"{vendor}/.cache:/bench/.cache:rw" in sim
    assert f"{tmp_path}/tmp/rd-4242/cache/gpu3/kit:/isaac-sim/kit/cache:rw" in sim  # RoboDojo's, on the host's disk
    assert "PYTHONPATH=/bench" in sim
    launch.close()
