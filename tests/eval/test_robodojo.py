"""RoboDojo's eval suites, checkout lookup and simulator wire, without Isaac Sim or a GPU."""

import json
import socket
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from roboskills.eval import runtime
from roboskills.eval.robodojo import launch, vendor
from roboskills.eval.robodojo.layouts import FULL5_TASKS, check_layout
from roboskills.eval.robodojo.sim import replay, vector
from roboskills.eval.robodojo.sim.server import ACTION
from roboskills.eval.wire import Sim, free_port, recv, send


@pytest.mark.parametrize(
    ("task", "layout", "mode", "suite"),
    [("general_pickup", 0, "eval", "legacy"), ("general_pickup", 10, "dev", "legacy"),
     ("press_by_number", 55, "eval", "legacy"), ("stack_blocks", 4, "eval", "full5"), ("stack_blocks", 13, "dev", "full5")],
)  # fmt: skip
def test_suites_admit_their_own_episodes(task, layout, mode, suite):
    check_layout(task, layout, 2, mode, suite)


@pytest.mark.parametrize(
    ("task", "layout", "seed", "mode", "suite"),
    [("general_pickup", 0, 2, "dev", "legacy"), ("general_pickup", 10, 2, "eval", "legacy"),
     ("general_pickup", 0, 1, "eval", "legacy"), ("stack_blocks", 0, 2, "eval", "legacy"),
     ("stack_blocks", 0, 2, "dev", "full5"), ("stack_blocks", 13, 2, "eval", "full5"),
     ("stack_blocks", 13, 3, "dev", "full5"), ("general_pickup", 10, 2, "dev", "full10")],
)  # fmt: skip
def test_eval_layouts_stay_in_eval_mode(task, layout, seed, mode, suite):
    with pytest.raises(ValueError):
        check_layout(task, layout, seed, mode, suite)


def test_full5_lists_the_canonical_tasks():
    assert len(FULL5_TASKS) == 42 and "stack_blocks" in FULL5_TASKS


def test_every_canonical_task_has_its_recipe():
    from roboskills.eval.robodojo.robot import RECIPES

    assert sorted(p.stem for p in RECIPES.glob("*.md")) == sorted(FULL5_TASKS)


def test_checkout_comes_from_the_environment(monkeypatch, tmp_path):
    monkeypatch.delenv("ROBODOJO_VENDOR", raising=False)
    with pytest.raises(RuntimeError, match="ROBODOJO_VENDOR"):
        vendor()
    assert vendor(tmp_path) == tmp_path.resolve()
    monkeypatch.setenv("ROBODOJO_VENDOR", str(tmp_path))
    assert vendor() == tmp_path.resolve()


def test_vector_launch_refuses_eval_layouts_before_docker(monkeypatch, tmp_path):
    monkeypatch.setattr(launch.subprocess, "run", lambda *a, **kw: pytest.fail("docker started"))
    with pytest.raises(ValueError):
        launch.make_vector("general_pickup", "0,10", "0", tmp_path, mode="dev", vendor=tmp_path)


class SimProcess:
    """The process `make_vector` starts for the sim: records its command and reports ready at once."""

    started, robots = [], []

    def __init__(self, cmd, stdout, env=None, **kw):
        SimProcess.started.append((cmd, env))
        stdout.write("ready vector_sim num_envs=1\n")
        stdout.flush()
        self.pid = 4242

    def poll(self):
        return None

    def terminate(self):
        pass

    def wait(self, timeout=None):
        return 0


def launch_one_env(monkeypatch, tmp_path, gpu="0", **kw):
    SimProcess.started, SimProcess.robots = [], []
    monkeypatch.setenv("HOME", str(tmp_path))
    for name in ("RD_RUNTIME", "RD_IMAGE", "RD_SANDBOX", "RD_CACHE_PREFIX", "RD_DOCKER_ARGS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(launch.os, "getuid", lambda: 4242)
    monkeypatch.setattr(launch.os, "getgid", lambda: 4343)
    monkeypatch.setattr(launch.subprocess, "Popen", SimProcess)
    monkeypatch.setattr(launch, "make", lambda **robot: SimProcess.robots.append(robot) or robot["layout"])
    assert launch.make_vector("general_pickup", "10", gpu, tmp_path, mode="dev", vendor=tmp_path, **kw) == [10]
    return SimProcess.started[0]


def test_one_image_runs_the_sim_as_whoever_launches_it_with_their_own_caches(monkeypatch, tmp_path):
    """Docker images and volumes are shared by everyone on a host; a volume is writable only by its creator."""
    docker, lib = [], "/usr/lib/x86_64-linux-gnu/libnvidia-gpucomp.so.595.99.02"
    monkeypatch.setattr(launch.subprocess, "run", lambda cmd, **kw: docker.append(cmd) or SimpleNamespace(returncode=0))
    monkeypatch.setattr(runtime.glob, "glob", lambda pattern: [lib] if "gpucomp" in pattern else [])
    sim, _ = launch_one_env(monkeypatch, tmp_path, runtime="docker")
    assert docker[0][-3:] == ["ghcr.io/openrobokit/roboskills-robodojo:0.2.0", "-c", "sleep infinity"]
    assert "type=volume,source=sk-rd-4242-kitcache,target=/isaac-sim/kit/cache" in docker[0]
    assert "type=volume,source=sk-rd-4242-optixcache,target=/var/tmp/OptixCache_robodojo" in docker[0]
    assert (
        f"type=bind,source={lib},target={lib},readonly" in docker[0]
    )  # Vulkan's compiler library, which older NVIDIA container toolkits omit
    assert sim[:4] == ["docker", "exec", "-u", "4242:4343"]
    launch.close()
    assert docker[-1][:3] == ["docker", "rm", "-f"]


def test_official_observation_reaches_the_sim_and_every_robot(monkeypatch, tmp_path):
    monkeypatch.setattr(launch.subprocess, "run", lambda cmd, **kw: SimpleNamespace(returncode=0))
    sim, _ = launch_one_env(monkeypatch, tmp_path, runtime="docker", obs="official")
    assert "--official" in sim and [r["obs"] for r in SimProcess.robots] == ["official"]
    launch.close()
    sim, _ = launch_one_env(monkeypatch, tmp_path, runtime="docker")
    assert "--official" not in sim and [r["obs"] for r in SimProcess.robots] == ["full"]
    assert sim[sim.index("--events") + 1] == str(tmp_path.resolve() / "sim_events.jsonl")
    launch.close()


def test_inside_the_sim_image_the_sim_runs_natively_on_the_given_gpu(monkeypatch, tmp_path):
    """A Kubernetes Pod is the image: Isaac Sim's python runs there, with no container to start or remove."""
    signals, python = [], tmp_path / "python.sh"
    python.touch()
    monkeypatch.setattr(launch, "PYTHON", str(python))
    monkeypatch.setattr(launch.subprocess, "run", lambda *a, **kw: pytest.fail("a container was started"))
    monkeypatch.setattr(launch.os, "killpg", lambda pid, sig: signals.append(sig))
    sim, env = launch_one_env(monkeypatch, tmp_path)
    assert sim[0] == str(python) and env["CUDA_VISIBLE_DEVICES"] == "0" and env["ACCEPT_EULA"] == "Y"
    launch.close()
    assert signals == [runtime.signal.SIGTERM, runtime.signal.SIGKILL]


def test_hosts_where_docker_cannot_pull_the_image_run_it_under_apptainer_on_one_gpu(monkeypatch, tmp_path):
    """Apptainer exposes every GPU unless NVIDIA's container CLI narrows it; Kit's caches stay off the network home."""
    signals, ngx, cmds = [], "/usr/lib/x86_64-linux-gnu/libnvidia-ngx.so.580.82.07", []
    (tmp_path / ".roboskills/runs/docker/kit-exts/isaacsim.asset.importer.urdf").mkdir(parents=True)
    monkeypatch.setattr(launch.shutil, "which", lambda name: "/usr/bin/docker")
    failing = lambda cmd: cmd[1] in ("image", "pull")  # noqa: E731  (no such image here, and none from the registry)
    monkeypatch.setattr(launch.subprocess, "run",
                        lambda cmd, **kw: cmds.append(cmd) or SimpleNamespace(returncode=int(failing(cmd))))  # fmt: skip
    monkeypatch.setattr(runtime.glob, "glob", lambda pattern: [ngx] if "ngx" in pattern else [])
    monkeypatch.setattr(launch.tempfile, "tempdir", str(tmp_path / "tmp"))
    monkeypatch.setattr(launch.subprocess, "check_output", lambda cmd, **kw: "GPU-6d1f\n" if "3" in cmd else "GPU-0\n")
    monkeypatch.setattr(launch.os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    sim, env = launch_one_env(monkeypatch, tmp_path, gpu="3")
    sandbox = sim.index("/isaac-sim/python.sh") - 1
    assert sim[:2] == ["apptainer", "exec"] and "--nvccli" in sim and "--cleanenv" in sim
    image, built = launch.IMAGE, str(tmp_path / ".roboskills/images/roboskills-robodojo-0.2.0")
    assert ["docker", "pull", image] in cmds and sim[sandbox] == built
    assert ["apptainer", "build", "--userns", "--sandbox", "--fix-perms", built, f"docker://{image}"] in cmds
    assert env["NVIDIA_VISIBLE_DEVICES"] == "GPU-6d1f"
    scratch = tmp_path / "tmp/rd-4242"
    assert f"{scratch}/cache/gpu3/kit:/isaac-sim/kit/cache:rw" in sim and f"{tmp_path}:{tmp_path}:rw" in sim
    assert f"{scratch}/cache/gpu3/var-tmp:/var/tmp:rw" in sim and "OPTIX_CACHE_PATH=/var/tmp/OptixCache_robodojo" in sim
    (run,) = (tmp_path / "tmp/sk-4242").glob("rd-harness-4242-*")
    assert (run / "isaac-sim-kit-data/Kit/Isaac-Sim/5.1/exts/3/isaacsim.asset.importer.urdf").is_dir()
    assert f"{ngx}:{ngx}:ro" in sim  # NGX (DLSS), which --nvccli does not mount: without it cameras render black
    assert f"{run}/tmp:/tmp:rw" in sim  # not --contain's 16 MiB session directory
    (run / "isaac-sim-kit-data/documents/Kit/shared/screenshots").mkdir(parents=True, mode=0)  # as Kit leaves it
    launch.close()
    assert [sig for _, sig in signals] == [runtime.signal.SIGTERM, runtime.signal.SIGKILL]
    assert not run.exists() and (scratch / "cache/gpu3/kit").is_dir()


def test_sim_requests_round_trip_arrays_and_report_errors():
    listener = socket.create_server(("127.0.0.1", 0))
    port = listener.getsockname()[1]

    def serve():
        conn, _ = listener.accept()
        with conn:
            while True:
                try:
                    msg = recv(conn)
                except EOFError:
                    return
                if msg["op"] == "observe":
                    send(conn, {"state": {"q": np.arange(3.0)}, "steps": 7})
                else:
                    send(conn, {"error": f"unknown op {msg['op']!r}"})
                    return

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    sim = Sim(port, timeout=10)
    rep = sim.request(op="observe")
    assert rep["steps"] == 7 and np.array_equal(rep["state"]["q"], np.arange(3.0))
    with pytest.raises(RuntimeError, match="simulator: unknown op"):
        sim.request(op="bogus")
    thread.join(5)
    with pytest.raises(RuntimeError, match="the simulator is gone"):
        sim.request(op="observe")
    assert sim.lost
    with pytest.raises(RuntimeError, match="the simulator is gone"):
        sim.request(op="observe")
    sim.close()
    listener.close()


def test_motion_lines_name_the_verb_and_turn_that_moved_the_robot():
    """The agent reads one line per motion naming the bot method and turn that asked for it."""
    from roboskills.core.agent import execute
    from roboskills.eval.robodojo.robot import RoboDojo
    from roboskills.tools.robokit.bot import Tools

    class Robot(RoboDojo):
        def __init__(self):
            self.steps, self.ended = 0, False

        def _follow(self, path, stop=None):
            self.steps += len(path)
            return SimpleNamespace(done=False, obs=SimpleNamespace(q={}, hand={}))

    bot = Tools.__new__(Tools)
    bot.robot, bot.model = Robot(), None
    bot.robot.arms = {}
    plan = SimpleNamespace(
        steps=None,
        ended=None,
        stopped=None,
        q_start=None,
        paths=[[0, 0, 0]],
        tcp_end=None,
        record=lambda follows, ended: None,
        pose=SimpleNamespace(arm=None, verb="drive"),
    )
    assert execute("bot.execute(plan)", {"bot": bot, "plan": plan}) == "steps 3 at 0 by execute<execute\n"


class Batch:
    """A vector batch without Isaac Sim: every tick, each live env's joint closes a tenth of its gap to its last
    command, so ticks where an env is idle move it too."""

    def __init__(self, n):
        self.num_envs, self.official, self.fixes, self.dev, self.step_lim = n, False, [], False, 1000
        self.q, self.target, self.instructions = [0.0] * n, [0.0] * n, ["reach"] * n
        self.take_action_cnt, self.end_flag, self.success = [0] * n, [False] * n, [False] * n
        self.idle_steps, self.physical_steps, self.commanded_steps = [0] * n, [0] * n, [0] * n
        self.running = True
        self.app = SimpleNamespace(is_running=lambda: self.running)

    def joint_actions(self, actions):
        for e, action in actions.items():
            self.target[e] = action["left_arm_joint_state"][0]
        for e in vector.count_step(self, actions):
            self.q[e] += 0.1 * (self.target[e] - self.q[e])

    def observe(self, e, render):
        obs = dict(state={"left_arm_joint_state": np.array([self.q[e]])}, steps=self.take_action_cnt[e],
                   step_limit=self.step_lim, done=self.end_flag[e])  # fmt: skip
        if render:
            obs["cams"] = {"cam_head": {"rgb": np.full((4, 4, 3), self.physical_steps[e] % 256, np.uint8)}}
        return obs

    def terminal(self, e, final=False):
        if final and not self.end_flag[e]:
            self.end_flag[e], self.success[e] = True, self.q[e] > 0.5

    def score(self, e):
        return 1.0 if self.success[e] else self.q[e]

    def poses(self, e):
        return {"block:b0": [self.q[e], 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]}


def test_a_batch_whose_cameras_render_black_does_not_start():
    """A camera left unlit stays black for the sim's life; the agents would act blind."""
    batch = Batch(2)
    batch.physical_steps = [100, 100]  # Batch's frames are lit by its step count
    vector.check_cameras(batch)
    batch.physical_steps[1] = 3
    with pytest.raises(RuntimeError, match="camera cam_head renders black in env 1"):
        vector.check_cameras(batch)


def test_a_vector_batch_replays_from_its_event_log(tmp_path):
    """Two agents at different paces share one clock; the log replays both episodes to the same endings and frames."""
    served, ports = Batch(2), [free_port(), free_port()]
    config = SimpleNamespace(
        robot=SimpleNamespace(robots=[dict(default_root_pos=[0, 0, 0], default_root_rot=[1, 0, 0, 0])])
    )
    log = vector.events(tmp_path / "sim_events.jsonl", task="t", layouts=[0, 1], seed=2)
    sim = threading.Thread(target=vector.serve, args=(served, config, ports, [0, 1], 2, "shared", log), daemon=True)
    sim.start()

    def agent(e, target, pause):
        time.sleep(0.2)  # the batch's sockets are listening
        client = Sim(ports[e], timeout=10)
        client.request(op="reset", layout=e, seed=2, fixes=[], official=False)
        action = {k: [target if k == "left_arm_joint_state" else 0.0] * n for k, n in ACTION.items()}
        for _ in range(3):
            client.request(op="act", actions=[action] * 2, render=True)
            time.sleep(pause)  # thinking: the shared clock keeps ticking
        client.request(op="finish")
        client.close()

    agents = [threading.Thread(target=agent, args=args) for args in ((0, 1.0, 0.02), (1, 0.4, 0.1))]
    for thread in agents:
        thread.start()
    for thread in agents:
        thread.join(20)
    served.running = False
    sim.join(5)
    events = [json.loads(line) for line in (tmp_path / "sim_events.jsonl").read_text().splitlines()]
    assert events[0]["op"] == "batch" and min(served.idle_steps) > 0

    results, frames = replay.replay(Batch(2), events[1:])
    for e in (0, 1):
        assert results[e]["same_outcome"] and results[e]["joints_rad"] == 0.0 and results[e]["objects_m"] == 0.0
        assert results[e]["success"] == [e == 0, e == 0]
        assert len(frames[e]) == sum(event == dict(op="observe", env=e) for event in events)


def test_scene_makes_world_geometry_on_the_host_from_raw_sim_poses():
    from roboskills.eval.robodojo.robot import RoboDojo

    pytest.importorskip("robokit")
    q = np.array([np.cos(0.4), 0.0, 0.0, np.sin(0.4)])  # 0.8 rad about z, wxyz
    local = np.array([[0.1, 0.0, 0.0], [0.0, 0.2, 0.05]])
    reply = dict(objects=[dict(id="cup_1", label="cup", position=[1, 2, 3], quaternion_wxyz=q.tolist(),
                               local_points=local.tolist(), geometry="native_mesh_surface")],
                 errors=[], frame="world", steps=7)  # fmt: skip
    robot = RoboDojo.__new__(RoboDojo)
    robot.information, robot._alias, robot._req = "privileged", {"cup_1": "cup_0"}, lambda **msg: reply
    scene = robot.scene()
    (o,) = scene["objects"]
    c, s = np.cos(0.8), np.sin(0.8)
    T = np.array([[c, -s, 0, 1], [s, c, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1.0]])
    P = local @ T[:3, :3].T + T[:3, 3]  # what sim/state.py computed in the container before
    assert o["id"] == "cup_0" and np.allclose(o["pose"], T) and np.allclose(o["points"], P)
    assert np.allclose(o["lower"], P.min(0)) and np.allclose(o["upper"], P.max(0))
    assert "quaternion_wxyz" not in o and "local_points" not in o
    assert scene["errors"] == []


def test_native_server_rejects_geometry_before_importing_simulator():
    from roboskills.eval.robodojo.sim.server import handle

    env = SimpleNamespace(allow_scene=False, end_flag=[False], take_action_cnt=[10])
    with pytest.raises(PermissionError):
        handle(env, None, {"op": "scene"})
