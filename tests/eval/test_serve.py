"""The episode server and its host-side Robot, over a toy adapter in a real server process, without a simulator."""

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from roboskills.eval import check, runtime, serve, wire
from roboskills.eval.interface import Arm, Trajectory
from roboskills.eval.libero.hand import finger_q


TOY = """
from pathlib import Path

import numpy as np

from roboskills.eval.interface import Arm, Body, Camera, Follow, Hand, Obs
from roboskills.eval.libero.hand import finger_q


class Toy:
    REMOTE = ("double", "names")

    def __init__(self, out):
        self.dt, self.ended, self.q, self.body = 0.05, False, np.zeros(2), Body(0.3, 0.8)
        hand = Hand("jaw", np.eye(4), np.ones(1), np.zeros(1), model_q=finger_q)
        self.arms = {"right": Arm("right", str(Path(out) / "arm.xml"), ["j1", "j2"], "ee", np.eye(4), -np.ones(2),
                                  np.ones(2), np.zeros(2), hand)}

    def obs(self, cams=True):
        cs = {"c": Camera(np.zeros((4, 5, 3), np.uint8))} if cams else {}
        return Obs(cs, {"right": self.q.copy()}, {"right": np.ones(1)}, int(self.q[0] * 10), 10, "do it")

    def reset(self, seed):
        self.q[:] = seed / 100
        self.ended = False
        self.arms["right"].home[:] = seed / 100
        self.arms["right"].T_world_base[0, 3] = seed

    def robot_mask(self, cam, obs):
        return np.ones((4, 5), bool)

    def follow(self, path, stop=None):
        for n, q in enumerate(path.q["right"], 1):
            self.q = q
            self.arms["right"].T_world_base[0, 3] += 1  # a mobile base carries the arm
            if stop and stop(self.obs(cams=False)):
                return Follow(n, False, True, self.obs(cams=False))
        self.ended = True
        return Follow(n, True, False, self.obs(cams=False))

    def double(self, x):
        return 2 * x

    def names(self):
        return {"a set", "JSON cannot carry"}

    def secret(self):
        return "simulator internals"

    def finish(self, out):
        (Path(out) / "video.mp4").write_text("frames")
        return {"success": bool(self.q[0] > 0)}


def episode(out):
    return Toy(out)
"""


@pytest.fixture(params=[None, sys.executable])
def robot(tmp_path, monkeypatch, request):
    (tmp_path / "toy_adapter.py").write_text(TOY)
    src = Path(serve.__file__).resolve().parents[2]
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(tmp_path), str(src)]))
    r = serve.remote("toy_adapter", dict(out=tmp_path / "ep"), out=tmp_path / "ep", image=None, boot_s=60,
                     python=request.param)  # fmt: skip
    yield r
    serve.close()


def test_reset_preserves_process_and_arm_identity_but_refreshes_episode_state(robot):
    pid, arm = robot.server.proc.pid, robot.arms["right"]
    robot.obs()
    assert robot._masks
    robot.reset(7)
    assert robot.server.proc.pid == pid and robot.arms["right"] is arm
    assert not robot.ended and not robot._masks
    np.testing.assert_allclose(arm.home, [0.07, 0.07])
    np.testing.assert_allclose(robot.obs(cams=False).q["right"], arm.home)
    assert arm.T_world_base[0, 3] == 7


def test_the_host_holds_only_a_proxy_with_the_robots_facts(robot):
    assert isinstance(robot, serve.Remote) and robot.dt == 0.05 and not hasattr(robot, "base")
    hand = robot.arms["right"].hand
    assert hand.model_q is finger_q and np.allclose(hand.model_q([0.5]), finger_q([0.5]))
    obs = robot.obs()
    assert obs.cams["c"].rgb.shape == (4, 5, 3) and obs.instruction == "do it"
    assert robot.robot_mask("c", obs).all() and robot.robot_mask("c", robot.obs(cams=False)) is None
    assert robot.double(21) == 42 and not hasattr(robot, "secret") and robot.body.hi == 0.8


def test_a_reply_the_wire_cannot_carry_fails_that_call_only(robot):
    with pytest.raises(RuntimeError, match="TypeError"):
        robot.names()
    assert robot.double(2) == 4


def test_follow_asks_the_hosts_stop_and_keeps_the_arms_current(robot):
    T = robot.arms["right"].T_world_base
    path = Trajectory(np.arange(1, 4) * 0.1, {"right": np.tile([[0.2], [0.6], [0.9]], 2)}, {}, {})
    f = robot.follow(path, stop=lambda obs: obs.q["right"][0] > 0.5)
    assert (f.steps, f.stopped) == (2, True) and T[0, 3] == 2 and robot.arms["right"].T_world_base is T
    with pytest.raises(ZeroDivisionError):  # the motion ends there, and the host's error surfaces
        robot.follow(path, stop=lambda obs: 1 / 0)
    assert robot.obs(cams=False).q["right"][0] == 0.2  # the server is still in step with the host
    with pytest.raises(RuntimeError, match="cannot call the robot"):  # its stop gets the proprioception instead
        robot.follow(path, stop=lambda obs: robot.obs(cams=False) is None)
    assert robot.double(3) == 6


def test_a_lost_simulator_stays_lost(robot):
    robot.sock.shutdown(2)  # a reply that may still come would answer the wrong request
    with pytest.raises(RuntimeError, match="gone"):
        robot.obs()
    with pytest.raises(RuntimeError, match="gone"):
        robot.double(1)


def test_finish_writes_where_the_host_asks(robot, tmp_path):
    robot.follow(Trajectory(np.array([0.1]), {"right": np.ones((1, 2))}, {}, {}))
    assert robot.ended
    assert robot.finish(tmp_path / "result") == {"success": True}
    assert (tmp_path / "result/video.mp4").read_text() == "frames"


def test_a_server_that_fails_to_start_says_where_to_look(tmp_path, monkeypatch):
    (tmp_path / "broken_adapter.py").write_text("def episode(**kw):\n    raise ValueError('no such task')\n")
    src = Path(serve.__file__).resolve().parents[2]
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(tmp_path), str(src)]))
    with pytest.raises(RuntimeError, match="did not start"):
        serve.remote("broken_adapter", {}, out=tmp_path / "ep", image=None, boot_s=60)
    assert "no such task" in (tmp_path / "ep/sim.log").read_text()


def test_arms_on_one_model_share_its_file(tmp_path):
    (tmp_path / "m.xml").write_text("<mujoco/>")
    arm = Arm("left", str(tmp_path / "m.xml"), ["j"], "ee", np.eye(4), -np.ones(1), np.ones(1), np.zeros(1), None)
    toy = SimpleNamespace(arms={"left": arm, "right": Arm(**{**arm.__dict__, "name": "right"})}, dt=0.1)
    arms = serve.facts(toy, tmp_path / "out", export=True)["arms"]
    assert arms["left"].model == arms["right"].model == str(tmp_path / "out/model/left/robot.xml")


def test_portable_inlines_includes_under_the_main_files_directories(tmp_path):
    (tmp_path / "meshes/robot").mkdir(parents=True)
    (tmp_path / "meshes/robot/a.stl").write_text("solid")
    (tmp_path / "sky.png").write_text("png")
    (tmp_path / "robot").mkdir()
    (tmp_path / "robot/arm.xml").write_text('<mujoco><asset><mesh file="robot/a.stl"/></asset></mujoco>')
    (tmp_path / "scene.xml").write_text(
        '<mujoco><compiler meshdir="meshes"/><include file="robot/arm.xml"/>'
        '<asset><texture type="skybox" fileup="sky.png" fileleft="sky.png"/></asset></mujoco>'
    )
    text = Path(serve.portable(tmp_path / "scene.xml", tmp_path / "out/robot.xml")).read_text()
    assert "<include" not in text and "meshdir" not in text and 'name="a"' in text
    assert str(tmp_path / "out/assets") in text and "a.stl" in text and 'fileup="sky.png"' not in text


def test_a_bound_method_cannot_cross_the_wire():
    with pytest.raises(TypeError, match="cannot cross"):
        wire.ref(serve.Remote.__init__.__get__(object()))


def test_docker_mounts_quote_paths_with_commas(monkeypatch):
    cmds = []
    monkeypatch.setattr(runtime.subprocess, "run", lambda cmd, **kw: cmds.append(cmd) or SimpleNamespace(returncode=0))
    monkeypatch.setattr(runtime.glob, "glob", lambda pattern: [])
    server = runtime.Server("docker", "img", 0, mounts=[("/runs/a-10,11", "/runs/a-10,11", "rw")])
    server.close()
    assert 'type=bind,"source=/runs/a-10,11","target=/runs/a-10,11"' in cmds[0]


def test_episodes_get_a_directory_of_their_own(tmp_path):
    a, b = (serve.episode_dir(None, tmp_path / "runs/robotwin/x-s1") for _ in range(2))
    assert a != b and a.parent == b.parent == tmp_path / "runs/robotwin" and a.name.startswith("x-s1-")
    assert serve.episode_dir(tmp_path / "given", "unused") == tmp_path / "given"


def test_check_reports_an_episode_that_raises(tmp_path, monkeypatch):
    (tmp_path / "raising_adapter.py").write_text(
        "def make(**kw):\n    raise OSError('no image')\n\ndef close():\n    pass\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    report = check.check("raising_adapter", [("t", {}, None)], ["native", "docker"], tmp_path / "out", 1e-4)
    assert report["failures"] == [f"t {r}: the no-op raised OSError: no image" for r in ("native", "docker")]


def test_portable_models_name_their_files_absolutely(tmp_path):
    (tmp_path / "m").mkdir()
    (tmp_path / "m/a.stl").write_text("solid")
    (tmp_path / "arm.xml").write_text('<mujoco><compiler meshdir="m"/><asset><mesh file="a.stl"/></asset></mujoco>')
    copied = serve.portable(tmp_path / "arm.xml", tmp_path / "out/robot.xml")
    text = Path(copied).read_text()
    assert str(tmp_path / "out/assets") in text and 'name="a"' in text and "meshdir" not in text
    serve.portable(tmp_path / "arm.xml", tmp_path / "same/robot.xml", copy=False)
    assert str(tmp_path / "m/a.stl") in (tmp_path / "same/robot.xml").read_text()


@pytest.mark.skipif(sys.platform != "linux", reason="process groups")
def test_close_stops_the_server(robot):
    proc = robot.server.proc
    serve.close()
    assert proc.poll() is not None


def test_docker_pulls_a_missing_image_once_and_apptainer_builds_its_sandbox_from_the_registry(monkeypatch, tmp_path):
    cmds, have = [], set()

    def run(cmd, **kw):
        cmds.append(cmd)
        if cmd[1] == "pull":
            have.add(cmd[2])
        return SimpleNamespace(returncode=0 if cmd[1] != "image" or cmd[3] in have else 1)

    monkeypatch.setattr(runtime.subprocess, "run", run)
    monkeypatch.setattr(runtime.shutil, "which", lambda name: f"/usr/bin/{name}")
    image = "ghcr.io/openrobokit/roboskills-libero:0.1.0"
    assert runtime.pick(None, image, "NO_SUCH_VAR") == "docker" and ["docker", "pull", image] in cmds
    cmds.clear()
    assert runtime.docker_has(image) and cmds == [["docker", "image", "inspect", image]]  # pulled already
    monkeypatch.setattr(runtime.shutil, "which", lambda name: None)
    assert runtime.pick(None, image, "NO_SUCH_VAR") == "apptainer"
    monkeypatch.setenv("HOME", str(tmp_path))
    built = tmp_path / ".roboskills/images/roboskills-libero-0.1.0"
    assert runtime.sandbox(image) == built
    assert cmds[-1] == ["apptainer", "build", "--userns", "--sandbox", "--fix-perms", str(built), f"docker://{image}"]
    built.mkdir()
    cmds.clear()
    assert runtime.sandbox(image) == built and runtime.sandbox("/sandboxes/mine") == Path("/sandboxes/mine")
    assert not cmds


def test_a_tool_runs_in_a_fresh_container_as_the_caller_seeing_only_its_mounts(monkeypatch, tmp_path):
    image = "ghcr.io/openrobokit/roboskills-hawor:0.3.0"
    monkeypatch.setattr(runtime, "docker_has", lambda name: name == image)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    monkeypatch.setenv("ROBOSKILLS_RUN", "run-1")
    monkeypatch.setenv("SHOWN", "1")
    monkeypatch.delenv("UNSET", raising=False)
    cmd = runtime.contained(image, ["python", "-c", "print(1)"], [tmp_path, ("/models", "/opt/models", "ro")],
                            name="tool-1", env=["SHOWN", "UNSET"])  # fmt: skip
    assert cmd[:3] == ["docker", "run", "--rm"] and cmd[cmd.index("--gpus") + 1] == '"device=3"'
    assert cmd[cmd.index("-u") + 1] == f"{os.getuid()}:{os.getgid()}"
    mounts = [cmd[i + 1] for i, x in enumerate(cmd) if x == "--mount"]
    assert mounts == [
        f"type=bind,source={tmp_path},target={tmp_path}",
        "type=bind,source=/models,target=/opt/models,readonly",
    ]
    assert "SHOWN" in cmd and "UNSET" not in cmd and cmd[cmd.index("--name") + 1] == "tool-1"
    assert cmd[cmd.index("--label") + 1] == "roboskills.run=run-1"  # a harness run removes what its agent left
    assert cmd[-5:] == ["--entrypoint", "python", image, "-c", "print(1)"]
    assert runtime.contained("ghcr.io/openrobokit/roboskills-other:0.1.0", ["python"]) is None  # no docker for it

    calls = []
    monkeypatch.setattr(runtime.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(runtime.subprocess, "run",
                        lambda cmd, **kw: calls.append(cmd) or SimpleNamespace(stdout="a1\nb2\n"))  # fmt: skip
    runtime.remove_containers("run-1")
    assert calls == [
        ["docker", "ps", "-aq", "--filter", "label=roboskills.run=run-1"],
        ["docker", "rm", "-f", "a1", "b2"],
    ]


def test_from_a_shell_a_tool_runs_where_it_is_called(monkeypatch, tmp_path):
    image = "ghcr.io/openrobokit/roboskills-sam3d:0.3.0"
    monkeypatch.setattr(runtime, "docker_has", lambda name: True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys, "argv", ["runtime", "--mount", "/models:/opt/models:ro", image, "python", "render.py", "-x"]
    )
    execs = []
    monkeypatch.setattr(runtime.os, "execvp", lambda path, cmd: execs.append(cmd))
    runtime.main()
    (cmd,) = execs
    assert f"type=bind,source={tmp_path},target={tmp_path}" in cmd and cmd[cmd.index("-w") + 1] == str(tmp_path)
    assert "type=bind,source=/models,target=/opt/models,readonly" in cmd
    assert cmd[-5:] == ["--entrypoint", "python", image, "render.py", "-x"]
