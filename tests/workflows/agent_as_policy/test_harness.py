"""Self-checks without a simulator or an LLM: python -m pytest tests/workflows/agent_as_policy"""

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from roboskills.core import agent, episode
from roboskills.core.llm import LAST
from roboskills.tools.robokit.state import merge


SHOW = "show({'head': np.zeros((4, 4, 3), np.uint8)})"


class FakeTools:
    """Tools for the agent loop: `run.local` builds them as Tools(robot, out=, sam=); the instruction is the env's name."""

    result = None

    def __init__(self, robot=None, out=None, sam=None):
        self.robot, self.out = robot, out

    def status(self):
        """steps"""
        return dict(steps=0, instruction=self.out.name if self.out is not None else "t")

    def finish(self):
        """end"""
        self.result = dict(success=True)
        return "episode finished"


def test_loop():
    replies = iter(
        [
            "no code",
            "```python\n1/0\n```",
            f"```python\nx = 41\n{SHOW}\n```",
            "```python\nprint(x + 1)\nbot.finish()\n```",
        ]
    )
    seen, pics = [], []

    def llm(system, messages):
        seen.append(messages[-1]["content"])
        pics.append([n for m in messages for n, _ in m.get("images", [])])
        return next(replies)

    with tempfile.TemporaryDirectory() as d:
        t = FakeTools()
        assert agent.run(agent.Repl(t), "t", d, llm, prompt="{api}") == (True, 4) and t.result == dict(success=True)
        log = open(f"{d}/log.jsonl").read().splitlines()
    assert len(log) == 4 and "No ```python" in seen[1] and "ZeroDivisionError" in seen[2] and "42" in log[3]
    assert "success" not in seen[-1]  # the score never reaches the model
    assert pics == [[], [], [], ["head"]]  # the image goes in with the receipt of the turn that made it


def env_tools(spans):
    """Tools for run.local whose `move` runs the robot (the planner lock let go) for 0.3 s."""
    import time

    class Env(FakeTools):
        def move(self):
            """moves for 0.3 s (the robot executing: no lock held)"""
            self.robot.follow(lambda: (spans.append(time.time()), time.sleep(0.3)))

    return Env


def vector_adapter(path, n):
    path.write_text(
        "class Robot:\n"
        "    def obs(self, cams=True): pass\n"
        "    def follow(self, motion): motion()\n"
        "    def finish(self, *args): return {'success': True}\n"
        f"def make_vector(**kw): return [Robot() for _ in range({n})]\n"
    )
    return str(path)


def test_parallel(tmp_path, monkeypatch):
    import time

    from roboskills.tools.robokit import bot
    from roboskills.tools.sam3 import sam3
    from roboskills.workflows.agent_as_policy import run

    spans = []
    monkeypatch.setattr(bot, "Tools", env_tools(spans))
    monkeypatch.setattr(sam3, "Sam", lambda: SimpleNamespace(detect=lambda *args: None, close=lambda: None))

    def llm(system, messages, model, cost):  # env 2 never finishes: the runner does
        cost["usd"] += 0.01
        assert "Notes on this task:\nR0BOT" in messages[0]["content"]  # the ROBOT.md beside the adapter
        if len(messages) > 1:
            return "```python\nfinish()\n```"
        task = messages[0]["content"].split()[1]
        return f"```python\nprint('{task}'); bot.move(); print('{task}')\n```" if task != "env2" else "no code"

    (tmp_path / "ROBOT.md").write_text("R0BOT")
    t0 = time.time()
    summary = run.local(vector_adapter(tmp_path / "a.py", 3), tmp_path / "out", {}, max_turns=2, llm=llm)
    assert time.time() - t0 < 2.0 and max(spans) - min(spans) < 0.25, spans  # the moves overlapped: no lock held
    assert summary["successes"] == 3 and summary["voids"] == 0  # finished by the agents, or by the runner
    assert [r["cost_usd"] for r in summary["envs"]] == [0.02, 0.02, 0.02]
    for i in (0, 1):  # each env's prints reach only its own receipt, though their turns overlapped
        receipt = json.loads((tmp_path / f"out/env{i}/log.jsonl").read_text().splitlines()[0])["receipt"]
        assert receipt.startswith(f"env{i}\nenv{i}\n") and f"env{1 - i}" not in receipt, receipt


def test_failed_llm_voids_the_episode(tmp_path, monkeypatch):

    from roboskills.tools.robokit import bot
    from roboskills.tools.sam3 import sam3
    from roboskills.workflows.agent_as_policy import run

    monkeypatch.setattr(bot, "Tools", env_tools([]))
    monkeypatch.setattr(sam3, "Sam", lambda: SimpleNamespace(detect=lambda *args: None, close=lambda: None))

    def llm(system, messages, model, cost):
        raise RuntimeError("claude -p failed: 429 session limit")

    summary = run.local(vector_adapter(tmp_path / "a.py", 1), tmp_path / "out", {}, llm=llm)
    assert summary["voids"] == 1 and summary["successes"] == 0 and "429" in summary["envs"][0]["void"], summary


def test_unlocked_inside_unlocked():

    plan = episode.Lock()
    obs = episode.unlocked(plan.mine, plan)
    follow = episode.unlocked(obs, plan)  # a motion observes with the lock already let go
    with plan:
        assert obs() is False and follow() is False and plan.mine()
    assert not plan.mine()


def test_append_only_logs_usage():
    sent = []

    def llm(system, messages):  # a backend that reports usage, as agent.claude does
        sent.append([dict(m) for m in messages])
        LAST.usage = dict(cache_read_input_tokens=len(messages))
        return f"```python\n{SHOW}\n```" if len(messages) < 4 else "```python\nbot.finish()\n```"

    with tempfile.TemporaryDirectory() as d:
        agent.run(agent.Repl(FakeTools()), "t", d, llm, prompt="{api}")
        log = [json.loads(x) for x in open(f"{d}/log.jsonl")]
    assert all(s[: len(p)] == p for p, s in zip(sent, sent[1:]))  # no message changes once sent
    assert [len(m["images"]) for m in sent[-1][2::2]] == [1, 1]  # every turn's image stays
    assert [r["usage"]["cache_read_input_tokens"] for r in log] == [1, 3, 5] and "llm_s" in log[0]
    m = [dict(role="user", content="task")] + [dict(role="user", content=str(i), images=[(f"{i}{j}", "P") for j in
                                                range(3)]) for i in range(15)]  # fmt: skip
    kept = [n for x in agent.append_only(m) for n, _ in x.get("images", [])]
    assert kept == [n for x in m[1:] for n, _ in x["images"]][20:] and len(m[7]["images"]) == 3  # 45 -> newest 25
    assert agent.append_only(m[:13])[1:] == m[1:13]  # 36 images: none dropped


def test_merge():
    rng = np.random.default_rng(0)
    a = rng.uniform(0, 0.05, (200, 3))
    dets = [
        ("cup", 0.9, a, "head"),
        ("cup", 0.8, a + 0.002, "wrist"),
        ("cup", 0.7, a + 1.0, "wrist"),
        ("bowl", 0.9, a, "wrist"),
    ]
    assert sorted(map(sorted, merge(dets))) == [[0, 1], [2], [3]]


def test_lift_check():

    from robokit.xform.numpy import rot_tl_to_tf_mat

    from roboskills.tools.robokit.perception import View, cloud
    from roboskills.tools.robokit.state import lift_check

    K = np.array([[500.0, 0, 64], [0, 500, 64], [0, 0, 1]])
    cam = SimpleNamespace(
        K=K, T_world_cam=rot_tl_to_tf_mat(np.diag([1.0, -1, -1]), np.array([0, 0, 1.0]))
    )  # 1 m above, looking down

    def shot(*parts):  # a depth camera's view of point sets (z-buffer)
        X = np.concatenate(parts)
        C = X @ cam.T_world_cam[:3, :3] - cam.T_world_cam[:3, 3] @ cam.T_world_cam[:3, :3]  # camera frame
        z = C[:, 2]
        uv = C[:, :2] / z[:, None] * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]
        u, v = np.round(uv).astype(int).T
        d = np.full((128, 128), np.inf)
        ok = (u >= 0) & (u < 128) & (v >= 0) & (v < 128)
        np.minimum.at(d, (v[ok], u[ok]), z[ok])
        c = SimpleNamespace(depth=np.where(np.isinf(d), 0.0, d), K=K, T_world_cam=cam.T_world_cam)  # 0: nothing seen
        return View(cams={"c": c}, clouds={"c": cloud(c)})

    g = np.arange(-0.12, 0.12, 0.001)
    table = np.stack(np.meshgrid(g, g, [0.0]), -1).reshape(-1, 3)
    b = np.arange(-0.02, 0.02, 0.001)
    box = np.stack(np.meshgrid(b, b, np.arange(0, 0.04, 0.001)), -1).reshape(-1, 3) + [0, 0, 0.0005]
    P = shot(table, box).clouds["c"].reshape(-1, 3)
    P = P[np.isfinite(P).all(1) & (P[:, 2] > 0.004)]  # the object as find sees it
    T0, T1 = rot_tl_to_tf_mat(tl=np.array([0, 0, 0.02])), rot_tl_to_tf_mat(tl=np.array([0, 0, 0.12]))
    up = lift_check(shot(table, box + [0, 0, 0.1]), P, 0.0, T0, T1)
    stayed = lift_check(shot(table, box), P, 0.0, T0, T1)
    view = shot(table, box + [0, 0, 0.1])
    view.clouds["c"][:] = np.nan  # the robot fills the view: no votes
    blind = lift_check(view, P, 0.0, T0, T1)
    assert up["lifted"] is True and stayed["lifted"] is False and blind["lifted"] is None, (up, stayed, blind)
    T2 = rot_tl_to_tf_mat(
        tl=np.array([0.05, 0, 0.08])
    )  # carried aside: the cameras see it in the hand and where it stood (as a cup shows)
    both = lift_check(shot(table, box, box + [0.05, 0, 0.06]), P, 0.0, T0, T2, gripping=True)
    slipped = lift_check(shot(table, box), P, 0.0, T0, T2, gripping=True)
    assert both["lifted"] is True and slipped["lifted"] is False, (both, slipped)


def test_sam_lets_planner_go():
    import threading
    import time

    plan, sam_lock, spans = episode.Lock(), episode.Lock(), {}
    detect = episode.unlocked(
        episode.locked(lambda: (time.sleep(0.3), threading.current_thread().name), sam_lock), plan
    )

    def env(segment):
        with plan:
            t = time.time()
            detect() if segment else time.sleep(0.05)  # segments, or plans
            spans[segment] = (t, time.time())

    threads = [threading.Thread(target=env, args=(s,), name=f"env{s}") for s in (True, False)]
    for th in threads:
        th.start()
        time.sleep(0.05)
    for th in threads:
        th.join()
    assert spans[False][1] < spans[True][1] - 0.1, spans  # env False planned while env True was in SAM


def test_grasp_backend_contract(monkeypatch):
    """A $GRASP_BACKEND sees the episode as S(robot, kin, out): kin is the RobotModel, and the RoboDojo backend sweeps
    its hand surface samples (kin.hand_points) along each proposal's approach."""
    from roboskills.tools.robokit import action
    from roboskills.tools.robokit.robot_model import RobotModel

    assert (
        callable(RobotModel.hand_points)
        and callable(RobotModel.tcp)
        and "S = SimpleNamespace(robot=" in __import__("inspect").getsource(action.grasp_poses)
    )
    from roboskills.eval.robodojo import grasp_backend as gb

    names = list(__import__("inspect").signature(gb.candidates).parameters)
    assert names[:10] == ["S", "a", "P", "tz", "obst", "tgt", "avoid_pts", "ax", "cn", "seed"]
    monkeypatch.setenv("GRASP_POLICY_PYTHON", "py")
    monkeypatch.setenv("GRASP_POLICY_RANK", "ranker_reach")
    assert gb._server_cmd("s")[:4] == ["py", "-m", "graspkit.policies.arx_x5_server", "--socket"]
    assert gb._server_cmd("s")[-2:] == ["--rank", "ranker_reach"]
    g = np.arange(-0.02, 0.021, 0.01)
    box = np.stack(np.meshgrid(g, g, [0.0, 0.04]), -1).reshape(-1, 3)
    down, up = np.eye(4), np.eye(4)
    down[:3, :3], up[:3, :3] = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]], [[0, 0, -1], [0, 1, 0], [1, 0, 0]]  # +x down / up
    down[:3, 3], up[:3, 3] = [0, 0, 0.03], [0, 0, -0.03]
    T = np.stack([down, up])
    T_pre = T.copy()
    T_pre[:, 2, 3] -= T[:, :3, 0][:, 2] * 0.08  # 8 cm back along the travel
    keep, why = gb.select(T, T_pre, None, box, -0.001, np.array([[0.0, 0.03, 0.0], [0.0, -0.03, 0.0]]))
    assert list(keep) == [0] and why["cone"] == 1, (keep, why)  # from above kept; from under the table not


def test_approach_cone():
    from roboskills.tools.robokit.action import _approach_cone as approach_cone

    assert approach_cone(None) == (None, 30.0)
    axis, cone = approach_cone("top")
    assert np.allclose(axis, [0, 0, -1]) and cone == 35.0
    axis, cone = approach_cone("side", toward=[0, 2.0])  # an array after "side": no string comparison may follow
    assert np.allclose(axis, [0, 1, 0]) and cone == 50.0
    axis, cone = approach_cone(np.array([0, 0, -2.0]))
    assert np.allclose(axis, [0, 0, -1]) and cone == 30.0


def test_lift_check_takes_the_agents_gripping_judgement():
    from roboskills.tools.robokit import state as hold
    from roboskills.tools.robokit.perception import View

    T, P = np.eye(4), np.random.default_rng(0).uniform(0, 0.05, (50, 3)) + [0, 0, 0.8]
    no_view = View(cams={}, clouds={})  # the cameras see nothing: the fingers decide
    assert hold.lift_check(no_view, P, 0.75, T, T, gripping=True)["lifted"] is True
    assert hold.lift_check(no_view, P, 0.75, T, T, gripping=False)["lifted"] is False
    assert hold.lift_check(no_view, P, 0.75, T, T)["lifted"] is None
    from unittest.mock import patch

    with patch.object(hold, "seen", lambda *a, **k: (0.9, 100)):  # a camera sees it in the hand at once
        assert hold.lift_check(no_view, P, 0.75, T, T, gripping=True)["lifted"] is True


def test_local_start_fails():
    import subprocess
    import sys

    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "a.py").write_text("def make_vector(**kw):\n    raise RuntimeError('sim did not start')\n")
        r = subprocess.run([sys.executable, "-m", "roboskills.workflows.agent_as_policy.run", "--adapter", f"{d}/a.py", "--out", d],
                           capture_output=True, text=True, timeout=120, check=False)  # fmt: skip
        assert r.returncode and "sim did not start" in r.stderr, r.stderr[-500:]  # exits, does not hang


def test_one_env_adapter_plays_an_episode_end_to_end(tmp_path, monkeypatch):
    """An adapter with only `make` runs through the same agent loop and records as a vector env."""

    from roboskills.tools.robokit import bot
    from roboskills.tools.sam3 import sam3
    from roboskills.workflows.agent_as_policy import run

    (tmp_path / "adapter.py").write_text(
        "class Robot:\n"
        "    def obs(self, cams=True): pass\n"
        "    def follow(self, *args): pass\n"
        "    def finish(self, *args): return {'success': True}\n"
        "def make(**kw): return Robot()\n"
    )
    monkeypatch.setattr(bot, "Tools", FakeTools)
    closed = []
    monkeypatch.setattr(sam3, "Sam", lambda: SimpleNamespace(detect=lambda *args: None, close=lambda: closed.append(1)))
    replies = iter([f"```python\n{SHOW}\n```", "```python\nfinish()\n```"])
    summary = run.local(str(tmp_path / "adapter.py"), tmp_path / "out", {}, llm=lambda s, m, model, cost: next(replies))
    assert summary["episodes"] == 1 and summary["successes"] == 1 and summary["voids"] == 0
    log = [json.loads(line) for line in (tmp_path / "out/env0/log.jsonl").read_text().splitlines()]
    assert [row["images"] for row in log] == [["head"], []] and "llm_s" in log[0]
    assert json.loads((tmp_path / "out/env0/result.json").read_text()) == {"success": True}
    assert summary["envs"][0]["turns"] == 2
    assert closed == [1]  # the SAM worker is let go


def test_agents_never_get_artifacts(tmp_path, monkeypatch):
    """No --notes; a ROBOT.md reaches agents only from an adapter outside artifacts/, and every prompt says so."""

    import pytest

    from roboskills.core import llm, skill
    from roboskills.workflows.agent_as_policy import run

    for folder in (tmp_path / "eval", tmp_path / "artifacts" / "bench"):
        folder.mkdir(parents=True)
        (folder / "ROBOT.md").write_text("R0BOT")
    assert episode.robot(SimpleNamespace(__file__=str(tmp_path / "eval" / "launch.py"))) == "R0BOT"
    assert episode.robot(SimpleNamespace(__file__=str(tmp_path / "artifacts" / "bench" / "launch.py"))) == ""
    monkeypatch.setattr("sys.argv", ["run", "--adapter", "a.py", "--out", str(tmp_path), "--notes", "N.md"])
    with pytest.raises(SystemExit):
        run.main()
    assert "artifacts/" in llm.PRIVATE and llm.claude_command("system")[-2:] == ["--append-system-prompt", llm.PRIVATE]
    motion = llm.agent_command(skill("motion_gen"), "model")
    assert llm.PRIVATE in Path(motion[motion.index("--append-system-prompt-file") + 1]).read_text()
    assert not any("PYTHON" in part for part in llm.agent_command("Run `PYTHON -m check`.", "model"))
    assert motion[motion.index("--disallowedTools") + 1].endswith("/artifacts/**)")
