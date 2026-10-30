"""The agent-free pipeline's status.json per tool, skipping and its inputs (roboskills.workflows.motion_gen.pipeline), on fake tools."""

import json

import numpy as np
import pytest

from roboskills.workflows.motion_gen import pipeline


status = lambda w, name: json.loads((w / name / "status.json").read_text())  # noqa: E731


@pytest.fixture
def fakes(monkeypatch):
    calls = []

    def tool(name, output):
        def function(w, o, p):
            calls.append(name)
            o.mkdir(parents=True, exist_ok=True)
            (w / output).write_text(json.dumps(p))

        return function, output

    tools = {n: tool(n, o) for n, o in (("gvhmr", "motion/smplx.npz"), ("retarget", "retarget/motion.npz"), ("blender", "blender/compare.mp4"))}  # fmt: skip
    monkeypatch.setattr(pipeline, "TOOLS", tools)
    monkeypatch.setattr(
        pipeline, "CHECKS", {"retarget": lambda w: dict(ok=False, reason="a limb strays", limb_mm=120.0)}
    )
    return calls


def test_every_tool_reports_and_a_finished_tool_is_skipped(tmp_path, fakes):
    pipeline.run(tmp_path, tools=["blender", "gvhmr", "retarget"], moving_camera=True, engine="eevee")
    assert fakes == ["gvhmr", "retarget", "blender"]
    assert status(tmp_path, "motion")["ok"] and status(tmp_path, "blender")["seconds"] is not None
    assert json.loads((tmp_path / "motion/smplx.npz").read_text()) == dict(moving_camera=True, engine="eevee")
    failed = status(tmp_path, "retarget")  # a check that fails is recorded; the later tools still run
    assert not failed["ok"] and failed["limb_mm"] == 120.0 and failed["reason"] == "a limb strays"
    pipeline.run(tmp_path, redo=["retarget"])
    assert fakes == ["gvhmr", "retarget", "blender", "retarget"]


def test_a_finished_tool_without_a_status_is_only_checked(tmp_path, fakes):
    (tmp_path / "retarget").mkdir()
    (tmp_path / "retarget/motion.npz").write_text("kept")
    pipeline.run(tmp_path, tools=["retarget"])
    assert fakes == [] and (tmp_path / "retarget/motion.npz").read_text() == "kept"
    assert status(tmp_path, "retarget")["seconds"] is None


def test_a_raise_is_recorded_and_stops_the_run(tmp_path, fakes, monkeypatch):
    def boom(w, o, p):
        o.mkdir(parents=True)
        raise RuntimeError("no person")

    monkeypatch.setitem(pipeline.TOOLS, "gvhmr", (boom, "motion/smplx.npz"))
    with pytest.raises(RuntimeError):
        pipeline.run(tmp_path)
    assert fakes == [] and "no person" in status(tmp_path, "motion")["reason"]


def test_an_fbx_becomes_input_character_glb_and_a_glb_is_copied(tmp_path, fakes, monkeypatch):
    imported = []
    monkeypatch.setattr(pipeline, "import_character", lambda source, out: imported.append((source, out)))
    (tmp_path / "C.FBX").write_text("rig")
    (tmp_path / "v.mp4").write_text("video")
    w = tmp_path / "w"
    pipeline.run(w, video=tmp_path / "v.mp4", character=tmp_path / "C.FBX", tools=[])
    assert imported == [(w / "input/character.fbx", w / pipeline.CHARACTER)]
    assert (w / pipeline.VIDEO).read_text() == "video"
    (tmp_path / "C.glb").write_text("glb")
    pipeline.run(w, character=tmp_path / "C.glb", tools=[])
    assert (w / pipeline.CHARACTER).read_text() == "glb" and len(imported) == 1


def test_check_video_needs_every_frame_at_the_motions_fps(tmp_path, monkeypatch):
    np.savez(tmp_path / "motion.npz", fps=30.0, turns=np.zeros((120, 2, 3, 3)))
    probe = type("Probe", (), dict(stdout='{"streams": [{"nb_read_frames": "119", "avg_frame_rate": "30/1"}]}'))
    monkeypatch.setattr(pipeline.subprocess, "run", lambda *a, **k: probe)
    report = pipeline.check_video(tmp_path / "v.mp4", tmp_path / "motion.npz")
    assert not report["ok"] and report["frames"] == 119 and "119 frames" in report["reason"]
    probe.stdout = '{"streams": [{"nb_read_frames": "120", "avg_frame_rate": "30/1"}]}'
    assert pipeline.check_video(tmp_path / "v.mp4", tmp_path / "motion.npz")["ok"]
