"""The runner's status.json per tool and its retries (roboskills.workflows.real2sim.run), on fake tools."""

import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from roboskills.workflows import real2sim


def number(o, p, key, default):  # a fake tool's output: o/value.txt, its number from the set's inputs
    o.mkdir(parents=True, exist_ok=True)
    if p.get(key, {}).get("boom"):
        raise RuntimeError("no object at frame 3")
    (o / "value.txt").write_text(str(p.get(key, {}).get("x", default)))


@pytest.fixture(autouse=True)
def in_process_sets(monkeypatch):  # the fake tools live in this process only: retry sets in threads, not processes
    monkeypatch.setattr(real2sim, "SETS", ThreadPoolExecutor)


read = lambda w, name: float((w / name / "value.txt").read_text())  # noqa: E731
status = lambda w, name: json.loads((w / name / "status.json").read_text())  # noqa: E731


def test_every_tool_reports_and_a_raise_stops_the_pipeline(tmp_path, monkeypatch):
    def boom(w, o, p):
        raise RuntimeError("SAM 3 found nothing")

    tools = dict(first=(lambda w, o, p: number(o, p, "-", 1), "first/value.txt"), second=(boom, "second/value.txt"),
                 third=(lambda w, o, p: number(o, p, "-", 3), "third/value.txt"))  # fmt: skip
    monkeypatch.setattr(real2sim, "TOOLS", tools)
    monkeypatch.setattr(real2sim, "CHECKS", dict(first=lambda w, p: dict(ok=False, reason="too small", size=read(w, "first"))))  # fmt: skip
    with pytest.raises(RuntimeError, match="SAM 3 found nothing"):
        real2sim.run(tmp_path, ["third", "second", "first"], retry=True)
    first, second = status(tmp_path, "first"), status(tmp_path, "second")
    assert (first["ok"], first["reason"], first["size"]) == (False, "too small", 1.0) and first["seconds"] >= 0
    assert not second["ok"] and second["reason"] == "second raised RuntimeError: SAM 3 found nothing"
    assert not (tmp_path / "third").exists()  # nothing after a raise
    monkeypatch.setitem(tools, "second", (lambda w, o, p: number(o, p, "-", 2), "second/value.txt"))
    real2sim.run(tmp_path, ["first", "second", "third"])
    assert (
        status(tmp_path, "second")["ok"] and status(tmp_path, "third")["ok"] and status(tmp_path, "first")["size"] == 1
    )


def test_retry_tries_every_set_and_keeps_the_best_allowed(tmp_path, monkeypatch):
    tools = dict(align=(lambda w, o, p: number(o, p, "align_kwargs", 1), "align/value.txt"),
                 track=(lambda w, o, p: number(o, p, "track_kwargs", read(w, "align")), "track/value.txt"),
                 after=(lambda w, o, p: number(o, dict(t=dict(x=read(w, "track"))), "t", 0), "after/value.txt"))  # fmt: skip
    checks = dict(align=lambda w, p: dict(ok=read(w, "align") < 5), track=lambda w, p: dict(ok=read(w, "track") > 9, score=read(w, "track")))  # fmt: skip
    sets = {"first": {}, "more": dict(track_kwargs=dict(x=3)), "boom": dict(track_kwargs=dict(boom=True)),
            "rescaled": dict(align_kwargs=dict(x=7, scale_mesh=True)), "less": dict(track_kwargs=dict(x=2))}  # fmt: skip
    monkeypatch.setattr(real2sim, "TOOLS", tools)
    monkeypatch.setattr(real2sim, "CHECKS", checks)
    monkeypatch.setattr(real2sim, "RETRIES", dict(track=(("align", "track"), max, False, sets)))
    real2sim.run(tmp_path, ["align", "track", "after"], retry=True)
    kept, retries = status(tmp_path, "track"), tmp_path / "retries/track"
    assert sorted(p.name for p in retries.iterdir()) == sorted(sets)  # every set tried, the first run moved there
    assert kept["kept"] == "more" and kept["sets"]["rescaled"] == dict(ok=False, score=7.0)  # its align is not ok
    assert kept["sets"]["boom"] == dict(ok=False, score=None) and "no object at frame 3" in status(retries / "boom", "track")["reason"]  # fmt: skip
    assert (read(tmp_path, "align"), read(tmp_path, "track"), read(tmp_path, "after")) == (
        1,
        3,
        3,
    )  # after: on the kept
    assert (retries / "more/align").is_dir() and (retries / "more/input").is_symlink()


def test_retry_stops_at_the_first_ok_set_and_feedback_starts_from_the_best_so_far(tmp_path, monkeypatch):
    def feedback(retarget, mpc, out, gain):  # the best set's reference, corrected
        shutil.copytree(retarget, out)
        (out / "value.txt").write_text(str(float((retarget / "value.txt").read_text()) - 4 * gain))

    mpc = lambda w, o, p: number(o, dict(m=dict(x=read(w, "retarget") - p.get("mpc_kwargs", {}).get("d", 0))), "m", 0)  # noqa: E731
    tools = dict(retarget=(lambda w, o, p: number(o, p, "retarget_kwargs", 9), "retarget/value.txt"), mpc=(mpc, "mpc/value.txt"))  # fmt: skip
    sets = {"first": {}, "seed": dict(mpc_kwargs=dict(d=1)), "lead": dict(retarget_kwargs=dict(x=6), mpc_kwargs=dict(d=1)),
            "feedback": dict(feedback=1.0), "never": dict(mpc_kwargs=dict(d=8))}  # fmt: skip
    monkeypatch.setattr(real2sim, "TOOLS", tools)
    monkeypatch.setattr(real2sim, "CHECKS", dict(mpc=lambda w, p: dict(ok=read(w, "mpc") <= 1, score=read(w, "mpc"))))
    monkeypatch.setattr(real2sim, "RETRIES", dict(mpc=(("retarget", "mpc"), min, True, sets)))
    monkeypatch.setattr(real2sim, "feedback", feedback)
    real2sim.run(tmp_path, ["retarget", "mpc"], retry=True)
    kept = status(tmp_path, "mpc")
    assert {k: v["score"] for k, v in kept["sets"].items()} == dict(
        first=9, seed=8, lead=5, feedback=1
    )  # never: not run
    assert kept["kept"] == "feedback" and kept["ok"] and status(tmp_path, "retarget")["source"] == "lead"
    assert (read(tmp_path, "retarget"), read(tmp_path, "mpc")) == (
        2,
        1,
    )  # lead's reference corrected, lead's MPC inputs


def test_parallel_runs_ready_tools_at_once_with_the_serial_results(tmp_path, monkeypatch):
    slow = lambda x: lambda w, o, p: time.sleep(0.3) or number(o, p, "-", x)  # noqa: E731
    tools = dict(sam3=(slow(1), "masks/value.txt"), moge3=(slow(2), "depth/value.txt"), geocalib=(slow(3), "gravity/value.txt"),
                 sam3d=(lambda w, o, p: number(o, dict(t=dict(x=read(w, "masks") + read(w, "depth"))), "t", 0), "shape/value.txt"))  # fmt: skip
    monkeypatch.setattr(real2sim, "TOOLS", tools)
    monkeypatch.setattr(real2sim, "CHECKS", {})
    seconds = {}
    for parallel in (False, True):
        start = time.monotonic()
        real2sim.run(tmp_path / str(parallel), list(tools), parallel=parallel)
        seconds[parallel] = time.monotonic() - start
        assert [read(tmp_path / str(parallel), f) for f in ("masks", "depth", "gravity", "shape")] == [1, 2, 3, 3]
    assert seconds[True] < 0.6 and seconds[False] >= 0.9  # SAM 3, MoGe-3 and GeoCalib at once; SAM 3D after both


def test_parallel_retry_runs_a_sweep_at_once_and_a_stopping_retry_in_order(tmp_path, monkeypatch):
    def slow(o, p, key, x):
        time.sleep(0.2)
        number(o, p, key, x)

    tools = dict(track=(lambda w, o, p: slow(o, p, "track_kwargs", 1), "track/value.txt"),
                 mpc=(lambda w, o, p: slow(o, p, "mpc_kwargs", 9), "mpc/value.txt"))  # fmt: skip
    sweep = {"first": {}, **{f"w{x}": dict(track_kwargs=dict(x=x)) for x in (3, 5, 4, 2)}}
    stops = {"first": {}, "seed": dict(mpc_kwargs=dict(x=5)), "squeeze": dict(mpc_kwargs=dict(x=1)), "lead": dict(mpc_kwargs=dict(x=0))}  # fmt: skip
    monkeypatch.setattr(real2sim, "TOOLS", tools)
    monkeypatch.setattr(real2sim, "CHECKS", dict(track=lambda w, p: dict(ok=False, score=read(w, "track")), mpc=lambda w, p: dict(ok=read(w, "mpc") <= 1, score=read(w, "mpc"))))  # fmt: skip
    monkeypatch.setattr(real2sim, "RETRIES", dict(track=(("track",), max, False, sweep), mpc=(("mpc",), min, True, stops)))  # fmt: skip
    kept, seconds = {}, {}
    for parallel in (False, True):
        start = time.monotonic()
        real2sim.run(tmp_path / str(parallel), ["track", "mpc"], retry=True, parallel=parallel)
        seconds[parallel] = time.monotonic() - start
        kept[parallel] = [status(tmp_path / str(parallel), t) for t in ("track", "mpc")]
    assert kept[True] == kept[False] and [k["kept"] for k in kept[True]] == ["w5", "squeeze"]
    assert set(kept[True][1]["sets"]) == {"first", "seed", "squeeze"}  # lead: never run, as in turn
    assert seconds[True] < seconds[False] - 0.5  # the four track sets at once: 0.2 s instead of 0.8 s


def test_the_pipeline_waits_for_a_retry_that_replaces_what_a_tool_reads(tmp_path, monkeypatch):
    order, ended = "sam3 moge3 geocalib sam3d hawor align fit track coacd mujoco events retarget mpc".split(), {}

    def tool(name):
        def run(w, o, p):
            time.sleep(0.05)
            number(o, p, "-", 1)
            ended.setdefault(name, []).append(time.monotonic())

        return run

    sets = {"first": {}, "wide": {}, "trim": {}}
    monkeypatch.setattr(real2sim, "TOOLS", {n: (tool(n), f"{n}/value.txt") for n in order})
    monkeypatch.setattr(real2sim, "CHECKS", dict(track=lambda w, p: dict(ok=False, score=1.0)))
    monkeypatch.setattr(real2sim, "RETRIES", dict(track=(("align", "fit", "track"), max, False, sets)))
    real2sim.run(tmp_path, order, retry=True)
    assert len(ended["track"]) == 3 and min(ended["coacd"]) > max(ended["track"])  # coacd reads the kept set's fit
    assert (
        all(status(tmp_path, n)["ok"] for n in order if n != "track") and status(tmp_path, "track")["kept"] == "first"
    )


def test_mpc_retries_fall_back_to_the_unsteadied_hand_first():
    sets = real2sim.RETRIES["mpc"][3]
    assert list(sets)[:2] == ["first", "rough"]
    assert sets["rough"] == dict(retarget_kwargs=dict(steady=0.0, calm=0))
