"""Unified source-to-native reconstruction; SKILL.md is the workflow. A list of TOOLS is the pipeline: they run over
one workspace, input/ (video.mp4, object.txt; a robot's robot.npz, camera.json) then a folder per tool, each reading
the folders of the listed tools before it (`inputs`, NEEDS: a tool starts once those are done, beside any other that
is ready) and reporting in its status.json whether it succeeded (`call`, with the tool's own check in CHECKS).
`--retry` reruns a tool that is not ok with other parameters (RETRIES) in retries/<tool>/<set>/, track's sweep at
once, and installs the set that checks best (`retry_tool`). `agent` has Codex fix what track's checks still fail
(agent.py). `--no-parallel` runs one tool or set at a time.

    python -m roboskills.workflows.real2sim --workspace W [--tools sam3 ...] [--support pedestal] [--retry]
    python -m roboskills.workflows.real2sim --workspace W --tools pnp --retry
"""

import json
import shutil
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from multiprocessing import get_context
from pathlib import Path
from typing import Literal

from roboskills.tools.blender.blender import render_scene
from roboskills.tools.coacd.coacd import decompose_convex
from roboskills.tools.geocalib.geocalib import estimate_gravity
from roboskills.tools.hawor.hawor import check_hands, reconstruct_hands
from roboskills.tools.moge3.moge3 import estimate_depth
from roboskills.tools.mujoco.mujoco import assemble_scene
from roboskills.tools.robokit.edit import check_track
from roboskills.tools.robokit.mask_alignment import fit_shape
from roboskills.tools.robokit.mpc import check_rollout, feedback, track_with_mpc
from roboskills.tools.robokit.retarget import retarget_hand
from roboskills.tools.sam3.sam3 import check_masks, segment_video
from roboskills.tools.sam3d.sam3d import reconstruct_object
from roboskills.workflows.real2sim.agent import agent
from roboskills.workflows.real2sim.align import align, check_scale
from roboskills.workflows.real2sim.events import events
from roboskills.workflows.real2sim.pnp import check_route, pick_and_place
from roboskills.workflows.real2sim.track import track


reference = lambda w: json.loads(p.read_text()) if (p := w / "input/reference.json").exists() else dict(frame=0)
VIDEO, F, ROBOT = "input/video.mp4", lambda w: reference(w)["frame"], lambda w: (w / "input/robot.npz").exists()
ACTOR, OBJECT = lambda w: "robot arm" if ROBOT(w) else "right hand", lambda w: (w / "input/object.txt").read_text()
TOOLS = {  # name, in running order -> (function of the workspace, its output folder and `inputs` `p`, the output that marks it done)
    "sam3": (lambda w, o, p: segment_video(w / VIDEO, [OBJECT(w), ACTOR(w)], o, frame=F(w), point=reference(w).get("point")), "masks/objects.json"),
    "moge3": (lambda w, o, p: estimate_depth(w / VIDEO, o, frame=F(w)), "depth/depth.npz"),
    "geocalib": (lambda w, o, p: estimate_gravity(w / VIDEO, o), "gravity/gravity.json"),
    "sam3d": (lambda w, o, p: reconstruct_object(w / VIDEO, w / "masks", w / "depth/depth.npz", o, frame=F(w)), "shape/pose.json"),
    "hawor": (lambda w, o, p: reconstruct_hands(w / VIDEO, w / "masks", w / "depth/depth.npz", o), "hands/hands.npz"),
    "align": (lambda w, o, p: align(w / "masks", w / "depth/depth.npz", w / "shape", w / "hands/hands.npz", o, frame=F(w), robot=w / "input/robot.npz" if ROBOT(w) else None, camera=w / "input/camera.json", **p.get("align_kwargs", {})), "align/align.json"),
    "fit": (lambda w, o, p: fit_shape(w / "masks", w / "align/depth.npz", w / "align", o, frame=F(w)), "fit/fit.json"),
    "track": (lambda w, o, p: track(w / "masks", w / p["depth"], w / p["shape"], o, frame=F(w), refit=p["refit"], **p.get("track_kwargs", {})), "track/poses.npz"),
    "coacd": (lambda w, o, p: decompose_convex(w / p["shape"] / "mesh.ply", o), "collision/parts.json"),
    "mujoco": (lambda w, o, p: assemble_scene(w / VIDEO, w / "masks", w / p["depth"], w / p["shape"], w / "collision", o, frame=F(w), support=p["support"],
                                              **{k: w / p[k] for k in ("gravity", "hands", "track") if k in p}, robot=w / "input/robot.npz" if ROBOT(w) else None, camera=w / "input/camera.json"), "scene/preview.jpg"),
    "agent": (agent, "agent/agent.json"),
    "events": (lambda w, o, p: events(w / "hands/hands.npz", w / "track/poses.npz", w / p["shape"], o), "events/events.json"),
    "blender": (lambda w, o, p: render_scene(w / "scene", w / VIDEO, o), "blender/compare_ext1.mp4"),
    "retarget": (lambda w, o, p: retarget_hand(w / VIDEO, w / "hands/hands.npz", w / "scene", o, robot=(w / "input/robot.txt").read_text().strip() if (w / "input/robot.txt").exists() else "sharpa", **p.get("retarget_kwargs", {})), "retarget/preview.jpg"),
    "mpc": (lambda w, o, p: track_with_mpc(w / VIDEO, w / "retarget", o, **p.get("mpc_kwargs", {})), "mpc/diagnostics.json"),
    "pnp": (lambda w, o, p: pick_and_place(w, o, **p.get("pnp_kwargs", {})), "pnp/report.json"),  # a robot clip's (eval/*/read.py)
}  # fmt: skip
CHECKS = {  # tool -> its own check of its outputs: ok, a reason, its numbers; a tool without one is ok once it finishes
    "sam3": lambda w, p: check_masks(w / "masks", F(w), [OBJECT(w).strip(), ACTOR(w)]),
    "hawor": lambda w, p: check_hands(w / "hands/hands.npz"),
    "align": lambda w, p: check_scale(w / "align/align.json"),
    "track": lambda w, p: check_track(w / "masks", w / p["depth"], w / p["shape"] / "mesh.ply", w / "track/poses.npz"),
    "mpc": lambda w, p: check_rollout(w / "retarget", w / "mpc"),
    "pnp": lambda w, p: check_route(w, w / "pnp"),  # route: the sim's objects end where the source's do
}  # fmt: skip
RETRIES = {  # tool -> (tools rerun, better score, stop at first ok, sets: inputs over the pipeline's, the first run's first)
    "track": (("align", "fit", "track"), max, False, {  # align's scale per clip, SAM 3D's mesh scaled; ICP window, trim
        f"{'clip' if clip else 'frame'}{'-mesh' if mesh else ''}-w{window}-t{trim}": dict(
            align_kwargs=dict(clip=clip, scale_mesh=mesh), track_kwargs=dict(window=window, trim=trim))
        for clip in (False, True) for mesh in (False, True) for window in (21, 31) for trim in (80, 90)}),
    "mpc": (("retarget", "mpc"), min, True, dict(first={}, rough=dict(retarget_kwargs=dict(steady=0.0, calm=0)), seed1=dict(mpc_kwargs=dict(seed=1)),
            squeeze=dict(mpc_kwargs=dict(squeeze=1.0)), penalty=dict(mpc_kwargs=dict(penetration=3000.0)),
            lead=dict(retarget_kwargs=dict(lead=25)), feedback=dict(feedback=1.0))),  # feedback: from the best so far
    "pnp": (("pnp",), max, True, dict(first={}, physics=dict(pnp_kwargs=dict(retry=True)))),  # physics ladder, SAM 3D samples
}  # fmt: skip
NEEDS = {  # tool -> the tools whose folders it reads; one not listed reads every tool before it (TOOLS' order)
    "sam3": "", "moge3": "", "geocalib": "", "sam3d": "sam3 moge3", "hawor": "sam3 moge3", "align": "sam3 moge3 sam3d hawor",
    "fit": "sam3 align", "track": "sam3 moge3 sam3d align fit", "coacd": "sam3d align fit",
    "mujoco": "sam3 moge3 geocalib sam3d hawor align fit track coacd", "agent": "sam3 moge3 align track coacd mujoco",
    "events": "sam3d hawor align fit track agent", "blender": "mujoco agent", "retarget": "hawor mujoco agent", "mpc": "retarget",
}  # fmt: skip
DEFAULT = "sam3 moge3 geocalib sam3d hawor align fit track coacd mujoco events blender retarget mpc".split()
Tool = Literal[tuple(TOOLS)]
folder = lambda name: TOOLS[name][1].split("/")[0]
SETS = lambda n: ProcessPoolExecutor(n, mp_context=get_context("spawn"))  # retry sets at once, a process each: Warp's
# and torch's GPU state is per process (two retargets in one process corrupt each other's memory)
status = lambda f: json.loads(s.read_text()) if (s := Path(f) / "status.json").exists() else {}
write = lambda f, record: f.mkdir(parents=True, exist_ok=True) or (f / "status.json").write_text(json.dumps(record, indent=1) + "\n")  # fmt: skip


def inputs(tools: list[str], support: str) -> dict:
    """The folders the listed `tools` hand on: the object from fit, else align, else SAM 3D (shape/); depth from align,
    else MoGe; gravity, hands and track where their tool runs; track refits to the masks only with fit; mujoco's
    `support` (table plane or Do as I Do's pedestals); and the `tools`."""
    has = set(tools).__contains__
    p = dict(shape="fit" if has("fit") else "align" if has("align") else "shape", refit=has("fit"), support=support,
             depth="align/depth.npz" if has("align") else "depth/depth.npz", tools=list(tools))  # fmt: skip
    found = (("geocalib", "gravity", "gravity/gravity.json"), ("hawor", "hands", "hands/hands.npz"), ("track", "track", "track/poses.npz"))  # fmt: skip
    return p | {key: path for tool, key, path in found if has(tool)}


def call(w: Path, name: str, p: dict, fresh: bool = True) -> dict:
    """Tool `name` over `w` with inputs `p` (unless not `fresh`: only checked), then its folder's status.json: `ok`, a
    `reason` and its CHECKS numbers (else ok once it finishes; a raise is recorded, then raised again) and `seconds`."""
    function, output = TOOLS[name]
    start, record = time.monotonic(), dict(ok=False)
    try:
        fresh and function(w, (w / output).parent, p)
        record = dict(ok=True, reason="") | CHECKS.get(name, lambda w, p: {})(w, p)
    except BaseException as error:
        record["reason"] = f"{name} raised {type(error).__name__}: {error}"
        raise
    finally:
        write((w / output).parent, dict(record, seconds=round(time.monotonic() - start, 1) if fresh else None))
    return record


def tried(w: Path, tool: str) -> dict:
    """`tool`'s tried sets in w/retries/<tool>/, in RETRIES' order: name -> (folder, its status); the first run's outputs are
    moved from w into the first set's folder once."""
    rerun, _, _, sets = RETRIES[tool]
    root, first = w / "retries" / tool, next(iter(sets))
    if not (root / first).exists():
        (root / first).mkdir(parents=True)
        for f in [folder(t) for t in rerun if (w / folder(t)).is_dir()]:
            shutil.move(w / f, root / first / f)
    order = sorted(root.iterdir(), key=lambda v: list(sets).index(v.name) if v.name in sets else len(sets))
    return {v.name: (v, status(v / folder(tool))) for v in order if status(v / folder(tool))}


def best(tool: str, found: dict) -> str:
    """The best-scoring of the `found` sets; one rescaling SAM 3D's mesh counts only with its align ok."""
    _, better, _, sets = RETRIES[tool]
    counts = [n for n, (v, s) in found.items() if s.get("score") is not None and (status(v / "align").get("ok") or not sets.get(n, {}).get("align_kwargs", {}).get("scale_mesh"))]  # fmt: skip
    return better(counts, key=lambda n: found[n][1]["score"])


def attempt(w: Path, tool: str, name: str, p: dict) -> dict:
    """Retry set `name` of `tool` (RETRIES) in w/retries/<tool>/<name>/, with links to w's other folders: the
    set's listed tools rerun with its inputs over `p`. Physics' `feedback` instead corrects the best set so far's
    reference by its error (feedback) and reruns mpc with that set's inputs. Its status (a raise: not ok)."""
    rerun, _, _, sets = RETRIES[tool]
    v, kwargs, rerun = w / "retries" / tool / name, sets[name], [t for t in rerun if t in p["tools"]]
    shutil.rmtree(v, ignore_errors=True)
    v.mkdir(parents=True)
    for f in w.iterdir():
        f.name in ("retries", *map(folder, rerun)) or (v / f.name).symlink_to(f.resolve())
    if "feedback" in kwargs:
        base = best(tool, found := tried(w, tool))
        feedback(found[base][0] / "retarget", found[base][0] / "mpc", v / "retarget", kwargs["feedback"])
        write(v / "retarget", dict(ok=True, reason="", feedback=kwargs["feedback"], source=base))
        kwargs, rerun = dict(sets.get(base, {}), **kwargs), ["mpc"]
    for t in rerun:
        try:
            call(v, t, dict(p, **kwargs))
        except Exception:  # a set that raises is not ok, its status says why; the others are still compared
            write(v / folder(tool), status(v / folder(t)))
            break
    return status(v / folder(tool))


def keep(w: Path, tool: str) -> dict:
    """Install `tool`'s best tried set in w (its tools' folders copied back) and record in its status.json the `kept`
    set and every set's ok and score."""
    found, rerun = tried(w, tool), RETRIES[tool][0]
    source = found[kept := best(tool, found)][0]
    for f in [folder(t) for t in rerun if (source / folder(t)).is_dir() and not (source / folder(t)).is_symlink()]:
        shutil.rmtree(w / f, ignore_errors=True)
        shutil.copytree(source / f, w / f, symlinks=True)  # pnp's retries: its samples link the inputs
    record = dict(found[kept][1], kept=kept, sets={n: dict(ok=s["ok"], score=s.get("score")) for n, (_, s) in found.items()})  # fmt: skip
    write(w / folder(tool), record)
    return record


def retry_tool(w: Path, tool: str, p: dict, parallel: bool = True) -> dict:
    """`tool`, not ok: its other RETRIES sets attempted afresh in order (mpc's stopping at the first that is ok), then
    the best kept. With `parallel`, a sweep that never stops (track's) runs its sets at once, a process each (SETS);
    one that stops stays in order: its sets share one GPU, where at once they took longer than in turn."""
    _, _, stop, sets = RETRIES[tool]
    shutil.rmtree(w / "retries" / tool, ignore_errors=True)  # the first run is what w now holds
    names = list(sets)[1:]
    batch = (
        []
        if stop or not parallel
        else names[: next((i for i, n in enumerate(names) if "feedback" in sets[n]), len(names))]
    )
    if batch:
        with SETS(len(batch)) as pool:
            list(pool.map(attempt, *zip(*[(w, tool, n, p) for n in batch])))
    for name in names[len(batch) :]:
        if attempt(w, tool, name, p).get("ok") and stop:
            break
    return keep(w, tool)


def run(workspace: Path, tools: list[Tool] = DEFAULT, support: Literal["table", "pedestal"] = "table",
        retry: bool = False, video: Path | None = None, object: str | None = None, frame: int | None = None,
        point: tuple[float, float] | None = None, robot: str | None = None, redo: list[Tool] = [],
        parallel: bool = True):  # fmt: skip
    """Run `tools` (the pipeline) over `workspace`, each once the tools it needs are done (NEEDS; one at a time in
    TOOLS' order without `parallel`), skipping finished tools not in `redo` (only checked when without a status); with
    `retry`, a tool that runs and is not ok is retried (`retry_tool`) and the tools after it rerun, so a tool reading
    one a retry may replace also waits for that retry. A raise stops the pipeline once the running tools finish.
    `video`, `object`, the reference `frame` (default 0), an object `point` there and the `robot` hand (sharpa) fill
    input/."""
    (workspace / "input").mkdir(parents=True, exist_ok=True)
    robot and (workspace / "input/robot.txt").write_text(robot)
    if video and Path(video).resolve() != (workspace / VIDEO).resolve():  # a rerun replaces a read-only copy
        (workspace / VIDEO).unlink(missing_ok=True)
        shutil.copyfile(video, workspace / VIDEO)
    object and (workspace / "input/object.txt").write_text(object)
    if frame is not None or point:
        (workspace / "input/reference.json").write_text(json.dumps(dict(frame=frame or 0, point=point)) + "\n")
    p, order, redo = inputs(tools, support), sorted(tools, key=list(TOOLS).index), set(redo)
    needs = {n: set(NEEDS[n].split() if n in NEEDS else order[: order.index(n)]) & set(order) for n in order}
    for i, n in enumerate(order):  # an earlier tool's retry replaces its RETRIES tools' folders: their readers wait
        needs[n] |= {t for t in order[:i] if retry and t in RETRIES and needs[n] & {*RETRIES[t][0]}}

    def step(i, name):
        fresh = name in redo or not (workspace / TOOLS[name][1]).exists()
        if not fresh and status(workspace / folder(name)):
            return
        if not call(workspace, name, p, fresh)["ok"] and retry and fresh and name in RETRIES:
            retry_tool(workspace, name, p, parallel)
            redo.update(order[i + 1 :])

    done, running = set(), {}
    with ThreadPoolExecutor(len(order) if parallel else 1) as pool:
        while len(done) < len(order):
            for i, name in enumerate(order):
                if name not in done | set(running) and needs[name] <= done and (parallel or not running):
                    running[name] = pool.submit(step, i, name)
            finished = wait(running.values(), return_when=FIRST_COMPLETED).done
            for name in [n for n, f in running.items() if f in finished]:
                running.pop(name).result()  # a raise stops the loop; `with` waits for the tools still running
                done.add(name)
