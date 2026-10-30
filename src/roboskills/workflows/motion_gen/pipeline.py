"""A video's person on a Mixamo character with no agent, in the shape of real2sim's code: TOOLS run over one workspace,
input/ (video.mp4, character.glb) then a folder per tool, each reporting in its status.json whether it succeeded
(`call`, with the tool's own check in CHECKS). Body only: GVHMR gives no fingers, and nothing cleans the motion.

    python -m roboskills.workflows.motion_gen --workspace W --video V --character C.fbx [--moving-camera]
"""

import json
import shutil
import subprocess
import time
from fractions import Fraction
from pathlib import Path
from typing import Literal

import numpy as np

from roboskills.tools.blender.blender import import_character, render_motion
from roboskills.tools.gvhmr.gvhmr import recover_motion
from roboskills.workflows.motion_gen.character.retarget import check_motion, retarget_motion


VIDEO, CHARACTER = "input/video.mp4", "input/character.glb"
SOLVED = "retarget/motion.npz"
TOOLS = {  # name, in running order -> (function of the workspace, its output folder and `inputs` `p`, the output that marks it done)
    "gvhmr": (lambda w, o, p: recover_motion(w / VIDEO, o, moving_camera=p["moving_camera"]), "motion/smplx.npz"),
    "retarget": (lambda w, o, p: retarget_motion(w / "motion/smplx.npz", w / CHARACTER, o), SOLVED),
    "blender": (lambda w, o, p: render_motion(w / CHARACTER, w / SOLVED, w / VIDEO, o, p["engine"]), "blender/compare.mp4"),
}  # fmt: skip
Tool = Literal[tuple(TOOLS)]
status = lambda f: json.loads(s.read_text()) if (s := Path(f) / "status.json").exists() else {}
write = lambda f, record: f.mkdir(parents=True, exist_ok=True) or (f / "status.json").write_text(json.dumps(record, indent=1) + "\n")  # fmt: skip


def check_video(video: Path, motion: Path) -> dict:
    """`ok` when `video` decodes to every frame of `motion` at its fps; `frames` and `fps`."""
    probe = ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
             "stream=nb_read_frames,avg_frame_rate", "-of", "json", str(video)]  # fmt: skip
    stream = json.loads(subprocess.run(probe, capture_output=True, text=True, check=True).stdout)["streams"][0]
    frames, fps, saved = int(stream["nb_read_frames"]), float(Fraction(stream["avg_frame_rate"])), np.load(motion)
    ok = frames == len(saved["turns"]) and abs(fps - float(saved["fps"])) < 0.01
    return dict(ok=ok, reason="" if ok else f"{frames} frames at {fps} do not match the motion", frames=frames, fps=fps)


CHECKS = {  # tool -> its own check of its outputs: ok, a reason, its numbers; a tool without one is ok once it finishes
    "retarget": lambda w: check_motion(w / SOLVED, w / CHARACTER),
    "blender": lambda w: check_video(w / "blender/video.mp4", w / SOLVED),
}


def call(w: Path, name: str, p: dict, fresh: bool = True) -> dict:
    """Tool `name` over `w` with inputs `p` (unless not `fresh`: only checked), then its folder's status.json: `ok`, a
    `reason` and its CHECKS numbers (else ok once it finishes; a raise is recorded, then raised again) and `seconds`."""
    function, output = TOOLS[name]
    start, record = time.monotonic(), dict(ok=False)
    try:
        fresh and function(w, (w / output).parent, p)
        record = dict(ok=True, reason="") | CHECKS.get(name, lambda w: {})(w)
    except BaseException as error:
        record["reason"] = f"{name} raised {type(error).__name__}: {error}"
        raise
    finally:
        write((w / output).parent, dict(record, seconds=round(time.monotonic() - start, 1) if fresh else None))
    return record


def run(workspace: Path, video: Path | None = None, character: Path | None = None, tools: list[Tool] = list(TOOLS),
        redo: list[Tool] = [], moving_camera: bool = False,
        engine: Literal["cycles", "eevee", "workbench"] = "cycles"):  # fmt: skip
    """Run `tools` over `workspace` in TOOLS' order, skipping finished tools not in `redo` (only checked when without a
    status). `video` and the Mixamo `character` (`.fbx`, or `.glb` as Blender writes it) fill input/; a raise stops the
    run. `moving_camera` is GVHMR's, `engine` the renderer's."""
    workspace = Path(workspace)
    (workspace / "input").mkdir(parents=True, exist_ok=True)
    if video and Path(video).resolve() != (workspace / VIDEO).resolve():  # a rerun replaces a read-only copy
        (workspace / VIDEO).unlink(missing_ok=True)
        shutil.copyfile(video, workspace / VIDEO)
    if character:
        copy = (workspace / CHARACTER).with_suffix(Path(character).suffix.lower())
        Path(character).resolve() == copy.resolve() or shutil.copyfile(character, copy)
        copy.suffix == ".fbx" and import_character(copy, workspace / CHARACTER)  # its textures unpack beside the copy
    p = dict(moving_camera=moving_camera, engine=engine)
    for name in sorted(tools, key=list(TOOLS).index):
        fresh = name in redo or not (workspace / TOOLS[name][1]).exists()
        if fresh or not status((workspace / TOOLS[name][1]).parent):
            call(workspace, name, p, fresh)
