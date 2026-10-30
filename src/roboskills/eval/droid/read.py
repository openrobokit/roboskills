"""A DROID clip's recorded robot for real2sim: input/robot.npz (`joint_position` per source frame, Franka Panda joints
1-7 in radians; `gripper_position`, Robotiq 2F-85 normalized 0 open to 1 closed) and camera.json (`extrinsics`
`T_base_from_camera`: the source camera in the robot base frame, OpenCV axes) as the MuJoCo tool's Panda and 2F-85
(tools/mujoco/franka/) and its calibrated source camera; `prepare`: the object's noun (one Codex call, `name_object`)
and, for a clip without telemetry, the quarter-rate masks, depth and gravity its robot from video needs (the pnp
tool's droid.py). DROID: https://droid-dataset.github.io

    python -m roboskills.eval.droid.read --w W
"""

import json
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from roboskills.tools.mujoco.mujoco import FRANKA


GRIPPER = np.array([1, 0, 1, -0.96] * 2) * 0.8  # the 2F-85's driver, coupler, spring link, follower joints when closed


def read_robot(robot: Path, camera: Path) -> tuple[np.ndarray, np.ndarray]:
    """FRANKA's qpos per source frame (the gripper's linkage as MuJoCo settles it) and world (robot base) from camera."""
    r, world = np.load(robot), json.loads(Path(camera).read_text())["extrinsics"]["T_base_from_camera"]
    return np.c_[r["joint_position"], np.outer(r["gripper_position"], GRIPPER)], np.array(world)


def render_robot(robot: Path, camera: Path, k: np.ndarray, size: tuple[int, int], out: Path) -> np.ndarray:
    """FRANKA's depth (m; inf off it) per source frame from the calibrated camera with intrinsics `k`, at `size`
    (height, width); out/robot.xml is its scene."""
    import mujoco

    from roboskills.tools.mujoco.engine import build, load

    qpos, world = read_robot(robot, camera)
    camera = dict(name="source", pos=world[:3, 3], xyaxes=[*world[:3, 0], *-world[:3, 1]], fovy=np.degrees(2 * np.arctan(size[0] / 2 / k[1, 1])))  # fmt: skip
    model, data = load(build(Path(out) / "robot.xml", FRANKA, cameras=[camera]))
    model.vis.global_.offwidth, model.vis.global_.offheight, depths = size[1], size[0], []
    with mujoco.Renderer(model, *size) as renderer:
        renderer.enable_depth_rendering()
        for data.qpos[:] in qpos:
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, "source")
            depths.append(renderer.render())
    return np.where(np.array(depths) < model.stat.extent * model.vis.map.zfar * 0.99, depths, np.inf)


def name_object(task: str, out: Path, model: str = "gpt-6.1-sol", video: Path | None = None) -> dict:
    """{object, container}: the noun phrases of the object `task` moves and of the container it goes into (none for a
    surface), from one tool-free Codex call (`codex exec`, no tools, read-only sandbox, low effort; with `video`, its
    first, middle and last frames attached and the object named as seen), logged with the prompt and tokens to
    out/noun.json; out/object.txt and out/container.txt (empty for none) hold them."""
    import re
    import tempfile

    import cv2

    seen = ("The images are the first, middle and last frames of the robot's video (the text may be vague: trust the "
            "frames); name the object you see the gripper take and move as a segmentation model would find it in the "
            "first frame (what it is and its colour, never just 'object'). " if video else "")  # fmt: skip
    prompt = (seen + "Reply with JSON only, {\"object\": str, \"container\": str|null}: lower-case noun phrases of 1-4 "
              "words (with a colour the text gives) naming the physical object a robot picks up or moves in this task, "
              "and the container (bowl, mug, basket, box, drawer...) it is put into; null when it goes onto a surface "
              f"such as a table, shelf or armrest. Task: {task!r}")  # fmt: skip
    with tempfile.TemporaryDirectory() as tmp:
        images = []
        if video:  # the first, middle and last frames
            capture = cv2.VideoCapture(str(video))
            n = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            for f in (0, n // 2, n - 1):
                capture.set(cv2.CAP_PROP_POS_FRAMES, f)
                image = capture.read()[1]  # a small frame enlarged to 960 pixels wide: small objects stay legible
                cv2.imwrite(f"{tmp}/frame_{f:05d}.png", cv2.resize(image, None, fx=max(1, 960 / image.shape[1]), fy=max(1, 960 / image.shape[1]), interpolation=cv2.INTER_CUBIC))  # fmt: skip
                images += ["-i", f"{tmp}/frame_{f:05d}.png"]
        command = ["codex", "exec", "--ignore-user-config", "--ignore-rules", "--skip-git-repo-check", "--sandbox",
                   "read-only", "--ephemeral", "--model", model, "-c", 'model_reasoning_effort="low"', "--json", *images,
                   "-o", f"{tmp}/reply.txt", "-"]  # fmt: skip
        started = time.monotonic()
        done = subprocess.run(command, input=prompt, text=True, cwd=tmp, check=True, capture_output=True, timeout=120)
        reply = Path(f"{tmp}/reply.txt").read_text()
    usage = next((e["usage"] for line in done.stdout.splitlines() if line.startswith("{")
                  and (e := json.loads(line)).get("type") == "turn.completed"), None)  # fmt: skip
    nouns = json.loads(re.search(r"\{.*\}", reply, re.S).group(0))
    nouns = {k: (nouns.get(k) or "").strip().lower().removeprefix("the ") for k in ("object", "container")}
    log = dict(task=task, prompt=prompt, model=model, frames=bool(video), reply=reply, **nouns, usage=usage,
               seconds=round(time.monotonic() - started, 1))  # fmt: skip
    (out / "noun.json").write_text(json.dumps(log, indent=2) + "\n")
    (out / "object.txt").write_text(nouns["object"])
    (out / "container.txt").write_text(nouns["container"])
    return nouns


def prepare(w: Path):
    """input/noun.json etc. (the one Codex call; the given input/object.txt when it fails) and, for a clip without
    telemetry, what its robot from video needs: cache/masks/ (SAM 3 tracking the object, 1, and the robot, 2) and
    depth/moge/depth.npz (MoGe), both on every fourth frame (DROID writes each 15 Hz step twice) and held over the frames
    between (MoGe on every eighth lost 11's arm), and cache/gravity/ (GeoCalib); prepare.json, the seconds of each."""
    from roboskills.tools.geocalib.geocalib import estimate_gravity
    from roboskills.tools.moge3.moge3 import estimate_depth
    from roboskills.tools.robokit.pnp_replay import read_frames
    from roboskills.tools.sam3.sam3 import segment_video

    w, times, started = Path(w).resolve(), {}, time.monotonic()
    if not (w / "input/noun.json").exists():  # the one Codex call: the object and its container (else the given object)
        try:
            name_object(
                json.loads((w / "input/metadata.json").read_text())["instruction"],
                w / "input",
                video=w / "input/video.mp4",
            )
        except (subprocess.SubprocessError, ValueError, AttributeError) as error:
            (w / "input/noun.json").write_text(json.dumps(dict(error=repr(error))) + "\n")
    times["noun"] = time.monotonic() - started
    if not (w / "input/robot.npz").exists():
        (w / "cache").mkdir(exist_ok=True)
        n = len(read_frames(w / "input/video.mp4"))
        quarter = w / "cache/quarter.mp4"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", w / "input/video.mp4", "-vf", "select=not(mod(n\\,4))",
                        "-vsync", "vfr", "-c:v", "libx264", "-qp", "0", quarter], check=True)  # fmt: skip
        started = time.monotonic()
        segment_video(quarter, [(w / "input/object.txt").read_text().strip(), "robot"], w / "cache/quarter_masks")
        tracked = sorted((w / "cache/quarter_masks").glob("*.png"))
        (w / "cache/masks").mkdir(exist_ok=True)
        for f in range(n):
            shutil.copy(tracked[min(f // 4, len(tracked) - 1)], w / f"cache/masks/{f:05d}.png")
        times["sam3"], started = time.monotonic() - started, time.monotonic()
        estimate_depth(quarter, w / "cache/quarter_depth")
        d = dict(np.load(w / "cache/quarter_depth/depth.npz"))
        rows = np.minimum(np.arange(n) // 4, len(d["depth"]) - 1)
        (w / "depth/moge").mkdir(parents=True, exist_ok=True)
        np.savez_compressed(w / "depth/moge/depth.npz", depth=d["depth"][rows], valid=d["valid"][rows], frames=np.arange(n), K=d["K"])  # fmt: skip
        times["moge3"], started = time.monotonic() - started, time.monotonic()
        estimate_gravity(w / "input/video.mp4", w / "cache/gravity")
        times["geocalib"] = time.monotonic() - started
    (w / "prepare.json").write_text(json.dumps({k: round(v, 1) for k, v in times.items()}) + "\n")


if __name__ == "__main__":
    import tyro

    tyro.cli(prepare)
