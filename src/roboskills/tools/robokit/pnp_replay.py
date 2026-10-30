"""Pick and place replaying a clip's robot (input/: video.mp4, object.txt, robot.npz, camera.json, e.g. from
eval/droid/read.py), seen from the calibrated source camera in two videos:

- scene/recon_ext1.mp4, kinematic: the recorded joints replayed; the object still until the gripper closes, carried
  with the pinch while closed, then set down on the table.
- scene/mujoco_ext1.mp4, native: the Panda's position servos track the recorded joints and the 2F-85 its recorded
  closing (`engine.rollout`); the object, a free body, moves only by contact.

`perceive` (GPU) runs SAM 3 on frame 0 (the object noun) and on a few frames (the robot, for the focal length), then
MoGe and SAM 3D on frame 0 only; a clip without telemetry gets its robot from video (`estimate_robot`). `assemble`
(CPU) fits the focal length to the robot's silhouette, scales MoGe by the robot's rendered depth, places the object
(SAM 3D's shape at the pinch where the gripper first closes, resting on the table) and puts the clip's background
(the per-pixel median of its frames without the robot) behind the scene as MoGe's depth mesh.
"""

import json
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Literal

import numpy as np

from roboskills.eval.droid.read import read_robot
from roboskills.tools.mujoco.mujoco import FRANKA
from roboskills.tools.mujoco.scene import backdrop_render, background, container_parts, decimate, depth_mesh, write_obj


VIDEO, LEAD = "input/video.mp4", 0.1  # servo lead (s)


FOCALS, PAD = np.arange(100.0, 182.0, 4.0), 24  # focal lengths tried (px); principal points up to PAD px off centre


def read_frames(video: Path) -> np.ndarray:
    import cv2

    capture, frames = cv2.VideoCapture(str(video)), []
    while (got := capture.read())[0]:
        frames.append(np.ascontiguousarray(got[1][..., ::-1]))
    return np.stack(frames)


def grasp_frames(w: Path) -> tuple[int, int] | None:
    """The first closing of the gripper and the frame it opens again, or None when it never closes."""
    g = np.load(w / "input/robot.npz")["gripper_position"]
    closed = g > max(0.15, g.max() / 2)
    if not closed.any():
        return None
    start = int(closed.argmax())
    return start, start + int(np.argmin(closed[start:])) - 1 if not closed[start:].all() else len(g) - 1


def source_camera(world: np.ndarray, k: np.ndarray | None = None, size: tuple[int, int] = (180, 320)) -> dict:
    """The source camera at `world` (base from camera, OpenCV axes): pinhole intrinsics `k` at `size` (height, width), or
    a 60 degree field of view to change at run time."""
    camera = dict(name="source", pos=world[:3, 3], xyaxes=[*world[:3, 0], *-world[:3, 1]])
    if k is None:
        return dict(camera, fovy=60.0)
    # MuJoCo's principal offset moves the image the other way from OpenCV's (cx, cy)
    return dict(camera, resolution=size[::-1], sensorsize=[size[1] * 1e-5, size[0] * 1e-5], focalpixel=[k[0, 0], k[1, 1]],
                principalpixel=[size[1] / 2 - k[0, 2], size[0] / 2 - k[1, 2]])  # fmt: skip


def load_robot(w: Path, k: np.ndarray | None = None, size: tuple[int, int] = (180, 320), out: Path | None = None):
    """(model, data, qpos per frame, base from camera) of the recorded robot seen by the source camera."""
    from roboskills.tools.mujoco.engine import build, load

    qpos, world = read_robot(w / "input/robot.npz", w / "input/camera.json")
    model, data = load(build((out or w / "camera") / "robot.xml", FRANKA, cameras=[source_camera(world, k, size)]))
    return model, data, qpos, world


def silhouettes(model, data, qpos, rows, size: tuple[int, int], focals=(None,)) -> np.ndarray:
    """The robot's silhouette [focal, row, H, W] at each focal length (px; None keeps the model's)."""
    import mujoco

    model.vis.global_.offwidth, model.vis.global_.offheight, out = size[1], size[0], []
    with mujoco.Renderer(model, *size) as renderer:
        renderer.enable_depth_rendering()
        for f in focals:
            if f is not None:
                model.cam_fovy[0] = np.degrees(2 * np.arctan(size[0] / 2 / f))
            for row in rows:
                data.qpos[: qpos.shape[1]] = qpos[row]
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, "source")
                out.append(renderer.render() < model.stat.extent * model.vis.map.zfar * 0.99)
    return np.array(out).reshape(len(focals), len(rows), *size)


def perceive(w: Path, pick: int | None = None, focals: np.ndarray = FOCALS, pad: int = PAD, focal: float | None = None,
             center: tuple[float, float] | None = None):  # fmt: skip
    """masks/sam3.npz (SAM 3's robot masks at `rows`, object masks and scores at frame 0), camera/camera.json (`K`,
    the focal length's robot `iou`), masks/00000.png (1 the object, 2 the robot), depth/ (MoGe, frame 0, metric by the
    robot) and shape/ (SAM 3D). `pick` forces a frame-0 detection of the object (masks/objects.json lists them); the
    camera fit tries `focals` and principal points up to `pad` px off centre, unless `focal` (px; and `center`, the
    principal point, default the image's) forces the intrinsics."""
    from PIL import Image

    from roboskills.tools.moge3.moge3 import estimate_depth
    from roboskills.tools.sam3.sam3 import PALETTE, Sam
    from roboskills.tools.sam3d.sam3d import reconstruct_object

    w = Path(w).resolve()
    frames, noun, times = read_frames(w / VIDEO), (w / "input/object.txt").read_text().strip(), {}
    rows, size = np.linspace(0, len(frames) - 1, 8).astype(int), frames.shape[1:3]
    started = time.monotonic()
    if not (w / "masks/sam3.npz").exists():  # cached: SAM 3's detections
        sam, gripper = Sam(), np.zeros((len(frames), *size), bool)
        try:
            robot = [[d["mask"] for d in sam.detect(frames[r], ["robot arm"])["robot arm"]] for r in rows]
            objects = sam.detect(frames[0], [noun])[noun] or sam.detect(frames[0], [noun.split()[-1]])[noun.split()[-1]]
            container = (w / "input/container.txt").read_text().strip() if (w / "input/container.txt").exists() else ""
            holder = (sam.detect(frames[0], [container])[container] or [dict(mask=np.zeros(size, bool))])[0]["mask"] if container else np.zeros(size, bool)  # fmt: skip
            tracked = (w / "cache/masks/00000.png").exists() and np.mean([(np.array(Image.open(w / f"cache/masks/{f:05d}.png")) == 2).any() for f in rows]) > 0.3  # prepare's video masks: 2 the robot, when it was found  # fmt: skip
            for f in range(0, len(frames), 8) if not (w / "input/robot.npz").exists() and not tracked else ():
                if found := sam.detect(frames[f], ["robot"])["robot"]:  # SAM 3 finds no "gripper": the robot's tip
                    gripper[f] = found[0]["mask"]  # the whole robot, at 3.75 Hz
            for f in range(0, len(frames), 4) if not (w / "input/robot.npz").exists() and tracked else ():
                gripper[f] = np.array(Image.open(w / f"cache/masks/{f:05d}.png")) == 2
        finally:
            sam.close()
        (w / "masks").mkdir(parents=True, exist_ok=True)
        np.savez_compressed(w / "masks/sam3.npz", rows=rows, objects=np.array([d["mask"] for d in objects]).reshape(-1, *size),
                            scores=[d["score"] for d in objects], gripper=gripper, container=holder, **{f"robot_{r}": np.array(m).reshape(-1, *size) for r, m in zip(rows, robot)})  # fmt: skip
    found = np.load(w / "masks/sam3.npz")
    robot = [list(found[f"robot_{r}"]) for r in rows]
    objects = [dict(mask=m, score=float(x)) for m, x in zip(found["objects"], found["scores"])]
    if not objects:
        raise ValueError(f"SAM 3 finds no {noun!r} at frame 0")
    times["sam3"] = time.monotonic() - started

    started, iou = time.monotonic(), lambda a, b: (a & b).sum() / max((a | b).sum(), 1)
    if not (w / "input/robot.npz").exists():  # no telemetry: the robot from the video, seen by MoGe's camera
        estimate_robot(w, found["gripper"], objects[0]["mask"])
        times["robot"] = time.monotonic() - started
    if focal is not None:
        cx, cy = center or (size[1] / 2, size[0] / 2)
        (w / "camera").mkdir(exist_ok=True)
        (w / "camera/camera.json").write_text(json.dumps(dict(K=[[focal, 0, cx], [0, focal, cy], [0, 0, 1]], forced=True)) + "\n")  # fmt: skip
    if not (w / "camera/camera.json").exists():
        fit_camera(w, robot, rows, size, focals, pad)
    k = np.array(json.loads((w / "camera/camera.json").read_text())["K"])
    model, data, qpos, world = load_robot(w, k, size)

    started = time.monotonic()
    (w / "depth").mkdir(exist_ok=True)
    first = w / "depth/frame0.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", w / VIDEO, "-frames:v", "1", "-c:v", "libx264", "-qp", "0", first], check=True)  # fmt: skip
    if not (w / "depth/moge/depth.npz").exists():  # cached: MoGe's
        estimate_depth(first, w / "depth/moge")
    d = dict(np.load(w / "depth/moge/depth.npz"))
    times["moge3"] = time.monotonic() - started

    # the object: among the detections within 25 px of the pinch where the gripper first closes (frame 0's view), the
    # one whose width across the jaws best fits the recorded gap there (a recorded robot), else the nearest; else the best
    grasp, recorded = grasp_frames(w), "estimated" not in json.loads((w / "input/camera.json").read_text())
    if grasp:
        pinch = world_point(model, data, qpos[grasp[0]], "pinch")
        u, v, z = point_pixel(np.linalg.inv(world) @ np.r_[pinch, 1], k)
    if grasp and pick is None:
        apart = np.array([np.hypot(*(np.argwhere(o["mask"]) - [v, u]).T).min() for o in objects])
        pick = int(np.argmin(apart))
        if recorded and (apart <= 25).sum() > 1:
            across = data.body("left_pad").xpos - data.body("right_pad").xpos
            across = np.linalg.inv(world)[:3, :3] @ across / np.linalg.norm(across)  # in the camera frame
            gap = 0.085 * (1 - np.load(w / "input/robot.npz")["gripper_position"][grasp[0] : grasp[1] + 1].max())
            z0, (y, x) = d["depth"][0].astype(float), np.indices(size)
            rays = np.stack([(x - k[0, 2]) / k[0, 0], (y - k[1, 2]) / k[1, 1], np.ones(size)], -1)
            width = []
            for o, near in zip(objects, apart):
                keep = o["mask"] & d["valid"][0] & (z0 > 0)
                p = (
                    (rays[keep] * z0[keep, None]) @ across * z / max(np.median(z0[keep]), 1e-6)
                    if keep.sum() > 5
                    else [0]
                )
                width.append(np.ptp(np.percentile(p, [5, 95])) if near <= 25 else np.inf)
            pick = int(np.argmin(np.abs(np.array(width) - gap)))
    pick = pick or 0
    labels, own = np.zeros(size, np.uint8), silhouettes(model, data, qpos, [0], size)[0, 0]  # on SAM 3's arm
    arm = max(robot[0], key=lambda m: iou(own, m), default=own)
    labels[own & (arm if iou(own, arm) > 0.2 else own)] = 2
    labels[objects[pick]["mask"]] = 1
    png = Image.fromarray(labels)
    png.putpalette(PALETTE)
    png.save(w / "masks/00000.png")
    (w / "masks/objects.json").write_text(json.dumps(dict(noun=noun, pick=pick, scores=[d["score"] for d in objects])) + "\n")  # fmt: skip
    times["focal"] = time.monotonic() - started

    ratio = robot = robot_ratio(model, data, qpos[0], size, d, labels)
    if grasp:  # the pinch where the gripper closes is in the object: its depth over MoGe's on the object around it
        z0, seen = d["depth"][0].astype(float), np.argwhere((labels == 1) & d["valid"][0])
        near = seen[np.hypot(*(seen - [v, u]).T) < max(8, np.hypot(*(seen - [v, u]).T).min() + 2)]
        ratio = float(z / np.median(z0[near[:, 0], near[:, 1]]))
    d["depth"], d["K"] = (d["depth"].astype(float) * ratio).astype(np.float16), k
    np.savez_compressed(w / "depth/depth.npz", **d)
    (w / "depth/scale.json").write_text(json.dumps(dict(scale=ratio, robot=robot)) + "\n")

    started = time.monotonic()
    cached = w / "cache/shape/pose.json"  # the shared cache's SAM 3D, when its object is this one
    if not (w / "shape/pose.json").exists() and cached.exists():
        other = np.array(Image.open(w / "cache/masks/00000.png")) == 1
        if (other & (labels == 1)).sum() > 0.5 * (other | (labels == 1)).sum():
            shutil.copytree(cached.parent, w / "shape")
    if not (w / "shape/pose.json").exists():  # cached: SAM 3D's (its draw decides grasps: see the DROID notes)
        reconstruct_object(first, w / "masks", w / "depth/depth.npz", w / "shape", samples=4, keep=True)
    times["sam3d"] = time.monotonic() - started
    (w / "perceive.json").write_text(json.dumps({k: round(v, 1) for k, v in times.items()}) + "\n")


def robot_tip(mask: np.ndarray, share: float = 0.08) -> np.ndarray:
    """The `share` of a robot's mask farthest from where it enters the view (its pixels on the image border; else
    from its centre): the gripper."""
    at = np.argwhere(mask)
    edge = at[(at == 0).any(1) | (at[:, 0] >= mask.shape[0] - 1) | (at[:, 1] >= mask.shape[1] - 1)]
    far = np.hypot(*(at - (edge if len(edge) else at).mean(0)).T)
    tip = np.zeros_like(mask)
    tip[tuple(at[far >= np.quantile(far, 1 - share)].T)] = True
    return tip


READY = np.array([0, -0.3, 0, -2.2, 0, 2.0, 0.785])  # the Panda reaching forward, elbow up, gripper down


def solve_arm(pinch: np.ndarray, steps: int = 30) -> np.ndarray:
    """The Panda's joints per frame putting the 2F-85's pinch at `pinch` (N, 3; base frame), pointing down, yaw free:
    damped least squares on MuJoCo's site Jacobian, each frame from the previous (the first from READY turned toward
    it), pulled lightly toward READY."""
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(FRANKA))
    data, site, (low, high) = mujoco.MjData(model), model.site("pinch").id, model.jnt_range[:7].T
    jp, jr, out = np.zeros((3, model.nv)), np.zeros((3, model.nv)), []
    q = READY.copy()
    q[0] = np.arctan2(pinch[0, 1], pinch[0, 0])
    for target in pinch:
        for _ in range(steps):
            data.qpos[:7] = q
            mujoco.mj_kinematics(model, data)
            mujoco.mj_comPos(model, data)
            mujoco.mj_jacSite(model, data, jp, jr, site)
            axis = data.site_xmat[site].reshape(3, 3)[:, 2]  # the approach axis, wanted straight down
            error = np.r_[target - data.site_xpos[site], np.cross(axis, [0, 0, -1])]
            J = np.r_[jp[:, :7], (np.eye(3) - np.outer([0, 0, 1], [0, 0, 1])) @ jr[:, :7]]
            step = J.T @ np.linalg.solve(J @ J.T + 1e-3 * np.eye(6), error)
            q = np.clip(q + step + 0.02 * (np.eye(7) - np.linalg.pinv(J) @ J) @ (READY - q), low, high)
        out.append(q.copy())
    return np.array(out)


def estimate_robot(w: Path, whole: np.ndarray, still: np.ndarray):
    """For a clip without telemetry, input/robot.npz and camera.json (`estimated`) and camera/camera.json (MoGe's K), in
    a gravity-aligned world (the shared cache's GeoCalib) with the table, around the object `still` at frame 0, at z = 0:
    the gripper's tip is the lowest fifth of the far end of SAM 3's robot mask (`whole`, some frames) in MoGe's depth
    (the cache's, all frames); the object's centre is the cache's object masks'. The pinch then follows a pick and place
    from the tip at the first frame: above the object, down onto it where the tip comes nearest it before it first moves
    (a quarter of the way to where it settles), closed, lifted, over to where it settles, down (onto what it settles on:
    robot.npz `support`, its height over the table when over 3 cm off), open, up, and to the tip at the last frame
    (smooth steps between; a tip under the table, missed, is the pose over the object instead).
    The base is the one, on the table within reach, whose arm (by IK) best overlaps SAM 3's robot masks."""
    from PIL import Image
    from scipy.ndimage import binary_dilation, median_filter

    d, n = dict(np.load(w / "depth/moge/depth.npz")), len(whole)
    k, down = d["K"], np.array(json.loads((w / "cache/gravity/gravity.json").read_text())["down"])
    v, u = np.indices(whole.shape[1:])
    lift = lambda f, keep: np.stack([(u - k[0, 2]) / k[0, 0], (v - k[1, 2]) / k[1, 1], np.ones_like(u * 1.0)], -1)[keep] * d["depth"][f].astype(float)[keep][:, None]  # fmt: skip
    x = np.cross(down, [0, 0, 1.0]) if abs(down[2]) < 0.9 else np.cross(down, [1.0, 0, 0])
    world = np.eye(4)  # world from camera: z up, the table at 0 under the object
    world[:3, :3] = np.stack([x / np.linalg.norm(x), np.cross(-down, x / np.linalg.norm(x)), -down])
    ring = binary_dilation(still, iterations=12) & ~binary_dilation(still, iterations=3) & d["valid"][0]
    table = np.median(lift(0, ring) @ world[:3, :3].T, 0)
    centre = np.median(lift(0, still & d["valid"][0]) @ world[:3, :3].T, 0)
    world[:3, 3] = -np.r_[centre[:2], table[2]]
    points = lambda f, keep: lift(f, keep & d["valid"][f]) @ world[:3, :3].T + world[:3, 3]
    smooth = lambda x, ok: median_filter(np.stack([np.interp(np.arange(n), np.flatnonzero(ok), c[ok]) for c in x.T], 1), (9, 1))  # fmt: skip

    seen, tip = np.flatnonzero(whole.reshape(n, -1).sum(1) > 50), np.full((n, 3), np.nan)
    for f in seen:
        p = points(f, robot_tip(whole[f]))
        if len(p) > 20:
            tip[f] = np.median(p[p[:, 2] <= np.percentile(p[:, 2], 20)], 0) + [0, 0, 0.01]
    tip = smooth(tip, np.isfinite(tip[:, 0])) if np.isfinite(tip[:, 0]).any() else np.tile([0, 0, 0.3], (n, 1))
    obj = np.array([np.median(p, 0) if len(p := points(f, np.array(Image.open(w / f"cache/masks/{f:05d}.png")) == 1)) > 20
                    else [np.nan] * 3 for f in range(n)])  # fmt: skip
    obj = smooth(obj, np.isfinite(obj[:, 0]))
    start, end = np.median(obj[: max(n // 10, 1)], 0), np.median(obj[-max(n // 10, 1) :], 0)
    moved = np.linalg.norm(obj - start, axis=1) > max(
        0.05, 0.25 * np.linalg.norm(end - start)
    )  # over the gripper's occlusion jitter
    first = int(moved.argmax()) if moved.any() else n // 2
    grasp = max(0, first - 60) + int(np.argmin(np.linalg.norm(tip - start, axis=1)[max(0, first - 60) : first + 1]))
    away = np.flatnonzero(np.linalg.norm(obj - end, axis=1) > 0.03)
    release = int(np.clip(away.max() + 1 if moved.any() and len(away) else (grasp + n) // 2, grasp + 30, n - 12))
    grasp = min(grasp, release - 30)
    # what it settles on, outside a container: frame 0's depth where it lies at the end (its masks over the last tenth,
    # what covers them at frame 0 left out); over 3 cm above or below the table it starts on, it is put down there (and
    # assemble builds that support), held as it was picked up
    first_labels, heights = np.array(Image.open(w / "cache/masks/00000.png")), []
    for f in (
        range(n - max(n // 10, 1), n, 4)
        if moved.any() and np.load(w / "masks/sam3.npz")["container"].sum() < 20
        else ()
    ):
        under = (np.array(Image.open(w / f"cache/masks/{f:05d}.png")) == 1) & ~binary_dilation(
            first_labels > 0, iterations=3
        )
        if len(p := points(0, under)) > 20:
            heights.append(np.median(p[:, 2]))
    rise = float(np.median(heights)) if heights and abs(np.median(heights)) > 0.03 else 0.0
    place = np.r_[end[:2], rise + start[2]] if rise else end
    up, lift_s = np.array([0, 0, 0.12]), 15  # the pick and place's keyframes (frame, pinch) and their smooth steps
    # from and back to the tip at the first and last frames, or over the object where that is under the table (missed)
    begin, finish = tip[0] if tip[0, 2] > -0.03 else start + up, tip[-1] if tip[-1, 2] > -0.03 else place + up
    keys = [(0, begin), (max(grasp - 20, 1), start + up), (grasp, start), (grasp + 8, start), (grasp + 8 + lift_s, start + up),
            (release - lift_s, place + up), (release, place + [0, 0, 0.01]), (release + 6, place + [0, 0, 0.01]),
            (min(release + 6 + lift_s, n - 1), place + up), (n - 1, finish)]  # fmt: skip
    keys = [(f, p) for i, (f, p) in enumerate(keys) if i == 0 or f > keys[i - 1][0]]
    pinch = np.zeros((n, 3))
    for (f0, p0), (f1, p1) in zip(keys, keys[1:]):
        a = (np.arange(f0, f1 + 1) - f0) / max(f1 - f0, 1)
        pinch[f0 : f1 + 1] = p0 + np.outer(a * a * (3 - 2 * a), p1 - p0)
    pinch[:, 2] = np.maximum(pinch[:, 2], min(rise, 0) + 0.01)
    closing = np.zeros(n)
    closing[grasp:release] = 1.0

    # the base: of bases on the table 0.4-0.75 m from the task, every pinch within reach, 0.5 m from the camera, the
    # one whose arm (IK at a few frames) best overlaps SAM 3's robot masks there
    from roboskills.tools.mujoco.engine import build, load

    camera = source_camera(np.eye(4), k, whole.shape[1:])
    model, data = load(build(w / "camera/estimate.xml", FRANKA, cameras=[camera]))
    model.vis.global_.offwidth, model.vis.global_.offheight = whole.shape[2], whole.shape[1]
    frames = seen[np.linspace(0, len(seen) - 1, min(8, len(seen))).astype(int)] if len(seen) else np.array([0])
    middle, best = pinch.mean(0), (-1.0, None)
    for heading in np.radians(np.arange(0, 360, 22.5)):
        for reach in (0.4, 0.5, 0.6, 0.75):
            base = np.r_[middle[:2] + reach * np.array([np.cos(heading), np.sin(heading)]), 0]
            if (
                np.linalg.norm(pinch - base - [0, 0, 0.333], axis=1).max() > 0.8
                or np.linalg.norm(base[:2] - world[:2, 3]) < 0.5
            ):
                continue
            frame = base_frame(base, middle)
            q = solve_arm((pinch[frames] - base) @ frame[:3, :3], steps=40)
            score = np.mean(silhouette_iou(model, data, q, np.linalg.inv(frame) @ world, whole[frames]))  # fmt: skip
            best = max(best, (score, heading, reach), key=lambda b: b[0])
    if best[1] is None:  # nothing within reach: the nearest base behind the task, away from the camera
        heading, reach = np.arctan2(*(middle[:2] - world[:2, 3])[::-1]), 0.5
    else:
        heading, reach = best[1], best[2]
    base = np.r_[middle[:2] + reach * np.array([np.cos(heading), np.sin(heading)]), 0]
    frame = base_frame(base, middle)
    np.savez(w / "input/robot.npz", joint_position=solve_arm((pinch - base) @ frame[:3, :3]), gripper_position=closing,
             pinch=pinch, tip=tip, object=obj, support=np.r_[(np.linalg.inv(frame) @ np.r_[end[:2], 0, 1])[:2], rise])  # fmt: skip
    camera = dict(extrinsics=dict(T_base_from_camera=(np.linalg.inv(frame) @ world).tolist()), estimated=True,
                  robot_iou=best[0], grasp=grasp, release=release)  # fmt: skip
    (w / "input/camera.json").write_text(json.dumps(camera, indent=1) + "\n")
    (w / "camera/camera.json").write_text(json.dumps(dict(K=k.tolist(), source="MoGe")) + "\n")


def base_frame(base: np.ndarray, toward: np.ndarray) -> np.ndarray:
    """The world from the robot base at `base`, facing `toward`."""
    yaw, frame = np.arctan2(*(toward[:2] - base[:2])[::-1]), np.eye(4)
    frame[:3, :3] = [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
    frame[:3, 3] = base
    return frame


def silhouette_iou(model, data, q: np.ndarray, base_from_camera: np.ndarray, masks: np.ndarray) -> list[float]:
    """The IoU of the arm at joints `q` (per mask), seen from `base_from_camera`, with each of `masks`."""
    import mujoco

    model.cam_pos[0] = base_from_camera[:3, 3]
    mujoco.mju_mat2Quat(model.cam_quat[0], (base_from_camera[:3, :3] @ np.diag([1, -1, -1])).flatten())
    out = []
    with mujoco.Renderer(model, *masks.shape[1:]) as renderer:
        renderer.enable_depth_rendering()
        for joints, mask in zip(q, masks):
            data.qpos[:7] = joints
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, "source")
            drawn = renderer.render() < model.stat.extent * model.vis.map.zfar * 0.99
            out.append((drawn & mask).sum() / max((drawn | mask).sum(), 1))
    return out


def fit_camera(w: Path, robot: list, rows: np.ndarray, size: tuple[int, int], focals=FOCALS, pad=PAD) -> np.ndarray:
    """camera/camera.json (`K`, `iou`): the focal length and principal point whose robot silhouettes best match SAM 3's
    robot masks at `rows`: each focal length's silhouettes drawn on a canvas `pad` pixels wider all round, then every crop
    (a principal point) scored at half size."""
    iou = lambda a, b: (a & b).sum() / max((a | b).sum(), 1)
    model, data, qpos, world = load_robot(w)
    wide = silhouettes(model, data, qpos, rows, (size[0] + 2 * pad, size[1] + 2 * pad), focals)[..., ::2, ::2]
    masks = [[m[::2, ::2] for m in r] for r in robot]
    h, wd, best = size[0] // 2, size[1] // 2, (-1.0, 0, 0, 0)
    for i, drawn in enumerate(wide):
        for dy in range(pad + 1):
            for dx in range(pad + 1):
                crop = drawn[:, dy : dy + h, dx : dx + wd]
                score = np.mean([max([iou(c, m) for m in ms] or [0]) for c, ms in zip(crop, masks)])
                best = max(best, (score, i, dx, dy))
    score, i, dx, dy = best
    f, cx, cy = float(focals[i]), size[1] / 2 + pad - 2 * dx, size[0] / 2 + pad - 2 * dy
    k = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
    (w / "camera/camera.json").write_text(json.dumps(dict(K=k.tolist(), iou=float(score))) + "\n")
    return k


def world_point(model, data, q: np.ndarray, site: str) -> np.ndarray:
    import mujoco

    data.qpos[: len(q)] = q
    mujoco.mj_kinematics(model, data)
    return data.site(site).xpos.copy()


def pinch_pose(model, data, q: np.ndarray) -> np.ndarray:
    """The gripper's pinch site in the base frame (4x4) at joints `q`."""
    pose = np.eye(4)
    pose[:3, 3] = world_point(model, data, q, "pinch")
    pose[:3, :3] = data.site("pinch").xmat.reshape(3, 3)
    return pose


def point_pixel(p: np.ndarray, k: np.ndarray) -> tuple[float, float, float]:
    """A camera-frame point's pixel and depth."""
    return p[0] / p[2] * k[0, 0] + k[0, 2], p[1] / p[2] * k[1, 1] + k[1, 2], p[2]


def robot_ratio(model, data, q, size, d, labels) -> float:
    """Metres per MoGe unit at frame 0: the robot's rendered depth over MoGe's at its visible pixels (label 2), 1 when
    under 20 pixels."""
    import mujoco

    data.qpos[: len(q)] = q
    mujoco.mj_forward(model, data)
    model.vis.global_.offwidth, model.vis.global_.offheight = size[1], size[0]
    with mujoco.Renderer(model, *size) as renderer:
        renderer.enable_depth_rendering()
        renderer.update_scene(data, "source")
        near = renderer.render()
    z = d["depth"][0].astype(float)
    keep = (labels == 2) & d["valid"][0] & (z > 0) & (near < model.stat.extent * model.vis.map.zfar * 0.99)
    return float(np.median(near[keep] / z[keep])) if keep.sum() >= 20 else 1.0


def wrist_turn(model, data, q: np.ndarray, vertices: np.ndarray, band: float = 0.015) -> float:
    """The turn (rad) of the Panda's last joint from `q` (the gripper open) that closes the 2F-85's jaws across the
    narrowest width of `vertices` (world) at the pads' height (`band` along the approach), within the joint's range;
    0 unless it narrows that width by over 1 cm."""
    import mujoco

    widths = {}
    for turn in np.radians(np.arange(-180, 181, 5)):
        if not model.jnt_range[6, 0] <= q[6] + turn <= model.jnt_range[6, 1]:
            continue
        data.qpos[: len(q)] = np.r_[q[:6], q[6] + turn, np.zeros(len(q) - 7)]
        mujoco.mj_kinematics(model, data)
        across = data.body("left_pad").xpos - data.body("right_pad").xpos
        near = vertices[
            np.abs((vertices - data.site("pinch").xpos) @ data.site("pinch").xmat.reshape(3, 3)[:, 2]) < band
        ]
        widths[turn] = float(np.ptp(near @ across) / np.linalg.norm(across)) if len(near) else 0.0
    best = min(widths, key=lambda t: (round(widths[t], 3), abs(t)))
    return float(best) if widths.get(0.0, 0.0) - widths[best] > 0.01 else 0.0


def assemble(w: Path, friction: float = 1.5, condim: int = 6, mass: float | None = None, grip: float = 5.0,
             lead: float = LEAD, collision: Literal["hull", "hollow", "box", "cylinder", "coacd"] = "hull", grasp_at: Literal["centre", "surface"] = "centre", squeeze: bool = True,
             scale: float = 1.0, shift: tuple[float, float, float] = (0, 0, 0), yaw: float = 0.0,
             sample: int | None = None):  # fmt: skip
    """scene/: scene.xml and scene.mjb (Panda and 2F-85, table, object, backdrop, source camera), recon_ext1.mp4 with
    recon.npz (kinematic qpos), mujoco_ext1.mp4 with native/ (`engine.rollout`'s trajectory and diagnostics), and
    layer.json (focal, scale, table height, grasp, object placement, seconds). The object's contacts: `friction`
    (sliding), `condim`, `mass` (kg; default 500 kg/m^3 of its hull, 0.02-0.5) and `collision` (its convex hull, or
    CoACD's parts); the 2F-85's force limit `grip` (N); the servos' `lead` (s); `grasp_at` the object's centre, or the
    surface point nearest the pinch (a rim or a handle), put at the pinch where the gripper first closes. The clip's
    background (the median of its frames without the robot) shows as a backdrop mesh (MoGe's frame-0 depth, the robot
    and object filled in from around them, the table flat). The task's container (input/container.txt, SAM 3's mask) is a
    static open cylinder: its centre, radius and height from the depth. A clip from video: what the object settles on,
    over 3 cm above or below where it starts (robot.npz `support`), is a static block, and the Panda's last joint turns
    the jaws across the object's narrowest side at the grasp (`wrist_turn`). Edits after the placement: the object's `scale`
    (about its foot, which stays on the table), `shift` (m, base frame) and `yaw` (rad, about its vertical centre line);
    `sample` builds the scene on another of SAM 3D's samples (shape/samples/<seed>/; shape/pose.json's `seed` is the
    chosen one). `collision` also takes the box or the cylinder (its local z the axis) around the mesh, or a floor and thin walls
    around its footprint ("hollow"). When a recorded gripper closes to under 0.4 of the object's width across its
    jaws (a cup held by its rim), the object is hollow and grasped at its surface. With `squeeze` the gripper is commanded fully closed while
    held (it stops on the object, at `grip`), not to its recorded opening."""
    import mujoco
    import trimesh
    from PIL import Image

    from roboskills.tools.mujoco.engine import build, load, render, rollout

    w = Path(w).resolve()
    started, out = time.monotonic(), w / "scene"
    out.mkdir(parents=True, exist_ok=True)
    frames, k = read_frames(w / VIDEO), np.array(json.loads((w / "camera/camera.json").read_text())["K"])
    size, f = frames.shape[1:3], k[0, 0]
    model, data, qpos, world = load_robot(w, k, size, out)
    labels = np.array(Image.open(w / "masks/00000.png"))
    grasp = grasp_frames(w)
    pinch = np.array([world_point(model, data, q, "pinch") for q in qpos])
    r = np.load(w / "input/robot.npz")  # a clip from video: where it settles, when over 3 cm above or below its start
    rise = float(r["support"][2]) if "support" in r.files and abs(r["support"][2]) > 0.03 else 0.0

    # the object in the robot base frame, SAM 3D's metric shape and pose
    shape = w / "shape" if sample is None else w / f"shape/samples/{sample}"  # SAM 3D's chosen sample, or `sample`
    mesh = trimesh.load(shape / "mesh.ply")
    seen = np.array(json.loads((shape / "pose.json").read_text())["camera_from_object"])
    placed = world @ seen
    vertices = trimesh.transform_points(mesh.vertices, placed)
    # the table: the median height of the background in a ring of pixels around the object, below the lowest pinch
    from scipy.ndimage import binary_dilation

    d = dict(np.load(w / "depth/depth.npz"))
    z, (v, u) = d["depth"][0].astype(float), np.indices(size)
    xyz = np.stack([(u - k[0, 2]) / k[0, 0] * z, (v - k[1, 2]) / k[1, 1] * z, z], -1)
    ring = binary_dilation(labels == 1, iterations=12) & ~binary_dilation(labels == 1, iterations=3)
    near = trimesh.transform_points(xyz[d["valid"][0] & (labels == 0) & (z > 0) & ring], world)
    table = ground = float(np.median(near[:, 2])) if len(near) > 20 else vertices[:, 2].min()
    table = min(table, float(pinch[: grasp[0] + 1 if grasp and rise < 0 else None, 2].min()) - 0.01)
    # slid along its camera ray, scaled with its distance (the same image), until its foot is on the table; the table
    # raised (a tray, a shelf) when the gripper closes above the object's middle, so it closes on the object
    a = trimesh.transform_points(mesh.vertices, seen) @ world[2, :3]
    for _ in range(2):
        slide = float(np.clip((table - world[2, 3]) / a.min(), 0.25, 4.0)) if a.min() < 0 else 1.0
        tall = slide * np.ptp(a)
        low = pinch[max(grasp[0] - 10, 0) : grasp[0] + 10, 2].min() if grasp else table
        if grasp and low > table + 0.75 * tall:
            table = float(low - 0.6 * tall)
    mesh.apply_scale(slide)
    seen[:3, 3] *= slide
    placed = world @ seen
    vertices = trimesh.transform_points(mesh.vertices, placed)
    centre, camera_from_object = vertices.mean(0), seen.copy()  # the latter as seen at frame 0, for its texture
    values, axes = np.linalg.eigh(np.cov((vertices - centre).T))
    if values[2] > 9 * values[1] and abs(axes[2, 2]) > 0.5:  # a long thin one (a pen) stood up: laid down on the table
        long = axes[:, 2] * np.sign(axes[2, 2])
        flat = np.r_[long[:2], 0] if np.linalg.norm(long[:2]) > 1e-3 else world[:3, 0] * [1, 1, 0]
        axis = np.cross(long, flat / np.linalg.norm(flat))
        lay = trimesh.transformations.rotation_matrix(np.arcsin(min(np.linalg.norm(axis), 1.0)), axis, centre)
        placed, vertices = lay @ placed, trimesh.transform_points(vertices, lay)
        vertices[:, 2] += table - vertices[:, 2].min()
        placed[2, 3] += table - trimesh.transform_points(mesh.vertices, placed)[:, 2].min()
    # at the pinch where the gripper first closes (when within 15 cm), the long axis across the jaws, on the table
    offset, recorded = np.zeros(3), "estimated" not in json.loads((w / "input/camera.json").read_text())
    if grasp and recorded and collision == "hull" and grasp_at == "centre":  # held by a rim or a handle: hollow
        data.qpos[: qpos.shape[1]] = qpos[grasp[0]]
        mujoco.mj_kinematics(model, data)
        across = data.body("left_pad").xpos - data.body("right_pad").xpos
        gap = 0.085 * (1 - np.load(w / "input/robot.npz")["gripper_position"][grasp[0] : grasp[1] + 1].max())
        values, axes = np.linalg.eigh(np.cov((vertices[:, :2] - centre[:2]).T))  # a long one turns across the jaws
        across = np.r_[axes[:, 0], 0] if values[1] > 4 * values[0] else across / np.linalg.norm(across)
        if gap < 0.4 * np.ptp(vertices @ across):
            collision, grasp_at = "hollow", "surface"
    if grasp and np.hypot(*(pinch[grasp[0], :2] - centre[:2])) < 0.15:
        held = (
            vertices[np.argmin(np.linalg.norm(vertices - pinch[grasp[0]], axis=1))] if grasp_at == "surface" else centre
        )
        offset[:2] = pinch[grasp[0], :2] - held[:2]
        data.qpos[: qpos.shape[1]] = qpos[grasp[0]]
        mujoco.mj_kinematics(model, data)
        across = data.body("left_pad").xpos - data.body("right_pad").xpos
        flat = vertices[:, :2] - centre[:2]
        values, axes = np.linalg.eigh(np.cov(flat.T))
        if values[1] > 4 * values[0] and np.linalg.norm(across[:2]) > 1e-3:  # long: turn its long axis across the jaws
            a = np.arctan2(across[1], across[0]) + np.pi / 2 - np.arctan2(axes[1, 1], axes[0, 1])
            turn = np.eye(4)
            turn[:2, :2] = [[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]
            turn[:3, 3] = centre - turn[:3, :3] @ centre
            placed, vertices = turn @ placed, trimesh.transform_points(vertices, turn)
    offset[2] = table - vertices[:, 2].min()
    placed[:3, 3] += offset
    vertices += offset
    edit = np.eye(4)  # the edits: yaw about the vertical centre line, scale about the foot (kept on the table), shift
    edit[:2, :2] = [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]
    foot = np.r_[vertices[:, :2].mean(0), table]
    edit[:3, :3] *= scale
    edit[:3, 3] = foot - edit[:3, :3] @ foot + np.asarray(shift, float)
    vertices = trimesh.transform_points(vertices, edit)
    placed = edit @ placed  # its rotation part now carries `scale`: taken out below, put into the mesh
    # a clip from video: the table at the lower of where it starts and settles (`rise` over the ground around it at
    # frame 0), a block under the higher (a shelf or an armrest it settles on, wider than it; a counter or a mug it
    # starts on, its own footprint)
    settle, (lo, hi) = ground + rise if rise else table, (vertices[:, :2].min(0), vertices[:, :2].max(0))
    floor, top, supports = min(table, settle), max(table, settle), []
    if rise:
        at, half = (r["support"][:2], (hi - lo) / 2 + 0.04) if rise > 0 else ((lo + hi) / 2, (hi - lo) / 2 - 0.005)
        supports.append(dict(type="box", pos=[*at, (floor + top) / 2], size=[*np.maximum(half, 0.01), (top - floor) / 2],
                             group=3, rgba=[0.6, 0.6, 0.6, 1]))  # fmt: skip
    turn = wrist_turn(model, data, qpos[grasp[0]], vertices) if grasp and not recorded else 0.0
    if turn:  # its yaw free: the wrist turned in before the grasp (and back after the release) so the jaws close
        ease, row = lambda a: np.clip(a, 0, 1) ** 2 * (3 - 2 * np.clip(a, 0, 1)), np.arange(len(qpos))
        qpos[:, 6] += turn * np.minimum(ease((row - grasp[0] + 40) / 25), ease((grasp[1] + 30 - row) / 25))

    # assets: the object textured by frame 0 (each vertex's source pixel), its convex hull, the backdrop
    small, faces = decimate(mesh.vertices, mesh.faces, max(mesh.extents.max() / 60, 1e-3))
    seen = trimesh.transform_points(small, camera_from_object)
    uv = np.c_[seen[:, 0] / seen[:, 2] * f + k[0, 2], seen[:, 1] / seen[:, 2] * f + k[1, 2]] / size[::-1]
    small, placed[:3, :3] = small * scale, placed[:3, :3] / scale
    write_obj(out / "object.obj", small, faces, np.c_[uv[:, 0], 1 - uv[:, 1]])
    hull = trimesh.Trimesh(small, faces).convex_hull
    write_obj(out / "hull.obj", hull.vertices, hull.faces)
    Image.fromarray(frames[0]).save(out / "frame0.png")
    robot = silhouettes(model, data, qpos, range(len(qpos)), size)[0]
    Image.fromarray(background(frames, robot, labels == 1)).save(out / "backdrop.png")
    write_obj(out / "backdrop.obj", *depth_mesh(d, labels, robot[0], k, world, floor))
    holder = container_parts(np.load(w / "masks/sam3.npz"), d, k, world, floor, frames[0], False)
    actor = ET.parse(FRANKA).getroot()
    actor.find("compiler").set("meshdir", str(FRANKA.parent))
    asset = actor.find("asset")
    for name in ("frame0", "backdrop"):
        ET.SubElement(asset, "texture", name=name, type="2d", file=str(out / f"{name}.png"))
        ET.SubElement(asset, "material", name=name, texture=name, emission="1", specular="0", shininess="0", rgba="0.6 0.6 0.6 1")  # fmt: skip
    parts, shapes = {"hull": out / "hull.obj"}, []
    low, high = small.min(0), small.max(0)
    if collision == "box":
        parts, shapes = {}, [dict(type="box", pos=(low + high) / 2, size=(high - low) / 2)]
    if collision == "cylinder":
        parts, shapes = {}, [dict(type="cylinder", pos=(low + high) / 2, size=[(high - low)[:2].max() / 2, (high - low)[2] / 2])]  # fmt: skip
    if collision == "hollow":  # a floor and thin walls around the footprint (pnp_hands.py: receptacles)
        from roboskills.tools.mujoco.scene import hollow_parts

        parts = {f"wall_{p.stem}": p for p in hollow_parts(trimesh.Trimesh(small, faces), placed, out / "hollow")}
    if collision == "coacd":  # CoACD's convex parts of the decimated mesh
        from roboskills.tools.coacd.coacd import decompose_convex

        write_obj(out / "solid.obj", small, faces)
        parts = {p.stem: p for p in decompose_convex(out / "solid.obj", out / "parts", faces=5000, hulls=16)}
    for name, file in {"object": out / "object.obj", "backdrop": out / "backdrop.obj", **parts}.items():
        ET.SubElement(asset, "mesh", name=name, file=str(file), **(dict(inertia="shell") if name == "backdrop" else {}))
    actor.find("actuator/general[@name='fingers_actuator']").set("forcerange", f"{-grip} {grip}")
    ET.SubElement(ET.SubElement(actor, "visual"), "headlight", ambient="0.4 0.4 0.4", diffuse="0.6 0.6 0.6", specular="0.1 0.1 0.1")  # fmt: skip
    ET.ElementTree(actor).write(out / "assets.xml")
    mass = float(np.clip(hull.volume * 500, 0.02, 0.5)) if mass is None else mass
    pose = placed
    build(out / "scene.xml", out / "assets.xml", cameras=[source_camera(world, k, size)], bodies=[
        dict(name="backdrop", parts=[dict(type="mesh", mesh="backdrop", material="backdrop", contype=0, conaffinity=0, group=0)]),
        dict(name="table", pos=[centre[0], centre[1], floor - 0.025], parts=[dict(type="box", size=[0.6, 0.6, 0.025], group=3, rgba=[0.6, 0.6, 0.6, 1])]),
        *([dict(name="container", parts=holder)] if holder else []),
        *([dict(name="support", parts=supports)] if supports else []),
        dict(name="object", free=True, pos=pose[:3, 3], xyaxes=[*pose[:3, 0], *pose[:3, 1]], parts=[
            dict(type="mesh", mesh="object", material="frame0", contype=0, conaffinity=0, density=0),
            *(dict(type="mesh", mesh=m, group=3, mass=mass / len(parts), friction=[friction, 0.01, 0.001], condim=condim, priority=2) for m in parts),
            *(dict(c, group=3, mass=mass, friction=[friction, 0.01, 0.001], condim=condim, priority=2) for c in shapes)]),
    ])  # fmt: skip
    model, data = load(out / "scene.xml")
    lit = [backdrop_render(model, data, qpos[0], size, e) for e in (0.0, 1.0)]  # brightness is lit + e * emitted
    keep = (labels == 0) & ~robot[0]
    target = np.array(Image.open(out / "backdrop.png")).astype(float)[keep]
    base, slope = lit[0][keep], lit[1][keep] - lit[0][keep]
    emission = float(np.clip(((target - base) * slope).sum() / max((slope**2).sum(), 1e-6), 0, 5))
    model.mat_emission[model.material("backdrop").id] = emission
    mujoco.mj_saveModel(model, str(out / "scene.mjb"))

    # kinematic: the object still, carried with the pinch from the grasp to the release, then set down on the table
    adr, n = model.jnt_qposadr[model.body("object").jntadr[0]], qpos.shape[1]
    poses = np.repeat(pose[None], len(qpos), 0)
    if grasp:
        held = np.linalg.inv(pinch_pose(model, data, qpos[grasp[0]])) @ pose
        for row in range(grasp[0], len(qpos)):
            poses[row] = pinch_pose(model, data, qpos[min(row, grasp[1])]) @ held
        low = trimesh.transform_points(small, poses[grasp[1]])[:, 2].min()
        poses[grasp[1] + 1 :, 2, 3] -= low - settle
    recon = np.repeat(model.qpos0[None], len(qpos), 0)
    recon[:, :n] = qpos
    for row, p in zip(recon, poses):
        row[adr : adr + 3] = p[:3, 3]
        mujoco.mju_mat2Quat(row[adr + 3 : adr + 7], p[:3, :3].flatten())
    kinematic = out / "recon"
    kinematic.mkdir(exist_ok=True)
    shutil.copy(out / "scene.mjb", kinematic / "scene.mjb")
    np.savez(kinematic / "trajectory.npz", qpos=recon)
    render(kinematic, camera="source", size=size[::-1], preview=6, source=w / VIDEO)
    shutil.copy(kinematic / "mujoco_ext1.mp4", out / "recon_ext1.mp4")

    # native: position servos to the recorded joints `lead` seconds ahead, the gripper to its recorded closing
    steps = np.rint(np.arange(len(qpos)) / 30 / model.opt.timestep).astype(int)
    t = np.arange(steps[-1]) * model.opt.timestep * 30 + lead * 30  # in source frames
    r = np.load(w / "input/robot.npz")
    arm = np.stack([np.interp(t, np.arange(len(qpos)), c) for c in qpos[:, :7].T], 1)  # the recorded joints (turned)
    close = np.zeros(
        len(qpos)
    )  # the gripper's command: fully closed from the grasp to the release (it stops on contact)
    if grasp:
        close[grasp[0] : grasp[1] + 1] = 1
    command = np.maximum(r["gripper_position"], close) if squeeze else r["gripper_position"]
    ctrl = np.c_[np.clip(arm, *model.actuator_ctrlrange[:7].T), 255 * np.interp(t, np.arange(len(qpos)), command)]
    native = out / "native"
    shutil.rmtree(native, ignore_errors=True)
    report = rollout(out / "scene.xml", native, ctrl, fps=30, frames=len(qpos), initial=dict(qpos=recon[0]),
                     camera="source", size=size[::-1], preview=6, source=w / VIDEO)  # fmt: skip
    shutil.copy(native / "mujoco_ext1.mp4", out / "mujoco_ext1.mp4")
    lifted = np.load(native / "trajectory.npz")["qpos"][:, adr + 2] - pose[2, 3]
    info = dict(focal=f, table=table, sample=json.loads((shape / "pose.json").read_text()).get("sample"), collision=collision, grasp_at=grasp_at, slide=slide, emission=emission, grasp=grasp, offset=offset.tolist(), rise=rise, wrist_turn=turn, mass=mass, lift_max=float(lifted.max()),
                end_shift=float(np.linalg.norm(np.load(native / "trajectory.npz")["qpos"][-1, adr : adr + 2] - poses[-1, :2, 3])),
                penetration=report["max_penetration_m"], seconds=round(time.monotonic() - started, 1))  # fmt: skip
    (out / "layer.json").write_text(json.dumps(info, indent=2) + "\n")
