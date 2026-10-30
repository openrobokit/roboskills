"""An EgoDex clip's Vision Pro hand tracking for real2sim: one hand's keypoints in the camera frame.

    python -m roboskills.eval.egodex.read CLIP.hdf5 [--hand left] --out JOINTS.npz
    python -m roboskills.eval.egodex.read prepare --w W --video V --task T [--gpus 0 1]

Writes `joints [frames, 21, 3]` (OpenPose order, metres, the camera frame) for `fit --joints`, and `fov`, the
camera's horizontal field of view in degrees from its calibrated intrinsics. `prepare` makes a clip's workspace for
the pnp step (tools/robokit/pnp_hands.py) from its video and task: the objects' nouns (one Codex call, `name_objects`) and
each object's perception by the workflow's tools, the camera following the head (W/camera/camera.npz). EgoDex
(Apple, CC BY-NC-ND): https://github.com/apple/ml-egodex
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from roboskills.tools.robokit.pnp_hands import RECEPTACLES, track_camera


COLOURS = ("white", "black", "red", "green", "blue", "yellow", "orange", "pink", "purple", "brown", "grey", "gray")


PARTS = ["Knuckle", "IntermediateBase", "IntermediateTip", "Tip"]  # CMC/MCP, MCP/PIP, IP/DIP, tip
FINGERS = ["Thumb", "IndexFinger", "MiddleFinger", "RingFinger", "LittleFinger"]


def name_objects(task: str, out: Path, model: str = "gpt-6.1-sol", video: Path | None = None) -> list[dict]:
    """The task's objects as SAM 3 prompts with their counts ([{noun, count}], most important first), by one tool-free
    Codex call at low effort; with `video`, its first, middle and last frames attached, the objects the hands move
    (`moved`, how many of them move) first, then what they come out of or go into or onto, named as seen. Prompt,
    reply, tokens and seconds kept in out/objects.json."""
    import re
    import tempfile

    import cv2

    prompt = ("List the physical objects the person manipulates in this task, or puts things into or onto: short "
              "lowercase noun phrases a segmentation model can find (never hands, arms, the table or the background), "
              "one per kind with how many there are, the most important first. Reply with JSON only: "
              f'[{{"noun": str, "count": int}}].\nTask: {task}')  # fmt: skip
    if video:
        prompt = ("The images are the first, middle and last frames of a video of a person's hands doing this task "
                  "(the text may be vague, or the reverse of what the video shows: trust the frames). List, as "
                  "lowercase noun phrases of 1-4 words a segmentation model can find in the first frame (the item you "
                  "see, e.g. 'bread roll' not 'food item', with a colour when it tells it from other things; never a "
                  "position such as left or middle, which the model cannot use, and never hands, arms, the table or the "
                  "background): each kind of object the hands move from where it is in the first frame (\"moved\": "
                  "true, count: how many of that kind move), then each kind one comes out of or goes into or onto "
                  "(\"moved\": false, count: how many of those there are; the same noun again when things of one kind "
                  "go onto each other). Reply with JSON only: "
                  f'[{{"noun": str, "count": int, "moved": bool}}].\nTask: {task}')  # fmt: skip
    began = time.monotonic()
    with tempfile.TemporaryDirectory() as tmp:  # a neutral directory: nothing but the prompt reaches the model
        images = []
        if video:  # the first, middle and last frames
            capture = cv2.VideoCapture(str(video))
            n = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            for f in (0, n // 2, n - 1):
                capture.set(cv2.CAP_PROP_POS_FRAMES, f)
                cv2.imwrite(f"{tmp}/frame_{f:05d}.png", capture.read()[1])
                images += ["-i", f"{tmp}/frame_{f:05d}.png"]
        command = ["codex", "exec", "--ignore-user-config", "--ignore-rules", "--skip-git-repo-check", "--ephemeral",
                   "--sandbox", "read-only", "--model", model, "-c", 'model_reasoning_effort="low"', "--json", *images,
                   "-o", f"{tmp}/reply.txt", "-"]  # fmt: skip
        done = subprocess.run(command, input=prompt, text=True, cwd=tmp, check=True, capture_output=True, timeout=300)
        reply = Path(f"{tmp}/reply.txt").read_text()
    usage = next((e["usage"] for line in done.stdout.splitlines() if line.startswith("{")
                  and (e := json.loads(line)).get("type") == "turn.completed"), None)  # fmt: skip
    nouns = json.loads(re.search(r"\[.*\]", reply, re.S).group(0))
    nouns.sort(key=lambda n: not n.get("moved", True))  # what moves first
    out.mkdir(parents=True, exist_ok=True)
    record = dict(task=task, model=model, prompt=prompt, frames=bool(video), reply=reply, nouns=nouns, usage=usage,
                  seconds=round(time.monotonic() - began, 1))  # fmt: skip
    (out / "objects.json").write_text(json.dumps(record, indent=2) + "\n")
    return nouns


def hull_shape(w: Path, label: int, out: Path, frame: int = 0):
    """out/mesh.ply and pose.json (`camera_from_object`): the convex hull of label `label`'s aligned depth points at
    `frame` and the same points dropped along gravity to the lowest of them (a solid down to its support), coloured by
    their pixels, in a frame at its centroid with the camera's axes."""
    import cv2
    import trimesh
    from PIL import Image

    d, down = np.load(w / "align/depth.npz"), np.array(json.loads((w / "gravity/gravity.json").read_text())["down"])
    row = int(np.searchsorted(d["frames"], frame))
    z, k, keep = d["depth"][row].astype(float), d["K"], np.array(Image.open(w / f"masks/{frame:05d}.png")) == label
    size = 9 if keep.sum() > 5000 else 3  # a small object keeps its few pixels
    keep = cv2.erode(keep.astype(np.uint8), np.ones((size, size), np.uint8)).astype(bool) & d["valid"][row] & (z > 0)
    v, u = np.nonzero(keep)
    points = np.c_[(u - k[0, 2]) / k[0, 0] * z[keep], (v - k[1, 2]) / k[1, 1] * z[keep], z[keep]]
    low, high = np.percentile(points, [3, 97], axis=0)  # mask edges bleeding onto what lies behind go
    points = points[((points > low - 0.01) & (points < high + 0.01)).all(1)]
    floor = points @ down
    mesh = trimesh.Trimesh(np.r_[points, points + np.outer(floor.max() - floor, down)]).convex_hull
    capture = cv2.VideoCapture(str(w / "input/video.mp4"))
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
    colour = capture.read()[1][keep][:, ::-1].mean(0)
    centre, pose = mesh.centroid, np.eye(4)
    pose[:3, 3] = centre
    mesh.apply_translation(-centre)
    mesh.visual.vertex_colors = np.r_[colour, 255].astype(np.uint8)
    out.mkdir(parents=True, exist_ok=True)
    mesh.export(out / "mesh.ply")
    (out / "pose.json").write_text(json.dumps(dict(camera_from_object=pose.tolist())) + "\n")


def follow_masks(w: Path, reference: np.ndarray, frame: int = 0) -> np.ndarray:
    """track/poses.npz (`camera_from_object` per frame, `frames`) for W's object (mask label 1, eroded) by translation
    alone: its fitted pose moved with the median of its aligned depth points, each frame's brought into the reference
    camera by `reference` (reference_from_camera); under 50 points holds the last. Seconds, not the minutes of
    `track`'s per-frame mask refit; the turn is left to the fit, as the scene keeps moving objects upright."""
    import cv2
    from PIL import Image

    d = np.load(w / "align/depth.npz")
    depths, valid, k = d["depth"], d["valid"], d["K"]
    v, u = np.indices(depths.shape[1:])
    placed = np.array(json.loads((w / "fit/pose.json").read_text())["camera_from_object"])
    centres, last = np.zeros((len(depths), 3)), None
    for f in range(len(depths)):
        keep = cv2.erode(
            (np.array(Image.open(w / f"masks/{f:05d}.png")) == 1).astype(np.uint8), np.ones((7, 7), np.uint8)
        ).astype(bool)
        keep &= valid[f] & (depths[f] > 0)
        if keep.sum() >= 50:
            z = depths[f][keep].astype(float)
            seen = np.median(np.c_[(u[keep] - k[0, 2]) / k[0, 0] * z, (v[keep] - k[1, 2]) / k[1, 1] * z, z], 0)
            last = reference[f, :3, :3] @ seen + reference[f, :3, 3]
        centres[f] = last if last is not None else np.nan
    seen = np.flatnonzero(np.isfinite(centres[:, 0]))
    centres = (
        centres[seen[np.clip(np.searchsorted(seen, np.arange(len(centres))), 0, len(seen) - 1)]]
        if len(seen)
        else np.zeros_like(centres)
    )
    poses = np.repeat(placed[None], len(depths), 0)
    poses[:, :3, 3] += centres - centres[frame]
    (w / "track").mkdir(exist_ok=True)
    np.savez(w / "track/poses.npz", camera_from_object=np.linalg.inv(reference) @ poses, frames=np.arange(len(depths)))
    return poses


def launch(call: str, gpu: str, **kwargs) -> subprocess.Popen:
    """`module.function(**kwargs)` in a Python of its own on CUDA device `gpu` (the tools follow
    CUDA_VISIBLE_DEVICES, in their uv environments and images too)."""
    import os
    import sys

    module, function = call.rsplit(".", 1)
    args = ", ".join(f"{k}=Path({str(v)!r})" if isinstance(v, Path) else f"{k}={v!r}" for k, v in kwargs.items())
    code = f"from pathlib import Path; from {module} import {function}; {function}({args})"
    return subprocess.Popen([sys.executable, "-c", code], env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu))


def finish(*runs: subprocess.Popen | bool, strict: bool = True) -> bool:
    """Wait for `runs` (True: done before); False if one failed and not `strict`."""
    failed = [run for run in runs if run is not True and run.wait()]
    if failed and strict:
        raise RuntimeError(f"{failed[0].args[-1][:160]} failed")
    return not failed


def follow_icp(w: Path, reference: np.ndarray, frame: int = 0, steps: int = 20) -> np.ndarray:
    """track/poses.npz for W's object in six degrees of freedom: from its mask-centre track (follow_masks), each
    frame ICP-fitted to its aligned depth points (mask label 1, eroded; the reference camera's frame), from the
    neighbour's pose shifted with the mask centre; `turned` marks whether the turn is to be trusted (no frame turns
    over 15 degrees from its neighbour). Point-to-point, the worst fifth of matches dropped; under 50 points the
    shifted pose is kept."""
    import cv2
    import trimesh
    from PIL import Image
    from scipy.spatial import cKDTree
    from scipy.spatial.transform import Rotation

    centred = reference @ np.load(w / "track/poses.npz")["camera_from_object"]  # the mask-centre track
    d = np.load(w / "align/depth.npz")
    depths, valid, k = d["depth"], d["valid"], d["K"]
    v, u = np.indices(depths.shape[1:])
    surface = cKDTree(trimesh.sample.sample_surface(trimesh.load(w / "fit/mesh.ply"), 2000, seed=0)[0])
    poses = centred.copy()

    def fit(f: int, pose: np.ndarray) -> np.ndarray:
        keep = cv2.erode((np.array(Image.open(w / f"masks/{f:05d}.png")) == 1).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)  # fmt: skip
        keep &= valid[f] & (depths[f] > 0)
        if keep.sum() < 50:
            return pose
        z = depths[f][keep].astype(float)
        x = (
            np.c_[(u[keep] - k[0, 2]) / k[0, 0] * z, (v[keep] - k[1, 2]) / k[1, 1] * z, z] @ reference[f, :3, :3].T
            + reference[f, :3, 3]
        )
        x, inverse = x[np.random.default_rng(0).permutation(len(x))[:1500]], np.linalg.inv(pose)
        for _ in range(steps):
            distance, index = surface.query(trimesh.transform_points(x, inverse))
            near = distance <= np.percentile(distance, 80)
            inverse = trimesh.registration.procrustes(x[near], surface.data[index[near]], reflection=False, scale=False, return_cost=False)  # fmt: skip
        return np.linalg.inv(inverse)

    for rows in (range(frame + 1, len(poses)), range(frame - 1, -1, -1)):
        last = frame
        for f in rows:
            start = poses[last].copy()
            start[:3, 3] += centred[f, :3, 3] - centred[last, :3, 3]
            poses[f], last = fit(f, start), f
    turns = np.degrees(np.linalg.norm(Rotation.from_matrix(np.einsum("nji,njk->nik", poses[:-1, :3, :3], poses[1:, :3, :3])).as_rotvec(), axis=1))  # fmt: skip
    np.savez(w / "track/poses.npz", camera_from_object=np.linalg.inv(reference) @ poses, frames=np.arange(len(poses)),
             turned=bool(turns.max() < 15))  # fmt: skip
    return poses


def prepare(w: Path, video: Path, task: str, frame: int = 0, gpus: tuple[str, ...] = ("0",), width: int = 960,
            grasped: int = 2, reuse: Path | None = None) -> dict:  # fmt: skip
    """Perception for every named object (each instance) and both hands with the workflow's tools, on the video
    resized to `width`, finished steps kept, two lanes of GPU work on `gpus` (one or two devices): MoGe then GeoCalib
    beside SAM 3 (the nouns, each its count, "arm" and "hand" twice; the smaller on top; a kind the hands move two
    more, the count moving farthest kept: sam3.keep_moved), found at the reference frame and, on the other
    lane, at the last (sam3.join_tracks: what the first loses on the way is re-found where it ends); then SAM 3D
    on the first noun's best instance beside HaWoR per hand (hands_<side>/, sides by the hands' boxes); SAM 3 tracks
    every second frame of a clip over 150 frames. Each object gets
    W/objects/<id>/ (name.txt, masks relabelled: it 1, the better-seen hand 2, the rest 3): the first noun's best
    instance its SAM 3D shape, the workflow's align (W/align) and fit; its other instances that mesh refit to their
    masks; every other object the hull of its depth points. Objects a hand touches follow their mask centres
    (follow_masks), the `grasped` that move farthest in six degrees of freedom: the workflow's track (each frame's
    mask refit) on clips up to 150 frames, side by side on the two lanes, else ICP alone (follow_icp); the rest hold
    still. CoACD parts for receptacles, convex hulls for the rest. `reuse`, an earlier workspace of the clip: the first
    noun's best instance takes its SAM 3D shape of the object whose frame-0 mask it overlaps (IoU at least 0.5).
    Returns seconds per step."""
    import concurrent.futures

    import cv2
    import trimesh
    from PIL import Image
    from scipy.ndimage import binary_dilation

    from roboskills.tools.coacd.coacd import decompose_convex
    from roboskills.tools.robokit.mask_alignment import fit_shape
    from roboskills.workflows.real2sim import run as reconstruct

    seconds, began, lanes = {}, time.monotonic(), (gpus[0], gpus[-1])

    def lap(name: str):
        seconds[name] = round(time.monotonic() - began - sum(seconds.values()), 1)

    (w / "input").mkdir(parents=True, exist_ok=True)
    clip = w / "input/video.mp4"
    if not clip.exists():
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(video), "-vf", f"scale={width}:-2", "-c:v", "libx264",
                        "-crf", "16", "-pix_fmt", "yuv420p", str(clip)], check=True)  # fmt: skip
    named = w / "input/objects.json"
    nouns = json.loads(named.read_text())["nouns"] if named.exists() else name_objects(task, w / "input", video=clip)
    merged = {}  # one prompt per noun: its counts summed, moved if any of them moves
    for n in (n for n in nouns if "table" not in n["noun"]):
        m = merged.setdefault(n["noun"], dict(n, count=0, moved=False))
        m["count"], m["moved"] = m["count"] + n["count"], m["moved"] or bool(n.get("moved"))
    nouns = list(merged.values())
    lap("input")
    colour = lambda n: n.split()[0] + " object" if n.split()[0] in COLOURS else "object"
    prompts = [f"{n['noun']}|{colour(n['noun'])}" for n in nouns] + ["arm", "hand"]  # a generic fallback each
    counts = [min(n["count"] + 2 * bool(n.get("moved")), 6) for n in nouns] + [2, 2]  # candidates for what moves
    depth = (w / "depth/depth.npz").exists() or launch("roboskills.tools.moge3.moge3.estimate_depth", lanes[0], video=clip, out=w / "depth", frame=frame)  # fmt: skip
    last = int(cv2.VideoCapture(str(clip)).get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    stride = 2 if last >= 150 else 1  # a long clip: half the tracking
    tracked = [] if (w / "masks/objects.json").exists() else [  # found at the reference frame and at the last
        (w / f"masks_{f}/objects.json").exists() or launch("roboskills.tools.sam3.sam3.segment_video", lane, video=clip, prompts=prompts, out=w / f"masks_{f}", frame=f, counts=counts, smaller_on_top=True, stride=stride)  # fmt: skip
        for f, lane in ((frame, lanes[1]), (last, lanes[0]))]  # fmt: skip
    finish(depth)
    gravity = (w / "gravity/gravity.json").exists() or launch("roboskills.tools.geocalib.geocalib.estimate_gravity", lanes[0], video=clip, out=w / "gravity")  # fmt: skip
    finish(*tracked)
    if tracked:  # one track, what moves the farthest of its candidates kept
        from roboskills.tools.sam3.sam3 import join_tracks, keep_moved

        moved = {n["noun"]: n["count"] for n in nouns if n["moved"]}
        join_tracks(w / f"masks_{frame}", w / f"masks_{last}", w / "masks", frame, add=tuple(moved))
        keep_moved(w / "masks", moved, frame)
    lap("moge3_sam3")
    found = {int(i): v for i, v in json.loads((w / "masks/objects.json").read_text()).items()}
    labels = np.array(Image.open(w / f"masks/{frame:05d}.png"))
    hands = sorted((np.nonzero(labels == i)[1].mean(), i) for i, v in found.items() if v["prompt"] == "hand" and (labels == i).any())  # fmt: skip
    sides = dict(zip(["left", "right"] if len(hands) == 2 else ["left" if hands[0][0] < labels.shape[1] / 2 else "right"], [i for _, i in hands])) if hands else {}  # fmt: skip
    others = [i for i, v in found.items() if v["prompt"] in ("hand", "arm")]
    items = {i: v["prompt"].split("|")[0] for i, v in found.items() if i not in others and (labels == i).sum() >= 50}
    primary = max((i for i in items if items[i] == items[min(items)]), key=lambda i: found[i]["score"])
    order = [primary] + [i for i in items if i != primary]
    hawor = [launch("roboskills.tools.hawor.hawor.reconstruct_hands", lanes[1], video=clip, masks=w / "masks", depth=w / "depth/depth.npz",
                    out=w / f"hands_{side}", label=label, side=side) for side, label in sides.items() if not (w / f"hands_{side}/hands.npz").exists()]  # fmt: skip
    every = {f: np.array(Image.open(path)) for f, path in enumerate(sorted((w / "masks").glob("*.png")))}
    touched = dict.fromkeys(items, 0)  # frames where a hand (dilated) meets the object
    for labels_f in every.values():
        near = binary_dilation(np.isin(labels_f[::2, ::2], list(sides.values())), np.ones((7, 7)))
        for i in items:
            touched[i] += bool((near & (labels_f[::2, ::2] == i)).any())
    hand = max(sides, key=lambda side: sum(int((m == sides[side]).sum()) for m in every.values()))  # the better seen
    for i in order:
        sub = w / "objects" / f"{i:02d}"
        (sub / "masks").mkdir(parents=True, exist_ok=True)
        (sub / "name.txt").write_text(items[i])
        for name, target in (("input", w / "input"), ("depth", w / "depth"), ("gravity", w / "gravity"), ("hands", w / f"hands_{hand}")):  # fmt: skip
            (sub / name).exists() or (sub / name).symlink_to(target.resolve())
        if not (sub / "masks/objects.json").exists():
            for f, labels_f in every.items():
                relabelled = np.select([labels_f == i, labels_f == sides[hand], labels_f > 0], [1, 2, 3], 0)
                Image.fromarray(relabelled.astype(np.uint8)).save(sub / "masks" / f"{f:05d}.png")
            (sub / "masks/objects.json").write_text(json.dumps({"1": found[i]}) + "\n")
    shape = w / "objects" / f"{primary:02d}"
    for old in (
        sorted(Path(reuse).glob("objects/*/shape/pose.json")) if reuse and not (shape / "shape").exists() else ()
    ):
        theirs = np.array(Image.open(old.parents[1] / f"masks/{frame:05d}.png")) == 1
        if (theirs & (labels == primary)).sum() >= 0.5 * (theirs | (labels == primary)).sum():  # the same object
            (shape / "shape").symlink_to(old.parent.resolve())
            break
    sam3d = (shape / "shape/pose.json").exists() or launch("roboskills.tools.sam3d.sam3d.reconstruct_object", lanes[0], video=clip, masks=shape / "masks",
                                                          depth=w / "depth/depth.npz", out=shape / "shape", frame=frame)  # fmt: skip
    finish(gravity, *hawor)
    if not finish(sam3d, strict=False) or not (shape / "shape/pose.json").exists():  # no SAM 3D shape: its depth hull
        (shape / "align").mkdir(parents=True, exist_ok=True)
        (shape / "align/depth.npz").exists() or (shape / "align/depth.npz").symlink_to(
            (w / "depth/depth.npz").resolve()
        )
        hull_shape(shape, 1, shape / "shape", frame)
        (shape / "align/depth.npz").unlink()
    lap("sam3d_hawor")
    reconstruct(shape, ["align", "fit"], frame=frame)
    (w / "align").exists() or (w / "align").symlink_to((shape / "align").resolve())
    lap("align_fit")
    for i in order[1:]:
        sub = w / "objects" / f"{i:02d}"
        (sub / "align").mkdir(exist_ok=True)
        (sub / "align/depth.npz").exists() or (sub / "align/depth.npz").symlink_to((w / "align/depth.npz").resolve())
        if (sub / "fit/pose.json").exists():
            continue
        if items[i] != items[primary]:  # the hull is its observation: no refit
            hull_shape(sub, 1, sub / "fit", frame)
            continue
        hull_shape(sub, 1, sub / "align", frame)  # the primary's mesh at this instance's place, refit
        mesh = trimesh.load(shape / "fit/mesh.ply")
        pose = np.array(json.loads((shape / "fit/pose.json").read_text())["camera_from_object"])
        placed = np.array(json.loads((sub / "align/pose.json").read_text())["camera_from_object"])
        pose[:3, 3] = placed[:3, 3] - pose[:3, :3] @ mesh.centroid
        mesh.export(sub / "align/mesh.ply")
        (sub / "align/pose.json").write_text(json.dumps(dict(camera_from_object=pose.tolist())) + "\n")
        fit_shape(sub / "masks", sub / "align/depth.npz", sub / "align", sub / "fit", frame=frame)
    lap("shapes")
    camera = w / "camera/camera.npz"
    reference = np.load(camera)["reference_from_camera"] if camera.exists() else track_camera(w, w / "camera", frame)
    d = np.load(w / "align/depth.npz")  # the reference frame alone, for the scene
    row = int(np.searchsorted(d["frames"], frame))
    np.savez_compressed(w / "camera/reference.npz", depth=d["depth"][row], valid=d["valid"][row], K=d["K"], frame=frame)
    lap("camera")
    far = {}
    for i in order:
        sub = w / "objects" / f"{i:02d}"
        if (sub / "track/poses.npz").exists():
            continue
        if touched[i] >= 5:
            moved = follow_masks(sub, reference, frame)[:, :2, 3]
            far[i] = float(np.linalg.norm(moved - moved[frame], axis=1).max())
        else:  # held still in the world: its reference pose seen from each frame's camera
            placed = np.array(json.loads((sub / "fit/pose.json").read_text())["camera_from_object"])
            (sub / "track").mkdir(exist_ok=True)
            np.savez(sub / "track/poses.npz", camera_from_object=np.linalg.inv(reference) @ placed, frames=np.arange(len(reference)))  # fmt: skip
    movers = sorted((i for i in far if far[i] > 0.06), key=far.get, reverse=True)[:grasped]
    if len(every) <= 150:  # the workflow's track (ICP, then each frame's mask refit), the movers side by side
        finish(*(launch("roboskills.workflows.real2sim.track.track", lane, masks=w / "objects" / f"{i:02d}" / "masks",
                        depth=w / "objects" / f"{i:02d}" / "align/depth.npz", shape=w / "objects" / f"{i:02d}" / "fit",
                        out=w / "objects" / f"{i:02d}" / "track", frame=frame) for i, lane in zip(movers, lanes)))  # fmt: skip
    else:  # a long clip: ICP alone, from the mask centres, in a fraction of the time
        for i in movers:
            follow_icp(w / "objects" / f"{i:02d}", reference, frame)
    lap("track")

    def collide(i: int):  # CoACD for what things go into, the convex hull for the rest
        sub = w / "objects" / f"{i:02d}"
        if (sub / "collision/parts.json").exists():
            return
        if any(word in items[i] for word in RECEPTACLES):
            decompose_convex(sub / "fit/mesh.ply", sub / "collision", threshold=0.08, faces=3000, hulls=10)
            return
        (sub / "collision").mkdir(exist_ok=True)
        mesh = trimesh.load(sub / "fit/mesh.ply", force="mesh")
        mesh.export(sub / "collision/visual.obj")
        mesh.convex_hull.export(sub / "collision/part_000.obj")
        (sub / "collision/parts.json").write_text(json.dumps(dict(parts=["part_000.obj"])) + "\n")

    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        list(pool.map(collide, order))
    lap("collision")
    record = dict(nouns=nouns, sides=sides, primary=primary, objects=items, touched=touched, moved=far, seconds=seconds)
    (w / "prepare.json").write_text(json.dumps(record, indent=2) + "\n")
    return seconds


def main():
    import sys

    import h5py

    if sys.argv[1:2] == ["prepare"]:
        import tyro

        return tyro.cli(prepare, args=sys.argv[2:])

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("clip", help="the clip's .hdf5")
    p.add_argument("--hand", choices=["left", "right"], default="right")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    f = h5py.File(a.clip)
    names = [f"{a.hand}Hand"] + [f"{a.hand}{finger}{part}" for finger in FINGERS for part in PARTS]
    camera = np.linalg.inv(f["transforms/camera"][:])
    joints = np.stack([(camera @ f[f"transforms/{n}"][:])[:, :3, 3] for n in names], 1)
    k = f["camera/intrinsic"][:]
    fov = float(np.degrees(2 * np.arctan(k[0, 2] / k[0, 0])))
    np.savez_compressed(a.out, joints=joints, fov=fov)
    print(f"{len(joints)} frames; field of view {fov:.1f} degrees")


if __name__ == "__main__":
    main()
