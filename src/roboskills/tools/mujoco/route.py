"""Real2sim routing: whether a cheap layer's delivered MuJoCo rollout passes, or goes to the full agent, judged
against the source video's own evidence rather than our object track (no judge, no GPU, seconds a clip).

Each object SAM 3 tracks through the source (per-frame label masks: workspace/masks/, or DROID prepare's
cache/masks/; arms, hands and the robot left out) is matched one to one to the sim's free bodies by their silhouettes,
projected through the source camera at frame 0: the assignment of least total centre distance (Hungarian), a pair
within one object scale (half its frame-0 box diagonal, at least 2% of the image diagonal). At the clip's end
(sampled frames in its last 10%, else 25%, where the source mask keeps VISIBLE of its frame-0 area; medians) two
centroid distances in object scales: the sim's silhouette from the source mask (`off`), and the frame-0 pose (the
object left where it was, seen by each frame's camera) from it (`moved`). An object the source moves (moved >= MOVED)
scores 1 - off / moved; one it leaves passes while off <= STILL.
With the source's depth (MoGe's, each frame scaled to frame 0's background as its camera sees it), the last 3 frames
also give, in metres: the source's 3D shift since frame 0 (its mask's median point), the sim's end from it (`error`)
and its height part (`rise`), and the sim's 2D offset at the source's depth (`lateral`). An object the source moves
under MOVED but at least SHIFT in depth and NUDGED in 2D (a plug pushed in or pulled out) passes while lateral <= NEAR;
a moved one scores 0 ending over RISE above or below the source's (on the support, not under it or still held up).
An object still held at the end (touching the robot, its nearest gripper at least GRIP closed) where the source's hands
end LEFT object scales off it (hand or arm masks) scores 0, as does one whose sim path strays from the source's (the
90th percentile of its per-frame 2D offset, all sampled frames) over PATH. The worst object is the score; it passes
at THRESHOLD. Calibrated on 211 judged dev rollouts (judge success >= 50), leave-one-clip-out: the threshold 0.65 in
23 of 24 folds; RISE, NEAR and PATH inside the ranges where the calibration is flat (0.06-0.12 m, 0.02-0.05 m, 3-3.5);
pass precision 0.83 -> 0.97, recall 0.71 -> 0.74. On 27 newer judged deliveries held out: 0.75 -> 1.0, 0.69 -> 0.77.
No evidence goes to the full agent: a source object with no sim body, one never visible at the end, no per-frame masks, or a source
that moves none of its tracked objects (the moving one is untracked: nothing to confirm the task by). So the masks
should follow what moves (sam3.py `keep_moved`, `join_tracks`).

    python -m roboskills.tools.mujoco.route --delivery-dir D --workspace W [--masks M] [--depth DEPTH_NPZ] [--also DIR ...]
"""

import json
import re
from pathlib import Path

import numpy as np


THRESHOLD, MOVED, STILL, VISIBLE, SAMPLES = 0.65, 1.5, 1.0, 0.05, 48
RISE, NEAR, SHIFT, NUDGED = 0.08, 0.05, 0.04, 0.5  # metres: height, small-move end tolerances, a small move; scales
LEFT, GRIP, PATH = 0.5, 0.1, 3.3  # scales: the source's hand lets go; closure: the sim's gripper holds; path, scales
SKIP = {"arm", "arms", "hand", "hands", "robot", "gripper"}
FLIP = np.diag([1.0, -1.0, -1.0])  # OpenCV camera axes from MuJoCo's


def body_points(model, body: int, cap: int = 3000) -> list[tuple[int, np.ndarray]]:
    """[(geom, points in its frame)]: the body's visual mesh geoms' vertices (at most ~`cap`), else its colliding
    meshes' and boxes' corners."""
    import mujoco

    geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] == body]
    seen = [g for g in geoms if not (model.geom_contype[g] | model.geom_conaffinity[g])]
    out = []
    for g in seen or geoms:
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            m = model.geom_dataid[g]
            v = model.mesh_vert[model.mesh_vertadr[m] : model.mesh_vertadr[m] + model.mesh_vertnum[m]]
            out.append((g, v[:: max(1, len(v) // cap)].astype(float)))
        elif model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
            out.append((g, np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * model.geom_size[g]))  # fmt: skip
    return out


def model_camera(model, data) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """The model's first camera (after mj_forward): intrinsics K, world from camera (OpenCV axes), (width, height)."""
    import mujoco

    mujoco.mj_camlight(model, data)
    if model.cam_sensorsize[0].any():  # pinhole intrinsics (a calibrated source camera)
        (w, h), s, i = model.cam_resolution[0], model.cam_sensorsize[0], model.cam_intrinsic[0]
        fx, fy, cx, cy = i[0] / s[0] * w, i[1] / s[1] * h, w / 2 - i[2] / s[0] * w, h / 2 - i[3] / s[1] * h
    else:  # a field of view (hands'), centred
        w, h = model.vis.global_.offwidth, model.vis.global_.offheight
        fx = fy = h / 2 / np.tan(np.radians(model.cam_fovy[0]) / 2)
        cx, cy = w / 2, h / 2
    pose = np.eye(4)
    pose[:3, :3], pose[:3, 3] = data.cam_xmat[0].reshape(3, 3) @ FLIP, data.cam_xpos[0]
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]), pose, (int(w), int(h))


def silhouette(data, points, k: np.ndarray, pose: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """A body's silhouette: the union of each geom's projected convex hull (geoms behind the camera skipped)."""
    import cv2

    mask, inv = np.zeros(size[::-1], np.uint8), np.linalg.inv(pose)
    for g, v in points:
        c = (v @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]) @ inv[:3, :3].T + inv[:3, 3]
        if (c[:, 2] > 0.02).all():
            uv = np.c_[k[0, 0] * c[:, 0] / c[:, 2] + k[0, 2], k[1, 1] * c[:, 1] / c[:, 2] + k[1, 2]]
            cv2.fillConvexPoly(mask, cv2.convexHull(np.round(uv * 16).astype(np.int32)), 1, shift=4)
    return mask.astype(bool)


def source_objects(masks: Path) -> tuple[dict[int, str], list[int]]:
    """Object labels of a per-frame masks folder (objects.json's prompts other than arms, hands and the robot; label 1
    alone without prompts) and its arms', hands' and the robot's labels."""
    found = json.loads((masks / "objects.json").read_text()) if (masks / "objects.json").exists() else {}
    if found and all(isinstance(v, dict) and "prompt" in v for v in found.values()):
        names = {int(k): v["prompt"].split("|")[0] for k, v in found.items()}
        return {k: v for k, v in names.items() if not set(v.split()) & SKIP}, [k for k, v in names.items() if set(v.split()) & SKIP]  # fmt: skip
    return {1: "object"}, []


def centre(mask: np.ndarray) -> np.ndarray | None:
    ys, xs = np.nonzero(mask)
    return np.array([xs.mean(), ys.mean()]) if len(xs) else None


def lift(z: np.ndarray, k: np.ndarray, mask: np.ndarray, step: int = 1) -> np.ndarray:
    """The 3D points (camera axes, metres) of `mask`'s pixels (every `step`-th) with depth `z`."""
    ys, xs = np.nonzero(mask[::step, ::step] & (z[::step, ::step] > 0))
    ys, xs = ys * step, xs * step
    return np.c_[(xs - k[0, 2]) / k[0, 0], (ys - k[1, 2]) / k[1, 1], np.ones(len(xs))] * z[ys, xs, None]


def holding(model, data, body: int) -> tuple[bool, float]:
    """In `data` (after mj_forward): whether `body` touches the robot (a body on a hinged chain), and the closure (0
    open, 1 closed) of the gripper (a `*driver_joint` and its `*pad`s) whose pads are nearest it."""
    import mujoco

    hinged = {int(model.jnt_bodyid[j]) for j in range(model.njnt) if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE}

    def robot(b: int) -> bool:
        while b > 0 and b not in hinged:
            b = int(model.body_parentid[b])
        return b > 0

    pairs = [(int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])) for c in data.contact[: data.ncon] if c.dist < 0.002]  # fmt: skip
    grips = []
    for j in (j for j in range(model.njnt) if model.joint(j).name.endswith("right_driver_joint")):
        side = model.joint(j).name[: -len("right_driver_joint")]
        pads = [data.xpos[b] for b in range(model.nbody) if model.body(b).name in (f"{side}right_pad", f"{side}left_pad")]  # fmt: skip
        if pads:
            grips.append((np.linalg.norm(np.mean(pads, 0) - data.xpos[body]), data.qpos[model.jnt_qposadr[j]] / model.jnt_range[j][1]))  # fmt: skip
    touch = any(body in (a, b) and robot(b if a == body else a) for a, b in pairs)
    return touch, float(min(grips, default=(0, 0.0))[1])


def score_rollout(model, qpos: np.ndarray, cameras: np.ndarray | None, masks: Path, depth: Path | None = None, **limits) -> dict:  # fmt: skip
    """{pass, score, reasons, objects} of native `qpos` (one row per source frame) in `model`, seen by `cameras`
    (world from camera per frame, OpenCV axes; None: the model's fixed camera), against per-frame label `masks` and the
    source's `depth` (a depth.npz: depth, valid, frames, K; metres in the scene's scale); `limits` override judge's."""
    import cv2
    import mujoco
    from PIL import Image
    from scipy.optimize import linear_sum_assignment

    n, data, rest = len(qpos), mujoco.MjData(model), mujoco.MjData(model)
    frames = sorted({0, *np.round(np.linspace(0, n - 1, SAMPLES)).astype(int).tolist()})
    bodies = [int(model.jnt_bodyid[j]) for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
    points, (labels, hands) = {b: body_points(model, b) for b in bodies}, source_objects(masks)
    d = dict(np.load(depth)) if depth is not None and Path(depth).exists() else {"frames": []}
    rows = {int(f): i for i, f in enumerate(d["frames"])}
    rest.qpos[:] = qpos[0]
    mujoco.mj_forward(model, rest)
    seen, lifted, ends = {}, {}, []  # frame: (source labels, {body: (sim silhouette, frame-0 pose's, sim position)})
    for f in sorted({*frames, *range(max(n - 3, 1), n)}):  # the 2D samples and the last 3 frames (3D, hands)
        if not (masks / f"{f:05d}.png").exists():
            continue
        data.qpos[:] = qpos[f]
        mujoco.mj_forward(model, data)
        k, pose, size = model_camera(model, data)  # intrinsics: the model's, fixed
        pose = cameras[f] if cameras is not None else pose
        raw = np.array(Image.open(masks / f"{f:05d}.png"))
        found = raw if raw.shape[::-1] == size else cv2.resize(raw, size, interpolation=cv2.INTER_NEAREST)
        seen[f] = (found, {b: (silhouette(data, points[b], k, pose, size), silhouette(rest, points[b], k, pose, size), data.xpos[b].copy()) for b in bodies})  # fmt: skip
        if f in rows:  # the source in 3D: depth, its labels, world from camera
            z = np.where(d["valid"][rows[f]], d["depth"][rows[f]].astype(np.float32), 0)
            lifted[f] = [z, raw if raw.shape == z.shape else cv2.resize(raw, z.shape[::-1], interpolation=cv2.INTER_NEAREST), pose]  # fmt: skip
        if f >= n - 3:  # contacts at the end
            ends.append([holding(model, data, b) for b in bodies])
    if 0 not in seen or len(seen) < 2:
        return {"pass": False, "score": None, "reasons": [f"no per-frame source masks in {masks}"], "objects": []}
    if 0 in lifted:  # each frame's depth scaled to frame 0's background (label 0) seen through it by its camera
        z0, at0, pose0 = lifted[0]
        back = pose0[:3, :3] @ lift(z0, d["K"], at0 == 0, 4).T + pose0[:3, 3:]  # world
        for z, at, pose in lifted.values():
            c = np.linalg.inv(pose)[:3, :3] @ back + np.linalg.inv(pose)[:3, 3:]  # this camera's
            uv = np.round(d["K"][:2] @ (c / c[2])).astype(int)
            ok = (c[2] > 0) & (uv[0] >= 0) & (uv[1] >= 0) & (uv[0] < z.shape[1]) & (uv[1] < z.shape[0])
            uv, c = uv[:, ok], c[:, ok]
            ok = (at[uv[1], uv[0]] == 0) & (z[uv[1], uv[0]] > 0)
            z *= np.median(c[2, ok] / z[uv[1, ok], uv[0, ok]]) if ok.sum() > 100 else 1.0
    world = lambda f, p: lifted[f][2][:3, :3] @ np.median(p, 0) + lifted[f][2][:3, 3]  # a point cloud's median, world
    objects, shown = [], {label: m for label in labels if (m := seen[0][0] == label).sum() >= 20}
    scales = {label: max(0.5 * np.hypot(*np.ptp(np.nonzero(m), 1) + 1), 0.02 * np.hypot(*m.shape)) for label, m in shown.items()}  # fmt: skip
    far = np.array([[np.linalg.norm(c - centre(m)) / scales[label] if (c := centre(seen[0][1][b][0])) is not None else np.inf for b in bodies] for label, m in shown.items()]).reshape(len(shown), len(bodies))  # fmt: skip
    pairs = zip(*linear_sum_assignment(np.where(far <= 1.0, far, 1e3)))  # one body each, least total distance
    match = {list(shown)[r]: bodies[c] for r, c in pairs if far[r, c] <= 1.0}
    for label, name in ((label, name) for label, name in labels.items() if label in shown):
        first, scale = shown[label], scales[label]
        if label not in match:
            objects.append(dict(name=name, body=None))
            continue
        body, i, steps = match[label], bodies.index(match[label]), []  # (frame, off, moved): sim way, source way
        for f in (f for f in frames if f in seen and (seen[f][0] == label).sum() >= max(VISIBLE * first.sum(), 5)):
            sim, still, at = centre(seen[f][1][body][0]), centre(seen[f][1][body][1]), centre(seen[f][0] == label)
            if sim is not None and still is not None:
                steps.append((f, np.linalg.norm(sim - at) / scale, np.linalg.norm(still - at) / scale))
        tail = [s[1:] for s in steps if s[0] >= 0.9 * (n - 1)] or [s[1:] for s in steps if s[0] >= 0.75 * (n - 1)]
        record = dict(name=name, body=model.body(body).name, scale=round(float(scale), 1))
        objects.append(record)
        if not tail:
            continue
        (off, moved), path = np.median(tail, 0), np.percentile([s[1] for s in steps], 90)
        record.update(off=round(float(off), 3), moved=round(float(moved), 3), path=round(float(path), 3))
        last, shift, error, lateral = [f for f in seen if f >= n - 3 and (seen[f][0] == label).sum() >= 5], [], [], []
        p0 = lift(lifted[0][0], d["K"], lifted[0][1] == label) if 0 in lifted else []
        for f in (f for f in last if f in lifted and len(p0) >= 20):  # the end in 3D: source shift, sim off it
            if len(p := lift(lifted[f][0], d["K"], lifted[f][1] == label)) >= 20:
                shift.append(world(f, p) - world(0, p0))
                error.append(seen[f][1][body][2] - rest.xpos[body] - shift[-1])
                if (c := centre(seen[f][1][body][0])) is not None:  # the 2D offset in metres at the source's depth
                    lateral.append(np.linalg.norm(c - centre(seen[f][0] == label)) * np.median(p[:, 2]) / k[0, 0])
        if error:
            record.update(error=round(float(np.median(np.linalg.norm(error, axis=1))), 3), rise=round(float(np.median(np.array(error)[:, 2])), 3),
                          shift=round(float(np.median(np.linalg.norm(shift, axis=1))), 3), **({"lateral": round(float(np.median(lateral)), 3)} if lateral else {}))  # fmt: skip
        if last and hands:  # the source's hands at the end: their gap to the object, object scales
            far = [cv2.distanceTransform((seen[f][0] != label).astype(np.uint8), cv2.DIST_L2, 3) for f in last]
            record["gap"] = round(float(np.median([g[np.isin(seen[f][0], hands)].min(initial=99 * scale) / scale for g, f in zip(far, last)])), 2)  # fmt: skip
        record.update(touch=any(e[i][0] for e in ends), closed=round(ends[-1][i][1], 2) if ends else 0.0)
    return {**judge(objects, **limits), "objects": objects}


def judge(objects: list[dict], threshold: float = THRESHOLD, moved: float = MOVED, still: float = STILL,
          release: bool = True, height: float | None = RISE, near: float | None = NEAR, path: float | None = PATH) -> dict:  # fmt: skip
    """{pass, score, reasons} of score_rollout's object records: `height` (m) the rise tolerance, `near` the small-motion
    end tolerance, `path` the path tolerance (object scales), `release` the release veto; None or False turns one off."""
    reasons, scores, moving = [], [], False
    for o in objects:
        if o["body"] is None:
            reasons.append(f"{o['name']}: no sim object at frame 0")
            continue
        if "off" not in o:
            reasons.append(f"{o['name']}: not visible at the end")
            continue
        small = near is not None and NUDGED <= o["moved"] < moved and o.get("shift", 0) >= SHIFT and "lateral" in o
        if o["moved"] >= moved:
            score, why = max(0.0, 1 - o["off"] / o["moved"]), f"moved {o['moved']:.1f}, sim off {o['off']:.1f}"
        elif small:  # a small move the depth confirms: the sim's end within `near` of the source mask's, in metres
            score, why = float(o["lateral"] <= near), f"moved {o['shift'] * 100:.0f} cm, sim {o['lateral'] * 100:.0f} cm off"  # fmt: skip
        else:
            score, why = float(o["off"] <= still), f"still ({o['moved']:.1f}), sim off {o['off']:.1f}"
        moving |= o["moved"] >= moved or small
        if height is not None and (o["moved"] >= moved or small) and abs(o.get("rise", 0)) > height:
            score, why = 0.0, why + f", ends {abs(o['rise']) * 100:.0f} cm {'above' if o['rise'] > 0 else 'below'} the source's"  # fmt: skip
        if release and o.get("gap", 0) >= LEFT and o.get("touch") and o.get("closed", 0) >= GRIP:
            score, why = 0.0, why + ", still held where the source lets go"
        if path is not None and (o.get("path") or 0) > path:
            score, why = 0.0, why + f", strays {o['path']:.1f} off the source's way"
        scores.append(score)
        reasons.append(f"{o['name']}: {why}")
    if not objects or len(scores) < len(objects) or not moving:
        return {"pass": False, "score": None, "reasons": reasons + ([] if moving else ["the source moves no tracked object"])}  # fmt: skip
    score = round(min(scores), 3)
    return {"pass": score >= threshold, "score": score, "reasons": reasons}


def find_rollout(delivery: Path, workspace: Path, also: tuple[Path, ...] = ()) -> tuple[Path, Path]:
    """(model, native trajectory) behind delivery/mujoco_ext1.mp4: its own folder's, else those beside a copy of the
    same video in the delivery, the workspace (layer 1's scene/native/), `also` (earlier layers' deliveries one may
    have kept) or a folder whose video the delivery's JSON names (a layer's attempt or round)."""
    video = delivery / "mujoco_ext1.mp4"
    same = lambda p: p.stat().st_size == video.stat().st_size and p.read_bytes() == video.read_bytes()
    named = {Path(p).parent for j in delivery.glob("*.json") for p in re.findall(r'"(/[^"]+/mujoco_ext1\.mp4)"', j.read_text())}  # fmt: skip
    roots = [delivery, workspace, *map(Path, also), *sorted(d for d in named if d.is_dir())]
    copies = [(delivery, video)] + [(root, p) for root in roots for p in sorted(root.rglob("mujoco_ext1.mp4")) if same(p)]  # fmt: skip
    for root, copy in copies:
        for d in (copy.parent, copy.parent / "native"):
            if (d / "trajectory.npz").exists():  # its scene: beside it or up to two folders up, else layer 1's
                near = [p for p in (d, d.parent, d.parent.parent) if p.is_relative_to(root)]
                found = [p / n for p in (*near, workspace / "scene") for n in ("scene.mjb", "scene.xml")]  # fmt: skip
                if model := next((p for p in found if p.exists()), None):
                    return model, d / "trajectory.npz"
    raise FileNotFoundError(f"no trajectory behind {video}")


def route(delivery_dir: Path, workspace: Path, masks: Path | None = None, depth: Path | None = None,
          also: tuple[Path, ...] = ()) -> dict:  # fmt: skip
    """{pass, score, reasons, objects, trajectory} for the rollout delivered in `delivery_dir` (its mujoco_ext1.mp4)
    of the clip in `workspace` (`also`: earlier layers' deliveries), against `masks` (default: the workspace's
    per-frame SAM 3 masks, masks/ or cache/masks/; the clip's best, any of its workspaces') and the depth its scene was
    built on (default: the workspace's align/depth.npz, hands', else depth/depth.npz)."""
    import mujoco

    delivery, workspace = Path(delivery_dir).resolve(), Path(workspace).resolve()
    model_path, trajectory = find_rollout(delivery, workspace, also)
    load = mujoco.MjModel.from_xml_path if model_path.suffix == ".xml" else mujoco.MjModel.from_binary_path
    model, qpos = load(str(model_path)), np.load(trajectory)["qpos"]
    cameras = None
    for d in (trajectory.parent, *trajectory.parent.parents):  # hands: the head-following camera
        if (d / "camera.npz").exists() and (d / "world.json").exists():
            world = np.array(json.loads((d / "world.json").read_text())["world_from_reference"])
            cameras = world @ np.load(d / "camera.npz")["reference_from_camera"]
            break
    if masks is None:
        masks = next((m for m in (workspace / "masks", workspace / "cache/masks") if len(list(m.glob("*.png"))) > 1), workspace / "masks")  # fmt: skip
    depth = depth or next((p for p in (workspace / "align/depth.npz", workspace / "depth/depth.npz") if p.exists()), None)  # fmt: skip
    return {**score_rollout(model, qpos, cameras, Path(masks), depth), "trajectory": str(trajectory), "depth": str(depth)}  # fmt: skip


if __name__ == "__main__":
    import tyro

    print(json.dumps(tyro.cli(route), indent=2))
