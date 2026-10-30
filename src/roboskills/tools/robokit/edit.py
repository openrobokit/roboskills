"""`r2s`, the `agent` tool's edits: small, fixed edits of one workspace's tracked object (its mesh and per-frame poses),
never rerunning a vision model. Each edit saves a new state in W/agent/state (the previous one in W/agent/history/),
logs itself to W/agent/edits.jsonl, prints the checks and redraws W/agent/look.png.

    python -m roboskills.tools.robokit.edit [-w W] {look,shape,scale,rotate,move,refit,undo} ...
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np


STATE, HISTORY = "agent/state", "agent/history"
SAMPLES = 4000  # surface points per mesh for projections and fits
PASS = dict(iou=0.8, residual=0.03, jitter=0.01)  # track is ok when it meets all three


class State:
    """The workspace's current mesh (object frame) and camera_from_object per frame, with what projecting them needs."""

    def __init__(self, w: Path):
        import trimesh

        self.w, s = w, w / STATE
        if not (s / "poses.npz").exists():  # the first state: track's outputs
            s.mkdir(parents=True, exist_ok=True)
            shutil.copy(w / "align/mesh.ply", s / "mesh.ply")
            p = np.load(w / "track/poses.npz")
            np.savez(s / "poses.npz", camera_from_object=p["camera_from_object"], frames=p["frames"])
        d = np.load(w / "align/depth.npz")
        self.k, self.size, self.frames = d["K"], d["depth"].shape[1:], d["frames"]
        self.poses = np.load(s / "poses.npz")["camera_from_object"].copy()
        self.mesh = trimesh.load(s / "mesh.ply", force="mesh")
        self.ref = int(np.searchsorted(self.frames, json.loads(p.read_text())["frame"])) if (p := w / "input/reference.json").exists() else 0  # fmt: skip
        points, faces = trimesh.sample.sample_surface(self.mesh, SAMPLES, seed=0)
        colors = self.mesh.visual.vertex_colors if self.mesh.visual.kind == "vertex" else None
        self.points, self.normals = np.asarray(points), self.mesh.face_normals[faces]
        self.colors = (
            colors[self.mesh.faces[faces]].mean(1)[:, :3] if colors is not None else np.full((SAMPLES, 3), 160.0)
        )
        obb = self.mesh.bounding_box_oriented.primitive
        self.obb, self.extents = np.array(obb.transform), np.array(obb.extents)
        self.centre = self.obb[:3, 3]

    def labels(self, row: int) -> np.ndarray:
        from PIL import Image

        return np.array(Image.open(self.w / "masks" / f"{int(self.frames[row]):05d}.png"))

    def project(self, row: int, points=None, pose=None):
        pose = self.poses[row] if pose is None else pose
        x = (self.points if points is None else points) @ pose[:3, :3].T + pose[:3, 3]
        return x[:, :2] / np.maximum(x[:, 2:], 1e-3) * np.diag(self.k)[:2] + self.k[:2, 2], x[:, 2]

    def silhouette(self, row: int, pose=None) -> np.ndarray:
        import cv2

        uv, z = self.project(row, pose=pose)
        image, uv = np.zeros(self.size, np.uint8), np.round(uv).astype(int)
        keep = (z > 0.01) & (uv >= 0).all(1) & (uv < self.size[::-1]).all(1)
        image[uv[keep, 1], uv[keep, 0]] = 1
        return cv2.morphologyEx(cv2.dilate(image, np.ones((3, 3), np.uint8)), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)) > 0  # fmt: skip

    def save(self, edit: dict, mesh=None):
        """The current state into history/<n>, then this one (poses and, when given, a new mesh) as the state."""
        history = self.w / HISTORY
        n = len(list(history.glob("*"))) if history.exists() else 0
        shutil.copytree(self.w / STATE, history / f"{n:03d}")
        np.savez(self.w / STATE / "poses.npz", camera_from_object=self.poses, frames=self.frames)
        mesh is not None and mesh.export(self.w / STATE / "mesh.ply")
        with open(self.w / "agent/edits.jsonl", "a") as f:
            f.write(json.dumps(edit) + "\n")


def check_track(masks: Path, depth: Path, mesh: Path, poses: Path) -> dict:
    """track's check of `mesh` at `poses` (camera_from_object): the means over frames with the object in its mask, and
    `score`, iou less residual and jitter; ok when they meet PASS."""
    import trimesh
    from PIL import Image
    from scipy import ndimage

    d, poses = dict(np.load(depth)), np.load(poses)["camera_from_object"]  # read once
    k, size, frames = d["K"], d["depth"].shape[1:], d["frames"]
    points = trimesh.sample.sample_surface(trimesh.load(mesh), 20000, seed=0)[0]
    iou, residual = [], []
    for row, f in enumerate(frames):
        labels, z_seen = np.array(Image.open(Path(masks) / f"{f:05d}.png")), d["depth"][row].astype(float)
        x = points @ poses[row, :3, :3].T + poses[row, :3, 3]
        uv, z = np.round(x[:, :2] / x[:, 2:] * np.diag(k)[:2] + k[:2, 2]).astype(int), np.full(size, np.inf)
        keep = (x[:, 2] > 0.01) & (uv >= 0).all(1) & (uv < size[::-1]).all(1)
        np.minimum.at(z, (uv[keep, 1], uv[keep, 0]), x[keep, 2])
        z = ndimage.grey_erosion(z, 3)  # the splatted samples' holes filled: the mesh's depth image
        drawn, seen, hand = np.isfinite(z), labels == 1, labels == 2
        if seen.sum() < 50:
            continue
        iou.append((drawn & seen).sum() / ((drawn | seen) & ~hand).sum())
        both = drawn & seen & d["valid"][row] & (z_seen > 0)
        residual.append(np.median(np.abs(z[both] / z_seen[both] - 1)) if both.sum() >= 50 else 1.0)
    t = poses[:, :3, 3]
    jitter = np.linalg.norm(t[2:] - 2 * t[1:-1] + t[:-2], axis=1) / t[1:-1, 2] if len(t) > 2 else np.zeros(1)
    c = dict(iou=float(np.mean(iou or [0])), residual=float(np.mean(residual or [1])), jitter=float(np.mean(jitter)))
    failed = [key for key, limit in PASS.items() if (c[key] < limit if key == "iou" else c[key] > limit)]
    missed = ", ".join(f"{k} {c[k]:.3g} (limit {PASS[k]})" for k in failed)
    return dict(
        c,
        score=c["iou"] - c["residual"] - c["jitter"],
        failed=failed,
        ok=not failed,
        reason=f"track missed {missed}" if failed else "",
    )


def checks(w: Path) -> dict:
    """Track's check (its status's numbers) of the state."""
    return check_track(w / "masks", w / "align/depth.npz", w / STATE / "mesh.ply", w / STATE / "poses.npz")


def report(w: Path, frames=None, render=False, out=None) -> str:
    """The checks, the mesh's size and depth and the overlay image's path."""
    s, c = State(w), checks(w)
    image = look(s, frames, render, out)
    centre = s.poses[s.ref, :3, :3] @ s.centre + s.poses[s.ref, :3, 3]
    lines = [
        f"checks: iou {c['iou']:.3f} (pass >= 0.8), residual {c['residual']:.3f} (<= 0.03), jitter {c['jitter']:.4f} (<= 0.01)",
        f"size: {' x '.join(f'{e:.1f}' for e in sorted(100 * s.extents, reverse=True))} cm (the mesh's box extents); "
        f"centre at reference frame {int(s.frames[s.ref])}: {np.round(centre, 3).tolist()} m (camera: x right, y down, z forward)",
        f"image: {image}",
    ]  # fmt: skip
    return "\n".join(lines)


def look(s: State, frames=None, render=False, out=None) -> Path:
    """agent/look.png: per frame a crop of the source with SAM 3's object mask outlined red, the mesh's silhouette green and
    the hand mask tinted blue (`render`: the source beside the mesh drawn in its colours over the darkened source)."""
    import cv2

    rows = list(np.linspace(0, len(s.frames) - 1, 4).round().astype(int))
    rows[int(np.argmin(np.abs(np.array(rows) - s.ref)))] = s.ref  # the reference frame and three more
    rows = [int(np.searchsorted(s.frames, f)) for f in frames] if frames else sorted(set(rows))
    video, images, want = cv2.VideoCapture(str(s.w / "input/video.mp4")), {}, {int(s.frames[r]) for r in rows}
    for i in range(int(max(want)) + 1):
        ok, image = video.read()
        if not ok:
            break
        if i in want:
            images[i] = image
    tiles = []
    for row in rows:
        f, labels, sil = int(s.frames[row]), s.labels(row), s.silhouette(row)
        image = images.get(f, np.zeros((*s.size, 3), np.uint8)).copy()
        obj, hand = labels == 1, labels == 2
        iou = (obj & sil).sum() / max(1, ((obj | sil) & ~hand).sum())
        ys, xs = np.nonzero(obj | sil)
        if len(ys) == 0:
            ys, xs = np.array([0, s.size[0] - 1]), np.array([0, s.size[1] - 1])
        cy, cx, half = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2, max(np.ptp(ys), np.ptp(xs)) * 0.65 + 20
        box = [
            int(max(0, cx - half)),
            int(max(0, cy - half)),
            int(min(s.size[1], cx + half)),
            int(min(s.size[0], cy + half)),
        ]
        overlay = image.copy()
        overlay[hand] = (0.6 * overlay[hand] + 0.4 * np.array([255, 80, 0])).astype(np.uint8)
        for mask, colour in ((obj, (0, 0, 255)), (sil, (0, 255, 0))):
            contours = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
            cv2.drawContours(overlay, contours, -1, colour, 1)
        panels = [overlay]
        if render:
            panels = [image, draw(s, row, image)]
        crops = [cv2.resize(p[box[1] : box[3], box[0] : box[2]], (256, 256)) for p in panels]
        for crop in crops:
            cv2.putText(crop, f"f{f}" + ("" if render else f" iou {iou:.2f}"), (4, 16), 0, 0.5, (255, 255, 255), 1)
        tiles.append(np.hstack(crops))
    columns = 1 if render else 2
    tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % columns)
    grid = np.vstack([np.hstack(tiles[i : i + columns]) for i in range(0, len(tiles), columns)])
    path = s.w / (out or "agent/look.png")
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), grid)
    return path


def draw(s: State, row: int, source: np.ndarray) -> np.ndarray:
    """The mesh at `row` in its colours, lit by its normals, nearest drawn last, over the darkened source frame (HaWoR's
    hand is left out: often misplaced, it would be judged instead of the object)."""
    pose = s.poses[row]
    uv, z = s.project(row)
    light = 0.45 + 0.55 * np.abs((s.normals @ pose[:3, :3].T)[:, 2])
    colour = (s.colors[:, ::-1] * light[:, None] * 1.2 + 25).clip(0, 255)  # BGR
    image = (source * 0.45).astype(np.uint8)
    order = np.argsort(-z)
    for dy in (0, 1, 2):
        for dx in (0, 1, 2):
            p = np.round(uv[order]).astype(int) + [dx - 1, dy - 1]
            keep = (z[order] > 0.01) & (p >= 0).all(1) & (p < s.size[::-1]).all(1)
            image[p[keep, 1], p[keep, 0]] = colour[order][keep]
    return image


def ray_factor(s: State, after: np.ndarray) -> float:
    """The slide along the camera ray (a factor on every frame's distance) at which the new mesh's silhouette best
    matches SAM 3's object mask (hand pixels ignored), over the five frames with the most object pixels."""
    before, s.points = s.points, after
    labels = {r: s.labels(r) for r in range(len(s.frames))}
    rows = sorted(labels, key=lambda r: -(labels[r] == 1).sum())[:5]
    centre, poses, best = s.poses[:, :3, :3] @ s.centre + s.poses[:, :3, 3], s.poses.copy(), (0, 1.0)
    for factor in np.arange(0.6, 1.62, 0.02):
        ious = []
        for r in rows:
            pose = poses[r].copy()
            pose[:3, 3] += (factor - 1) * centre[r]
            sil, obj, hand = s.silhouette(r, pose), labels[r] == 1, labels[r] == 2
            ious.append((sil & obj).sum() / max(1, ((sil | obj) & ~hand).sum()))
        best = max(best, (float(np.mean(ious)), float(factor)))
    s.points = before
    return best[1]


def resize(s: State, mesh, keep_depth: bool) -> dict:
    """Swap in `mesh` (object frame, about the old mesh's centre), its visible face kept where it was, then (unless
    `keep_depth`) slide it along its ray to best fit the mask."""
    import trimesh

    new_obb = mesh.bounding_box_oriented.primitive
    pose, view = s.poses[s.ref], s.poses[s.ref, :3, :3] @ s.centre + s.poses[s.ref, :3, 3]
    view = pose[:3, :3].T @ (view / np.linalg.norm(view))  # the viewing ray in the object frame
    axes, grow = s.obb[:3, :3], np.sort(new_obb.extents) - np.sort(s.extents)  # growth per old axis, by rank
    grow = grow[np.argsort(np.argsort(s.extents))]
    shift = sum(g / 2 * (a @ view) * a for g, a in zip(grow, axes.T))  # thickening goes behind the visible face
    mesh.apply_translation(shift)
    after = np.asarray(trimesh.sample.sample_surface(mesh, SAMPLES, seed=0)[0])
    factor = 1.0 if keep_depth else ray_factor(s, after)
    centre = s.poses[:, :3, :3] @ s.centre + s.poses[:, :3, 3]
    s.poses[:, :3, 3] += (factor - 1) * centre
    return dict(ray_factor=round(factor, 4), shift_cm=np.round(100 * shift, 2).tolist())


def primitive(s: State, kind: str, dims) -> "trimesh.Trimesh":
    """A box (`dims` its three extents) or a cylinder (diameter, height) on the old mesh's box axes, matched by rank (a
    cylinder's axis is the old box's most distinct one), or the old mesh stretched to the extents ('mesh')."""
    import trimesh

    e, rank = s.extents, np.argsort(np.argsort(s.extents))
    if kind == "cylinder":
        d, h = dims
        axis = int(np.argmax([abs(e[i] - np.mean(np.delete(e, i))) for i in range(3)]))
        target = np.full(3, d)
        target[axis] = h
    else:
        target = np.sort(dims)[rank]
    if kind == "mesh":
        mesh = s.mesh.copy()
        stretch = s.obb @ np.diag([*target / e, 1]) @ np.linalg.inv(s.obb)
        return mesh.apply_transform(stretch)
    if kind == "box":
        mesh = trimesh.creation.box(extents=target)
    else:
        mesh = trimesh.creation.cylinder(radius=target[(axis + 1) % 3] / 2, height=target[axis], sections=48)
        turn = np.eye(4)
        turn[:3, :3] = [[[0, 0, 1], [0, 1, 0], [-1, 0, 0]], [[1, 0, 0], [0, 0, 1], [0, -1, 0]], np.eye(3)][
            axis
        ]  # z onto it
        mesh.apply_transform(turn)
    mesh.apply_transform(s.obb)
    mesh.visual.vertex_colors = np.r_[s.colors.mean(0), 255].astype(np.uint8)
    return mesh


def refit(s: State, window: int = 5, fix_depth: bool = True, frames=None, weight: float = 0.3) -> dict:
    """Each frame's pose fitted to its SAM 3 object mask's outline (two-sided: the silhouette inside object or hand
    pixels, the outline away from the hand covered), out from the reference frame, each from its neighbour's correction,
    held to it with `weight`; then a `window`-frame median. `fix_depth` keeps each frame's depth (z): the silhouette
    barely sees it, and MoGe's focal length, short by about a tenth, would bias it."""
    import cv2
    from scipy import ndimage, optimize, spatial
    from scipy.spatial.transform import Rotation

    start, fitted, ious = s.poses.copy(), s.poses.copy(), []
    points = s.points[:: max(1, len(s.points) // 1500)]
    rows = [r for r in range(len(s.frames)) if frames is None or s.frames[r] in frames]
    order = [r for r in rows if r >= s.ref] + [r for r in rows[::-1] if r < s.ref]
    correction, last = np.eye(4), None
    for row in order:
        if last is not None and (row < s.ref) != (last < s.ref):
            correction = fitted[s.ref] @ np.linalg.inv(
                start[s.ref]
            )  # the backward pass starts from the reference again
        inits = [correction @ start[row]] + ([fitted[last]] if last is not None else [])  # tracked, or held still
        labels = s.labels(row)
        obj, hand = labels == 1, labels == 2
        contours = cv2.findContours(obj.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
        q = np.concatenate([c[:, 0] for c in contours]) if contours else np.zeros((0, 2), int)
        near = cv2.dilate(hand.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        h, w = s.size
        q = (
            q[~near[q[:, 1], q[:, 0]] & (q[:, 0] > 2) & (q[:, 1] > 2) & (q[:, 0] < w - 3) & (q[:, 1] < h - 3)]
            if len(q)
            else q
        )
        if obj.sum() < 150 or len(q) < 30:
            fitted[row], last = inits[0], row
            continue
        q = q[:: max(1, len(q) // 300)].astype(float)
        outside = cv2.distanceTransform((~(obj | hand)).astype(np.uint8), cv2.DIST_L2, 5)

        def pose_of(x):
            centre = init[:3, :3] @ s.centre + init[:3, 3]
            r = Rotation.from_rotvec(x[:3]).as_matrix()
            p = init.copy()
            p[:3, :3], p[:3, 3] = (
                r @ init[:3, :3],
                r @ (init[:3, 3] - centre) + centre + [x[3], x[4], 0 if fix_depth else x[5]],
            )
            return p

        def residuals(x):  # pixels, clipped at 30, as RMS per term; the pull in px per rad and per m (depth weaker)
            uv, _ = s.project(row, points, pose_of(x))
            r1 = ndimage.map_coordinates(outside, [uv[:, 1], uv[:, 0]], order=1, mode="nearest").clip(0, 30) / np.sqrt(
                len(uv)
            )
            r2 = spatial.cKDTree(uv).query(q)[0].clip(0, 30) / np.sqrt(len(q))
            return np.r_[r1, r2, weight * 20 * x[:3], weight * 300 * x[3:5], weight * 60 * x[5:]]

        tries = []
        for init in inits:
            fit = optimize.least_squares(residuals, np.zeros(6), diff_step=3e-3, max_nfev=40)
            tries.append((fit.cost, pose_of(fit.x)))
        fitted[row], last = min(tries, key=lambda t: t[0])[1], row
        correction = fitted[row] @ np.linalg.inv(start[row])
        sil = s.silhouette(row, fitted[row])
        ious.append((obj & sil).sum() / max(1, ((obj | sil) & ~hand).sum()))
    if window > 1:
        r = Rotation.from_matrix(fitted[:, :3, :3]).as_quat()
        r *= np.cumprod(np.r_[1, np.where((r[1:] * r[:-1]).sum(1) < 0, -1, 1)])[:, None]
        smooth = ndimage.median_filter(np.c_[r, fitted[:, :3, 3]], (window, 1), mode="nearest")
        fitted[:, :3, :3], fitted[:, :3, 3] = Rotation.from_quat(smooth[:, :4]).as_matrix(), smooth[:, 4:]
    s.poses = fitted
    return dict(fitted_frames=len(ious), mean_fit_iou=round(float(np.mean(ious)), 3) if ious else None)


def rotation(s: State, axis: str, degrees: float):
    """Turn the object about its centre by `degrees` about camera axis `axis` at the reference frame; every frame turns
    by the same object-frame rotation, so the tracked motion is kept."""
    from scipy.spatial.transform import Rotation

    r = s.poses[s.ref, :3, :3]
    turn = np.eye(4)
    turn[:3, :3] = r.T @ Rotation.from_euler(axis, degrees, degrees=True).as_matrix() @ r
    about = np.eye(4)
    about[:3, 3] = s.centre
    s.poses = s.poses @ about @ turn @ np.linalg.inv(about)


def main(argv=None):
    p = argparse.ArgumentParser(prog="r2s", description="Edit the object's reconstruction (mesh, pose per frame). Every edit prints the checks and redraws agent/look.png.")  # fmt: skip
    p.add_argument("-w", "--workspace", type=Path, default=Path.cwd(), help=argparse.SUPPRESS)
    sub = p.add_subparsers(dest="command", required=True)
    c = sub.add_parser("look", help="print the checks, size and depth; draw agent/look.png")
    c.add_argument("--frames", type=int, nargs="+", help="frames to draw (default: the reference and 3 more)")
    c.add_argument("--render", action="store_true", help="source beside the mesh drawn in its colours")
    c.add_argument("--out", help=argparse.SUPPRESS)
    c = sub.add_parser(
        "shape",
        help="set the object's real size (centre's depth kept): box A B C, cylinder D H, or mesh A B C (the mesh stretched)",
    )
    c.add_argument("kind", choices=("box", "cylinder", "mesh"))
    c.add_argument(
        "dims", type=float, nargs="+", help="cm; box/mesh: three extents (any order), cylinder: diameter height"
    )
    c = sub.add_parser(
        "scale", help="scale the mesh by S and its distance from the camera by S: the same image, S times deeper"
    )
    c.add_argument("factor", type=float)
    c.add_argument("--keep-depth", action="store_true", help="scale the mesh only (it changes in the image)")
    c = sub.add_parser("rotate", help="turn the object about its centre, about a camera axis at the reference frame")
    c.add_argument("axis", choices=("x", "y", "z"), help="x right, y down, z forward (into the image)")
    c.add_argument("degrees", type=float)
    c = sub.add_parser("move", help="shift the object in every frame by DX DY DZ cm (camera axes)")
    c.add_argument("offset", type=float, nargs=3)
    c = sub.add_parser(
        "refit", help="fit each frame's pose (depth kept) to the SAM 3 mask outline (hand pixels excluded), then smooth"
    )
    c.add_argument("--window", type=int, default=5, help="median smoothing window in frames (default 5)")
    c.add_argument("--frames", type=int, nargs=2, metavar=("FIRST", "LAST"), help="only these frames")
    c.add_argument("--weight", type=float, default=0.3, help="pull toward the current poses, 0.05 to 1 (default 0.3)")
    sub.add_parser("undo", help="revert the last edit")
    a = p.parse_args(argv)
    w = a.workspace.resolve()
    if a.command == "undo":
        history = sorted((w / HISTORY).glob("*")) if (w / HISTORY).exists() else []
        if not history:
            sys.exit("nothing to undo")
        shutil.rmtree(w / STATE)
        shutil.move(history[-1], w / STATE)
        with open(w / "agent/edits.jsonl", "a") as f:
            f.write(json.dumps(dict(command="undo")) + "\n")
        print(report(w))
        return
    if a.command == "look":
        print(report(w, a.frames, a.render, a.out))
        return
    s, mesh, edit = State(w), None, dict(command=a.command, **{k: v for k, v in vars(a).items() if k not in ("workspace", "command")})  # fmt: skip
    if a.command == "shape":
        if len(a.dims) != (2 if a.kind == "cylinder" else 3):
            sys.exit("box/mesh take three extents, cylinder its diameter and height (cm)")
        mesh = primitive(s, a.kind, np.array(a.dims) / 100)
        edit |= resize(s, mesh, keep_depth=True)  # depth by silhouette: biased near by MoGe's short focal length
    elif a.command == "scale":
        mesh = s.mesh.copy().apply_translation(-s.centre).apply_scale(a.factor).apply_translation(s.centre)
        if not a.keep_depth:
            centre = s.poses[:, :3, :3] @ s.centre + s.poses[:, :3, 3]
            s.poses[:, :3, 3] += (a.factor - 1) * centre
    elif a.command == "rotate":
        rotation(s, a.axis, a.degrees)
    elif a.command == "move":
        s.poses[:, :3, 3] += np.array(a.offset) / 100
    elif a.command == "refit":  # unpulled (weight 0), single silhouettes flip and overfit
        frames = None if a.frames is None else range(a.frames[0], a.frames[1] + 1)
        edit |= refit(s, a.window, True, frames, float(np.clip(a.weight, 0.05, 1.0)))
    s.save(edit, mesh)
    extra = {k: v for k, v in edit.items() if k in ("ray_factor", "fitted_frames", "mean_fit_iou")}
    print((json.dumps(extra) + "\n" if extra else "") + report(w))


if __name__ == "__main__":
    main()
