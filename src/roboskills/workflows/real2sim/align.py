"""Metric scale for MoGe's depth, frame by frame: a DROID robot's depth rendered from its calibrated camera at its joints
(input/robot.npz, camera.json), or HaWoR's MANO hand of known size, over MoGe's at the actor's own pixels (mask label 2)
rescales each frame's depth (the reference frame's scale where it is not seen; `clip`: all by their median), and that scale
moves the SAM 3D object along its camera ray, keeping SAM 3D's size (`scale_mesh`: scaled too). No actor seen: scale 1."""

import json
import shutil
from pathlib import Path

import numpy as np

from roboskills.eval.droid.read import render_robot


SCALES = (0.9, 1.5)  # hand scales trusted to rescale SAM 3D's mesh: MoGe's depth runs short by up to half


def align(masks: Path, depth: Path, shape: Path, hands: Path | None, out: Path, frame: int = 0, robot: Path | None = None,
          camera: Path | None = None, clip: bool = False, scale_mesh: bool = False):  # fmt: skip
    """align/: depth.npz rescaled per frame, shape/'s pose.json moved and mesh.ply unchanged, and align.json (`scale` and
    the actor's `pixels` at the reference frame, `scales` per frame), the actor HaWoR's `hands` or DROID's `robot`."""
    import trimesh
    from PIL import Image

    out.mkdir(parents=True, exist_ok=True)
    k, size = (d := dict(np.load(depth)))["K"], d["depth"].shape[1:]  # the robot's depth per frame, or the hand's vertices  # fmt: skip
    actor = render_robot(robot, camera, k, size, out) if robot else np.load(hands)["vertices"]

    def ratio(row: int):  # the actor's depth over MoGe's at its visible mask pixels, and how many; None under 20
        f, z, valid = int(d["frames"][row]), d["depth"][row].astype(float), d["valid"][row]
        near, labels = actor[f] if robot else np.full(size, np.inf), np.array(Image.open(masks / f"{f:05d}.png"))
        if not robot and np.isfinite(hand := actor[f]).all():  # the hand's nearest vertex per pixel
            uv = np.round(hand[:, :2] / hand[:, 2:] * np.diag(k)[:2] + k[:2, 2]).astype(int)
            order = np.argsort(hand[:, 2])
            front = order[np.unique(uv[order], axis=0, return_index=True)[1]]
            front = front[(uv[front] >= 0).all(1) & (uv[front] < size[::-1]).all(1)]
            near[uv[front, 1], uv[front, 0]] = hand[front, 2]
        keep = (labels == 2) & valid & (z > 0) & np.isfinite(near)
        if keep.sum() < 20:
            return None, 0
        return float(np.median(near[keep] / z[keep])), int(keep.sum())

    found = [ratio(row) for row in range(len(d["frames"]))]
    scale, used = found[int(np.searchsorted(d["frames"], frame))]
    scale = float(np.median(seen)) if clip and (seen := [s for s, _ in found if s]) else scale or 1.0
    scales = np.array([scale if clip else s or scale for s, _ in found])
    d["depth"] = (d["depth"].astype(float) * scales[:, None, None]).astype(np.float16)
    np.savez_compressed(out / "depth.npz", **d)
    placed = np.array((pose := json.loads((shape / "pose.json").read_text()))["camera_from_object"])
    placed[:3, 3] *= scale
    (out / "pose.json").write_text(json.dumps(dict(pose, camera_from_object=placed.tolist()), indent=2) + "\n")
    trimesh.load(shape / "mesh.ply").apply_scale(scale).export(out / "mesh.ply") if scale_mesh else shutil.copy(shape / "mesh.ply", out / "mesh.ply")  # fmt: skip
    (out / "align.json").write_text(json.dumps(dict(scale=scale, pixels=used, scales=scales.round(4).tolist())) + "\n")


def check_scale(path: Path) -> dict:
    """align's hand scale (align.json); ok inside SCALES."""
    s = json.loads(Path(path).read_text())["scale"]
    inside = SCALES[0] <= s <= SCALES[1]
    return dict(ok=inside, reason="" if inside else f"align's hand scale {s:.3g} lies outside {SCALES}", scale=s)
