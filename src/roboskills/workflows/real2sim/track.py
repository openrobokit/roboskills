"""The object's pose at every frame, each in that frame's camera (an egocentric camera moves): the object's mesh
(shape/) stays rigid and is fit by ICP to its nearest `trim` % of the object's depth points (mask label 1), frame by frame
out from the reference frame, each from its neighbour's pose (kept under 50 points), then smoothed by a `window`-frame
median. With `refit`, each pose is then refit to its frame's SAM 3 mask (RoboKit's mask alignment; the hand occludes)."""

import json
from pathlib import Path

import numpy as np

from roboskills.tools.robokit.mask_alignment import fit_object_to_masks


def track(masks: Path, depth: Path, shape: Path, out: Path, frame: int = 0, refit: bool = True, window: int = 21, trim: float = 80):  # fmt: skip
    """track/poses.npz: `camera_from_object` per frame (the reference frame's from shape/pose.json), `seen` (depth
    points), `frames` and, with `refit`, silhouette `iou` and `iou_start` (after and before it); icp.npz: before it."""
    import trimesh
    from PIL import Image
    from scipy import ndimage, spatial

    out.mkdir(parents=True, exist_ok=True)
    k, depths, valid = (d := np.load(depth))["K"], d["depth"].astype(float), d["valid"] & (d["depth"] > 0)
    labels = np.stack([np.array(Image.open(masks / f"{f:05d}.png")) for f in d["frames"]])
    ref, (v, u) = int(np.searchsorted(d["frames"], frame)), np.indices(depths.shape[1:])
    mesh = spatial.cKDTree(trimesh.sample.sample_surface(trimesh.load(shape / "mesh.ply"), 3000, seed=0)[0])
    placed = np.array(json.loads((shape / "pose.json").read_text())["camera_from_object"])

    def icp(row: int, pose: np.ndarray):  # point-to-point ICP of the row's object points onto the mesh, from `pose`
        keep, z = valid[row] & (labels[row] == 1), depths[row]
        if keep.sum() < 50:
            return pose, False
        x = np.stack([(u - k[0, 2]) / k[0, 0] * z, (v - k[1, 2]) / k[1, 1] * z, z], -1)[keep]
        x, inverse = x[np.random.default_rng(0).permutation(len(x))[:2000]], np.linalg.inv(pose)  # object from camera
        for _ in range(30):
            distance, index = mesh.query(trimesh.transform_points(x, inverse))
            near = distance <= np.percentile(distance, trim)  # drop the worst pairs: mask edges, occlusion
            inverse = trimesh.registration.procrustes(x[near], mesh.data[index[near]], reflection=False, scale=False, return_cost=False)  # fmt: skip
        return np.linalg.inv(inverse), True

    poses, seen = np.repeat(placed[None], len(depths), 0), np.zeros(len(depths), bool)
    poses[ref], seen[ref] = icp(ref, placed)
    for row in [*range(ref + 1, len(poses)), *range(ref - 1, -1, -1)]:  # out from the reference, from each neighbour
        poses[row], seen[row] = icp(row, poses[row - 1 if row > ref else row + 1])
    poses = poses @ np.linalg.inv(poses[ref]) @ placed  # the fitted motion, applied to the reference pose
    q = (r := spatial.transform.Rotation.from_matrix(poses[:, :3, :3]).as_quat()) * np.cumprod(np.r_[1, np.where((r[1:] * r[:-1]).sum(1) < 0, -1, 1)])[:, None]  # quaternions on one hemisphere  # fmt: skip
    s = ndimage.median_filter(np.c_[q, poses[:, :3, 3]], (window, 1), mode="nearest")  # depth, ICP jitter; keeps ramps
    poses[:, :3, :3], poses[:, :3, 3] = spatial.transform.Rotation.from_quat(s[:, :4]).as_matrix(), s[:, 4:]
    np.savez(out / "icp.npz", camera_from_object=poses, frames=d["frames"])
    refit and fit_object_to_masks(shape / "mesh.ply", out / "icp.npz", masks, depth, out)
    np.savez(out / "poses.npz", **np.load(out / ("poses.npz" if refit else "icp.npz")), seen=seen)
