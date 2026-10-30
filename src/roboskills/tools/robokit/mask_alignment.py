# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = ["robokit[pt-cu126]>=1.0.0rc1", "torch==2.9.1", "nvdiffrast==0.3.3+pt2.9.1cu126", "numpy", "pillow", "trimesh"]
#
# [tool.uv.sources]
# torch = { index = "pytorch-cu126" }
# nvdiffrast = { index = "torch-packages-builder" }
#
# [[tool.uv.index]]
# name = "pytorch-cu126"
# url = "https://download.pytorch.org/whl/cu126"
# explicit = true
#
# [[tool.uv.index]]
# name = "torch-packages-builder"
# url = "https://jianglongye.github.io/torch_packages_builder"
# explicit = true
# ///
"""A rigid mesh's pose (and scale) fitted to a mask, frame by frame, by RoboKit's mask alignment (its hand-eye
calibrator, the mesh a one-link robot and its pose the camera extrinsic). Run as a script, this file is the model side,
in its own environment (`uv run` of its inline dependencies: the RoboKit pin of pyproject.toml, torch and a prebuilt
nvdiffrast wheel), with PYTHONSAFEPATH so this directory's modules shadow nothing."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


def fit_object_to_masks(mesh: Path, poses: Path, masks: Path, depth: Path, out: Path, label: int = 1,
                        scale: bool = False):  # fmt: skip
    """out/poses.npz: `camera_from_object` [N, 4, 4] (OpenCV camera frame) and `scale` [N] fitted from poses.npz's
    `camera_from_object` at its `frames` so the mesh, times `scale`, covers label `label` of masks/<frame>.png (other
    labels occlude it), with `iou` and `iou_start` (soft silhouette IoU after and before; NaN where the mask is empty,
    the pose kept); depth.npz gives `K`. The depth of the object's origin stays: the silhouette barely sees it."""
    Path(out).mkdir(parents=True, exist_ok=True)
    argv = [*(Path(p).resolve() for p in (mesh, poses, masks, depth, out)), label, int(scale)]
    env = dict(os.environ, PYTHONSAFEPATH="1")
    subprocess.run(["uv", "run", "--quiet", __file__, *map(str, argv)], check=True, env=env)


def fit_shape(masks: Path, depth: Path, shape: Path, out: Path, frame: int = 0):
    """out/: shape/ (mesh.ply, pose.json at `frame`) refit to its mask there by `fit_object_to_masks` with `scale`:
    mesh.ply scaled, pose.json (`camera_from_object`) and fit.json (`scale`, silhouette `iou` after and `iou_start`
    before); depth.npz gives `K`. A real2sim object's shape/ in, the same layout out."""
    import trimesh

    out.mkdir(parents=True, exist_ok=True)
    mesh, placed = shape / "mesh.ply", json.loads((shape / "pose.json").read_text())["camera_from_object"]
    np.savez(out / "start.npz", camera_from_object=[placed], frames=[frame])
    fit_object_to_masks(mesh, out / "start.npz", masks, depth, out, scale=True)
    r = {key: value[0].tolist() for key, value in np.load(out / "poses.npz").items() if key != "frames"}
    trimesh.load(mesh).apply_scale(r["scale"]).export(out / "mesh.ply")
    (out / "pose.json").write_text(json.dumps(dict(camera_from_object=r["camera_from_object"]), indent=2) + "\n")
    (out / "fit.json").write_text(json.dumps({key: r[key] for key in ("scale", "iou", "iou_start")}, indent=2) + "\n")


def fit(mesh, poses, masks, depth, out, label, scale):
    """`fit_object_to_masks`'s model side. RoboKit fits all six degrees of freedom; a pinhole camera sees the object
    scaled by s at s times its translation the same, so the fit is slid along its ray back to the starting depth, the
    object scaled with it (`scale`) or not."""
    import tempfile
    from dataclasses import replace

    import torch
    import trimesh
    from PIL import Image
    from robokit.helpers.hand_eye_calibration import HandEyeCalibrator, compute_robot_masks, presets
    from robokit.robo import Robot

    start, label, scale = np.load(poses), int(label), bool(int(scale))
    labels = [np.array(Image.open(Path(masks) / f"{f:05d}.png")) for f in start["frames"]]
    (h, w), k = labels[0].shape, torch.tensor(np.load(depth)["K"], dtype=torch.float32, device="cuda")
    with tempfile.TemporaryDirectory() as tmp:  # the mesh as a one-link robot, loaded eagerly
        trimesh.load(mesh, force="mesh").export(f"{tmp}/mesh.obj")
        geometry = f'<geometry><mesh filename="{tmp}/mesh.obj"/></geometry>'
        Path(f"{tmp}/object.urdf").write_text(f'<robot name="object"><link name="object"><visual>{geometry}</visual></link></robot>')  # fmt: skip
        robot = Robot.load(f"{tmp}/object.urdf", load_meshes=True)
    # RoboKit's RGB preset (coarse to fine, early stopping) from the start alone: no sampled candidates, no restarts
    config = replace(presets.rgb, solver=replace(presets.rgb.solver, num_candidates=[1] * 5), max_restarts=0)
    calibrator, q = HandEyeCalibrator(config, robot, k, h, w), torch.zeros(1, 0, device="cuda")

    def iou(pose: np.ndarray, target, keep) -> float:  # soft silhouette IoU outside the occluders
        a = compute_robot_masks(robot, q, k, torch.tensor(pose, dtype=torch.float32, device="cuda"), h, w)[0]
        return float((a * target * keep).sum() / ((a + target - a * target) * keep).sum().clamp(min=1))

    fitted, scales, after, before = start["camera_from_object"].copy(), np.ones(len(labels)), [], []
    for i, label_map in enumerate(labels):
        m = torch.tensor(label_map.astype(np.int64), device="cuda")
        target, occluder = (m == label).float(), ((m > 0) & (m != label)).float()
        if target.sum() < 20:  # not seen: the pose kept
            after.append(np.nan), before.append(np.nan)
            continue
        before.append(iou(fitted[i], target, 1 - occluder))
        # occluded pixels at 0.5: a silhouette edge crossing them neither gains nor loses
        initial = torch.tensor(fitted[i], dtype=torch.float32, device="cuda")
        pose = calibrator.solve(initial, target_masks=(target + occluder / 2)[None], q=q).cpu().numpy().astype(float)
        s, along = fitted[i][2, 3] / pose[2, 3], pose.copy()
        along[:3, 3] *= s  # the fit at the starting depth
        score = iou(pose if scale else along, target, 1 - occluder)
        if score > before[-1]:  # else the start kept
            fitted[i], scales[i] = along, s if scale else 1.0
        after.append(max(score, before[-1]))
    np.savez(
        Path(out) / "poses.npz",
        camera_from_object=fitted,
        scale=scales,
        frames=start["frames"],
        iou=after,
        iou_start=before,
    )
    print(json.dumps(dict(frames=len(labels), iou_start=np.nanmean(before), iou=np.nanmean(after))))


if __name__ == "__main__":
    fit(*sys.argv[1:])
