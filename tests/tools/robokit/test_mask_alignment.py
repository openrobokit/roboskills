"""The mask fit recovers a known pose and size."""

import json
import os
import subprocess
import sys

import numpy as np
import pytest


@pytest.mark.gpu
def test_a_perturbed_occluded_box_returns_to_its_mask(tmp_path):
    import cv2
    import trimesh
    from PIL import Image
    from scipy.spatial.transform import Rotation

    box, k = trimesh.creation.box([0.10, 0.06, 0.04]), np.array([[600.0, 0, 320], [0, 600, 240], [0, 0, 1]])
    truth = np.eye(4)
    truth[:3, :3], truth[:3, 3] = Rotation.from_euler("xyz", [0.4, -0.3, 0.2]).as_matrix(), [0.02, -0.01, 0.5]
    x = trimesh.transform_points(box.vertices, truth)
    uv, labels = np.round(x[:, :2] / x[:, 2:] * 600 + [320, 240]).astype(np.int32), np.zeros((480, 640), np.uint8)
    for face in box.faces:
        cv2.fillConvexPoly(labels, uv[face], 1)
    labels[230:, 330:380] = 2  # a hand in front
    (tmp_path / "masks").mkdir()
    Image.fromarray(labels).save(tmp_path / "masks/00000.png")
    start = truth.copy()
    start[:3, :3], start[:3, 3] = (
        Rotation.from_rotvec([0.1, 0.12, -0.08]).as_matrix() @ truth[:3, :3],
        [0.028, -0.016, 0.5],
    )
    np.savez(tmp_path / "start.npz", camera_from_object=start[None], frames=[0])
    np.savez(tmp_path / "depth.npz", K=k)
    box.apply_scale(0.8).export(tmp_path / "mesh.ply")
    paths = [tmp_path / n for n in ("mesh.ply", "start.npz", "masks", "depth.npz", "fit")]
    code = f"from roboskills.tools.robokit.mask_alignment import fit_object_to_masks; fit_object_to_masks(*{list(map(str, paths))}, scale=True)"
    subprocess.run([sys.executable, "-c", code], check=True, timeout=1200, env=os.environ)
    fit = np.load(tmp_path / "fit/poses.npz")
    turn = Rotation.from_matrix(fit["camera_from_object"][0, :3, :3] @ truth[:3, :3].T).magnitude()
    assert fit["iou"][0] > 0.95 > fit["iou_start"][0] and np.degrees(turn) < 3
    assert (
        np.abs(fit["camera_from_object"][0, :3, 3] - truth[:3, 3]).max() < 0.002 and abs(fit["scale"][0] - 1.25) < 0.03
    )
    assert json.dumps(fit["iou"].tolist())
