"""One ARCTIC sequence's ground truth as a real2sim track: both hands and the articulated object, from one camera.

    python -m roboskills.eval.arctic.read ROOT s01/laptop_use_01 --view 4 --frames 80 175 --out DIR [--images DIR]

ROOT holds ARCTIC's `raw_seqs` and `meta` (licensed: https://arctic.is.tue.mpg.de). Writes DIR/track.npz in the
format `fit` writes, in allocentric camera VIEW's frame (OpenCV), metres: MANO keypoints of the left and right hand
(`hand_joints [frames, 2, 21, 3]`, OpenPose order, RoboKit's MANO from `--mano`), the object's mesh with its top
part on ARCTIC's hinge (`part`, `hinge`, `angles`: the top turns about the template's -z axis) and pose, gravity,
K and image size. With `--images` (the sequence's images zip unpacked: `VIEW/NNNNN.jpg`; mocap frame i is image
i + ioi_offset) it writes DIR/video.mp4 at 30 fps, scaled by `--scale` like K.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def mano_joints(params, side, mano):
    """Keypoints [frames, 21, 3] (OpenPose order) of ARCTIC's MANO fit, world frame, metres."""
    import torch
    from robokit.smplx import load_mano
    from robokit.smplx.torch_lbs import mano_forward_pca_mm

    spec = load_mano(Path(mano) / f"MANO_{side.upper()}.npz", side=side)
    pose = torch.tensor(np.concatenate([params["rot"], params["pose"]], 1), dtype=torch.float32)
    betas = torch.tensor(np.broadcast_to(params["shape"], (len(pose), 10)), dtype=torch.float32)
    transl = torch.tensor(params["trans"], dtype=torch.float32)
    _, joints = mano_forward_pca_mm(spec, pose, betas, transl, flat_hand_mean=False, use_pca=False)
    return joints.numpy() / 1000


def main():
    from PIL import Image
    from scipy.spatial.transform import Rotation

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("root", type=Path)
    p.add_argument("sequence", help="SUBJECT/SEQUENCE, e.g. s01/laptop_use_01")
    p.add_argument("--view", type=int, required=True, help="allocentric camera, 1-8")
    p.add_argument("--frames", type=int, nargs=2, required=True, help="first and last+1 mocap frame")
    p.add_argument("--mano", type=Path, default=Path.home() / ".robokit/body_models/mano", help="MANO_{RIGHT,LEFT}.npz")
    p.add_argument("--images", type=Path)
    p.add_argument("--scale", type=float, default=0.5, help="video and K scale")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    subject, name = a.sequence.split("/")
    frames = slice(*a.frames)
    misc = json.loads((a.root / "meta/misc.json").read_text())[subject]
    world2cam = np.array(misc["world2cam"][a.view - 1])
    K = np.array(misc["intris_mat"][a.view - 1]) * [[a.scale], [a.scale], [1]]
    size = np.round(np.array(misc["image_size"][a.view]) * a.scale).astype(int)
    seq = a.root / "raw_seqs" / subject / name
    mano = np.load(f"{seq}.mano.npy", allow_pickle=True).item()
    hands = np.stack([mano_joints(mano[side], side, a.mano)[frames] for side in ("left", "right")], 1)
    hands = hands @ world2cam[:3, :3].T + world2cam[:3, 3]
    obj = np.load(f"{seq}.object.npy")[frames]  # articulation, axis-angle, translation in mm
    template = a.root / "meta/object_vtemplates" / name.split("_")[0]
    lines = [line.split() for line in (template / "mesh.obj").read_text().splitlines()]
    vertices = np.array([line[1:4] for line in lines if line[:1] == ["v"]], float) / 1000
    faces = np.array([[int(i.split("/")[0]) - 1 for i in line[1:4]] for line in lines if line[:1] == ["f"]])
    poses = np.tile(np.eye(4), (len(obj), 1, 1))
    poses[:, :3, :3] = world2cam[:3, :3] @ Rotation.from_rotvec(obj[:, 1:4]).as_matrix()
    poses[:, :3, 3] = obj[:, 4:7] / 1000 @ world2cam[:3, :3].T + world2cam[:3, 3]
    a.out.mkdir(parents=True, exist_ok=True)
    texture = np.asarray(Image.open(template / "material.jpg").convert("RGB"))
    np.savez_compressed(
        a.out / "track.npz",
        poses=poses,
        vertices=vertices,
        faces=faces,
        colors=texture.reshape(-1, 3)[:: max(texture.size // 30000, 1)],
        part=np.array(json.loads((template / "parts.json").read_text()), bool),
        hinge=np.array([0, 0, 0, 0, 0, -1.0]),
        angles=obj[:, 0],
        hand_joints=hands,
        hand=np.array(["left", "right"]),
        gravity=-world2cam[:3, 2],  # ARCTIC's world is z-up
        rests=True,
        K=K,
        size=size,
        frame=0,
    )
    if a.images:
        import cv2

        folder = a.images / str(a.view)
        video = cv2.VideoWriter(str(a.out / "video.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, tuple(map(int, size)))
        for i in range(*a.frames):
            video.write(
                cv2.resize(cv2.imread(str(folder / f"{i + misc['ioi_offset']:05d}.jpg")), tuple(map(int, size)))
            )
        video.release()
    print(
        json.dumps(
            {"frames": len(obj), "size": size.tolist(), "angle": [float(obj[:, 0].min()), float(obj[:, 0].max())]}
        )
    )


if __name__ == "__main__":
    main()
