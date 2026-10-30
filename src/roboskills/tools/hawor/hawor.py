"""A hand's MANO joints and vertices per video frame with HaWoR's video steps (motion estimation, then the infiller with
the camera still, as Do as I Do runs it), boxed by its masks, its root path smoothed, in docker/hawor's image with its
weights and MANO mounted. This file is self-contained: the image runs its text.

    python -m roboskills.tools.hawor --video V --masks masks/ --depth depth/depth.npz --out hands/ [--label 2]
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


IMAGE = "ghcr.io/openrobokit/roboskills-hawor:0.3.0"
REPO = "/opt/hawor"  # the HaWoR checkout in that image; weights/ (a link to /weights/hawor) and MANO resolve under it
WEIGHTS = Path.home() / ".roboskills/weights/hawor"  # docker/weights.sh hawor; mounted read-only at /weights/hawor
MANO = "/opt/hawor/_DATA/data/mano/MANO_RIGHT.pkl", "/opt/hawor/_DATA/data_left/mano_left/MANO_LEFT.pkl"


def reconstruct_hands(video: Path, masks: Path, depth: Path, out: Path, label: int = 2, side: str = "right",
          mano: Path = Path.home() / ".robokit/body_models/mano"):  # fmt: skip
    """out/hands.npz: `joints` (N, 21, 3) and `vertices` (N, 778, 3) in the OpenCV camera frame (NaN where HaWoR's
    infiller does not reach), `seen` frame indices (the mask's), `side`, and `other_joints`, the other hand's (boxed by
    HaWoR's detector); the hand is label `label` of masks/<frame>.png, the focal length depth.npz's `K`'s. `mano` holds
    the licensed MANO_RIGHT.pkl and MANO_LEFT.pkl."""
    from roboskills.eval.runtime import contained

    if not WEIGHTS.is_dir():
        raise SystemExit(f"HaWoR's weights are not in {WEIGHTS}: fill it once with docker/weights.sh hawor")
    paths = [Path(p).resolve() for p in (video, masks, depth, out)]
    paths[3].mkdir(parents=True, exist_ok=True)
    mounts = sorted({str(p.parent) for p in paths}) + [(Path(mano).resolve() / Path(m).name, m, "ro") for m in MANO]
    mounts.append((WEIGHTS.resolve(), "/weights/hawor", "ro"))
    argv = ["python", "-c", Path(__file__).read_text(), *map(str, [*paths, label, side])]
    if (cmd := contained(IMAGE, argv, mounts)) is None:
        raise SystemExit(f"docker cannot run {IMAGE} here")
    subprocess.run(cmd, check=True)


def check_hands(path: Path) -> dict:
    """HaWoR's frames with a right hand, of all; ok with any."""
    joints = np.load(path)["joints"]
    n = int(np.isfinite(joints).all((1, 2)).sum())
    return dict(ok=n > 0, reason="" if n else "HaWoR saw no right hand", frames=n, of=len(joints))


def smooth_path(trans, rotvec, window=7, order=3, jump=3.0):
    """A hand's root path (N, 3) translations and (N, 3) axis-angles smoothed as Do as I Do smooths an object's
    (reconstruction/scripts/smooth_trajectory.py): a frame whose steps in and out both exceed `jump` times the median
    step, in translation or rotation, is interpolated from the frames around it, then a Savitzky-Golay filter (`window`,
    `order`) smooths the translation and the sign-continuous quaternion."""
    from scipy.signal import savgol_filter
    from scipy.spatial.transform import Rotation, Slerp

    def continuous(q):
        for i in range(1, len(q)):
            q[i] *= np.sign(q[i] @ q[i - 1]) or 1
        return q

    def spikes(x):
        step = np.linalg.norm(np.diff(x, axis=0), axis=1)
        big = step > max(jump * np.median(step), 1e-6)
        return set(np.flatnonzero(big[:-1] & big[1:]) + 1)

    trans, quat = np.array(trans, float), continuous(Rotation.from_rotvec(rotvec).as_quat())
    if out := sorted(spikes(trans) | spikes(np.asarray(rotvec))) if len(trans) > 2 else []:
        keep = np.setdiff1d(np.arange(len(trans)), out)
        trans[out] = np.stack([np.interp(out, keep, trans[keep, j]) for j in range(3)], 1)
        quat[out] = Slerp(keep, Rotation.from_quat(quat[keep]))(out).as_quat()
        quat = continuous(quat)
    if (width := min(window, len(trans) - 1 + len(trans) % 2)) > order:
        trans, quat = savgol_filter(trans, width, order, axis=0), savgol_filter(quat, width, order, axis=0)
    return trans, Rotation.from_quat(quat / np.linalg.norm(quat, axis=1, keepdims=True)).as_rotvec()


def run(video, masks, depth, out, label, side):
    """`reconstruct_hands`' model side, as Do as I Do runs HaWoR's video demo (modules/HaWoR/demo.py --static_camera):
    motion estimation over each run of frames with a box (the mask's for `side`, HaWoR's detector's for the other hand),
    the infiller over the frames between with the camera held still (no SLAM: the camera frame is the world), then
    `smooth_path` on each hand's root."""
    os.chdir(REPO)
    sys.path.insert(0, REPO)
    from types import SimpleNamespace

    import cv2
    import torch
    from hawor.utils.process import run_mano, run_mano_left
    from lib.pipeline.tools import detect_track
    from PIL import Image
    from scripts.scripts_test_video import hawor_video

    capture, frames = cv2.VideoCapture(video), []
    while (got := capture.read())[0]:
        frames.append(got[1])
    hand = np.stack([np.array(Image.open(p)) == int(label) for p in sorted(Path(masks).glob("*.png"))])
    seen = [i for i in range(len(frames)) if hand[i].sum() > 50]
    joints, vertices = np.full((len(frames), 21, 3), np.nan), np.full((len(frames), 778, 3), np.nan)
    if len(seen) < 2:  # motion estimation needs the hand in two frames: every frame NaN
        np.savez_compressed(Path(out) / "hands.npz", joints=joints, vertices=vertices, other_joints=joints, seen=np.array(seen), side=side)  # fmt: skip
        return
    n, right, k = len(frames), side == "right", np.load(depth)["K"]
    with tempfile.TemporaryDirectory() as folder:  # hawor_video's layout: <folder>/v.mp4's frames, tracks in <folder>/v
        seq = Path(folder) / "v"
        (seq / "extracted_images").mkdir(parents=True)
        (seq / f"tracks_0_{n}").mkdir()
        for i, frame in enumerate(frames):
            cv2.imwrite(f"{seq}/extracted_images/{i:06d}.jpg", frame)
        track = []
        for i in seen:  # one track of the mask's boxes, its handedness `side`
            rows, cols = np.nonzero(hand[i])
            box = np.array([[cols.min(), rows.min(), cols.max(), rows.max(), 1.0]])
            track.append(dict(frame=i, det=True, det_box=box, det_handedness=np.array([float(right)])))
        found = detect_track(sorted(map(str, (seq / "extracted_images").glob("*.jpg"))), thresh=0.2)[1].item()
        is_right = lambda t: np.mean([d["det_handedness"][0] > 0 for d in t]) >= 0.5  # as motion estimation sorts them
        other = {i: t for i, t in found.items() if is_right(t) != right}  # the detector's tracks of the other hand
        np.save(seq / f"tracks_0_{n}/model_tracks.npy", {-1: track} | other)
        args = SimpleNamespace(video_path=f"{folder}/v.mp4", img_focal=float(k[0, 0]),
                               checkpoint="./weights/hawor/checkpoints/hawor.ckpt", infiller_weight="./weights/hawor/checkpoints/infiller.pt")  # fmt: skip
        chunks, _ = hawor_video.hawor_motion_estimation(args, 0, n, str(seq))
        still = torch.eye(3).repeat(n, 1, 1), torch.zeros(n, 3)
        hawor_video.load_slam_cam = lambda path: (*still, *still)  # --static_camera: the camera never moves
        trans, rot, pose, betas, valid = hawor_video.hawor_infiller(args, 0, n, chunks)
    hands = []
    for h in (int(right), 1 - int(right)):  # HaWoR's hands: 0 left (mirrored back), 1 right; `side` first
        done, t, r = valid[h] > 0, trans[h].numpy(), rot[h].numpy()  # the frames it estimated or filled
        if done.any():
            t[done], r[done] = smooth_path(t[done], r[done])
        mano = (run_mano_left, run_mano)[h](torch.tensor(t[None]), torch.tensor(r[None]), pose[h : h + 1], betas=betas[h : h + 1])  # fmt: skip
        hands.append([np.where(done[:, None, None], mano[key][0].cpu().numpy()[:, :m], np.nan) for key, m in (("joints", 21), ("vertices", 778))])  # fmt: skip
    (joints, vertices), (other, _) = hands
    np.savez_compressed(Path(out) / "hands.npz", joints=joints, vertices=vertices, other_joints=other, seen=np.array(seen), side=side)  # fmt: skip


if __name__ == "__main__":
    run(*sys.argv[1:])
