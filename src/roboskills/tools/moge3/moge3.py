# /// script
# requires-python = ">=3.10,<3.14"  # triton fails on 3.14 (ast.Num)
# dependencies = [
#   "moge @ git+https://github.com/microsoft/MoGe.git@74fbce054ebed49800de42d0ad0e83495065719a",
#   "utils3d-moge @ git+https://github.com/EasternJournalist/utils3d-moge.git@62f09d58509485564e24d5d9f6aac9ee9ebc0c37",
#   "flex-gemm @ git+https://github.com/JeffreyXiang/FlexGEMM.git@b2fadb29d41846c7981ade6801ffc689fae119cf",
#   "torch==2.9.1", "torchvision==0.24.1", "opencv-python-headless", "numpy",
# ]
# ///
"""MoGe-3 depth and intrinsics for video frames, in its own environment (`uv run --no-sources` of this file: MoGe's
own sources pin another torch index; the weights download once).

    python -m roboskills.tools.moge3 --video V --out depth/ [--frame 0] [--frames 0 12 ...] [--fov DEG]
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np


WEIGHTS = "Ruicheng/moge-3-vitl"


def estimate_depth(video: Path, out: Path, frame: int = 0, frames: list[int] | None = None, fov: float | None = None):
    """out/depth.npz: `depth` [N, H, W] float16 (metres, estimated; 0 where invalid), `valid` [N, H, W] bool, `frames`
    [N] (source frame indices, ascending: `frames` and `frame`, default every one) and `K` (3, 3) in source pixels;
    every frame reuses the reference `frame`'s field of view, which `fov` (horizontal, degrees) fixes when known."""
    Path(out).mkdir(parents=True, exist_ok=True)
    rows = "all" if frames is None else ",".join(map(str, frames))
    argv = [Path(video).resolve(), Path(out).resolve() / "depth.npz", frame, rows, "" if fov is None else fov]
    subprocess.run(["uv", "run", "--quiet", "--no-sources", __file__, *map(str, argv)], check=True)


def read_video(path) -> np.ndarray:
    import cv2

    capture, frames = cv2.VideoCapture(str(path)), []
    try:
        while (got := capture.read())[0]:
            frames.append(got[1][..., ::-1])
    finally:
        capture.release()
    if not frames:
        raise ValueError(f"No readable video frames: {path}")
    return np.stack(frames)


def estimate(video, out, frame, rows, fov):
    """`estimate_depth`'s model side."""
    import torch
    from moge.model.v3 import MoGeModel

    if fraction := os.environ.get("ROBOSKILLS_MEMORY_FRACTION"):
        torch.cuda.set_per_process_memory_fraction(float(fraction))
    frames, frame = read_video(video), int(frame)
    if not 0 <= frame < len(frames):
        raise ValueError("Reference frame outside the source video")
    model = MoGeModel.from_pretrained(WEIGHTS).cuda().eval()
    image = lambda f: torch.from_numpy(f.copy()).cuda().permute(2, 0, 1).float() / 255
    rows = range(len(frames)) if rows == "all" else sorted({frame, *map(int, rows.split(","))})
    with torch.no_grad():
        reference = model.infer(image(frames[frame]), fov_x=float(fov) if fov else None, use_fp16=True)
        fov = float(np.degrees(2 * np.arctan(0.5 / float(reference["intrinsics"][0, 0]))))
        depths, valid = [], []
        for i in rows:
            p = reference if i == frame else model.infer(image(frames[i]), fov_x=fov, use_fp16=True)
            depths.append(torch.nan_to_num(p["depth"], nan=0, posinf=0, neginf=0).cpu().numpy().astype(np.float16))
            valid.append(p["mask"].cpu().numpy())
    height, width = frames.shape[1:3]
    K = reference["intrinsics"].cpu().numpy() * [[width], [height], [1]]
    np.savez_compressed(out, depth=np.stack(depths), valid=np.stack(valid), frames=np.array(list(rows)), K=K)


if __name__ == "__main__":
    estimate(*sys.argv[1:])
