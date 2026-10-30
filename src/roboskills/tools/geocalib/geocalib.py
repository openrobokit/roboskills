# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "geocalib @ git+https://github.com/cvg/GeoCalib.git@1dedb29f94fc11b896c5f572005978aeebccd67f",
#   "torch", "torchvision", "opencv-python-headless", "numpy",
# ]
#
# [tool.uv.sources]
# torch = { index = "pytorch-cu126" }
# torchvision = { index = "pytorch-cu126" }
#
# [[tool.uv.index]]
# name = "pytorch-cu126"
# url = "https://download.pytorch.org/whl/cu126"
# explicit = true
# ///
"""A video's gravity direction in the camera frame with GeoCalib, in its own environment (`uv run` of this file; the
pinhole weights download once into torch.hub's cache).

    python -m roboskills.tools.geocalib --video V --out gravity/
"""

import json
import os
import subprocess
import sys
from pathlib import Path


def estimate_gravity(video: Path, out: Path):
    """out/gravity.json: `down`, the unit gravity direction in the OpenCV camera frame (x right, y down, z forward), the
    median of eight evenly spaced frames' estimates (`up` each)."""
    Path(out).mkdir(parents=True, exist_ok=True)
    argv = ["uv", "run", "--quiet", __file__, str(Path(video).resolve()), str(Path(out).resolve())]
    subprocess.run(argv, check=True, env=dict(os.environ, PYTHONSAFEPATH="1"))  # so `import geocalib` is not this file


def estimate(video, out):
    """`estimate_gravity`'s model side."""
    import cv2
    import numpy as np
    import torch
    from geocalib import GeoCalib

    capture, frames = cv2.VideoCapture(video), []
    while (got := capture.read())[0]:
        frames.append(np.ascontiguousarray(got[1][..., ::-1]))
    model = GeoCalib().cuda()
    with torch.no_grad():
        ups = [model.calibrate(torch.from_numpy(f).cuda().permute(2, 0, 1).float() / 255)["gravity"].vec3d[0].cpu().numpy()
               for f in frames[:: max(len(frames) // 8, 1)]]  # fmt: skip
    up = np.median(ups, 0)  # GeoCalib's vector points up
    record = dict(down=(-up / np.linalg.norm(up)).tolist(), up=np.round(ups, 5).tolist())
    (Path(out) / "gravity.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    estimate(*sys.argv[1:])
