# /// script
# requires-python = "==3.11.*"
# dependencies = ["gvhmr[preproc,rtmpose]==1.6.1", "torch==2.14.1", "torchvision==0.29.1"]
#
# [tool.uv]
# extra-build-dependencies = { chumpy = ["pip", "numpy"] }  # chumpy's setup imports both
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
"""A person's SMPL-X body motion from a video with GVHMR, in its own environment (`uv run` of this file; the demo
checkpoints download once into ~/.roboskills/weights/gvhmr, SMPL-X is read from `body_models`).

    python -m roboskills.tools.gvhmr --video V --out motion/ [--moving-camera]
"""

import os
import subprocess
import sys
from pathlib import Path


WEIGHTS = Path.home() / ".roboskills/weights/gvhmr"  # GVHMR's, HMR2's, ViTPose's and YOLO's checkpoints


def recover_motion(video: Path, out: Path, moving_camera: bool = False,
                   body_models: Path = Path.home() / ".robokit/body_models"):  # fmt: skip
    """out/smplx.npz: `world_*` and `camera_*` `global_orient`, `body_pose` (21 joints), `betas`, `transl`, plus
    `intrinsics` and `fps`; world Y-up in metres, ground near y = 0. GVHMR's working files stay in out/. `body_models`
    holds the licensed SMPL-X (`smplx/SMPLX_*.npz`)."""
    Path(out).mkdir(parents=True, exist_ok=True)
    argv = ["uv", "run", "--quiet", __file__, str(Path(video).resolve()), str(Path(out).resolve()), str(moving_camera)]
    env = dict(os.environ, GVHMR_CHECKPOINTS=str(WEIGHTS), GVHMR_BODY_MODELS=str(Path(body_models).resolve()))
    subprocess.run(argv, check=True, env=dict(env, PYTHONSAFEPATH="1"))  # so `import gvhmr` is not this file


def recover(video, out, moving_camera):
    """`recover_motion`'s model side."""
    import gvhmr

    static = moving_camera == "False"
    r = gvhmr.recover(video, static_camera=static, device="cuda", output_root=out, output_dir=out, progress=False)
    r.save_npz(Path(out) / "smplx.npz")


if __name__ == "__main__":
    recover(*sys.argv[1:])
