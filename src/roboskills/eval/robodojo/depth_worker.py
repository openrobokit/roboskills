# /// script
# requires-python = ">=3.10,<3.14"  # as MoGe-3: triton fails on 3.14
# dependencies = ["torch==2.9.1+cu126", "torchvision==0.24.1+cu126", "numpy", "moge @ git+https://github.com/microsoft/MoGe.git@b942f00bdc2a2a23ebb474fbe034d487e6dcceec"]
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
"""Monocular depth (MoGe-2 ViT-L) kept loaded in a worker process with its own dependencies (like SAM 3's worker).

One JSON request per stdin line {images: npz path with rgb_<i>, fov_x: [deg per image], out: npz path}; one reply per
stdout line, "DEPTH " + JSON {ok, n}; the depths (H, W float32, relative: the caller fixes the scale) go to `out`.

    uv run src/roboskills/eval/robodojo/depth_worker.py
"""

import json
import sys

import numpy as np
import torch


def main():
    from moge.model.v2 import MoGeModel

    model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to("cuda").eval()
    print(
        "DEPTH "
        + json.dumps(
            {
                "ok": True,
                "ready": True,
                "torch": str(torch.__version__),
                "cuda": torch.version.cuda,
                "model": "Ruicheng/moge-2-vitl-normal",
            }
        ),
        flush=True,
    )
    for line in sys.stdin:
        try:
            req = json.loads(line)
            imgs = np.load(req["images"])
            out = {}
            for i, fov in enumerate(req["fov_x"]):
                x = torch.tensor(imgs[f"rgb_{i}"] / 255.0, dtype=torch.float32).permute(2, 0, 1).cuda()
                with torch.no_grad():
                    d = model.infer(x, fov_x=float(fov), use_fp16=True)["depth"]
                out[f"depth_{i}"] = d.float().cpu().numpy()
            np.savez(req["out"], **out)
            print("DEPTH " + json.dumps({"ok": True, "n": len(out)}), flush=True)
        except Exception as e:  # noqa: BLE001
            print("DEPTH " + json.dumps({"ok": False, "error": repr(e)[:500]}), flush=True)


if __name__ == "__main__":
    main()
