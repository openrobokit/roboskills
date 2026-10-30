---
name: moge3
description: Metric depth and pinhole intrinsics for every video frame with MoGe-3, in its own uv environment (python -m roboskills.tools.moge3).
---

# MoGe-3

**Runs in** `uv run --no-sources` of [`moge3.py`](moge3.py) (its inline dependencies: MoGe at a pinned commit,
torch 2.9.1; the MoGe-3 ViT-L weights download once into the Hugging Face cache) on `$CUDA_VISIBLE_DEVICES`. The
caller's Python only launches it. RoboDojo's `rgbd.MonoDepth` runs MoGe-2 in a `uv run` worker of its own for its
camera depth.

**Call:** `python -m roboskills.tools.moge3 --video V --out depth/ [--frame 0] [--frames 0 12 ...] [--fov DEG]` (or
`estimate_depth(video, out, frame=0, frames=None, fov=None)`). `--frame` is the reference frame, estimated first; every
other frame reuses its field of view, which `--fov` (horizontal, degrees) fixes when the camera's is known; `--frames`
estimates only those frames (and the reference).
`ROBOSKILLS_MEMORY_FRACTION` caps the model's share of the GPU.

**Returns** `depth/depth.npz`: `depth` [N, H, W] float16, metres (estimated; 0 where invalid), `valid` [N, H, W] bool,
`frames` [N] (every source frame, ascending) and `K` (3, 3) in source pixels.

**GPU memory:** MoGe-3 peaks at 3.9 GiB and runs out at 2.4 and 3 GiB caps.

**Known failures:** an unreadable or empty video and a reference frame outside it raise `ValueError`. Check the metric
scale against a known size.

**Upstream:** [MoGe](https://github.com/microsoft/MoGe).
