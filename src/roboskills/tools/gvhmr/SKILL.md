---
name: gvhmr
description: A person's SMPL-X body motion (no hands) in world and camera coordinates from a video with GVHMR, in its own uv environment with SMPL-X read from ~/.robokit/body_models (python -m roboskills.tools.gvhmr).
---

# GVHMR

**Runs in** `uv run` of [`gvhmr.py`](gvhmr.py) (its inline dependencies: `gvhmr[preproc,rtmpose]` 1.6.1, torch
2.14.1 cu126) on `$CUDA_VISIBLE_DEVICES`; the demo checkpoints (GVHMR, HMR2, ViTPose, YOLO; 5.2 GB) download once into
`~/.roboskills/weights/gvhmr` (`$GVHMR_CHECKPOINTS`). SMPL-X is licensed: `body_models` names a directory with
`smplx/SMPLX_*.npz` (`$GVHMR_BODY_MODELS`, ours `~/.robokit/body_models`).

**Call:** `python -m roboskills.tools.gvhmr --video V --out motion/ [--moving-camera]` (or `recover_motion(...)`). The
motion_gen workflow's `gvhmr.sh` (written by its `setup.sh`) names `$GVHMR_PYTHON`, this environment's python, and
both directories, for scripts of its own:

    $GVHMR_PYTHON -c 'import gvhmr; r = gvhmr.recover("in.mp4", static_camera=True, device="cuda", output_root="out", output_dir="out/in", progress=False); r.save_npz("out/in/smplx.npz")'

**Returns** `motion/smplx.npz` (`save_npz`): `world_*` and `camera_*` `global_orient`, `body_pose` (21 joints), `betas`, `transl`, plus
`intrinsics` and `fps` (30); world Y-up in metres, ground near y = 0. `transl` is not the pelvis: run the SMPL-X model
for joints.

**GPU memory:** about 7 GB and 25 s per 4 s clip.

**Known failures:** it keeps the largest person's track (write `preprocess/bbx.pt` to choose one); its wrists are
guesses; Triton compiles a launcher at run time, so the host needs GCC and libc headers.

**Upstream:** [GVHMR](https://github.com/zju3dv/GVHMR).
