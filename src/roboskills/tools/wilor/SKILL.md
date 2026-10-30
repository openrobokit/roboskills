---
name: wilor
description: Hands in a video with WiLoR, per frame a detector's boxes (left or right) and per box MANO's turns, keypoints and camera, in its own uv environment with MANO linked (python -m roboskills.tools.wilor).
---

# WiLoR

**Runs in** `uv run` of [`wilor.py`](wilor.py) (its inline dependencies: the verified environment's versions, torch
2.6.0 cu126, chumpy from a pinned commit) on `$CUDA_VISIBLE_DEVICES`. WiLoR is not a package: `prepare()` makes its
checkout once in `~/.roboskills/weights/wilor` (pinned commit, reading MANO from its own `mano_data/` from any
directory) with the released `wilor_final.ckpt` and hand detector `detector.pt` (SHA-256 checked) in
`pretrained_models/`, and links `mano_data/MANO_RIGHT.pkl` to ours (`~/.robokit/body_models/mano/`; MANO is
licensed); a left hand is read with the right model mirrored.

**Call:** `python -m roboskills.tools.wilor --video V --out hands/ [--mano DIR]` (or `detect_hands(...)`). For workers
of your own, the motion_gen workflow's `wilor.sh` names `$WILOR_PYTHON` (this environment's python with the checkout
on its path) and `$WILOR` (the checkout), and shows a minimal verified worker: `load_wilor("$WILOR/pretrained_models/
wilor_final.ckpt", ".../model_config.yaml")` loads the model and `ultralytics.YOLO(".../detector.pt")` the detector;
`wilor.datasets.vitdet_dataset.ViTDetDataset` cuts a crop per box (`rescale_factor=2.0`) and the model runs on a batch
of crops from any frames.

**Returns** `hands/hands.npz`, one row per detected hand: `frame`, `box`, `right` (the detector's label), MANO's
`global_orient` (3, 3), `hand_pose` (15, 3, 3, each joint in its parent) and `betas`, `keypoints_3d` (21, 3) in WiLoR's
camera frame (focal `focal`, centred) and `keypoints_2d` (21, 2) in image pixels. In a worker, per box
`pred_mano_params` and `pred_keypoints_3d`, and `pred_cam`; `wilor.utils.renderer.cam_crop_to_full` places a hand in
the full image (flip `pred_cam`'s x for a left hand).

**GPU memory:** about 3.2 GB with batches of 32 crops. Batched on an RTX 3090, the detector takes 20-40 ms per 1080p
frame and the model about 50 ms per hand. Cutting a crop on the CPU took 0.7-3 s per hand on a loaded shared host, the
largest cost: prepare crops in parallel. Loading takes about 10 s.

**Known failures:** without `MANO_RIGHT.pkl` the model does not load. A small, blurred or covered hand is often missed
or read flipped, and the detector's left/right label is often wrong.

**Upstream:** [WiLoR](https://github.com/rolpotamias/WiLoR) (models CC BY-NC-ND; MANO under its own license).
