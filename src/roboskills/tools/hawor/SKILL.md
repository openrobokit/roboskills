---
name: hawor
description: A hand's MANO joints and vertices per video frame with HaWoR, boxed by its masks, in its own image with MANO mounted (python -m roboskills.tools.hawor).
---

# HaWoR

**Runs in** `ghcr.io/openrobokit/roboskills-hawor:0.3.0` ([`docker/hawor`](../../../../docker/hawor/Dockerfile)): the
HaWoR checkout at `/opt/hawor` (the working directory; weights resolve under it) with its verified environment first
on `PATH`. Its weights (the hand checkpoint, model config, infiller and hand detector) are not in the image:
[`docker/weights.sh hawor`](../../../../docker/weights.sh) fills `~/.roboskills/weights/hawor` once, and the call
mounts it read-only at `/weights/hawor`, where `/opt/hawor/weights` links. MANO is licensed and never in the image:
mount `MANO_RIGHT.pkl` at `/opt/hawor/_DATA/data/mano/` and `MANO_LEFT.pkl` at `/opt/hawor/_DATA/data_left/mano_left/`
(ours are in `~/.robokit/body_models/mano/`).

**Call:** `python -m roboskills.tools.hawor --video V --masks masks/ --depth depth/depth.npz --out hands/ [--label 2]
[--side right] [--mano ~/.robokit/body_models/mano]` (or `reconstruct_hands(...)` from [`hawor.py`](hawor.py)); it mounts the
weights and the two MANO files.

**Runs** HaWoR's video steps as Do as I Do does (`demo.py --static_camera`): motion estimation over each run of frames
where label `label` of `masks/<frame>.png` covers more than 50 pixels (its box), and over HaWoR's detector's tracks of
the other hand; the infiller over the frames between, the camera held still (no SLAM); then each hand's root path
(translation, orientation) loses its jump-and-back frames and is smoothed (Savitzky-Golay, window 7, order 3).

**Returns** `hands/hands.npz`: `joints` (N, 21, 3) and `vertices` (N, 778, 3) in the OpenCV camera frame, NaN where the
infiller does not reach (and in every frame when the mask has the hand in fewer than two); `seen`, the frames the mask
has the hand in; `side`; `other_joints`, the other hand's joints.

**GPU memory:** 6.4 GiB in use on the card at peak (74 frames, Quadro RTX 6000).

**Known failures:** without the two MANO files mounted at those paths the model does not load; without
`~/.roboskills/weights/hawor` the call stops, naming `docker/weights.sh hawor`.

**Upstream:** [HaWoR](https://github.com/ThunderVVV/HaWoR).
