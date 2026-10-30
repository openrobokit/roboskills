---
name: geocalib
description: A video's gravity direction in the camera frame with GeoCalib, in its own uv environment (python -m roboskills.tools.geocalib).
---

# GeoCalib

**Runs in** `uv run` of [`geocalib.py`](geocalib.py) (its inline dependencies: GeoCalib at a pinned commit, torch; the
pinhole weights download once into torch.hub's cache) on `$CUDA_VISIBLE_DEVICES`.

**Call:** `python -m roboskills.tools.geocalib --video V --out gravity/` (or `estimate_gravity(video, out)`).

**Returns** `gravity/gravity.json`: `down`, the unit gravity direction in the OpenCV camera frame (x right, y down,
z forward), the median of eight evenly spaced frames' estimates, and those estimates (`up`).

**GPU memory:** small (one image at a time).

**Known failures:** a few degrees of error is common; a fitted support plane is a better local up.

**Upstream:** [GeoCalib](https://github.com/cvg/GeoCalib).
