---
name: sam3d
description: An object's mesh and camera pose from one video frame, its mask and MoGe depth with SAM 3D Objects, in its own image (python -m roboskills.tools.sam3d).
---

# SAM 3D

**Runs in** `ghcr.io/openrobokit/roboskills-sam3d:0.3.0` ([`docker/sam3d`](../../../../docker/sam3d/Dockerfile)): the
sam-3d-objects checkout at `/opt/sam3d` (entry point `notebook/inference.py`, `checkpoints/pipeline.yaml`) with its
verified environment first on `PATH`. The weights (the gated checkpoints, MoGe and DINOv2) are not in the image:
[`docker/weights.sh sam3d`](../../../../docker/weights.sh) fills `~/.roboskills/weights/sam3d` once, and the call
mounts it read-only at `/weights/sam3d` (`/opt/sam3d/checkpoints` links there), read offline.

**Call:** `python -m roboskills.tools.sam3d --video V --masks masks/ --depth depth/depth.npz --out shape/ [--frame 0]
[--label 1] [--samples 8] [--no-offload] [--gaussians]` (or `reconstruct_object(...)` from [`sam3d.py`](sam3d.py)): label `label` of
`masks/<frame>.png` on the MoGe point map; the best of `samples` pose samples by mask IoU and depth residual, then its
mesh, decoded alone unless `--gaussians` (SAM 3D's default decodes its Gaussians too, which only texture baking reads).

**Returns** `shape/mesh.ply` (the object's frame, metres, vertex colours; 0.25 M vertices for the sugar box) and `shape/pose.json`
(`camera_from_object`, 4x4 in the OpenCV camera frame; `scale`, the mesh's from SAM 3D's own frame; `seed`; each
sample's [seed, IoU, residual]); with `--gaussians`, `shape/gaussians.ply` too (SAM 3D's Gaussians, in its own frame).

**GPU memory:** by default (`offload`) the models load on the CPU and only the running stage's go to the GPU (sparse
structure, structured latent, decoders; MoGe never does): 8.44 GiB peak allocation, 8.9 GiB in use at most on a free
RTX 3090 (229 s for sugar-box), so it fits beside other jobs on a 24 GB card. The moves cost about 20 s per call and
loading on the CPU is slower; the samples, chosen seed and pose are identical. The mesh is decoded alone unless
`--gaussians`; either way decoding comes after the peak, so memory and time barely change (223 s for main, which
decoded both, on the same card). On one machine the pose, seed and samples match main's exactly and the mesh within
run-to-run noise: 0.0025 mm mean Chamfer distance, as between two runs of main (0.0022 mm). With torch's
deterministic kernels (a test harness only), main and this code wrote byte-identical meshes; two runs of main still
can differ, since some SAM 3D kernels have no deterministic version.
`--no-offload` keeps every model on the GPU: 17.26 GiB peak allocation, 17.8 GB in use on an RTX 3090 (19.63 GB on a
48 GB RTX 6000 Ada), for a card with that much free.

**Known failures:** the import path must include the checkout root; licensed or gated weights never download at run
time, so without `~/.roboskills/weights/sam3d` the call stops, naming `docker/weights.sh sam3d`.

**Upstream:** [SAM 3D Objects](https://github.com/facebookresearch/sam-3d-objects).
