---
name: coacd
description: Convex collision parts for one mesh with CoACD (decimated visual mesh plus convex hulls in its frame), on the CPU (python -m roboskills.tools.coacd).
---

# CoACD

**Runs in** its own `uv run` environment ([`coacd.py`](coacd.py)'s inline dependencies: `coacd`, `trimesh`,
`fast-simplification`), on the CPU; no image.

**Call:** `python -m roboskills.tools.coacd --mesh M --out DIR [--threshold 0.05] [--faces 20000] [--hulls 32]` (or
`decompose_convex(mesh, out, threshold=0.05, faces=20000, hulls=32)`).

**Returns** the part paths; `DIR` holds `visual.obj` (the input decimated to at most `faces`), `part_000.obj` ... (convex
hulls in the mesh's frame, for MuJoCo mesh geoms) and `parts.json` (part names, threshold, faces, seconds).

**Known failures:** a non-watertight or single-view mesh still decomposes, but its hidden side is whatever the mesh
invents; check openings and thin walls before trusting contacts. Lower `threshold` for concave objects (mugs, bowls).

**Upstream:** [CoACD](https://github.com/SarahWeiii/CoACD).
