# /// script
# requires-python = ">=3.11"
# dependencies = ["coacd", "trimesh", "fast-simplification", "numpy"]
# ///
"""Convex collision parts for one mesh with CoACD, on the CPU in its own environment (`uv run` of this file; its name
is CoACD's, so it runs with PYTHONSAFEPATH).

    python -m roboskills.tools.coacd --mesh M --out DIR [--threshold 0.05] [--faces 20000] [--hulls 32]
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path


def decompose_convex(mesh: Path, out: Path, threshold: float = 0.05, faces: int = 20000, hulls: int = 32) -> list[Path]:
    """out/visual.obj (the mesh, decimated to at most `faces`), out/part_000.obj ... (at most `hulls` convex hulls in the
    same frame; a lower CoACD concavity `threshold` gives more, finer parts) and out/parts.json (part files, threshold,
    faces, seconds); returns the part paths. The caller's Python needs neither CoACD nor trimesh."""
    Path(out).mkdir(parents=True, exist_ok=True)
    argv = [Path(mesh).resolve(), Path(out).resolve(), threshold, faces, hulls]
    subprocess.run(
        ["uv", "run", "--quiet", __file__, *map(str, argv)], check=True, env=dict(os.environ, PYTHONSAFEPATH="1")
    )
    return [Path(out) / name for name in json.loads((Path(out) / "parts.json").read_text())["parts"]]


def decompose(mesh, out, threshold, faces, hulls):
    """`decompose_convex`'s CoACD side."""
    import coacd
    import trimesh

    start, mesh, out = time.time(), trimesh.load(mesh, force="mesh", process=True), Path(out)
    if len(mesh.faces) > int(faces):
        mesh = mesh.simplify_quadric_decimation(face_count=int(faces))
    mesh.export(out / "visual.obj")
    coacd.set_log_level("error")
    parts = coacd.run_coacd(
        coacd.Mesh(mesh.vertices, mesh.faces), threshold=float(threshold), max_convex_hull=int(hulls), seed=0
    )
    names = [f"part_{i:03d}.obj" for i in range(len(parts))]
    for name, (vertices, triangles) in zip(names, parts):
        trimesh.Trimesh(vertices, triangles).export(out / name)
    record = dict(parts=names, threshold=float(threshold), faces=len(mesh.faces), seconds=round(time.time() - start, 1))
    (out / "parts.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    decompose(*sys.argv[1:])
