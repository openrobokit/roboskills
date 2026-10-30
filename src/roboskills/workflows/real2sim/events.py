"""Grasp and release frames from the hand's distance to the object, both in the camera frame: HaWoR's MANO vertices
against the object's mesh at its tracked pose. A contact is under `near` metres, its gaps and blips under `hold` frames
removed; it starts with a grasp and ends with a release."""

import json
from pathlib import Path

import numpy as np


def events(hands: Path, track: Path, shape: Path, out: Path, near: float = 0.02, hold: int = 3):
    """events/events.json from hands.npz, track's poses.npz and shape/mesh.ply: `events` ([{frame, event: grasp|release}],
    in order), the per-frame hand-object `distance_m` (None where the hand is not seen), `near` and `hold`."""
    import trimesh
    from scipy.ndimage import binary_closing, binary_opening
    from scipy.spatial import cKDTree

    out.mkdir(parents=True, exist_ok=True)
    hands, poses = np.load(hands)["vertices"], np.load(track)["camera_from_object"]
    surface = trimesh.load(shape / "mesh.ply").sample(3000)
    distance = np.array([cKDTree(trimesh.transform_points(surface, pose)).query(hand)[0].min() if np.isfinite(hand).all()
                         else np.nan for hand, pose in zip(hands, poses)])  # fmt: skip
    gaps = np.ones(hold)  # contact gaps and blips under `hold` frames go
    contact = binary_opening(binary_closing(np.nan_to_num(distance, nan=np.inf) < near, gaps), gaps)
    edges = np.diff(np.r_[0, contact.astype(int), 0])
    found = [dict(frame=int(f), event=e) for s, t in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))
             for f, e in ((s, "grasp"), (t, "release")) if f < len(distance)]  # fmt: skip
    record = dict(events=found, distance_m=[None if np.isnan(x) else round(float(x), 4) for x in distance])
    (out / "events.json").write_text(json.dumps(dict(record, near=near, hold=hold), indent=2) + "\n")
