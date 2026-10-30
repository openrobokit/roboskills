"""Grasp and release come from the hand's distance to the tracked object, short gaps and blips ignored."""

import json

import numpy as np
import trimesh

from roboskills.workflows.real2sim.events import events


def test_one_contact_gives_one_grasp_and_one_release(tmp_path):
    (tmp_path / "fit").mkdir()
    (tmp_path / "hands").mkdir()
    (tmp_path / "track").mkdir()
    trimesh.creation.box([0.1, 0.1, 0.1]).export(tmp_path / "fit/mesh.ply")
    poses = np.repeat(np.eye(4)[None], 16, 0)
    poses[:, 2, 3] = 0.5
    np.savez(tmp_path / "track/poses.npz", camera_from_object=poses)
    hand = np.zeros((16, 778, 3)) + [0.0, 0.0, 0.5]
    gap = [
        0.3,
        0.3,
        0.3,
        0.06,
        0.055,
        0.06,
        0.2,
        0.055,
        0.06,
        0.05,
        0.3,
        0.3,
        0.3,
        0.06,
        0.3,
        0.3,
    ]  # vertex x: a 1-frame gap, a blip
    hand[:, :, 0] = np.array(gap)[:, None]
    hand[15] = np.nan
    np.savez(tmp_path / "hands/hands.npz", vertices=hand)
    events(tmp_path / "hands/hands.npz", tmp_path / "track/poses.npz", tmp_path / "fit", tmp_path / "events")
    record = json.loads((tmp_path / "events/events.json").read_text())
    assert record["events"] == [dict(frame=3, event="grasp"), dict(frame=10, event="release")]
    assert record["distance_m"][15] is None and abs(record["distance_m"][4] - 0.005) < 0.002
