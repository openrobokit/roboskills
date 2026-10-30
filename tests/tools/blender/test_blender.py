"""A MuJoCo scene goes to Blender as meshes, colours, per-state poses and its source camera, without Blender in tests."""

import numpy as np
import pytest

from roboskills.eval import runtime
from roboskills.tools.blender import blender
from roboskills.tools.mujoco import engine


mujoco = pytest.importorskip("mujoco")
cv2 = pytest.importorskip("cv2")


def test_render_scene_exports_states_and_the_source_camera(monkeypatch, tmp_path):
    scene = tmp_path / "scene"
    engine.build(scene / "scene.xml", cameras=[dict(name="source", pos=[0, -1, 0.5], xyaxes=[1, 0, 0, 0, 0.5, 1], fovy=50)],
                 bodies=[dict(name="box", free=True, parts=[dict(type="box", size=[0.1, 0.2, 0.3], rgba=[1, 0, 0, 1])])])  # fmt: skip
    model, _ = engine.load(scene / "scene.xml")
    mujoco.mj_saveModel(model, str(scene / "scene.mjb"))
    qpos = np.repeat(model.qpos0[None], 2, 0)
    qpos[1, :3] = [0.1, 0.2, 0.3]
    np.savez(scene / "trajectory.npz", qpos=qpos)
    writer = cv2.VideoWriter(str(tmp_path / "v.mp4"), cv2.VideoWriter.fourcc(*"mp4v"), 10, (64, 48))
    writer.write(np.zeros((48, 64, 3), np.uint8))
    writer.release()
    runs = []
    monkeypatch.setattr(runtime, "contained", lambda image, argv, mounts: ["docker", image, *argv[:2]])
    monkeypatch.setattr(blender.subprocess, "run", lambda cmd, **kw: runs.append(cmd))
    blender.render_scene(scene, tmp_path / "v.mp4", tmp_path / "out")
    d = np.load(tmp_path / "out/scene.npz")
    assert d["poses"].shape == (2, 1, 4, 4) and np.allclose(d["poses"][1, 0, :3, 3], [0.1, 0.2, 0.3])
    assert np.allclose(d["camera"][:3, 3], [0, -1, 0.5]) and np.allclose(d["camera"][:3, 0], [1, 0, 0])
    assert d["rgba"].tolist() == [[1, 0, 0, 1]] and d["size"].tolist() == [32, 24] and d["v0"].shape == (8, 3)
    assert runs[0][:2] == ["docker", blender.IMAGE] and str(runs[-1][-1]).endswith("compare_ext1.mp4")
