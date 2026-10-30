"""MoGe-3 depth keeps every source frame and the reference frame's field of view, without the model in tests."""

import contextlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from roboskills.tools.moge3 import moge3


def test_empty_video_is_explicit_and_capture_closes(monkeypatch):
    closed = []
    capture = SimpleNamespace(read=lambda: (False, None), release=lambda: closed.append(True))
    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace(VideoCapture=lambda path: capture))
    with pytest.raises(ValueError, match="No readable video frames"):
        moge3.read_video("empty.mp4")
    assert closed == [True]


@pytest.mark.parametrize("fraction", [None, "0.30"])
def test_resource_cap_precedes_model_loading(monkeypatch, tmp_path, fraction):
    calls = []
    if fraction is None:
        monkeypatch.delenv("ROBOSKILLS_MEMORY_FRACTION", raising=False)
    else:
        monkeypatch.setenv("ROBOSKILLS_MEMORY_FRACTION", fraction)
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(set_per_process_memory_fraction=calls.append))
    )
    for name in ("moge", "moge.model", "moge.model.v3"):
        monkeypatch.setitem(sys.modules, name, SimpleNamespace(MoGeModel=None))
    monkeypatch.setattr(moge3, "read_video", lambda path: calls.append("model") or np.zeros((0, 2, 2, 3)))
    with pytest.raises(ValueError, match="Reference frame"):
        moge3.estimate("v.mp4", tmp_path / "depth.npz", "0", "all", "")
    assert calls == ([0.30, "model"] if fraction else ["model"])


def test_invalid_resource_fraction_stops_before_model_loading(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setenv("ROBOSKILLS_MEMORY_FRACTION", "invalid")
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(set_per_process_memory_fraction=calls.append))
    )
    for name in ("moge", "moge.model", "moge.model.v3"):
        monkeypatch.setitem(sys.modules, name, SimpleNamespace(MoGeModel=None))
    monkeypatch.setattr(moge3, "read_video", lambda path: calls.append("model"))
    with pytest.raises(ValueError, match="float"):
        moge3.estimate("v.mp4", tmp_path / "depth.npz", "0", "all", "")
    assert not calls


class Tensor:
    def __init__(self, data):
        self.data = np.asarray(data)

    def cpu(self):
        return self

    def numpy(self):
        return self.data

    def tolist(self):
        return self.data.tolist()

    def __getitem__(self, key):
        return Tensor(self.data[key])

    def __float__(self):
        return float(self.data)


def test_depth_reuses_reference_and_shared_camera_fov(monkeypatch, tmp_path):
    calls, loaded = [], []

    class ImageTensor(Tensor):
        def cuda(self):
            return self

        def permute(self, *axes):
            return ImageTensor(np.transpose(self.data, axes))

        def float(self):
            return ImageTensor(self.data.astype(float))

        def __truediv__(self, value):
            return ImageTensor(self.data / value)

    class Model:
        @classmethod
        def from_pretrained(cls, weights):
            loaded.append(weights)
            return cls()

        def cuda(self):
            return self

        def eval(self):
            return self

        def infer(self, image, *, fov_x=None, use_fp16):
            index = round(image.data[0, 0, 0] * 255)
            calls.append((index, fov_x))
            return dict(
                depth=Tensor([[index, np.inf], [np.nan, -np.inf]]),
                mask=Tensor([[True, False], [False, False]]),
                intrinsics=Tensor([[0.5, 0, 0.5], [0, 1, 0.5], [0, 0, 1]]),
            )

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            no_grad=contextlib.nullcontext,
            from_numpy=ImageTensor,
            nan_to_num=lambda tensor, **kwargs: Tensor(np.nan_to_num(tensor.data, **kwargs)),
        ),
    )
    for name in ("moge", "moge.model", "moge.model.v3"):
        monkeypatch.setitem(sys.modules, name, SimpleNamespace(MoGeModel=Model))
    frames = np.stack([np.full((5, 6, 3), i, np.uint8) for i in range(3)])
    monkeypatch.setattr(moge3, "read_video", lambda path: frames)
    out = tmp_path / "depth.npz"
    moge3.estimate("v.mp4", out, "1", "all", "")
    assert loaded == [moge3.WEIGHTS]
    assert calls == [(1, None), (0, 90), (2, 90)]
    with np.load(out) as stored:
        np.testing.assert_array_equal(stored["depth"][:, 0, 0], [0, 1, 2])
        np.testing.assert_array_equal(stored["depth"][:, 1, :], 0)
        np.testing.assert_array_equal(stored["valid"][:, 0, 0], True)
        np.testing.assert_allclose(stored["K"], [[3, 0, 3], [0, 5, 2.5], [0, 0, 1]])
        np.testing.assert_array_equal(stored["frames"], [0, 1, 2])
    calls.clear()  # chosen frames (the reference among them) at a known field of view
    moge3.estimate("v.mp4", tmp_path / "some.npz", "1", "2", "70")
    assert calls == [(1, 70.0), (2, 90)]
    with np.load(tmp_path / "some.npz") as stored:
        np.testing.assert_array_equal(stored["frames"], [1, 2])


@pytest.mark.gpu
def test_depth_protocol_with_the_real_model(tmp_path):
    """The real model on a tiny clip: the keys, shapes, dtypes and units SKILL.md promises."""
    import subprocess

    import cv2

    video, out = tmp_path / "clip.mp4", tmp_path / "depth"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter.fourcc(*"mp4v"), 10, (64, 48))
    for i in range(3):
        frame = np.full((48, 64, 3), 90, np.uint8)
        frame[:, : 20 + 10 * i] = 200  # an edge, so the frames differ
        writer.write(frame)
    writer.release()
    command = ["-m", "roboskills.tools.moge3", "--video", str(video), "--out", str(out), "--frame", "2"]
    subprocess.run([sys.executable, *command], check=True, timeout=600)
    with np.load(out / "depth.npz") as result:
        assert set(result.files) == {"depth", "valid", "frames", "K"}
        assert result["depth"].shape == result["valid"].shape == (3, 48, 64)  # every frame
        assert result["depth"].dtype == np.float16 and result["valid"].dtype == bool
        assert (
            result["frames"].tolist() == [0, 1, 2]
            and np.isfinite(result["depth"]).all()
            and (result["depth"] >= 0).all()
        )
        K = result["K"]
        assert K.shape == (3, 3) and 0 < K[0, 2] < 64 and 0 < K[1, 2] < 48 and K[0, 0] > 0  # pixels of this video
