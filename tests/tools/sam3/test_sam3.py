"""Perception launchers must release their child before GPU allocation reuse."""

import base64
import io
import json
import os
import signal
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image

from roboskills.tools.sam3 import sam3


def test_close_reaps_a_launcher_and_its_child_on_eof():
    child = "import os,sys; print(os.getpid(),flush=True); sys.stdin.read()"
    launcher = "import subprocess,sys; sys.exit(subprocess.call([sys.executable,'-c',sys.argv[1]],stdin=sys.stdin))"
    process = subprocess.Popen(
        [sys.executable, "-c", launcher, child],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    worker = object.__new__(sam3.Sam)
    worker.p, worker._closed = process, False
    child_pid = int(process.stdout.readline())
    try:
        worker.close()
        assert process.returncode == 0 and worker._closed
        with pytest.raises(ProcessLookupError):
            os.kill(child_pid, 0)
        worker.close()  # idempotent: no stale process group is signaled
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def test_constructor_failure_closes_the_worker(monkeypatch):
    process = subprocess.Popen(
        [sys.executable, "-c", "print('initialization failed')"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    arguments = []

    def launch(*args, **kwargs):
        arguments.append(kwargs)
        return process

    monkeypatch.setattr(sam3.subprocess, "Popen", launch)
    with pytest.raises(RuntimeError, match="SAM 3 worker exited"):
        sam3.Sam()
    assert arguments[0]["start_new_session"]
    assert process.returncode == 0 and process.stdout.closed and process.stdin.closed


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group escalation")
def test_close_escalates_only_the_dedicated_worker_group(monkeypatch):
    process = Mock(pid=1234, stdin=io.StringIO(), stdout=io.StringIO())
    process.wait.side_effect = [subprocess.TimeoutExpired("model", 15), 0]
    process.poll.return_value = 0
    signals = []
    monkeypatch.setattr(sam3.os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    worker = object.__new__(sam3.Sam)
    worker.p, worker._closed = process, False
    worker.close()
    assert signals == [(1234, signal.SIGKILL)]
    assert worker._closed and process.stdout.closed


def test_frames_and_masks_travel_over_the_pipe():
    npz = io.BytesIO()
    np.savez_compressed(npz, **{"0": np.eye(2, dtype=bool)})
    hit = dict(id=0, score=0.9, box_px=[0, 0, 1, 1], area_px=2)
    reply = dict(ok=True, detections={"cup": [hit]}, masks=base64.b64encode(npz.getvalue()).decode())
    worker = object.__new__(sam3.Sam)
    worker.p = SimpleNamespace(stdin=io.StringIO(), stdout=io.StringIO("SAM3 " + json.dumps(reply) + "\n"))
    got = worker.detect(np.zeros((2, 3, 3)), ["cup"], threshold=0.4)
    sent = json.loads(worker.p.stdin.getvalue())
    assert Image.open(io.BytesIO(base64.b64decode(sent["png"]))).size == (3, 2)
    assert sent["prompts"] == ["cup"] and sent["threshold"] == 0.4
    assert got["cup"][0]["mask"].tolist() == [[True, False], [False, True]]


@pytest.mark.gpu
def test_detect_protocol_with_the_real_model():
    """The real worker on a tiny image: the keys, shapes and ranges SKILL.md promises."""
    image = np.zeros((48, 64, 3), np.uint8)
    image[12:36, 16:48] = (220, 30, 30)
    worker = sam3.Sam()
    try:
        found = worker.detect(image, ["red square"], threshold=0.0)
    finally:
        worker.close()
    assert list(found) == ["red square"]
    for hit in found["red square"]:
        assert {"id", "score", "box_px", "area_px", "mask"} <= set(hit) and 0 <= hit["score"] <= 1
        assert hit["mask"].shape == (48, 64) and hit["mask"].dtype == bool and len(hit["box_px"]) == 4
