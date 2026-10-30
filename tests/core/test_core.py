"""What every launcher relies on: receipts, loaders, process cleanup, native outcomes and the propose-evaluate loop."""

import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from roboskills.core.process import load, terminate
from roboskills.eval import records


def test_save_is_atomic_refuses_nan_and_converts_with_default(tmp_path):
    path = tmp_path / "deep/receipt.json"
    records.save(path, {"b": 1, "a": [1.5]})
    assert json.loads(path.read_text()) == {"b": 1, "a": [1.5]} and list(json.loads(path.read_text())) == ["b", "a"]
    with pytest.raises(ValueError):
        records.save(path, {"x": math.nan})
    assert json.loads(path.read_text())["b"] == 1  # a refused write leaves the old receipt whole
    records.save(path, {"q": np.arange(2)}, default=lambda v: v.tolist())
    assert json.loads(path.read_text()) == {"q": [0, 1]} and sorted(p.name for p in path.parent.iterdir()) == [
        "receipt.json"
    ]


def test_digests_ignore_key_order_and_files_hash_their_bytes(tmp_path):
    assert records.digest({"a": 1, "b": 2}) == records.digest({"b": 2, "a": 1})
    (tmp_path / "f").write_bytes(b"payload")
    assert records.file_digest(tmp_path / "f") == records.digest(b"payload")


def test_load_takes_modules_files_and_attributes(tmp_path):
    (tmp_path / "adapter.py").write_text("def make(): return 7\n")
    assert load("json") is json and load("json:dumps") is json.dumps
    assert load(str(tmp_path / "adapter.py")).make() == 7
    assert load(f"{tmp_path / 'adapter.py'}:make")() == 7


def test_terminate_stops_the_whole_group_even_a_child_that_ignores_term(tmp_path):
    marker = tmp_path / "child.pid"
    code = (
        "import os, signal, subprocess, sys, time\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import signal, time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)'])\n"
        f"open({str(marker)!r}, 'w').write(str(child.pid))\n"
        "time.sleep(60)\n"
    )
    process = subprocess.Popen([sys.executable, "-c", code], start_new_session=True)
    deadline = time.monotonic() + 10
    while not marker.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    child = int(marker.read_text())
    terminate(process, grace=0.5)
    assert process.returncode is not None
    deadline = time.monotonic() + 5
    while alive(child) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not alive(child)


def alive(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] not in ("Z", "X")
    except (OSError, IndexError):
        return False
