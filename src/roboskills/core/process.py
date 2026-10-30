"""Adapters and child processes: `load` imports an adapter by module or file; `terminate` stops a process group."""

import contextlib
import importlib
import importlib.util
import os
import signal
import subprocess
from pathlib import Path


def load(spec, name="loaded"):
    """A module, file or attribute: `package.module`, `path/file.py`, or either followed by `:attribute`."""
    target, _, attribute = spec.rpartition(":") if ":" in spec else (spec, "", "")
    if target.endswith(".py") or Path(target).exists():
        module_spec = importlib.util.spec_from_file_location(name, target)
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
    else:
        module = importlib.import_module(target)
    return getattr(module, attribute) if attribute else module


def terminate(process, grace=2.0):
    """Stop a process started with `start_new_session` and its whole group: TERM, then KILL after `grace` s. The group
    is always killed at the end (a child may ignore TERM after its parent has exited)."""
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
    with contextlib.suppress(subprocess.TimeoutExpired):
        process.wait(timeout=grace)
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGKILL)
    process.wait()
