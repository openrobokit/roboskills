"""Receipts and hashes: atomic JSON files and SHA-256 identities."""

import hashlib
import json
import os
import tempfile
from pathlib import Path


def digest(value):
    """Hash bytes or canonical JSON; dictionary order never changes identity."""
    if not isinstance(value, bytes):
        value = json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
    return hashlib.sha256(value).hexdigest()


def file_digest(path):
    """SHA-256 of a file's bytes, read in blocks."""
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def save(path, value, *, default=None):
    """Write JSON atomically: a reader sees the old file or the new one, never part of either. NaN is refused;
    `default` converts values JSON cannot (e.g. NumPy arrays)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, indent=2, allow_nan=False, default=default) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
