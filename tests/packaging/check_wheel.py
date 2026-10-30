"""Install a built wheel in a fresh environment and exercise it outside the checkout."""

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from zipfile import ZipFile


if __name__ == "__main__":
    wheel = Path(sys.argv[1]).resolve()
    with ZipFile(wheel) as archive:
        names = archive.namelist()
        assert all(name.startswith(("roboskills/", "roboskills-0.1.0.dist-info/")) for name in names)
        assert not any(name.startswith("roboskills/artifacts/") for name in names)
        assert not any(name.endswith((".mp4", ".webm", ".mov")) for name in names)
    with tempfile.TemporaryDirectory(prefix="roboskills-wheel-") as directory:
        directory = Path(directory)
        python = directory / "venv/bin/python"
        subprocess.run(["uv", "venv", "--python", sys.executable, str(python.parent.parent)], check=True)
        subprocess.run(["uv", "pip", "install", "--python", str(python), str(wheel)], check=True)
        subprocess.run(
            [str(python), str(Path(__file__).with_name("smoke.py").resolve())],
            cwd=directory,
            env=dict(os.environ, PYTHONPATH="", PYTHONNOUSERSITE="1"),
            check=True,
        )
