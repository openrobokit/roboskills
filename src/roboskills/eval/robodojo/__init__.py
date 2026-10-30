"""RoboDojo behind the shared `Robot` contract (see SKILL.md): `launch` (the sim container, `make_vector`), `robot` (one
env's `Robot`, over `roboskills.eval.wire`), `layouts` (eval suites), `rgbd` (depth for RoboDojo's official
observation), `evaluation` (saved RGB evidence), `grasp_backend` (the learned grasp policy as $GRASP_BACKEND), and `sim` (the container side, run by Isaac Sim's python)."""

import os
from pathlib import Path


def vendor(path=None):
    """The prepared RoboDojo checkout (setup.sh): `path`, else $ROBODOJO_VENDOR. It never lives inside this package."""
    path = path or os.environ.get("ROBODOJO_VENDOR")
    if not path:
        raise RuntimeError(
            "set ROBODOJO_VENDOR to the prepared RoboDojo checkout (see roboskills/eval/robodojo/SKILL.md)"
        )
    return Path(path).resolve()
