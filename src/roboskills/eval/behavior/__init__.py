"""BEHAVIOR-1K (the 2026 BEHAVIOR Challenge: R1Pro on OmniGibson 3.9.2 / Isaac Sim 5.1) behind the shared Robot contract;
see SKILL.md. `launch` (the simulator in the official image, `make`), `robot` (the launcher side's `Robot`), `hand` (the
gripper and the robot model), `goal` (BDDL goals in words), `sim` (the image side, run by OmniGibson's python)."""

import os
from pathlib import Path


# The cohort: five challenge tasks whose goals are a few pick/place/open/switch-on facts about a handful of objects within
# the arms' reach with the torso held (0.5-1.5 m up), each needing the base to move. task: (the challenge's step limit,
# 1.5x its mean human demonstration at 30 Hz; the challenge's own sentence, or None: goal.describe of its BDDL goal).
TASKS = {
    "turning_on_radio": (3224, "Turn on the radio receiver that's on the table in the living room."),
    "store_honey": (10150, None),
    "composting_waste": (5457, None),
    "installing_a_scanner": (6387, None),
    "dispose_of_glass": (13792, None),
}
INSTANCES = {"1": 301, "2": 302, "3": 303}  # seed -> the challenge's public test instance (its first three)


def data(path=None):
    """The BEHAVIOR dataset (setup.sh): `path`, else $BEHAVIOR_DATA. It holds omnigibson-robot-assets/,
    behavior-1k-assets/, 2026-challenge-task-instances/ and omnigibson.key, and never lives inside this package."""
    path = path or os.environ.get("BEHAVIOR_DATA")
    if not path:
        raise RuntimeError("set BEHAVIOR_DATA to the BEHAVIOR dataset (see roboskills/eval/behavior/SKILL.md)")
    return Path(path).resolve()
