"""HumanoidArena (Unitree G1 29-DoF with Dex3 hands under the frozen GEAR-SONIC whole-body controller, Isaac Sim 6.1)
behind the shared Robot contract; see SKILL.md. `launch` (the simulator in its image, `make`), `robot` (the launcher side's
`Robot`), `hand` (the Dex3 hands and the arm model), `sim` (the image side: `server.py`, the planner gait `gait.py`)."""

import hashlib
import os
from pathlib import Path


# task: (Gym id, upstream task package, official test config, step limit at 50 Hz (upstream's SONIC eval MAX_STEPS),
# instruction: RAX's objective sentence for the task)
TASKS = {
    "warehouse_navigation": ("Isaac-Move-SmallWarehouse-VisionNavigation-G129-Dex3-Wholebody",
                             "move_small_warehouse_vision_navigation_g1_29dof_dex3_wholebody", "vision_navi", 1800,
                             "Navigate through the warehouse to the designated destination."),
    "boxing_bag": ("Isaac-Move-Boxing-Bag-G129-Dex3-Wholebody", "move_boxing_bag_g1_29dof_dex3_wholebody", "boxing", 900,
                   "Move a punching body into the boxing bag's designated target region."),
    "opendoor": ("Isaac-Move-Open-Door-G129-Dex3-Wholebody", "move_open_door_g1_29dof_dex3_wholebody", "open_door", 1800,
                 "Physically operate the handle or latch and open the door leaf while remaining standing."),
    "football_single": ("Isaac-Move-Football-Single-G129-Dex3-Wholebody", "move_football_single_g1_29dof_dex3_wholebody",
                        "football_single", 2000, "Move the football into the single goal."),
    "sit_sofa": ("Isaac-Move-Sit-Sofa-G129-Dex3-Wholebody", "move_sit_sofa_g1_29dof_dex3_wholebody", "sit_sofa", 2000,
                 "Sit on the sofa in the task's target region and posture."),
}  # fmt: skip
GROUPS = {"1": 0, "2": 1}  # seed -> the official test group seed (its first repeat)


def episode_seed(task, seed):
    """The official test episode's environment seed: upstream's `_derive_episode_seed` for group GROUPS[seed], repeat 0
    (the last 4 bytes of sha256("<gym id>|<group>|<repeat>"), masked to a positive int32)."""
    if str(seed) not in GROUPS:
        raise ValueError(f"seed is one of {tuple(GROUPS)} (official test groups 0 and 1, repeat 0)")
    payload = f"{TASKS[task][0]}|{GROUPS[str(seed)]}|0".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[-4:], "big") & 0x7FFFFFFF


def vendor(path=None):
    """The vendor directory (setup.sh): `path`, else $HA_VENDOR. It holds HumanoidArena/ (the pinned checkout with its
    assets) and sonic/ (encoder, decoder, planner), and never lives inside this package."""
    path = path or os.environ.get("HA_VENDOR")
    if not path:
        raise RuntimeError(
            "set HA_VENDOR to the HumanoidArena vendor directory (see roboskills/eval/humanoidarena/SKILL.md)"
        )
    return Path(path).resolve()
