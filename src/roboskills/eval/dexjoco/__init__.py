"""DexJoCo's native MuJoCo runtime behind the shared Robot contract; see SKILL.md."""

import os
from pathlib import Path


TASKS = (
    "pick_bucket",
    "water_plant",
    "hammer_nail",
    "click_mouse",
    "pinch_tongs",
    "fold_glasses",
    "bimanual_photograph",
    "bimanual_hanoi",
    "bimanual_microwave_cook",
    "bimanual_unlock_ipad",
)
LIMITS = dict(zip(TASKS, (1200, 1000, 1000, 1200, 1000, 1200, 1000, 1500, 1100, 1200)))


def vendor(path=None):
    """The prepared checkout: an explicit path, else $DEXJOCO_VENDOR, outside the Python package."""
    path = path or os.environ.get("DEXJOCO_VENDOR")
    if not path:
        raise RuntimeError("set DEXJOCO_VENDOR to the prepared DexJoCo checkout (see eval/dexjoco/SKILL.md)")
    return Path(path).resolve()
