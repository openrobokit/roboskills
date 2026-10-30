"""EmbodiedSWE-Bench (RoboBench on Isaac Lab 2.3.2 / Isaac Sim 5.1) behind the shared Robot contract; see SKILL.md.
`launch` (the simulator in its image, `make`), `robot` (the launcher side's `Robot`), `hand` (the Panda gripper and the
G1's three-finger hands), `sim` (the image side, run by Isaac Sim's python).

Imports only the standard library: the simulator's image reads `TASKS` too."""

import os
from pathlib import Path


# The cohort: RoboBench presets (suite.scene.robot.mode) in their joint-target modes. No preset defines a step limit, so
# each gets SECONDS_PER_GOAL of simulated time per sub-goal (a pick-and-place, a door or drawer moved, a press, a walk
# around the table) plus SECONDS_EXTRA. `views`: the fixed external cameras, {name: (eye, target, focal mm)} in the
# world (metres; focal on Isaac's 20.955 mm aperture); every robot also carries its own (EGO).
SECONDS_PER_GOAL, SECONDS_EXTRA = 45.0, 30.0
TASKS = {
    # 4-5 organics (of a lemon, a small lemon, a lime, an orange, a small pumpkin) into the bin: 5 placements
    "packing.clear_organic_objects.franka.joint": dict(goals=5, views={
        "front": ((0.95, -0.85, 1.30), (0.12, 0.18, 0.55), 18.0),
        "shoulder": ((-0.65, -0.65, 1.45), (0.15, 0.25, 0.55), 18.0)}),
    # (the chest variant) three drawers each pulled open, given its tool and pushed shut: 9
    "packing.tool_packing.franka.joint": dict(goals=9, views={
        "front": ((0.75, -0.85, 1.30), (-0.08, -0.08, 0.62), 18.0),
        "shoulder": ((-0.95, -0.50, 1.40), (-0.05, -0.10, 0.60), 16.0)}),
    # cover open, capsule in, cover shut, mug under the spout, START, mug back to the tray, and the brew's wait: 7
    "puzzle.coffee.franka.joint": dict(goals=7, views={
        "front": ((0.95, -0.90, 1.30), (0.15, -0.05, 0.65), 18.0),
        "shoulder": ((-0.60, -0.95, 1.40), (0.20, 0.0, 0.62), 16.0)}),
    # three blocks onto their zones: 3
    "puzzle.classify_objects.g1.joint": dict(goals=3, views={
        "front": ((0.60, 0.05, 1.30), (0.0, -0.38, 0.72), 18.0)}),
    # the lemon picked, carried around the table (two walks and a turn), placed: 4, and the slow turn on the spot
    "locomanip.fruit_delivery.g1.loco_joint": dict(goals=6, views={
        "front": ((1.90, -1.90, 2.10), (0.10, 0.0, 0.70), 14.0)}),
    # (the fallback for a G1 that cannot walk usefully) four blocks into one tower: 4
    "puzzle.stack_blocks.g1.joint": dict(goals=4, views={
        "front": ((0.60, 0.05, 1.30), (0.04, -0.36, 0.72), 18.0)}),
}  # fmt: skip
# Robot-mounted cameras: {robot: {name: (link, eye, target, focal mm)}}, eye and target in the link's frame. The Panda's
# is RoboBench's own wrist view (FrankaRobot.CAMERAS); the G1's sits where its head's RealSense D435 does (Unitree's
# d435_joint on torso_link), looking ahead and 47.6 degrees down, and turns with the torso.
EGO = {
    "franka": {"wrist": ("panda_hand", (0.0465, -0.02, 0.036), (0.0465, -0.02, 0.3), 10.5)},
    "g1": {"head": ("torso_link", (0.0576, 0.0175, 0.4299), (0.7314, 0.0175, -0.3088), 11.0)},
}
SEEDS = ("1", "2")  # RoboBench has no official seeds: ours, passed to env.reset(seed=...), one env per process


def step_limit(task, dt):
    """Control steps the episode gets (see TASKS), at `dt` seconds per step."""
    return int(round((SECONDS_PER_GOAL * TASKS[task]["goals"] + SECONDS_EXTRA) / dt))


def vendor(path=None):
    """The EmbodiedSWE checkout with its assets (setup.sh): `path`, else $ESWE_VENDOR."""
    path = path or os.environ.get("ESWE_VENDOR")
    if not path:
        raise RuntimeError("set ESWE_VENDOR to the EmbodiedSWE checkout (see roboskills/eval/embodiedswe/SKILL.md)")
    return Path(path).resolve()
