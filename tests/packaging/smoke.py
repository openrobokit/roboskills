"""Installed-package checks; no simulator, GPU, model credentials or source imports."""

import importlib
import importlib.util
import subprocess
import sys
from importlib.resources import files
from pathlib import Path

import numpy as np

import roboskills
from roboskills.eval.robodojo.evaluation import check_rgb_evidence
from roboskills.eval.robodojo.layouts import development_layouts


# runtimes with a launch.py, ROBOT.md and setup files
BENCHMARKS = ("behavior", "dexjoco", "embodiedswe", "humanoidarena", "libero", "robocasa365", "robosuite", "robotwin")
WORKFLOWS = ("agent_as_policy", "agentic_sim", "auto_research", "motion_gen", "real2sim")
TOOLS = ("blender", "coacd", "geocalib", "graspkit", "gvhmr", "hawor", "kimodo", "moge3", "mujoco", "robokit", "sam3",
         "sam3d", "wilor")  # fmt: skip
checkout = Path(__file__).resolve().parents[2]
assert not Path(roboskills.__file__).resolve().is_relative_to(checkout)
assert importlib.util.find_spec("artifacts") is None
for launcher in ("agent_as_policy", "agentic_sim", "auto_research"):  # runnable from the wheel alone
    assert callable(importlib.import_module(f"roboskills.workflows.{launcher}.run").main)
for namespace, resource in [
    *[("roboskills.workflows", f"{name}/{doc}") for name in WORKFLOWS for doc in ("SKILL.md", "README.md")],
    *[("roboskills.tools", f"{name}/SKILL.md") for name in TOOLS],
    ("roboskills.core", "README.md"),
    ("roboskills.workflows.agentic_sim", "build.md"),
    ("roboskills.workflows.agentic_sim", "setup.sh"),
    ("roboskills.workflows.agentic_sim", "solve.md"),
    ("roboskills.workflows.motion_gen", "setup.sh"),
    ("roboskills.workflows.motion_gen.character", "SKILL.md"),
    ("roboskills.tools.robokit", "README.md"),
    ("roboskills.tools.mujoco", "engine.py"),
    ("roboskills.tools.mujoco", "mano/right.xml"),
    ("roboskills.tools.mujoco", "franka/panda_2f85.xml"),
    ("roboskills.eval.robodojo", "SKILL.md"),
    ("roboskills.eval.video2world", "SKILL.md"),
    *[(f"roboskills.eval.{name}", doc) for name in BENCHMARKS for doc in ("SKILL.md", "ROBOT.md")],
]:
    assert files(namespace).joinpath(resource).read_text().strip()
assert "# /// script" in files("roboskills.tools.sam3").joinpath("sam3.py").read_text()
character_skill = files("roboskills.workflows.motion_gen.character").joinpath("SKILL.md").read_text()
assert character_skill.startswith("---\nname: character\n")
for resource in ["mano/meshes/right_meshes/palm_vis.obj", "franka/2f85/base.stl", "franka/panda/LICENSE"]:
    assert files("roboskills.tools.mujoco").joinpath(resource).is_file(), resource
for resource in ["cams.json", "full_eval_set.json", "setup.sh"]:
    assert files("roboskills.eval.robodojo").joinpath(resource).is_file(), resource
for name in BENCHMARKS:
    for resource in ("setup.sh", "pyproject.toml", "uv.lock"):
        assert files(f"roboskills.eval.{name}").joinpath(resource).is_file(), (name, resource)
assert files("roboskills.workflows.real2sim").joinpath("SKILL.md").read_text().startswith("---\nname: real2sim\n")
assert "{api}" in files("roboskills.workflows.agent_as_policy").joinpath("SKILL.md").read_text()
assert development_layouts([35, 41], stop=55) == [35, 41]
check_rgb_evidence(
    dict(simulator_depth=False, simulator_camera_calibration=False, simulator_object_poses=False),
    {"cam_head_rgb": np.zeros((2, 3, 3), dtype=np.uint8)},
)

for module in (
    "roboskills.workflows.auto_research.policy",
    "roboskills.workflows.motion_gen.character.character",
    "roboskills.tools.robokit.action",
    "roboskills.tools.robokit.bot",
    "roboskills.tools.robokit.perception",
    "roboskills.tools.robokit.planner",
    "roboskills.tools.sam3.sam3",
    "roboskills.tools.mujoco.engine",
    "roboskills.eval.robodojo.launch",
    *[f"roboskills.eval.{name}.launch" for name in BENCHMARKS],
):
    importlib.import_module(module)

for module in ["roboskills.workflows.agentic_sim.check", "roboskills.tools.moge3"]:
    subprocess.run([sys.executable, "-m", module, "--help"], check=True, capture_output=True, text=True)

assert "torch" not in sys.modules and "robokit" not in sys.modules
print("Installed wheel: canonical imports, resources and CLIs passed")
