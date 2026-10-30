"""Simulator integrations must stay outside the shared execution workflow."""

import ast
from pathlib import Path


def test_workflow_has_no_benchmark_integrations_or_simulator_imports():
    from roboskills.workflows import agentic_sim
    from roboskills.workflows.agentic_sim import run

    root = Path(agentic_sim.__file__).parent
    assert not any(p.is_dir() and p.name not in {"tests", "__pycache__"} for p in root.iterdir())
    forbidden = {"artifacts", "sapien", "isaacsim", "mujoco", "envs", "robokit", "torch"}
    for path in [*root.glob("*.py"), Path(run.__file__)]:
        for node in ast.walk(ast.parse(path.read_text())):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            assert all(name.split(".")[0] not in forbidden for name in names), path
