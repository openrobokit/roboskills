"""Size and ownership contracts for the real2sim workflow and the two tools it calls."""

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
FOLDERS = ("src/roboskills/workflows/real2sim", "src/roboskills/tools/mujoco", "src/roboskills/tools/moge3")
AGENT = ROOT / "src/roboskills/workflows/real2sim/agent.py"  # the `agent` tool calls Codex


def sources(extension):
    return [p for folder in FOLDERS for p in (ROOT / folder).rglob(f"*.{extension}") if "__pycache__" not in p.parts]


def test_real2sim_workflow_stays_below_800_python_lines():
    found = [p for p in (ROOT / FOLDERS[0]).rglob("*.py") if "__pycache__" not in p.parts]
    sizes = {str(path.relative_to(ROOT)): len(path.read_text().splitlines()) for path in found}
    assert sizes, "Missing real2sim Python sources"
    assert sum(sizes.values()) < 800, f"Python line budget exceeded: {sizes}"


def test_core_never_imports_another_workflow_or_deployment_infrastructure():
    foreign = ("roboskills.workflows", "roboskills.core")
    violations = []
    for path in sources("py"):
        allowed = ("roboskills.core",) if path == AGENT else ()  # only this drives a model CLI
        if path.is_relative_to(ROOT / FOLDERS[0]):
            allowed += (FOLDERS[0].removeprefix("src/").replace("/", "."),)  # its own modules
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [f"{node.module}.{alias.name}" for alias in node.names]
            else:
                continue
            within = lambda module, prefixes: any(module == p or module.startswith(p + ".") for p in prefixes)
            internal = lambda module: module.split(".")[0].endswith("_internal")
            violations += [
                f"{path.name}:{node.lineno}: {m}"
                for m in modules
                if internal(m) or (within(m, foreign) and not within(m, allowed))
            ]
    assert not violations, "Core logic must stay in real2sim: " + ", ".join(violations)
