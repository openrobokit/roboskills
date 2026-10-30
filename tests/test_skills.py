"""Every tool's and workflow's SKILL.md: frontmatter with a name and a description, and only images docker/ defines."""

import re
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SKILLS = sorted(p for kind in ("tools", "workflows") for p in (ROOT / "src/roboskills" / kind).rglob("SKILL.md"))
IMAGE = re.compile(r"ghcr\.io/openrobokit/roboskills-([a-z0-9]+):([0-9.]+)")


def test_every_tool_and_workflow_has_one():
    folders = {p.parent.relative_to(ROOT / "src/roboskills").parts[:2] for p in SKILLS}
    for kind in ("tools", "workflows"):
        for folder in (ROOT / "src/roboskills" / kind).iterdir():
            if folder.is_dir() and folder.name != "__pycache__":
                assert (kind, folder.name) in folders, f"{kind}/{folder.name} has no SKILL.md"


@pytest.mark.parametrize("path", SKILLS, ids=lambda p: str(p.parent.relative_to(ROOT / "src/roboskills")))
def test_frontmatter_and_images(path):
    text = path.read_text()
    assert text.startswith("---\n") and "\n---\n\n" in text, "frontmatter, then a blank line"
    meta = yaml.safe_load(text[4 : text.index("\n---\n")])
    assert isinstance(meta.get("name"), str) and meta["name"].strip()
    assert isinstance(meta.get("description"), str) and meta["description"].strip()
    images = (ROOT / "docker/README.md").read_text()
    for name, version in IMAGE.findall(text):
        assert (ROOT / "docker" / name / "Dockerfile").is_file(), f"no docker/{name}"
        assert f"| `{name}` {version} |" in images, f"docker/README.md lists no {name} {version}"
