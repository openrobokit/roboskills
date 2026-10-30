"""Shared by every launcher: `llm` (model CLIs), `agent` (turn loop and code execution), `episode` (live episodes)
and `process` (adapters and child processes); receipts and hashes are `roboskills.eval.records`. Nothing here knows a
simulator, robot or benchmark."""

from pathlib import Path

import roboskills.workflows


def skill(name, file="SKILL.md"):
    """A workflow's instructions: its SKILL.md (or another file of its folder) after the frontmatter and the blank line
    that follows it."""
    text = (Path(roboskills.workflows.__file__).parent / name / file).read_text()
    return text.split("\n---\n\n", 1)[1] if text.startswith("---\n") else text
