#!/usr/bin/env bash
# Prepare the host environment; CaP-X's robosuite fork and its assets install from git (pyproject.toml). Safe to rerun.
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
uv sync -q --project "$here" --frozen
uv pip install -q --python "$here/.venv/bin/python" --no-deps -e "$here/../../../.."
echo "Robosuite ready: $here/.venv"
