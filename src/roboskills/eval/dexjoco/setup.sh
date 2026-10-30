#!/usr/bin/env bash
set -euo pipefail
# Prepare the native benchmark outside the package and the host venv (pyproject.toml/uv.lock). Safe to rerun.
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
vendor="${DEXJOCO_VENDOR:?set DEXJOCO_VENDOR to the DexJoCo checkout path (outside src/)}"
[ -d "$vendor" ] || git clone https://github.com/brave-eai/dexjoco.git "$vendor"
git -C "$vendor" fetch origin
git -C "$vendor" checkout 8d23b0fab23b17a58c4b55f3942e17013aaf8267
uv sync -q --project "$here" --frozen
echo "DexJoCo ready: $vendor, $here/.venv"
