#!/usr/bin/env bash
# Set up RoboTwin 2.0 for the runtime: pinned checkout outside src/, clean-config assets, the venv. Safe to rerun.
# RoboTwin's own scripts/_install.sh also patches two site-packages files; this path needs neither: sapien's URDF
# loader reads UTF-8 under PYTHONUTF8=1 (run.py sets it for every sim), and mplib's screw planner is unused
# (actions are joint targets; the aloha-agilex planner is CuRobo).
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
vendor="${ROBOTWIN:?set ROBOTWIN to the RoboTwin checkout path (outside src/), e.g. \$PWD/runs/vendor/RoboTwin}"
export PYTHONUTF8=1

REV=6dde57155eafa3e4ebf6ad1f93a7cf7d5d41a755
[ -d "$vendor/.git" ] || git clone -q https://github.com/RoboTwin-Platform/RoboTwin.git "$vendor"
git -C "$vendor" checkout -q "$REV"

uv sync -q --project "$here" --frozen
uv pip install -q --python "$here/.venv/bin/python" --no-deps -e "$here/../../../.."

cd "$vendor/assets"
for z in embodiments objects; do  # background_texture (11 GB) only matters for randomized configs
  if [ ! -d "$z" ]; then
    [ -f "$z.zip" ] || uv run --project "$here" python -c "from huggingface_hub import hf_hub_download as d; d('TianxingChen/RoboTwin2.0', '$z.zip', repo_type='dataset', local_dir='.')"
    unzip -q "$z.zip" && rm "$z.zip"
  fi
done
cd "$vendor"
uv run --project "$here" python scripts/update_embodiment_config_path.py

echo "RoboTwin ready at $vendor ($(git rev-parse --short HEAD))"
