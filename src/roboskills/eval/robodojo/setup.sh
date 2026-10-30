#!/usr/bin/env bash
# Set up RoboDojo itself for this skill at $ROBODOJO_VENDOR (outside the package): the benchmark's code (tasks, layouts,
# success checks) at a pinned commit and its robot/object/layout assets at a pinned Hugging Face revision. Safe to
# rerun. On a shared home run it once; launch.py mounts the checkout read-only into the sim container.
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
vendor="${ROBODOJO_VENDOR:?set ROBODOJO_VENDOR to the RoboDojo checkout path (outside src/)}"

REV=726e9aabfaa642203722eb126f5eaf0f37f3e1ad     # github.com/RoboDojo-Benchmark/RoboDojo
HF_REV=91f76c28d93dd20c5fa46ce6a5a1d96a4f384acd  # huggingface.co/datasets/RoboDojo-Benchmark/RoboDojo (Assets/)
[ -d "$vendor" ] || git clone -q https://github.com/RoboDojo-Benchmark/RoboDojo.git "$vendor"
git -C "$vendor" fetch -q origin && git -C "$vendor" checkout -q "$REV"
[ -d "$vendor/third_party/IsaacLab/source" ] || git -C "$vendor" submodule update -q --init third_party/IsaacLab  # for the Dockerfile

if [ ! -d "$vendor/Assets/Eval_Layout" ]; then
  uv run --no-project --with huggingface_hub python -c "from huggingface_hub import snapshot_download as s; \
s('RoboDojo-Benchmark/RoboDojo', repo_type='dataset', revision='$HF_REV', allow_patterns=['Assets/**'], local_dir='$vendor')"
fi

uv sync -q --project "$here" --frozen  # the host venv

echo "RoboDojo ready at $vendor ($(git -C "$vendor" rev-parse --short HEAD))"
