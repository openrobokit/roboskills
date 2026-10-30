#!/usr/bin/env bash
# Prepare EmbodiedSWE-Bench for this runtime. Safe to rerun; each part is skipped when present.
#   1. the host venv (the launcher side: roboskills, RoboKit) at .venv here;
#   2. the simulator image (docker hosts): docker/embodiedswe (pip's isaaclab 2.3.2 on Isaac Sim 5.1), pulled;
#   3. the EmbodiedSWE checkout at $ESWE_VENDOR at the pinned commit, with its assets at the manifest's revision
#      (about 2.6 GB from the Hugging Face dataset EmbodiedSWE/robobench-assets, fetched by its own script in the image).
# A shared home holds one checkout.
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd -- "$here/../../../.." && pwd)"
commit=046f7bf48865f3d3b48e64614350777fb5d5d163
image="${ESWE_IMAGE:-ghcr.io/openrobokit/roboskills-embodiedswe:0.2.0}"

uv sync -q --project "$here" --frozen
uv pip install -q --python "$here/.venv/bin/python" --no-deps -e "$repo"
echo "host venv: $here/.venv"

vendor="${ESWE_VENDOR:?set ESWE_VENDOR to the EmbodiedSWE checkout (outside src/)}"
if [ ! -d "$vendor/.git" ]; then
  git clone -q https://github.com/EmbodiedSWE/EmbodiedSWE "$vendor"
  git -C "$vendor" checkout -q "$commit"
fi
[ "$(git -C "$vendor" rev-parse HEAD)" = "$commit" ] || { echo "$vendor is not at $commit"; exit 2; }
mkdir -p "$vendor/.cache"

if command -v docker >/dev/null && docker info >/dev/null 2>&1; then
  docker image inspect "$image" >/dev/null 2>&1 || docker pull "$image"
  echo "image: $(docker image inspect "$image" --format '{{.Id}}')"
  docker run --rm --network host -u "$(id -u):$(id -g)" -e HOME=/tmp -e PYTHONPATH=/bench -v "$vendor":/bench -w /bench \
    --entrypoint /isaac-sim/python.sh "$image" -m robobench.scripts.fetch_assets
fi
echo "checkout: $vendor ($(git -C "$vendor" rev-parse --short HEAD), assets $(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['revision'][:8])" "$vendor/robobench/assets_manifest.json"))"
