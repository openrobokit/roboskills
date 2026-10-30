#!/usr/bin/env bash
# Prepare BEHAVIOR-1K for this runtime. Safe to rerun; each part is skipped when present.
#   1. the host venv (the launcher side: roboskills, RoboKit) at .venv here;
#   2. the official simulator image by digest (docker hosts);
#   3. the dataset at $BEHAVIOR_DATA (about 35 GB) with BEHAVIOR's own downloader, run in the image. It asks you to accept
#      the BEHAVIOR Data Bundle licence (no redistribution): pass --accept-license to accept it non-interactively.
# A shared home holds one copy of each; other docker hosts pull the image once.
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
image="${BH_IMAGE:-docker.io/stanfordvl/behavior:3.9.2@sha256:2702900192741904e69a573aa7869ea51c592586f9791050796efc37faa893fb}"

uv sync -q --project "$here" --frozen
uv pip install -q --python "$here/.venv/bin/python" --no-deps -e "$here/../../../.."
echo "host venv: $here/.venv"

if command -v docker >/dev/null && docker info >/dev/null 2>&1; then
  docker image inspect "$image" >/dev/null 2>&1 || docker pull "$image"
  echo "image: $(docker image inspect "$image" --format '{{.Id}}')"
fi

data="${BEHAVIOR_DATA:?set BEHAVIOR_DATA to the dataset directory (outside src/)}"
if [ ! -f "$data/omnigibson.key" ] || [ ! -d "$data/behavior-1k-assets" ] || [ ! -d "$data/2026-challenge-task-instances" ]; then
  [ "${1:-}" = "--accept-license" ] || { echo "pass --accept-license to accept the BEHAVIOR Data Bundle licence and download"; exit 2; }
  mkdir -p "$data"
  docker run --rm --network host -u "$(id -u):$(id -g)" -e HOME=/tmp -e USER="$(id -un)" -e LOGNAME="$(id -un)" \
    -e HF_HOME=/tmp/hf -e OMNIGIBSON_DATA_PATH=/data -e OMNIGIBSON_APPDATA_PATH=/tmp/appdata -e OMNI_KIT_ACCEPT_EULA=YES \
    -v "$data":/data --entrypoint /opt/conda/envs/behavior/bin/python "$image" -m omnigibson.utils.asset_utils \
    --download_omnigibson_robot_assets --download_behavior_1k_assets --download_2026_challenge_task_instances --accept_license
fi
echo "dataset: $data (robot assets $(cat "$data/omnigibson-robot-assets/VERSION"), BEHAVIOR-1K assets $(cat "$data/behavior-1k-assets/VERSION"))"
