#!/usr/bin/env bash
# Prepare HumanoidArena for this runtime. Safe to rerun; each part is skipped when present and checked against its pin.
#   1. the host venv (the launcher side: roboskills, RoboKit) at .venv here;
#   2. the simulator image (docker/humanoidarena: Isaac Sim 6.1, Isaac Lab patched for 6.1, onnxruntime), pulled on a
#      docker host; that accepts the NVIDIA Isaac Sim EULA: pass --accept-nvidia-eula (HA_RUNTIME=apptainer skips it);
#   3. the vendor directory $HA_VENDOR (default runs/vendor/humanoidarena of this checkout, outside src/): upstream
#      HumanoidArena at its pinned commit, its released simulation assets (2.7 GB from Google Drive, unpacked into
#      isaaclab_twist2_g1/assets), GEAR-SONIC's encoder, decoder and kinematic planner (Hugging Face, 0.9 GB) and the
#      G1's URDF and meshes (cuRobo v0.8.0, for the arm model).
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd -- "$here/../../../.." && pwd)"
vendor="${HA_VENDOR:-$repo/runs/vendor/humanoidarena}"
image="${HA_IMAGE:-ghcr.io/openrobokit/roboskills-humanoidarena:0.2.0}"

ARENA_URL=https://github.com/William-wAng618/HumanoidArena.git
ARENA_REV=68479287a784a69be9ce6ad739311d2f11f75ef9  # release/open-source-prep, as RAX pins it
CUROBO_REV=4ea77366ca48ee453e7df139e39fa6532af49f3b  # cuRobo v0.8.0: its G1 rev 1.0 + Dex3 URDF and meshes
ASSETS_URL='https://drive.usercontent.google.com/download?id=1TCa_aVRmFrZs_l4wlxkqanNebvDtChNk&export=download&confirm=t'
ASSETS_SHA=d0a0c6be2e47ac9a3b161ec9ca11a9d67908ef755fe0fd96445404938376447b
SONIC=https://huggingface.co/nvidia/GEAR-SONIC/resolve/6733128a3d8a523b1418b06bca3cdf61c8b0987f
declare -A PINS=(
  [model_encoder.onnx]=013ab0287236aa2721e13f1e936d699db982302d0de0bfcdae76d5c3245362d3
  [model_decoder.onnx]=c7241a123eaa36b5d64bad19540efde93cac1ad443bd4572fd12ca99898118ed
  [observation_config.yaml]=466d05947c78af6c76388adfb86e3a2a77b2a1d921a64883ed3d085ebf58de1b
  [planner_sonic.onnx]=39b553e197f62f077975ba38512bc04781a3fc37c2af7c6756e04629f760edea
  [LICENSE]=24ab66be50d1aca4fc5e029ef76ce4ceaac6557ea21665caf4b140695a76ffee
)
check() { [ "$(sha256sum "$1" | cut -d' ' -f1)" = "$2" ]; }

if command -v uv >/dev/null; then
  uv sync -q --project "$here" --frozen
  uv pip install -q --python "$here/.venv/bin/python" --no-deps -e "$repo"
  echo "host venv: $here/.venv"
fi

if [ "${HA_RUNTIME:-}" != apptainer ] && command -v docker >/dev/null && docker info >/dev/null 2>&1 && ! docker image inspect "$image" >/dev/null 2>&1; then
  [ "${1:-}" = "--accept-nvidia-eula" ] || { echo "pass --accept-nvidia-eula to accept the NVIDIA Isaac Sim EULA and pull $image"; exit 2; }
  docker pull "$image"
fi
command -v docker >/dev/null && docker image inspect "$image" >/dev/null 2>&1 && echo "image: $image $(docker image inspect "$image" --format '{{.Id}}')"

mkdir -p "$vendor/sonic"
arena="$vendor/HumanoidArena"
[ -d "$arena/.git" ] || git clone -q "$ARENA_URL" "$arena"
git -C "$arena" cat-file -e "$ARENA_REV" 2>/dev/null || git -C "$arena" fetch -q origin "$ARENA_REV"
git -C "$arena" -c advice.detachedHead=false checkout -q "$ARENA_REV"
assets="$arena/isaaclab_twist2_g1/assets"
if [ ! -f "$assets/.sha256" ] || [ "$(cat "$assets/.sha256")" != "$ASSETS_SHA" ]; then
  zip="$vendor/humanoidarena-assets.zip"
  if ! { [ -f "$zip" ] && check "$zip" "$ASSETS_SHA"; }; then
    curl -L --fail --retry 3 -o "$zip.partial" "$ASSETS_URL"
    check "$zip.partial" "$ASSETS_SHA" || { echo "the assets archive does not match its pin" >&2; exit 1; }
    mv "$zip.partial" "$zip"
  fi
  rm -rf "$vendor/unpack" && mkdir -p "$vendor/unpack" && unzip -q "$zip" -d "$vendor/unpack"
  top="$vendor/unpack"; [ -d "$top/assets" ] && top="$top/assets"
  [ -d "$top/objects" ] && [ -d "$top/robots" ] || { echo "unexpected assets archive layout" >&2; exit 1; }
  rm -rf "$assets" && mv "$top" "$assets" && rm -rf "$vendor/unpack" "$zip"
  echo "$ASSETS_SHA" > "$assets/.sha256"
fi
for name in "${!PINS[@]}"; do
  f="$vendor/sonic/$name"
  { [ -f "$f" ] && check "$f" "${PINS[$name]}"; } && continue
  curl -L --fail --retry 3 -o "$f.partial" "$SONIC/$name"
  check "$f.partial" "${PINS[$name]}" || { echo "$name does not match its pin" >&2; exit 1; }
  mv "$f.partial" "$f"
done
# The G1 description the simulator's robot matches (rev 1.0 with the current Dex3; upstream's own URDF is an older
# revision, 1 cm off at the shoulders): Unitree's, as cuRobo ships it, at RAX's pin.
g1="$vendor/curobo"
[ -d "$g1/.git" ] || git clone -q --filter=blob:none --no-checkout https://github.com/NVlabs/curobo.git "$g1"
git -C "$g1" sparse-checkout set curobo/content/assets/robot/g1
git -C "$g1" cat-file -e "$CUROBO_REV" 2>/dev/null || git -C "$g1" fetch -q origin "$CUROBO_REV"
git -C "$g1" -c advice.detachedHead=false checkout -q "$CUROBO_REV"
chmod -R a+rX "$vendor/sonic" "$arena" "$g1"  # (the simulator may run as another user: a container's root on a network home)
echo "vendor: $vendor (HumanoidArena $(git -C "$arena" rev-parse --short HEAD), assets, SONIC encoder/decoder/planner, G1 $(git -C "$g1" rev-parse --short HEAD))"
