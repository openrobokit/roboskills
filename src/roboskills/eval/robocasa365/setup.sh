#!/usr/bin/env bash
# Prepare the host environment: the venv (pyproject.toml), RoboCasa 1.0.1 from a checkout outside src/ on the venv's
# path, and RoboCasa's kitchen assets unpacked inside that checkout, where RoboCasa reads them (robocasa/models/assets).
# ROBOCASA picks the checkout (default runs/vendor/robocasa); 23 GB once unpacked. Each archive is checked against its
# SHA-256 and unpacked once (a marker records it). Safe to rerun.
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd -- "$here/../../../.." && pwd)"
vendor="${ROBOCASA:-$repo/runs/vendor/robocasa}"

REV=4f8a2980def75a55dff96b990745b83540425f09  # robocasa 1.0.1, as EFE's RoboCasa365 image pins it
[ -d "$vendor/.git" ] || git clone -q https://github.com/robocasa/robocasa.git "$vendor"
git -C "$vendor" checkout -q "$REV"

uv sync -q --project "$here" --frozen
py="$here/.venv/bin/python"
uv pip install -q --python "$py" --no-deps -e "$repo"
# RoboCasa's dependencies are in pyproject.toml; a path entry stands in for an editable install, whose build would walk
# every asset file its manifest names (minutes, and far longer on a network home).
site="$("$py" -I -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
echo "$vendor" > "$site/robocasa.pth"
# RoboCasa and robosuite warn on every import without a private macro file; theirs are the defaults.
[ -f "$vendor/robocasa/macros_private.py" ] || cp "$vendor/robocasa/macros.py" "$vendor/robocasa/macros_private.py"
rs="$("$py" -I -c 'import importlib.util as u; print(u.find_spec("robosuite").submodule_search_locations[0])')"
[ -f "$rs/macros_private.py" ] || cp "$rs/macros.py" "$rs/macros_private.py"

# EFE's RoboCasa365 archives and hashes (images/robocasa365/fetch_assets.sh): textures, AI-generated textures, the
# Lightwheel fixtures, and the Objaverse, AI-generated and Lightwheel objects.
assets="$vendor/robocasa/models/assets"
hf=https://huggingface.co/datasets/robocasa/robocasa-assets/resolve/1b92c3d02ca4354984fec961357db0bff7b32166
box=https://utexas.box.com/shared/static
fetch() {  # url sha256 destination
  local mark="$3/.robocasa365-$2"
  [ -f "$mark" ] && return
  mkdir -p "$3"
  local zip="$vendor/.download.zip"
  curl -fsSL --retry 3 "$1" -o "$zip"
  echo "$2  $zip" | sha256sum -c --quiet -
  "$py" -I -m zipfile -e "$zip" "$3"
  rm "$zip"
  touch "$mark"
}
fetch $hf/textures.zip ba3a4dcef96e199cbd601c7cdb7eb46077fb6672611b7ac74c25a191bf0be392 "$assets"
fetch $hf/generative_textures.zip bc05095af306222d9a799a9b0b5b16662aa546998622ee8fd3072566ea4c31ec "$assets"
fetch $box/idbncsadpnaz1jfl4i6m8qejawk7p9pi.zip 62c1c87776edda9b7c42dda7b00781a1e5a7f90acc3e49d06c39c9545cbd6c3e "$assets"
fetch $hf/objaverse.zip 66a7eebef3bac855301964d35d905fc2cb44d691f007f97af08d96e83db5cc08 "$assets/objects"
fetch $hf/aigen_objs.zip 1f88792322ccce79f775e88482626c324c6e07d002adf23320a124b72764dd19 "$assets/objects"
fetch $box/vckqvvkh1z8t69k8qcpcmee6k66stii4.zip a1db739b0e80fc4d56d4be27d89dba3aa60000e9648c5607874700e0f14b7955 "$assets/objects"

echo "RoboCasa365 ready: $here/.venv, RoboCasa at $vendor ($(git -C "$vendor" rev-parse --short HEAD))"
