#!/usr/bin/env bash
# Prepare LIBERO-Pro outside the package: pinned SDK checkouts, their asset config and the host venv. Safe to rerun.
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
vendor="${LIBERO_VENDOR:?set LIBERO_VENDOR to the SDK checkout directory (outside src/), e.g. \$PWD/runs/vendor/libero}"
mkdir -p "$vendor/config"
for spec in 'LIBERO-PRO https://github.com/uynitsuj/LIBERO-PRO.git 47aaa8038930bcdc84ab9ea2867e2ffc8039ab4a' \
  'robosuite https://github.com/Max-Fu/robosuite.git a498b087d4bc5a3981e3d27030d09bc537a537f3'; do
  read -r name url revision <<< "$spec"
  [ -d "$vendor/$name/.git" ] || git clone -q --filter=blob:none "$url" "$vendor/$name"
  git -C "$vendor/$name" checkout -q --detach "$revision"
done
# robosuite's optional file log is a shared /tmp file that can belong to another user; disable it in this checkout.
macros="$vendor/robosuite/robosuite/macros_private.py"
[ -f "$macros" ] || printf 'from . import macros\nmacros.FILE_LOGGING_LEVEL = None\n' > "$macros"
# LIBERO reads its asset paths from $LIBERO_CONFIG_PATH/config.yaml (env.sh) instead of asking interactively.
base="$vendor/LIBERO-PRO/libero/libero"
cat > "$vendor/config/config.yaml" <<YAML
benchmark_root: $base
bddl_files: $base/bddl_files
init_states: $base/init_files
datasets: $vendor/LIBERO-PRO/libero/datasets
assets: $base/assets
YAML
uv sync -q --project "$here" --frozen
echo "LIBERO-Pro ready: $vendor, $here/.venv"
