#!/bin/sh
# Weights into a tool's image: one Hugging Face repo at a pinned revision, in a RUN step of its Dockerfile.
#
#   hf.sh REPO REVISION [hf download args...]
#
# Without --local-dir the files land in $HF_HOME's cache, refs/main naming the revision, so the tool loads REPO offline
# ($HF_HUB_OFFLINE=1) as its main; with it they land in that directory, without the download's metadata. A gated repo
# needs the build's secret `hf`, a token with access (RUN --mount=type=secret,id=hf). What lacks it is made
# world-readable (containers run as their caller), and nothing else is touched: a chmod would copy an earlier layer's
# files into this one.
set -eu
repo=$1 rev=$2
shift 2
token=$(cat /run/secrets/hf 2>/dev/null || true)
[ -n "$token" ] && export HF_TOKEN="$token"
HF_HUB_OFFLINE=0 uvx -q --from huggingface_hub==1.33.0 hf download "$repo" --revision "$rev" "$@" >/dev/null
dir=
while [ $# -gt 0 ]; do
  [ "$1" = --local-dir ] && dir=$2
  shift
done
if [ -n "$dir" ]; then
  rm -rf "$dir/.cache"
else
  dir=$HF_HOME
  refs="$HF_HOME/hub/models--$(echo "$repo" | sed 's#/#--#g')/refs"
  mkdir -p "$refs" && printf %s "$rev" > "$refs/main"
fi
find "$dir" \( ! -perm -a+r -o -type d ! -perm -a+x \) -exec chmod a+rX {} +
