#!/usr/bin/env bash
# A tool's weights, kept out of its image: ~/.roboskills/weights/NAME, filled once per home with the files and pinned revisions its image bundled up to 0.2.0; the tool mounts it read-only at /weights/NAME.
#
#   docker/weights.sh sam3d   # 15.6 GB: facebook/sam-3d-objects' checkpoints (gated), MoGe ViT-L, DINOv2 ViT-L/14 reg
#   docker/weights.sh hawor   # 3.3 GB: HaWoR's hand checkpoint and model config
#
# Gated repos read a Hugging Face token with access from ~/.cache/huggingface/token ($HF_TOKEN_PATH), never printed.
# Needs uv (hf.sh runs Hugging Face's CLI through uvx), git and curl. It fills NAME.partial, which a rerun resumes, and
# renames it to NAME when complete; a complete NAME is left alone.
set -euo pipefail
case ${1:-} in
  sam3d | hawor) name=$1 ;;
  *) echo "usage: $0 sam3d|hawor (the tools whose weights are mounted)" >&2; exit 2 ;;
esac
hf="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/hf.sh"
root=$HOME/.roboskills/weights
dest=$root/$name part=$root/$name.partial
if [ -d "$dest" ]; then
  echo "$dest is already filled"
  exit 0
fi
command -v uvx > /dev/null || { echo "needs uv's uvx on PATH" >&2; exit 1; }
export HF_TOKEN_PATH=${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}
mkdir -p "$part"
if [ "$name" = sam3d ]; then
  if [ ! -s "$HF_TOKEN_PATH" ]; then
    echo "facebook/sam-3d-objects is gated: a token with access goes in $HF_TOKEN_PATH" >&2
    exit 1
  fi
  sh "$hf" facebook/sam-3d-objects 2e73555018d2741ccd486e56c24fac41155a1dc6 --include "checkpoints/*" \
    --local-dir "$part"
  HF_HOME=$part/hf sh "$hf" Ruicheng/moge-vitl ad326bfb61facd6c52b5a825bc1e34d7c97d9672
  # DINOv2 as torch.hub caches it: its source at a pinned commit, the checkpoint by SHA-256, the repo trusted
  hub=$part/torch/hub
  dino=$hub/facebookresearch_dinov2_main ckpt=$hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth
  if [ ! -f "$dino/hubconf.py" ]; then
    rm -rf "$dino" && git init -q "$dino"
    git -C "$dino" fetch -q --depth 1 https://github.com/facebookresearch/dinov2.git \
      7764ea0f912e53c92e82eb78a2a1631e92725fc8
    git -C "$dino" checkout -q FETCH_HEAD && rm -rf "$dino/.git"
  fi
  sum="36e4deffbaef061a2576705b0c36f93621e2ae20bf6274694821b0b492551b51  $ckpt"
  if ! echo "$sum" | sha256sum -c --status - 2>/dev/null; then
    mkdir -p "${ckpt%/*}"
    curl -fsSL -o "$ckpt" https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth
    echo "$sum" | sha256sum -c --quiet -
  fi
  printf 'facebookresearch_dinov2\n' > "$hub/trusted_list"
else
  sh "$hf" ThunderVVV/HaWoR da6335f47f9806308992d5ae1002a4cc5f7252c2 --include hawor/checkpoints/hawor.ckpt \
    --include hawor/model_config.yaml --local-dir "$part"
fi
# readable as in the images; Docker's daemon resolves the mount as root, which an NFS home squashes
chmod -R a+rX "$part" && chmod a+x "$HOME/.roboskills" "$root"
mv "$part" "$dest"
du -sh "$dest"
