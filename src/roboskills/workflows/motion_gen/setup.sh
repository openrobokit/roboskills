#!/usr/bin/env bash
# Make the tools SKILL.md names (Kimodo, GVHMR, Blender, WiLoR, SAM 3) ready, each in its own uv environment (its file's
# inline dependencies in src/roboskills/tools/<tool>/), and write their setup files into runs/motion_gen/env/.
#
#   bash src/roboskills/workflows/motion_gen/setup.sh [ENV_DIR]
#
# Needs uv, the repository environment (uv sync), SMPL-X body models in ~/.robokit/body_models/smplx and MANO's right
# hand in ~/.robokit/body_models/mano/MANO_RIGHT.pkl: they are licensed, so GVHMR and WiLoR read them there. Weights
# download on first use: Kimodo's and SAM 3's into the Hugging Face cache (Llama 3, Kimodo's text encoder, and
# facebook/sam3 are gated: a token with access), GVHMR's and WiLoR's into ~/.roboskills/weights/.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../../../.." && pwd)
E=$(mkdir -p "${1:-$REPO/runs/motion_gen/env}/bin" && cd "${1:-$REPO/runs/motion_gen/env}" && pwd)
TOOLS=$REPO/src/roboskills/tools
[ -x "$REPO/.venv/bin/python" ] || { echo "run uv sync in $REPO first" >&2; exit 1; }

# Each tool's python: its file's own environment, made once
python_of() {  # python_of SCRIPT
  uv sync --quiet --script "$1" && uv python find --script "$1"
}
KIMODO=$(python_of "$TOOLS/kimodo/kimodo.py")
GVHMR=$(python_of "$TOOLS/gvhmr/gvhmr.py")
BLENDER=$(python_of "$TOOLS/blender/motion_studio.py")
uv sync --quiet --script "$TOOLS/sam3/sam3.py"
# WiLoR's checkout (not a package) and weights, made once, go on its python's path
WILOR=$("$REPO/.venv/bin/python" -c 'from roboskills.tools.wilor.wilor import prepare; print(prepare())')
printf '#!/bin/sh\nPYTHONPATH=%s TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYOPENGL_PLATFORM=egl PYGLET_HEADLESS=1 exec %s "$@"\n' \
  "$WILOR" "$(python_of "$TOOLS/wilor/wilor.py")" > "$E/bin/wilor-python"
chmod +x "$E/bin/wilor-python"

cat > "$E/kimodo.sh" <<EOF
# Kimodo: text-to-motion, native Unitree G1, SOMA human and SMPL-X human (tools/kimodo: source 1aece8c1; the models
# from the Hugging Face cache). When TEXT_ENCODER_URL is already set, an encoder is serving there: use it rather
# than starting another. Otherwise start the text encoder once per machine (about 15 GB of GPU memory, ready after
# about 40 s) on port 9571; another run on the same machine needs another port, GRADIO_SERVER_PORT for the encoder and
# TEXT_ENCODER_URL to match for generate.
#   \$KIMODO_PYTHON -m kimodo.scripts.run_text_encoder_server --tmp-folder /tmp/encoder &
# Generate (writes STEM.npz and STEM.csv; with the SOMA model, --bvh writes STEM.bvh; --model Kimodo-SMPLX-RP-v1 writes
# STEM_amass.npz, an AMASS SMPL-X file robokit.helpers.humanoid_retargeting.loaders.load_smplx_file reads):
#   \$KIMODO_PYTHON -m kimodo.scripts.generate "PROMPT" --model \$KIMODO_MODEL --duration 4 --seed 0 --output STEM [--constraints c.json]
export KIMODO_PYTHON=$KIMODO KIMODO_MODEL=Kimodo-G1-RP-v1 MOTION_KIMODO_HUMAN_MODEL=Kimodo-SMPLX-RP-v1
export TEXT_ENCODER_MODE=\${TEXT_ENCODER_MODE:-api} GRADIO_SERVER_NAME=127.0.0.1 GRADIO_TEMP_DIR=/tmp/gradio
export GRADIO_SERVER_PORT=\${GRADIO_SERVER_PORT:-9571} TEXT_ENCODER_URL=\${TEXT_ENCODER_URL:-http://127.0.0.1:9571}
EOF
cat > "$E/gvhmr.sh" <<EOF
# GVHMR 1.6.1 (tools/gvhmr) for video to SMPL-X motion. For example:
#   \$GVHMR_PYTHON -c 'import gvhmr; r=gvhmr.recover("in.mp4", static_camera=True, device="cuda", output_root="out", output_dir="out/in", progress=False); r.save_npz("out/in/smplx.npz")'
export GVHMR_PYTHON=$GVHMR OMP_NUM_THREADS=\${OMP_NUM_THREADS:-16}
export GVHMR_CHECKPOINTS=$HOME/.roboskills/weights/gvhmr GVHMR_BODY_MODELS=$HOME/.robokit/body_models
EOF
cat > "$E/blender.sh" <<EOF
# Blender 4.5 as the bpy module (tools/blender). Cycles uses the run's GPUs; EEVEE and Workbench open the first. For example:
#   \$BLENDER_PYTHON -c 'import bpy; bpy.ops.import_scene.fbx(filepath="char.fbx"); print([o.name for o in bpy.data.objects])'
export BLENDER_PYTHON=$BLENDER
export MOTION_STUDIO="$REPO/src/roboskills/tools/blender/motion_studio.py"
EOF
cat > "$E/character.sh" <<EOF
# The character retargeting's functions run on the host (this repository's environment).
. "$E/gvhmr.sh"
. "$E/blender.sh"
export ROBOSKILLS_PYTHON="$REPO/.venv/bin/python"
export CHARACTER_SKILL="$REPO/src/roboskills/workflows/motion_gen/character/SKILL.md"
EOF
cat > "$E/wilor.sh" <<EOF
# WiLoR (tools/wilor: source fcb91131; its checkout and weights in $WILOR, on WILOR_PYTHON's path, MANO linked) for a
# video's hands: per image, a detector's hand boxes (cls 1 right, 0 left), and per box MANO's global_orient, hand_pose
# (15 joints, each in its parent), pred_keypoints_3d [21, 3] and pred_cam. The model is a 641 M-parameter ViT-H
# (2.6 GB), slow to load: load it once. Batching crops (from any frames) runs the model several times faster per hand
# than one crop at a time, and the detector likewise on lists of frames; cutting each crop out on the CPU
# (ViTDetDataset) then costs the most, so prepare crops in parallel. Save results as you go. For example:
#   "\$WILOR_PYTHON" - frame.jpg <<'PY'
#   import sys, cv2, torch; from ultralytics import YOLO; from wilor.models import load_wilor; from wilor.datasets.vitdet_dataset import ViTDetDataset
#   W = "$WILOR/pretrained_models/"; model, cfg = load_wilor(W + "wilor_final.ckpt", W + "model_config.yaml"); model = model.cuda().eval()
#   image = cv2.imread(sys.argv[1]); found = YOLO(W + "detector.pt")(image, conf=0.3, verbose=False)[0].boxes
#   data = ViTDetDataset(cfg, image, found.xyxy.cpu().numpy(), found.cls.cpu().numpy(), rescale_factor=2.0)
#   batch = {k: v.cuda() for k, v in torch.utils.data.default_collate([data[i] for i in range(len(data))]).items()}
#   with torch.no_grad(): out = model(batch)  # out["pred_mano_params"]["global_orient"] [N,1,3,3], ["hand_pose"] [N,15,3,3]
#   PY
export WILOR_PYTHON=$E/bin/wilor-python WILOR=$WILOR
EOF
cat > "$E/sam3.sh" <<EOF
# SAM 3 (tools/sam3: facebook/sam3 from the Hugging Face cache) for an object's masks from text prompts, per image;
# detect frame by frame. roboskills.tools.sam3.sam3.Sam, from this repository's python, keeps the model loaded in its
# own uv environment's worker process on the run's first GPU; images go in over a pipe.
# For example:
#   "\$ROBOSKILLS_PYTHON" - frame.png <<'PY'
#   import sys, numpy as np; from PIL import Image; from roboskills.tools.sam3.sam3 import Sam
#   sam = Sam()  # loads once: detect every image you need, then close
#   found = sam.detect(np.asarray(Image.open(sys.argv[1]).convert("RGB")), ["umbrella", "parasol"], threshold=0.3)
#   sam.close()  # found: {prompt: [{score, box_px, area_px, mask (H, W) bool}]}, best first
#   PY
export ROBOSKILLS_PYTHON="$REPO/.venv/bin/python"
EOF
echo "Tools ready in $E"
