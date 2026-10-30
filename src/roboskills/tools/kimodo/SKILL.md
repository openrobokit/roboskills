---
name: kimodo
description: Text-to-motion for the Unitree G1 (Kimodo-G1-RP-v1) or a SOMA or SMPL-X human skeleton with Kimodo, in its own uv environment (python -m roboskills.tools.kimodo).
---

# Kimodo

**Runs in** `uv run` of [`kimodo.py`](kimodo.py) (its inline dependencies: Kimodo at a pinned commit, its C++ motion
correction built with CMake and the host's compiler, torch 2.11.0 cu128) on `$CUDA_VISIBLE_DEVICES`. The models
(Kimodo-G1-RP-v1, the human Kimodo-SOMA-RP-v1.1 and Kimodo-SMPLX-RP-v1) and their text encoder (Llama 3 8B Instruct
with LLM2Vec, about 16 GB) download once into the Hugging Face cache when first used; Llama 3 is gated, so the token
in `~/.cache/huggingface/token` needs access.

**Call:** `python -m roboskills.tools.kimodo --prompt "PROMPT" --out motion/ [--duration 4] [--seed 0] [--model
Kimodo-G1-RP-v1]` (or `generate_motion(...)`) for one motion, the prompt encoded in process unless an encoder serves
at `$TEXT_ENCODER_URL`. The motion_gen workflow's `kimodo.sh` names `$KIMODO_PYTHON`, this environment's python, and
the encoder's port. Start the text encoder once per machine, then generate:

    $KIMODO_PYTHON -m kimodo.scripts.run_text_encoder_server --tmp-folder /tmp/encoder &
    $KIMODO_PYTHON -m kimodo.scripts.generate "PROMPT" --model $KIMODO_MODEL --duration 4 --seed 0 --output STEM

The encoder listens on `GRADIO_SERVER_PORT` (9571) and the generator finds it at `TEXT_ENCODER_URL`; a second run on
the same machine needs another port; when `TEXT_ENCODER_URL` is already set, an encoder serves there. The human models
are `--model Kimodo-SMPLX-RP-v1`, the SMPL-X body (22 joints; `$MOTION_KIMODO_HUMAN_MODEL`), and
`Kimodo-SOMA-RP-v1.1`, the SOMA skeleton.
`TEXT_ENCODER_MODE=local` encodes in process instead.

**Returns** `STEM.csv` (`motion/motion.csv`) (per frame root xyz, root wxyz and the G1's 29 joint angles; Z-up, +X forward, MuJoCo's frame)
and `STEM.npz` (joints, rotations and foot contacts in Kimodo's Y-up frame), 30 fps; `--bvh` with a human model writes
a SOMA BVH. The SMPL-X model also writes `STEM_amass.npz`, an AMASS file (`trans`, `root_orient`, `pose_body`,
`pose_hand`, `pose_jaw`, `pose_eye`, `betas`, `gender`, `mocap_frame_rate`; Z-up, +Y forward) that RoboKit's
`robokit.helpers.humanoid_retargeting.loaders.load_smplx_file` reads directly.

**GPU memory:** the text encoder about 15 GB (ready after about 40 s); a generation about 10 s.

**Known failures:** trained on clips of at most 10 s; stop an encoder you started when done.

**Upstream:** [Kimodo](https://github.com/nv-tlabs/kimodo).
