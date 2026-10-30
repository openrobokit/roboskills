---
name: sam3
description: Instance masks for text prompts with SAM 3, per image with the model kept loaded (roboskills.tools.sam3.sam3.Sam) or tracked through a video into one indexed PNG per frame (python -m roboskills.tools.sam3).
---

# SAM 3

**Runs in** `uv run` of [`sam3.py`](sam3.py) on `$CUDA_VISIBLE_DEVICES`: its inline dependencies (transformers, torch;
prebuilt wheels), and the gated `facebook/sam3` weights, downloaded once into the Hugging Face cache with a token that
has access.

**Video:** `python -m roboskills.tools.sam3 --video V --prompts "sugar box" "right hand" --out masks/ [--frame 0]`
(or `segment_video(...)`) writes `masks/00000.png` ... (indexed: 0 background, i the i-th prompt's object, later prompts on top)
and `masks/objects.json` (id -> prompt, detected text, score, box). A prompt not found at `--frame` falls back to its
last word, else is left out.

**Call:** `sam = Sam()` (loads once), `sam.detect(rgb, prompts, threshold=0.3)` for an (H, W, 3) uint8 image and a list
of noun phrases; `sam.close()` when done. The robot tools' `bot.segment` wraps it per camera.

**Returns** `{prompt: [{id, score, box_px, area_px, mask}]}`, best first: `score` in [0, 1], `box_px` [x0, y0, x1, y1]
and `area_px` in image pixels, `mask` (H, W) bool. The worker's wire is one JSON line per request (`png` base64,
`prompts`, `threshold`) and one `SAM3 `-prefixed JSON reply (`ok`, `detections`, `masks`: a base64 `.npz` by id).

**GPU memory:** the model stays resident while `Sam` is open; close it before reusing its GPU allocation.

**Known failures:** a worker that exits raises "the SAM 3 worker exited"; a failed request raises "SAM 3 failed";
`close()` kills a stuck worker's own process group.

**Upstream:** [SAM 3](https://github.com/facebookresearch/sam3).
