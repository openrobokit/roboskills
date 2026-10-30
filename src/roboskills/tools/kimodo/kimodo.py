# /// script
# requires-python = "==3.10.*"
# dependencies = [
#   "kimodo @ git+https://github.com/nv-tlabs/kimodo.git@1aece8c124d73d255ceff5086d983b844c9f4e94",
#   "torch==2.11.0",
# ]
#
# [tool.uv]
# extra-build-dependencies = { kimodo = ["cmake"] }  # its C++ motion correction builds with CMake
#
# [tool.uv.sources]
# torch = { index = "pytorch-cu128" }
#
# [[tool.uv.index]]
# name = "pytorch-cu128"
# url = "https://download.pytorch.org/whl/cu128"
# explicit = true
# ///
"""Text to motion with Kimodo, in its own environment (`uv run` of this file; the models and their text encoder, Llama 3
8B Instruct with LLM2Vec, download once into the Hugging Face cache: Llama 3 is gated, so the token needs access).

    python -m roboskills.tools.kimodo --prompt "a person waves with the right hand" --out motion/ [--duration 4]
"""

import subprocess
import sys
from pathlib import Path


MODEL = "Kimodo-G1-RP-v1"


def generate_motion(prompt: str, out: Path, duration: float = 4.0, seed: int = 0, model: str = MODEL):
    """out/motion.csv (per frame root xyz, root wxyz and the G1's 29 joint angles; Z-up, +X forward, MuJoCo's frame)
    and out/motion.npz (joints, rotations and foot contacts in Kimodo's Y-up frame), 30 fps. The prompt is encoded by
    the text encoder serving at $TEXT_ENCODER_URL, else in this process (about 15 GB more GPU memory)."""
    Path(out).mkdir(parents=True, exist_ok=True)
    stem = Path(out).resolve() / "motion"
    argv = [prompt, "--model", model, "--duration", duration, "--seed", seed, "--output", stem]
    subprocess.run(["uv", "run", "--quiet", __file__, *map(str, argv)], check=True)


if __name__ == "__main__":
    sys.path.pop(0)  # this file's folder, so `import kimodo` is the package (Python 3.10 has no PYTHONSAFEPATH)
    from kimodo.scripts.generate import main

    main()
