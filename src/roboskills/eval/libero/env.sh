#!/usr/bin/env bash
# Source before running LIBERO: the vendored SDKs and this checkout on PYTHONPATH, their config, headless rendering.
export PATH="$HOME/.local/bin:$PATH"
libero_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)"
libero_vendor="${LIBERO_VENDOR:-$libero_root/runs/vendor/libero}"
export LIBERO_PYTHON="$libero_root/src/roboskills/eval/libero/.venv/bin/python"
export PYTHONPATH="$libero_root/src:$libero_root:$libero_vendor/robosuite:$libero_vendor/LIBERO-PRO/libero${PYTHONPATH:+:$PYTHONPATH}"
export LIBERO_CONFIG_PATH="$libero_vendor/config"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_PLATFORM=device
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
