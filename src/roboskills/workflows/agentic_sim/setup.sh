#!/usr/bin/env bash
# The simulators a build agent may choose from (MuJoCo, PyBullet, Newton with MuJoCo Warp), two renderers (Blender's
# Cycles as bpy, NVIDIA's RTX as ovrtx), robokit and this package, in one venv:
#
#   bash src/roboskills/workflows/agentic_sim/setup.sh [ENV_DIR]    # default runs/agentic_sim/env; then ENV_DIR/bin/python
#
# Needs uv. Rerunning reinstalls into the same venv.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../../../.." && pwd)
E=${1:-$REPO/runs/agentic_sim/env}
[[ -x $E/bin/python ]] || uv venv -q -p 3.13 "$E"  # bpy has wheels for 3.13, not 3.12
uv pip install -q -p "$E/bin/python" "newton[sim,importers]==1.7.0" "pybullet==3.2.7" "bpy==5.2.3" \
    "imageio==2.38.1" "imageio-ffmpeg==0.6.0" \
    "robokit>=1.0.0rc1" -e "$REPO"
# ovrtx's wheels need glibc 2.35; on older hosts the venv goes without it.
uv pip install -q -p "$E/bin/python" "ovrtx==0.6.0.1791947691" 2>/dev/null || echo "ovrtx skipped: no wheel for this host"
"$E/bin/python" -c "import mujoco, pybullet, newton, bpy, robokit; print('mujoco', mujoco.__version__, 'newton', newton.__version__)"
