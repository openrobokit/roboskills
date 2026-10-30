"""motion_studio.py in its own environment (`uv run` of its inline dependencies, Blender's python): this Python never
imports bpy.

    python -m roboskills.tools.blender SCENE.glb MOTION.npz --out DIR [motion_studio.py options]
"""

import subprocess
import sys
from pathlib import Path


studio = Path(__file__).with_name("motion_studio.py")
raise SystemExit(subprocess.run(["uv", "run", "--quiet", str(studio), *sys.argv[1:]], check=False).returncode)
