---
name: blender
description: Render scenes and animated characters with Blender 4.5's bpy module in its own uv environment, including motion_gen's shared renderer roboskills/tools/blender/motion_studio.py.
---

# Blender

**Runs in** `uv run` of [`motion_studio.py`](motion_studio.py) (its inline dependencies: Blender 4.5 as the `bpy`
module, NumPy; Python 3.11). Cycles renders on the GPU through CUDA and OptiX; EEVEE and Workbench open EGL on the
first GPU. `bpy` loads the host's X11 and OpenGL libraries (`libxi6`, `libxxf86vm1`, `libxfixes3`, `libxkbcommon0`,
`libsm6`, `libgl1`).

**Call:** `python -m roboskills.tools.blender SCENE.glb MOTION.npz --out DIR [options]` runs it there. The motion_gen
workflow's `blender.sh` names `$BLENDER_PYTHON` (that environment's python) and `$MOTION_STUDIO`
([`motion_studio.py`](motion_studio.py), which imports only NumPy and bpy):

    $BLENDER_PYTHON "$MOTION_STUDIO" SCENE.glb MOTION.npz --out DIR [--frames 0,12 | --final] [--camera source]
        [--view AZ,EL] [--frame-on NAMES [--follow]] [--camera-from DIR] [--environment studio|outdoor|warm [--kitchen]]
        [--engine cycles|eevee|workbench]

**MuJoCo scenes:** [`blender.py`](blender.py)'s `render_scene(scene, video, out)`, the real2sim workflow's blender
step, renders a saved scene (`scene.mjb`, `trajectory.npz`, its `source` camera) with Cycles in the
docker image its `IMAGE` names, its boxes and meshes in their MuJoCo colours: `out/blender_ext1.mp4`
and `out/compare_ext1.mp4` (the video | Blender | `scene/mujoco_ext1.mp4`).

**A character and a motion:** [`blender.py`](blender.py)'s `import_character(source, out)` makes a Mixamo FBX the GLB
`character.load_character` reads, and `render_motion(character, motion, video, out)`, the motion_gen workflow's blender
step, bakes the motion's turns into `out/scene.glb` ([`rig.py`](rig.py), `turns`, `positions`, `joint_names`,
`rest_joints` in its MOTION, as `character/retarget.py` writes them), renders `--final` and encodes `out/video.mp4`
and `out/compare.mp4` (the video | the render).

**Returns** in `DIR`: by default twelve stills spread over the clip, `--frames` the listed saved frames, `--final`
every saved frame, as resumable numbered PNGs (960×540, 32 samples by default), a contact sheet and `render.json`
(preset, camera, per-frame timing). The camera is one fixed camera framing everything that moves over the clip,
from `--view` (degrees around the vertical from `studio_camera_direction`, else the source camera's side, and above
the ground); `--frame-on` frames named joints, links, GLB objects or the prop and `--follow` re-frames on them every
frame; `--camera source` draws from the video's camera at its aspect; `--camera-from` reuses an earlier render's
camera. `MOTION.npz` holds `fps`, `turns` (frame count) and `positions` (framing), optionally the source camera,
a renderer-drawn prop, `studio_bounds` and `studio_camera_direction`. `--engine eevee` (lit like Cycles) or
`workbench` (flat studio shading of the textures, with shadows) renders the same camera faster; give each its own
`DIR`, since existing PNGs are kept.

**GPU memory:** not separately recorded. Per 960×540 frame on a shared RTX 3090: Cycles about 1.5–2 s whatever the
samples, EEVEE about 0.6 s after about 35 s compiling shaders on its first frame, Workbench about 0.15 s.

**Known failures:** EEVEE and Workbench need NVIDIA's EGL vendor file
(`/usr/share/glvnd/egl_vendor.d/10_nvidia.json`); without it they abort EGL initialization (seen with drivers 570
and 580), so use Cycles there. Where OptiX cannot create its denoiser, the renderer switches to CUDA and
OpenImageDenoise by itself. Set the scene fps before importing an animated GLB.

**Upstream:** [Blender Python API](https://docs.blender.org/api/4.5/).
