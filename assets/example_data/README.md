# Example data

Inputs for the tool examples in each `src/roboskills/tools/<name>/README.md`; their outputs go to `runs/examples/`
(gitignored), and each README's preview is in [`assets/tools/`](../tools).

| File | What | Used by |
|---|---|---|
| `sugar-box.mp4` | 70 frames at 640×480, 30 fps: a right hand picks up a sugar box (real2sim's example clip) | SAM 3, MoGe-3, GeoCalib, SAM 3D, HaWoR, WiLoR, then CoACD on their outputs |
| `sugar-box.glb`, `sugar-box-still.npz` | SAM 3D's sugar box decimated to 20,000 faces (glTF, Y-up) standing on the floor, and 30 still frames framing it | Blender |
| `tennis.mp4` | 120 frames at 812×720, 30 fps: the first 4 s of [GVHMR](https://github.com/zju3dv/GVHMR)'s `docs/example_video/tennis.mp4` | GVHMR |
