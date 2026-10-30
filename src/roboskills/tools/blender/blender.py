"""Bounded character previews in Blender; reuse the exported GLB, motion and camera.

Run with Blender's Python. Default: twelve stills, four Cycles samples and a contact sheet.
`--final` renders every saved frame at full resolution, sixteen samples, resumable PNGs.

`render_scene` renders a MuJoCo scene's saved states in docker/blender's image (this file's text runs there);
`import_character` and `render_motion` turn a Mixamo character and a saved motion into a clip, in Blender's uv
environment.
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


IMAGE = "ghcr.io/openrobokit/roboskills-blender:0.2.0"


def render_scene(scene: Path, video: Path, out: Path, samples: int = 16, scale: float = 0.5):
    """out/blender_ext1.mp4: Cycles renders of scene/trajectory.npz (scene.mjb's states) from its `source` camera at
    `scale` of the video's size, the geoms MuJoCo shows (groups 0-2: boxes and meshes) in their colours; and
    out/compare_ext1.mp4: the video | Blender | scene/mujoco_ext1.mp4."""
    import cv2
    import mujoco
    import trimesh

    from roboskills.eval.runtime import contained
    from roboskills.tools.mujoco.engine import load

    out, scene = Path(out).resolve(), Path(scene).resolve()
    (out / "frames").mkdir(parents=True, exist_ok=True)
    (model, data), capture = load(scene / "scene.mjb"), cv2.VideoCapture(str(video))
    size = (np.array([capture.get(cv2.CAP_PROP_FRAME_WIDTH), capture.get(cv2.CAP_PROP_FRAME_HEIGHT)]) * scale).round()
    geoms = [g for g in range(model.ngeom) if model.geom_group[g] < 3 and model.geom_type[g] in (6, 7)]  # box, mesh
    shapes, pose = {}, lambda m, p: np.r_[np.c_[m.reshape(3, 3), p], [[0, 0, 0, 1]]]
    for i, g in enumerate(geoms):
        if model.geom_type[g] == 6:
            box = trimesh.creation.box(extents=2 * model.geom_size[g])
            shapes[f"v{i}"], shapes[f"f{i}"] = box.vertices, box.faces
        else:
            m = model.mesh(model.geom_dataid[g])
            shapes[f"v{i}"] = model.mesh_vert[m.vertadr[0] : m.vertadr[0] + m.vertnum[0]]
            shapes[f"f{i}"] = model.mesh_face[m.faceadr[0] : m.faceadr[0] + m.facenum[0]]
    poses = []
    for data.qpos[:] in np.load(scene / "trajectory.npz")["qpos"]:
        mujoco.mj_forward(model, data)  # geom and camera poses
        poses.append([pose(data.geom_xmat[g], data.geom_xpos[g]) for g in geoms])
    camera = model.camera("source").id
    view = pose(data.cam_xmat[camera], data.cam_xpos[camera])  # MuJoCo's camera axes are Blender's
    rgba = [model.mat_rgba[model.geom_matid[g]] if model.geom_matid[g] >= 0 else model.geom_rgba[g] for g in geoms]
    np.savez(out / "scene.npz", poses=poses, rgba=rgba, camera=view, fovy=model.cam_fovy[camera], size=size, **shapes)
    argv = ["python", "-c", Path(__file__).read_text(), "scene", out / "scene.npz", out / "frames", samples]
    if (cmd := contained(IMAGE, argv, [str(out)])) is None:
        raise SystemExit(f"docker cannot run {IMAGE} here")
    subprocess.run(cmd, check=True)
    ffmpeg = ["ffmpeg", "-y", "-loglevel", "error"]
    encode = ["-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
    fps = str(capture.get(cv2.CAP_PROP_FPS))
    subprocess.run(
        [*ffmpeg, "-framerate", fps, "-i", out / "frames/%04d.png", *encode, out / "blender_ext1.mp4"], check=True
    )
    stack = "".join(f"[{i}:v]scale=-2:{int(size[1])}[v{i}];" for i in range(3)) + "[v0][v1][v2]hstack=inputs=3"
    inputs = [x for v in (video, out / "blender_ext1.mp4", scene / "mujoco_ext1.mp4") for x in ("-i", v)]
    subprocess.run([*ffmpeg, *inputs, "-filter_complex", stack, *encode, out / "compare_ext1.mp4"], check=True)


def blender_script(name: str, *argv) -> None:
    """A script beside this file in its own uv environment (inline dependencies: Blender's Python)."""
    subprocess.run(["uv", "run", "--quiet", str(Path(__file__).with_name(name)), *map(str, argv)], check=True)


def import_character(source: Path, out: Path):
    """`out`: a Mixamo FBX as the GLB `character.load_character` reads."""
    blender_script("rig.py", "import", Path(source).resolve(), Path(out).resolve())


def render_motion(character: Path, motion: Path, video: Path, out: Path, engine: str = "cycles"):
    """out/scene.glb: the `character` GLB animated by `motion`; out/frames/: every frame in motion_studio.py;
    out/video.mp4; and out/compare.mp4: the source `video` | the render."""
    character, motion, out = Path(character).resolve(), Path(motion).resolve(), Path(out).resolve()
    shutil.rmtree(out / "frames", ignore_errors=True)  # the renderer keeps the frames it finds
    out.mkdir(parents=True, exist_ok=True)
    blender_script("rig.py", "animate", character, motion, out / "scene.glb")
    blender_script(
        "motion_studio.py", out / "scene.glb", motion, "--out", out / "frames", "--final", "--engine", engine
    )
    ffmpeg = ["ffmpeg", "-y", "-loglevel", "error"]
    encode = ["-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
    fps = float(np.load(motion)["fps"])
    frames = ["-framerate", str(fps), "-start_number", "0", "-i", out / "frames/%04d.png"]
    subprocess.run([*ffmpeg, *frames, *encode, out / "video.mp4"], check=True)
    stack = f"[0:v]fps={fps},scale=-2:540[v0];[1:v]scale=-2:540[v1];[v0][v1]hstack=inputs=2"
    inputs = ["-i", video, "-i", out / "video.mp4"]
    subprocess.run([*ffmpeg, *inputs, "-filter_complex", stack, *encode, out / "compare.mp4"], check=True)


def render_states(npz, frames, samples):
    """`render_scene`'s Blender side: one PNG per saved state."""
    import bpy
    from mathutils import Matrix

    d = np.load(npz)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    s = bpy.context.scene
    s.render.engine, s.cycles.samples, s.cycles.use_denoising, s.cycles.device = "CYCLES", int(samples), True, "GPU"
    preferences = bpy.context.preferences.addons["cycles"].preferences
    preferences.compute_device_type = "CUDA"
    preferences.get_devices()
    for device in preferences.devices:
        device.use = device.type == "CUDA"
    s.render.resolution_x, s.render.resolution_y = (int(x) for x in d["size"])
    s.view_settings.view_transform, s.world = "Standard", bpy.data.worlds.new("world")
    s.world.color = (0.5, 0.5, 0.5)
    sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", type="SUN"))
    sun.data.energy, sun.rotation_euler = 1.5, (0.4, 0.3, 0.0)
    s.collection.objects.link(sun)
    parts = []
    for i, rgba in enumerate(d["rgba"]):
        mesh = bpy.data.meshes.new(str(i))
        mesh.from_pydata(d[f"v{i}"].tolist(), [], d[f"f{i}"].tolist())
        mesh.shade_smooth() if len(mesh.polygons) > 12 else None
        material = bpy.data.materials.new(str(i))
        material.use_nodes = True
        material.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = [float(x) for x in rgba]
        mesh.materials.append(material)
        parts.append(bpy.data.objects.new(str(i), mesh))
        s.collection.objects.link(parts[-1])
    camera = bpy.data.objects.new("camera", bpy.data.cameras.new("camera"))
    camera.data.sensor_fit, camera.data.angle_y = "VERTICAL", np.radians(float(d["fovy"]))
    camera.matrix_world = Matrix(d["camera"].tolist())
    s.collection.objects.link(camera)
    s.camera = camera
    for t, poses in enumerate(d["poses"]):
        for part, pose in zip(parts, poses):
            part.matrix_world = Matrix(pose.tolist())
        s.render.filepath = f"{frames}/{t:04d}.png"
        bpy.ops.render.render(write_still=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("character", type=Path, help="Animated GLB written by retarget")
    p.add_argument("motion", type=Path, help="Saved character motion NPZ")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--reference", type=Path, help="Fixed baseline camera/framing")
    p.add_argument("--frames", help="Comma separated saved frame indices")
    p.add_argument("--final", action="store_true")
    p.add_argument("--side", action="store_true")
    p.add_argument("--focus", help="Diagnostic closeup bone roles, optionally floor")
    p.add_argument("--at", type=int, default=0)
    a = p.parse_args(argv)
    import bpy
    from mathutils import Matrix, Quaternion, Vector

    a.out.mkdir(parents=True, exist_ok=True)
    d = np.load(a.motion)
    reference = np.load(a.reference) if a.reference else d
    width, height = (int(reference.get(k, default)) for k, default in [("image_width", 640), ("image_height", 480)])
    fps = float(d["fps"])
    n = len(d["turns"])
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    s = bpy.context.scene
    s.render.fps = round(fps)
    s.render.fps_base = round(fps) / fps
    s.render.threads_mode = "FIXED"
    s.render.threads = 2
    bpy.ops.import_scene.gltf(filepath=str(a.character.resolve()))
    rig = next(x for x in s.objects if x.type == "ARMATURE")
    first = float(rig.animation_data.action.frame_range[0])
    s.render.engine = "CYCLES"
    s.cycles.samples = 16 if a.final else 4
    s.cycles.use_denoising = True
    s.cycles.device = "GPU"
    pref = bpy.context.preferences.addons["cycles"].preferences
    pref.compute_device_type = "CUDA"
    pref.get_devices()
    for dev in pref.devices:
        dev.use = dev.type == "CUDA"
    s.render.resolution_x = width
    s.render.resolution_y = height
    s.render.resolution_percentage = 100 if a.final else 50
    s.render.image_settings.file_format = "PNG"
    s.world.color = (0.5, 0.5, 0.5)
    s.view_settings.view_transform = "Standard"
    bpy.ops.mesh.primitive_plane_add(size=200, location=(0, 0, -0.0005))
    floor = bpy.context.object
    m = bpy.data.materials.new("floor")
    m.diffuse_color = (0.72, 0.72, 0.72, 1)
    floor.data.materials.append(m)
    bpy.ops.object.light_add(type="AREA", location=(2, -3, 7))
    bpy.context.object.data.energy = 1200
    bpy.context.object.data.shape = "DISK"
    bpy.context.object.data.size = 6
    bpy.context.object.rotation_euler = (
        (Vector((0, 0, 1)) - bpy.context.object.location).to_track_quat("-Z", "Y").to_euler()
    )
    bpy.ops.object.camera_add()
    cam = bpy.context.object
    s.camera = cam
    cam.data.sensor_width = 36
    cam.data.sensor_fit = "HORIZONTAL"
    source_camera = not a.side and "camera_poses" in reference
    if not source_camera:
        if a.focus:
            names = [str(x).split(":")[-1] for x in reference["joint_names"]]
            roles = a.focus.split(",")
            points = reference["positions"][a.at, [names.index(x) for x in roles if x != "floor"]]
            if "floor" in roles:
                points = np.r_[points, [[*points.mean(0)[:2], 0.0]]]
            centre = points.mean(0)
            extent = float(np.ptp(points, axis=0).max()) + 0.6
            distance = 1.1 * extent * 40 / (36 * int(d.get("image_height", 480)) / int(d.get("image_width", 640)))
            direction = np.array([-1.0, -1.0, 0.25])
            direction /= np.linalg.norm(direction)
            cam.location = Vector(centre + direction * distance)
        else:
            points = reference["positions"].reshape(-1, 3)
            if "object_vertices" in reference:
                r = reference["object_poses"]
                if r.shape[-2:] == (4, 4):
                    points = np.r_[
                        points,
                        (
                            np.einsum("tij,vj->tvi", r[:, :3, :3], reference["object_vertices"]) + r[:, None, :3, 3]
                        ).reshape(-1, 3),
                    ]
                else:
                    reach = np.linalg.norm(reference["object_vertices"], axis=1).max()
                    points = np.r_[points, r[:, :3] + reach, r[:, :3] - reach]
            centre = (points.min(0) + points.max(0)) / 2
            radius = np.linalg.norm(points.max(0) - points.min(0)) / 2 + 0.15
            distance = radius / np.sin(np.arctan(36 * s.render.resolution_y / s.render.resolution_x / 80))
            cam.location = Vector(centre) + Vector((-1, 0, 0.2)).normalized() * distance
        cam.rotation_euler = (Vector(centre) - cam.location).to_track_quat("-Z", "Y").to_euler()
        cam.data.lens = 40
        (a.out / "camera.json").write_text(
            json.dumps(
                dict(
                    location=list(cam.location),
                    rotation=list(cam.rotation_euler),
                    lens=cam.data.lens,
                    reference=str(a.reference),
                    at=a.at,
                    focus=a.focus,
                ),
                indent=2,
            )
            + "\n"
        )
    obj = None
    if "object_vertices" in d:
        mesh = bpy.data.meshes.new("prop")
        mesh.from_pydata(d["object_vertices"].tolist(), [], d["object_faces"].tolist())
        mesh.update()
        obj = bpy.data.objects.new("prop", mesh)
        s.collection.objects.link(obj)
        m = bpy.data.materials.new("prop")
        m.diffuse_color = (0.85, 0.25, 0.05, 1)
        obj.data.materials.append(m)
    frames = (
        [int(x) for x in a.frames.split(",")]
        if a.frames
        else np.unique(np.linspace(0, n - 1, 12).astype(int)).tolist()
        if not a.final
        else list(range(n))
    )
    if not frames or min(frames) < 0 or max(frames) >= n:
        raise ValueError("Preview frame indices must lie inside the saved clip")
    times = []
    start = time.perf_counter()
    for i in frames:
        f = a.out / f"{i:04d}.png"
        if f.exists():
            continue
        before = time.perf_counter()
        s.frame_set(int(first) + i, subframe=first - int(first))
        if source_camera:
            T = reference["camera_poses"][i].copy()
            T[:3, :3] = T[:3, :3] @ np.diag([1, -1, -1])
            cam.matrix_world = Matrix(T.tolist())
            # A one-sided studio floor cannot obscure an aligned camera below its estimated plane.
            floor.visible_camera = bool(T[2, 3] >= -0.0005)
            K = reference["camera_intrinsics"][i]
            w, h = width, height
            cam.data.lens = float(K[0, 0]) * 36 / w
            cam.data.shift_x = 0.5 - float(K[0, 2]) / w
            ratio = float(K[0, 0] / K[1, 1])
            s.render.pixel_aspect_x, s.render.pixel_aspect_y = max(1.0, 1 / ratio), max(1.0, ratio)
            cam.data.shift_y = (float(K[1, 2]) - h / 2) / w * ratio
        if obj:
            at = d["object_poses"][i]
            if at.shape == (7,):
                transform = Quaternion(at[3:].tolist()).to_matrix().to_4x4()
                transform.translation = Vector(at[:3])
                obj.matrix_world = transform
            else:
                obj.matrix_world = Matrix(at.tolist())
        s.render.filepath = str(f.resolve())
        bpy.ops.render.render(write_still=True)
        times.append(dict(frame=i, seconds=time.perf_counter() - before))
    # A contact sheet without another image dependency in the Blender environment.
    sheet_frames = [frames[k] for k in np.unique(np.linspace(0, len(frames) - 1, min(12, len(frames))).astype(int))]
    images = [bpy.data.images.load(str((a.out / f"{i:04d}.png").resolve()), check_existing=False) for i in sheet_frames]
    w, h = images[0].size
    cols = min(3, len(images))
    rows = (len(images) + cols - 1) // cols
    pixels = np.ones((rows * (h + 4), cols * (w + 4), 4), np.float32)
    for k, im in enumerate(images):
        x, y = k % cols * (w + 4), (rows - 1 - k // cols) * (h + 4)
        pixels[y : y + h, x : x + w] = np.asarray(im.pixels[:], np.float32).reshape(h, w, 4)
    sheet = bpy.data.images.new("sheet", width=pixels.shape[1], height=pixels.shape[0], alpha=True)
    sheet.pixels.foreach_set(pixels.ravel())
    sheet.filepath_raw = str((a.out / "sheet.png").resolve())
    sheet.file_format = "PNG"
    sheet.save()
    result = dict(
        total_s=time.perf_counter() - start,
        frames=times,
        sheet_frames=sheet_frames,
        final=a.final,
        source_camera=source_camera,
        camera_below_floor_frames=np.flatnonzero(reference["camera_poses"][:, 2, 3] < -0.0005).tolist()
        if source_camera
        else [],
    )
    if times or not (a.out / "timing.json").exists():
        (a.out / "timing.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    render_states(*sys.argv[2:]) if sys.argv[1:2] == ["scene"] else main()
