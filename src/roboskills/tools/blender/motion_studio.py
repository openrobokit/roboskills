# /// script
# requires-python = "==3.11.*"
# dependencies = ["bpy==4.5.14", "numpy==1.26.4"]
# ///
"""The motion_gen renderer: an animated GLB scene and its saved motion, rendered by one Blender process (GPU Cycles;
`--engine eevee` or `workbench` for quicker previews, in their own `--out`). Its inline dependencies are Blender's
python: `uv run` this file, or `$BLENDER_PYTHON` (`uv sync --script` of it, then `uv python find --script`).

    $BLENDER_PYTHON motion_studio.py SCENE.glb MOTION.npz --out DIR [--frames 0,30,60 | --final] [options]

Twelve stills spread over the clip by default, `--frames` the listed saved frames, `--final` every saved frame:
numbered PNGs (existing ones are kept, so an interrupted render resumes), a contact sheet and render.json.

The look is Street dance's approved studio: matte materials, three soft lights, AgX. `--environment outdoor` keeps
authored ground and prop colours under daylight; `warm` is the paper-folding interior (`--kitchen` adds its wall,
cabinetry and worktop behind the action).

The camera is one fixed camera for the clip, framing everything that moves on every saved frame, from `--view AZ,EL`
degrees: AZ around the vertical, counterclockwise seen from above, from MOTION's studio_camera_direction, else the
video camera's side, else world -X; EL above the ground (default 22). `--frame-on` frames only the named positions
(MOTION's joint_names, or a robot's link_names), GLB objects or `prop`; `--follow` re-frames on them every frame, a
close-up that moves with them. `--camera source` draws from the video's camera at the video's aspect. `--camera-from
DIR` reuses the camera of an earlier render. MOTION holds fps, turns [T, ...] (frame count) and positions [T, J, 3];
optionally camera_poses
[T, 4, 4] (world from camera; x right, y down, z ahead) and camera_intrinsics [T, 3, 3] for an image_width x
image_height picture, object_vertices/object_faces/object_poses ([T, 4, 4] or xyz+wxyz) for a prop the renderer
draws, and studio_bounds [2, 3] kept in the default framing.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector


WIDTH, HEIGHT, LENS, SENSOR = 960, 540, 40.0, 36.0
BACKGROUND = {"studio": ((0.68, 0.70, 0.75, 1), 0.4), "outdoor": ((0.48, 0.66, 0.88, 1), 0.6),
              "warm": ((0.65, 0.60, 0.54, 1), 0.55)}  # fmt: skip
LIGHTS = {  # name, offset from the action (turned with the camera), energy, colour, size
    "studio": [("Soft key", (-3, -4, 6), 750, (1, 0.96, 0.92), 5), ("Soft fill", (4, -2, 4), 250, (0.86, 0.92, 1), 4),
               ("Soft rim", (1, 3, 5), 650, (1, 1, 1), 3)],
    "warm": [("Window", (-3, -4, 6), 560, (1.0, 0.97, 0.92), 5), ("Warm fill", (4, -1, 3), 140, (1.0, 0.90, 0.80), 4),
             ("Soft rim", (1, 3, 5), 260, (1.0, 0.95, 0.88), 3)],
    "outdoor": [],
}  # fmt: skip
FLOOR = {"studio": ((0.72, 0.73, 0.74, 1), 0.95, 0.1), "warm": ((0.86, 0.85, 0.83, 1), 0.32, 0.18)}
PROP = {"studio": (0.85, 0.85, 0.82, 1), "warm": (0.86, 0.84, 0.78, 1)}
PRESET = {"studio": "approved-street-dance-matte", "outdoor": "outdoor-daylight", "warm": "paper-kitchen-v1"}


def parse(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("scene", type=Path, help="animated GLB")
    p.add_argument("motion", type=Path, help="saved motion NPZ")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--frames", help="comma separated saved frame indices")
    p.add_argument("--final", action="store_true", help="every saved frame")
    p.add_argument("--camera", choices=("fixed", "source"), default="fixed")
    p.add_argument("--view", default="0,22", help="AZ,EL in degrees (a negative AZ as --view=-90,10)")
    p.add_argument("--frame-on", help="comma separated bones, GLB objects or prop")
    p.add_argument("--follow", action="store_true", help="re-frame on --frame-on every frame")
    p.add_argument("--camera-from", type=Path, help="an earlier render's DIR, for its camera")
    p.add_argument("--environment", choices=("studio", "outdoor", "warm"), default="studio")
    p.add_argument("--kitchen", action="store_true", help="warm only: wall, cabinetry and worktop behind the action")
    p.add_argument("--exposure", type=float, default=0.0, help="stops")
    p.add_argument("--engine", choices=("cycles", "eevee", "workbench"), default="cycles")
    p.add_argument("--samples", type=int, default=32)
    p.add_argument(
        "--size",
        type=int,
        nargs=2,
        default=(WIDTH, HEIGHT),
        metavar=("W", "H"),
        help="picture size (default 960 540); a source camera keeps its video's aspect, as long as W",
    )
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--device", choices=("optix", "cuda"), default="optix", help="optix falls back to cuda")
    a = p.parse_args(argv)
    a.azimuth, a.elevation = (float(x) for x in a.view.split(","))
    if not -85 < a.elevation < 85:
        raise ValueError("--view elevation must lie between -85 and 85 degrees")
    if a.follow and not a.frame_on:
        raise ValueError("--follow needs --frame-on")
    if a.kitchen and a.environment != "warm":
        raise ValueError("--kitchen needs --environment warm")
    if a.camera == "source" and (a.frame_on or a.camera_from):
        raise ValueError("--camera source takes the video's camera; --frame-on and --camera-from are for a fixed one")
    return a


def setup_render(scene: bpy.types.Scene, a: argparse.Namespace, size: tuple[int, int]) -> list[str]:
    scene.render.resolution_x, scene.render.resolution_y = size
    scene.render.resolution_percentage = 100
    scene.render.threads_mode = "FIXED"
    scene.render.threads = a.threads
    scene.render.use_persistent_data = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.filter_size = 1.0
    if a.engine == "workbench":  # flat studio shading of the image textures (base colours where there are none)
        scene.render.engine = "BLENDER_WORKBENCH"
        shading = scene.display.shading
        shading.light, shading.color_type, shading.show_shadows, shading.show_cavity = "STUDIO", "TEXTURE", True, True
        return []
    if a.engine == "eevee":
        scene.render.engine = "BLENDER_EEVEE_NEXT"
        scene.eevee.taa_render_samples = a.samples
        return []
    scene.render.engine = "CYCLES"
    scene.cycles.samples = a.samples
    scene.cycles.use_denoising = True
    scene.cycles.max_bounces, scene.cycles.diffuse_bounces, scene.cycles.glossy_bounces = 4, 2, 2
    scene.cycles.device = "GPU"
    for kind in ("OPTIX", "CUDA") if a.device == "optix" else ("CUDA",):
        devices = use_devices(scene, kind)
        if devices:
            return devices
    return []


def use_devices(scene: bpy.types.Scene, kind: str) -> list[str]:
    """Render on every GPU of `kind` (OPTIX denoises with OptiX, CUDA with OpenImageDenoise); the devices' names."""
    pref = bpy.context.preferences.addons["cycles"].preferences
    pref.compute_device_type = kind
    pref.get_devices()
    for device in pref.devices:
        device.use = device.type == kind
    scene.cycles.denoiser = "OPTIX" if kind == "OPTIX" else "OPENIMAGEDENOISE"
    return [d.name for d in pref.devices if d.use]


def look(scene: bpy.types.Scene, a: argparse.Namespace) -> None:
    """World, colour management and matte materials; the lights follow the camera (`light`)."""
    scene.world.use_nodes = True
    nodes, links = scene.world.node_tree.nodes, scene.world.node_tree.links
    background = nodes.get("Background")
    background.inputs["Color"].default_value, background.inputs["Strength"].default_value = BACKGROUND[a.environment]
    hdr = Path(bpy.utils.system_resource("DATAFILES")) / "studiolights/world/interior.exr"
    if a.environment == "warm" and hdr.exists():
        env = nodes.new("ShaderNodeTexEnvironment")
        env.image = bpy.data.images.load(str(hdr), check_existing=True)
        mapping = nodes.new("ShaderNodeMapping")
        mapping.inputs["Rotation"].default_value = (0, 0, np.radians(110))
        coord = nodes.new("ShaderNodeTexCoord")
        links.new(coord.outputs["Generated"], mapping.inputs["Vector"])
        links.new(mapping.outputs["Vector"], env.inputs["Vector"])
        links.new(env.outputs["Color"], background.inputs["Color"])
    if a.engine == "workbench":  # its studio light is meant for the plain view transform
        scene.view_settings.view_transform = "Standard"
    else:
        scene.view_settings.view_transform = "AgX"
        scene.view_settings.look = "AgX - Medium High Contrast"
    scene.view_settings.exposure = a.exposure - (0.5 if a.environment == "warm" else 0.0)
    for material in bpy.data.materials:
        material.use_nodes = True
        for node in material.node_tree.nodes:
            if node.type != "BSDF_PRINCIPLED":
                continue
            for name, value in [("Roughness", 0.8), ("Metallic", 0.0), ("Specular IOR Level", 0.12)]:
                for link in list(node.inputs[name].links):
                    material.node_tree.links.remove(link)
                node.inputs[name].default_value = value
            if material.name.startswith("floor") and a.environment in FLOOR:
                colour, roughness, specular = FLOOR[a.environment]
                node.inputs["Base Color"].default_value = colour
                node.inputs["Roughness"].default_value = roughness
                node.inputs["Specular IOR Level"].default_value = specular
            if material.name.startswith("prop") and a.environment in PROP:
                node.inputs["Base Color"].default_value = PROP[a.environment]
            if "Coat Weight" in node.inputs:
                node.inputs["Coat Weight"].default_value = 0


def light(scene: bpy.types.Scene, a: argparse.Namespace, centre: np.ndarray, yaw: float) -> None:
    """Replace the scene's lights with the environment's, turned to face the camera's side of `centre`."""
    for obj in [o for o in scene.objects if o.type == "LIGHT"]:
        bpy.data.objects.remove(obj, do_unlink=True)
    c, s = np.cos(yaw), np.sin(yaw)
    turn = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    for name, offset, energy, colour, size in LIGHTS[a.environment]:
        data = bpy.data.lights.new(name, "AREA")
        data.energy, data.color, data.shape, data.size = energy, colour, "DISK", size
        obj = bpy.data.objects.new(name, data)
        scene.collection.objects.link(obj)
        obj.location = Vector(centre + turn @ offset)
        obj.rotation_euler = (Vector(centre + [0, 0, 1]) - obj.location).to_track_quat("-Z", "Y").to_euler()
    if a.environment == "outdoor":
        data = bpy.data.lights.new("Sun", "SUN")
        data.energy, data.angle, data.color = 2.5, 0.08, (1.0, 0.96, 0.88)
        sun = bpy.data.objects.new("Sun", data)
        scene.collection.objects.link(sun)
        sun.rotation_euler = np.radians([35, -25, -30])


def kitchen(points: np.ndarray, centre: np.ndarray, camera: np.ndarray) -> None:
    """The warm kitchen's plaster wall, sage cabinetry and quartz worktop, behind the action as the camera sees it."""
    front = camera - np.r_[centre[:2], 0.0]
    front[2] = 0
    front /= np.linalg.norm(front)
    right = np.cross([0, 0, 1], front)
    span = max(2.5, float(np.ptp(points[:, :2], axis=0).max()) * 0.8)
    height = max(2.6, float(np.ptp(points[:, 2])) * 1.4)
    back = np.r_[centre[:2], min(0.0, float(points[:, 2].min()))] - front * (span + 0.6)
    yaw = np.arctan2(right[1], right[0])

    def cube(name: str, at: np.ndarray, dimensions: tuple, colour: tuple, roughness: float, coat: float = 0.0) -> None:
        mat = bpy.data.materials.new(name)
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        bsdf.inputs["Base Color"].default_value = (*colour, 1)
        bsdf.inputs["Roughness"].default_value = roughness
        bsdf.inputs["Coat Weight"].default_value = coat
        bpy.ops.mesh.primitive_cube_add(size=1, location=at, rotation=(0, 0, yaw))
        obj = bpy.context.object
        obj.name, obj.dimensions = name, dimensions
        obj.data.materials.append(mat)
        bevel = obj.modifiers.new("Soft edges", "BEVEL")
        bevel.width, bevel.segments = 0.012, 3

    cube("Warm plaster wall", back + [0, 0, height / 2], (2.8 * span, 0.08, height), (0.80, 0.77, 0.72), 0.8)
    cabinet = back + 0.32 * front + [0, 0, 0.43]
    cube("Sage cabinetry", cabinet, (2.2 * span, 0.58, 0.86), (0.32, 0.40, 0.36), 0.55)
    cube("Quartz worktop", cabinet + [0, 0, 0.46], (2.25 * span, 0.66, 0.06), (0.86, 0.85, 0.83), 0.28, 0.2)


def prop_vertices(d: np.lib.npyio.NpzFile) -> np.ndarray:
    """[T, V, 3] the renderer-drawn prop in the world."""
    poses = d["object_poses"]
    if poses.shape[-2:] == (4, 4):
        rotations, translation = poses[:, :3, :3], poses[:, :3, 3]
    else:
        rotations = np.array([Quaternion(p[3:].tolist()).to_matrix() for p in poses])
        translation = poses[:, :3]
    return np.einsum("tij,vj->tvi", rotations, d["object_vertices"]) + translation[:, None]


def animated(obj: bpy.types.Object) -> bool:
    if obj.type != "MESH":
        return False
    if any(m.type == "ARMATURE" for m in obj.modifiers):
        return True
    while obj:
        if obj.animation_data:
            return True
        obj = obj.parent
    return False


def targets(scene: bpy.types.Scene, d: np.lib.npyio.NpzFile, names: list[str] | None, first: float) -> list[np.ndarray]:
    """Per saved frame, the points a fixed camera keeps in frame: by default the joints, every animated mesh's actual
    bounds (hair and hats included), the prop and studio_bounds; else the named bones (padded to cover the skin),
    GLB objects and `prop`."""
    n, count = len(d["turns"]), d["positions"].shape[1]
    # positions are a character's joints or a robot's links; their names are the list of the same length
    listed = [d[k] for k in ("joint_names", "link_names") if k in d and len(d[k]) == count]
    joints = [str(x).split(":")[-1] for x in listed[0]] if listed else []
    if names is None:
        bones, objects = list(range(d["positions"].shape[1])), [o for o in scene.objects if animated(o)]
        prop, pad = "object_vertices" in d, 0.0
        extra = d["studio_bounds"].reshape(-1, 3) if "studio_bounds" in d else np.zeros((0, 3))
    else:
        bare = [x.split(":")[-1] for x in names]  # a rig's prefix (mixamorig5:LeftHand) or none
        unknown = [x for x, b in zip(names, bare) if x != "prop" and b not in joints and x not in scene.objects]
        if unknown:
            raise ValueError(f"--frame-on {unknown}: not in {joints} or objects {list(scene.objects.keys())}")
        bones = [joints.index(b) for b in bare if b in joints]
        objects = [scene.objects[x] for x, b in zip(names, bare) if b not in joints and x != "prop"]
        prop, pad, extra = "prop" in names, 0.1, np.zeros((0, 3))
    corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * pad
    out = [(d["positions"][t, bones][:, None] + corners).reshape(-1, 3) for t in range(n)]
    if prop:
        out = [np.r_[p, v] for p, v in zip(out, prop_vertices(d))]
    for t in range(n) if objects else ():
        scene.frame_set(int(first) + t, subframe=first - int(first))
        graph = bpy.context.evaluated_depsgraph_get()
        for obj in objects:
            ev = obj.evaluated_get(graph)
            out[t] = np.r_[out[t], [list(ev.matrix_world @ Vector(c)) for c in ev.bound_box]]
    return [np.r_[p, extra] for p in out]


def direction(d: np.lib.npyio.NpzFile, centre: np.ndarray, azimuth: float, elevation: float) -> np.ndarray:
    """The unit vector from the action toward the camera."""
    if "studio_camera_direction" in d:
        base = np.asarray(d["studio_camera_direction"], float).copy()
    elif "camera_poses" in d:
        base = d["camera_poses"][:, :3, 3].mean(0) - centre
    else:
        base = np.array([-1.0, 0.0, 0.0])
    base[2] = 0
    if np.linalg.norm(base) < 1e-6:
        raise ValueError("the camera's base direction needs a horizontal component")
    c, s = np.cos(np.radians(azimuth)), np.sin(np.radians(azimuth))
    out = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ (base / np.linalg.norm(base))
    out[2] = np.tan(np.radians(elevation))
    return out / np.linalg.norm(out)


def fit(points: np.ndarray, outward: np.ndarray, aspect: float) -> tuple[np.ndarray, np.ndarray]:
    """The camera location and look-at centre that frame `points` from direction `outward` through the lens."""
    centre = (points.min(0) + points.max(0)) / 2
    forward = -outward
    right = np.cross(forward, [0, 0, 1])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    relative = points - centre
    tan_x = SENSOR / 2 / LENS
    distance = max(
        np.max(1.12 * np.abs(relative @ right) / tan_x - relative @ forward),
        np.max(1.12 * np.abs(relative @ up) / (tan_x * aspect) - relative @ forward),
    ) + 0.15  # fmt: skip
    return centre + outward * distance, centre


def glide(x: np.ndarray, window: int) -> np.ndarray:
    """A moving average over `window` frames along axis 0, the ends held, so a following camera does not shake."""
    padded = np.pad(x, ((window // 2, window - 1 - window // 2), (0, 0)), mode="edge")
    return np.stack([np.convolve(padded[:, k], np.ones(window) / window, "valid") for k in range(x.shape[1])], 1)


def look_at(location: np.ndarray, centre: np.ndarray) -> list[float]:
    return list((Vector(centre) - Vector(location)).to_track_quat("-Z", "Y").to_euler())


def main(argv: list[str]) -> None:
    a = parse(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    d = np.load(a.motion)
    fps, n = float(d["fps"]), len(d["turns"])
    source = a.camera == "source"
    if source and not {"camera_poses", "camera_intrinsics", "image_width", "image_height"} <= set(d.files):
        raise ValueError("--camera source needs camera_poses, camera_intrinsics, image_width and image_height")
    width, height = (int(d["image_width"]), int(d["image_height"])) if source else tuple(a.size)
    long = max(a.size)
    size = (round(width * long / max(width, height)), round(height * long / max(width, height)))

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    scene = bpy.context.scene
    scene.render.fps = round(fps)
    scene.render.fps_base = round(fps) / fps
    bpy.ops.import_scene.gltf(filepath=str(a.scene.resolve()))
    # Saved frame 0 is the first key of the characters' action; a rigid scene's animation starts at frame 0.
    rigs = [o for o in scene.objects if o.type == "ARMATURE" and o.animation_data and o.animation_data.action]
    first = float(rigs[0].animation_data.action.frame_range[0]) if rigs else 0.0
    devices = setup_render(scene, a, size)
    bpy.ops.mesh.primitive_plane_add(size=200, location=(0, 0, -0.0005))
    floor = bpy.context.object
    floor.data.materials.append(bpy.data.materials.new("floor"))
    prop = None
    if "object_vertices" in d:
        mesh = bpy.data.meshes.new("prop")
        mesh.from_pydata(d["object_vertices"].tolist(), [], d["object_faces"].tolist())
        mesh.update()
        prop = bpy.data.objects.new("prop", mesh)
        scene.collection.objects.link(prop)
        prop.data.materials.append(bpy.data.materials.new("prop"))
    look(scene, a)

    bpy.ops.object.camera_add()
    cam = bpy.context.object
    scene.camera = cam
    cam.data.sensor_width, cam.data.sensor_fit, cam.data.lens = SENSOR, "HORIZONTAL", LENS
    locations, rotations, centre = None, None, np.zeros(3)
    if a.camera_from:
        saved = json.loads((a.camera_from / "render.json").read_text())["camera"]
        if saved["mode"] == "source" or len(saved["locations"]) not in (1, n):
            raise ValueError(f"{a.camera_from} holds no fixed camera for this clip's {n} frames")
        locations, rotations, centre = np.asarray(saved["locations"]), saved["rotations"], np.asarray(saved["centre"])
    elif not source:
        everything = np.concatenate(targets(scene, d, None, first))
        points = targets(scene, d, a.frame_on.split(","), first) if a.frame_on else [everything]
        outward = direction(d, (everything.min(0) + everything.max(0)) / 2, a.azimuth, a.elevation)
        if a.follow:
            fits = [fit(p, outward, size[1] / size[0]) for p in points]
            window = max(1, round(fps / 2))
            locations, aims = (glide(np.array([f[k] for f in fits]), window) for k in (0, 1))
            rotations, centre = [look_at(p, c) for p, c in zip(locations, aims)], aims.mean(0)
        else:
            location, centre = fit(np.concatenate(points), outward, size[1] / size[0])
            locations, rotations = location[None], [look_at(location, centre)]
        if a.kitchen:
            kitchen(everything, centre, locations[0])
    if locations is None:  # the video's camera: lights as authored for the source view
        light(scene, a, np.zeros(3), 0.0)
    else:
        toward = locations.mean(0) - centre
        light(scene, a, centre * [1, 1, 0], np.arctan2(toward[1], toward[0]) - np.pi)

    frames = (
        [int(x) for x in a.frames.split(",")] if a.frames
        else list(range(n)) if a.final
        else np.unique(np.linspace(0, n - 1, 12).astype(int)).tolist()
    )  # fmt: skip
    if not frames or min(frames) < 0 or max(frames) >= n:
        raise ValueError("frame indices must lie inside the saved clip")
    times, start = [], time.perf_counter()
    for i in frames:
        path = a.out / f"{i:04d}.png"
        if path.exists():
            continue
        before = time.perf_counter()
        scene.frame_set(int(first) + i, subframe=first - int(first))
        if source:
            T = d["camera_poses"][i].copy()
            T[:3, :3] = T[:3, :3] @ np.diag([1, -1, -1])
            cam.matrix_world = Matrix(T.tolist())
            K = d["camera_intrinsics"][i]
            ratio = float(K[0, 0] / K[1, 1])
            # A horizontal sensor fit measures lens and shift against the picture's width, upright or not.
            cam.data.lens = float(K[0, 0]) * SENSOR / width
            cam.data.shift_x = 0.5 - float(K[0, 2]) / width
            cam.data.shift_y = (float(K[1, 2]) - height / 2) / width * ratio
            scene.render.pixel_aspect_x, scene.render.pixel_aspect_y = max(1.0, 1 / ratio), max(1.0, ratio)
            z = T[2, 3]
        else:
            k = i if len(locations) > 1 else 0
            cam.location, cam.rotation_euler = Vector(locations[k]), rotations[k]
            z = locations[k][2]
        floor.visible_camera = bool(z >= -0.0005)  # a camera below the one-sided floor still sees the action
        if prop:
            at = d["object_poses"][i]
            if at.shape == (7,):
                transform = Quaternion(at[3:].tolist()).to_matrix().to_4x4()
                transform.translation = Vector(at[:3])
                prop.matrix_world = transform
            else:
                prop.matrix_world = Matrix(at.tolist())
        scene.render.filepath = str(path.resolve())
        try:
            bpy.ops.render.render(write_still=True)
        except RuntimeError as error:  # some containers render with OptiX but cannot create its denoiser
            if "OptiX denoiser" not in str(error):
                raise
            devices = use_devices(scene, "CUDA")
            bpy.ops.render.render(write_still=True)
        times.append(dict(frame=i, seconds=round(time.perf_counter() - before, 2)))

    # A contact sheet without another image dependency in Blender's Python.
    sheet = [frames[k] for k in np.unique(np.linspace(0, len(frames) - 1, min(12, len(frames))).astype(int))]
    images = [bpy.data.images.load(str((a.out / f"{i:04d}.png").resolve()), check_existing=False) for i in sheet]
    w, h = images[0].size
    cols = min(3, len(images))
    rows = (len(images) + cols - 1) // cols
    pixels = np.ones((rows * (h + 4), cols * (w + 4), 4), np.float32)
    for k, image in enumerate(images):
        x, y = k % cols * (w + 4), (rows - 1 - k // cols) * (h + 4)
        pixels[y : y + h, x : x + w] = np.asarray(image.pixels[:], np.float32).reshape(h, w, 4)
    contact = bpy.data.images.new("sheet", width=pixels.shape[1], height=pixels.shape[0], alpha=True)
    contact.pixels.foreach_set(pixels.ravel())
    contact.filepath_raw = str((a.out / "sheet.png").resolve())
    contact.file_format = "PNG"
    contact.save()

    camera = dict(mode="source") if source else dict(
        mode="follow" if len(locations) > 1 else "fixed", locations=np.asarray(locations).tolist(),
        rotations=[list(map(float, r)) for r in rotations], centre=list(map(float, centre)),
        view=[a.azimuth, a.elevation], frame_on=a.frame_on, lens=LENS, reused_from=a.camera_from and str(a.camera_from),
    )  # fmt: skip
    record = dict(
        preset=PRESET[a.environment], environment=a.environment, kitchen=a.kitchen, exposure=a.exposure,
        engine=a.engine, samples=a.samples, threads=a.threads,
        denoiser=scene.cycles.denoiser if a.engine == "cycles" else None, devices=devices, width=size[0],
        height=size[1], final=a.final, sheet_frames=sheet, total_s=round(time.perf_counter() - start, 1), frames=times,
        camera_below_floor_frames=np.flatnonzero(d["camera_poses"][:, 2, 3] < -0.0005).tolist() if source else [],
        camera=camera,
    )  # fmt: skip
    if times or not (a.out / "render.json").exists():
        (a.out / "render.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({k: v for k, v in record.items() if k not in ("frames", "camera")}), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)  # every frame and the record are written; skip bpy's teardown, which can hang
