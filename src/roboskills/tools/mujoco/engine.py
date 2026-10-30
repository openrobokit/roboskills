"""Native MuJoCo scene, replay and render tools for any articulation; robokit supplies IK and SimMPC."""

import json
import subprocess
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from roboskills.eval.records import file_digest


TAGS = (("geom", "parts"), ("joint", "joints"), ("site", "sites"), ("camera", "cameras"), ("body", "children"))


def load(scene):
    """(model, data) from XML or MJB."""
    import mujoco

    path = str(Path(scene).resolve())
    model = (mujoco.MjModel.from_binary_path if path.endswith(".mjb") else mujoco.MjModel.from_xml_path)(path)
    return model, mujoco.MjData(model)


def build(path, actor_xml=None, bodies=(), cameras=(), option=None):
    """Extend actor MJCF with named primitive compounds and `option` attributes; schema is in the real2sim workflow's SKILL.md."""
    root = ET.parse(actor_xml).getroot() if actor_xml else ET.Element("mujoco")
    compiler, settings, world, contact = (root.find(t) if root.find(t) is not None else ET.SubElement(root, t)
                                          for t in ("compiler", "option", "worldbody", "contact"))  # fmt: skip
    text = lambda v: str(v).lower() if isinstance(v, bool) else " ".join(map(str, np.ravel(v)))
    settings.attrib.update({k: text(v) for k, v in dict(dict(timestep=0.002), **(option or {})).items()})
    if actor_xml:
        base = Path(actor_xml).resolve().parent
        for key in ("meshdir", "texturedir"):
            compiler.set(key, str((base / compiler.get(key, ".")).resolve()))
        for include in root.iter("include"):
            include.set("file", str((base / include.get("file")).resolve()))

    def add(node, values):
        for tag, key in TAGS:
            for child in values.get(key, ()):
                attributes = {k: v for k, v in child.items() if k != "free" and k not in dict(TAGS).values()}
                element = ET.SubElement(node, tag, {k: text(v) for k, v in attributes.items()})
                if child.get("free"):
                    ET.SubElement(element, "freejoint")
                add(element, child)

    add(world, dict(children=bodies, cameras=cameras))
    (path := Path(path).resolve()).parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(path, encoding="unicode")
    model, _ = load(path)  # Fail on invalid geometry/assets before any replay.
    # MuJoCo skips parent-child contacts unless the parent is welded to the world; skip those pairs too.
    for child in range(1, model.nbody):
        parent = model.body_parentid[child]
        if parent and not model.body_weldid[parent] and model.body_weldid[child] == child:
            ET.SubElement(contact, "exclude", body1=model.body(parent).name, body2=model.body(child).name)
    ET.ElementTree(root).write(path, encoding="unicode")
    return path


def rollout(scene, out, ctrl, *, fps, frames, initial=None, camera=None, size=(640, 480), video=True, preview=12,
            source=None):  # fmt: skip
    """Step per-step controls from `initial` in CPU MuJoCo; save, diagnose and render each source frame (real2sim SKILL.md)."""
    import mujoco

    started, (model, data), ctrl = time.monotonic(), load(scene), np.asarray(ctrl, float)
    steps = np.rint(np.arange(frames) / fps / model.opt.timestep).astype(int)
    if frames < 2 or np.any(np.diff(steps) <= 0) or ctrl.shape != (steps[-1], model.nu):
        raise ValueError(f"Need {frames} frames at a rate below the step rate and ctrl of shape ({steps[-1]}, nu)")
    low, high = model.actuator_ctrlrange.T
    if not np.isfinite(ctrl).all() or np.any(model.actuator_ctrllimited & ((ctrl < low - 1e-6) | (ctrl > high + 1e-6))):
        raise ValueError("Controls must be finite and inside native ctrlrange; nothing is clipped silently")
    for key, value in (initial or {}).items():
        getattr(data, key)[:] = value
    data.ctrl[:] = ctrl[0]
    mujoco.mj_forward(model, data)
    (out := Path(out)).mkdir(parents=True, exist_ok=False)
    mujoco.mj_saveModel(model, str(out / "scene.mjb"))
    saved, contacts, penetration = {k: [] for k in ("qpos", "qvel", "ctrl", "time")}, [], 0.0
    for frame, target in enumerate(steps):
        for row in ctrl[steps[frame - 1] if frame else 0 : target]:
            data.ctrl[:] = row
            mujoco.mj_step(model, data)
            penetration = max(penetration, max((-c.dist for c in data.contact), default=0.0))
        if not all(np.isfinite(x).all() for x in (data.qpos, data.qvel, data.qacc)):
            raise RuntimeError(f"Nonfinite physics before source frame {frame}")
        for key, values in saved.items():
            values.append(np.copy(getattr(data, key)))
        pairs, force = {}, np.zeros(6)
        for i, c in enumerate(data.contact):
            mujoco.mj_contactForce(model, data, i, force)
            names = tuple(model.geom(g).name or str(g) for g in sorted((c.geom1, c.geom2)))
            total = pairs.setdefault(names, [0.0, np.zeros(3)])
            total[0] += force[0]
            total[1] += c.frame[:3] * force[0] * (1 if c.geom2 > c.geom1 else -1)  # on the second geom
        contacts.append([dict(geoms=k, normal_force_N=f, force_world_N=v.tolist()) for k, (f, v) in pairs.items()])
    np.savez_compressed(out / "trajectory.npz", **{key: np.asarray(value) for key, value in saved.items()},
                        source_frame_indices=np.arange(frames), source_time=np.arange(frames) / fps)  # fmt: skip
    report = dict(frames=frames, physical_seconds=float(data.time), max_penetration_m=penetration,
                  warnings={mujoco.mjtWarning(i).name: int(w.number) for i, w in enumerate(data.warning) if w.number},
                  equality_constraints=int(model.neq), mocap_bodies=int(model.nmocap),
                  contact_frames=contacts, scene_sha256=file_digest(out / "scene.mjb"),
                  state_sha256=file_digest(out / "trajectory.npz"), task_success=None)  # fmt: skip
    render(out, camera=camera, fps=fps, size=size, video=video, preview=preview, source=source)
    report["seconds"] = time.monotonic() - started
    (out / "diagnostics.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def measure(out, rows, objects, reference=None, projection=None):
    """Kinematics of saved `rows` in `out` without stepping: poses in the `reference` frame and pixels (real2sim SKILL.md)."""
    import mujoco

    (model, data), result, P = load(Path(out) / "scene.mjb"), [], None if projection is None else np.asarray(projection)
    pixel = lambda p: None if P is None or (h := P @ np.r_[p, 1])[2] <= 0 else (h[:2] / h[2]).tolist()
    for data.qpos[:] in np.load(Path(out) / "trajectory.npz")["qpos"][list(rows)]:
        mujoco.mj_kinematics(model, data)
        poses = {k: (getattr(data, kind)(name).xpos.copy(), getattr(data, kind)(name).xmat.reshape(3, 3).copy())
                 for k, (kind, name) in objects.items()}  # fmt: skip
        origin, basis = poses[reference] if reference else (np.zeros(3), np.eye(3))
        result.append({k: dict(xyz=(basis.T @ (p - origin)).tolist(), rotation=(basis.T @ m).tolist(), pixel=pixel(p))
                       for k, (p, m) in poses.items()})  # fmt: skip
    return result


def render(out, *, camera=None, fps=30, size=(640, 480), video=True, preview=12, source=None):
    """Render `out`/trajectory.npz to mujoco_ext1.mp4; preview.jpg tiles `preview` rows (count or list), with source."""
    import cv2
    import mujoco
    from PIL import Image, ImageDraw

    out, (width, height) = Path(out), size
    hashes = dict(scene_sha256=file_digest(out / "scene.mjb"), state_sha256=file_digest(out / "trajectory.npz"))
    model, data = load(out / "scene.mjb")
    with np.load(out / "trajectory.npz") as saved:
        qpos = saved["qpos"]
    rows = set(np.linspace(0, len(qpos) - 1, min(preview, len(qpos)), dtype=int) if np.ndim(preview) == 0 else preview)
    if not (video or rows):
        return
    capture, tiles = cv2.VideoCapture(str(source)) if source is not None else None, []
    command = f"ffmpeg -y -loglevel error -f rawvideo -pix_fmt rgb24 -s {width}x{height} -r {fps} -i - -an -c:v libx264"
    command += " -threads 2 -pix_fmt yuv420p -movflags +faststart"
    ffmpeg = video and subprocess.Popen([*command.split(), str(out / "mujoco_ext1.mp4")], stdin=subprocess.PIPE)
    model.vis.global_.offwidth, model.vis.global_.offheight = width, height
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        for index in range(len(qpos)) if video else sorted(rows):
            data.qpos[:] = qpos[index]
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=-1 if camera is None else camera)
            rgb = renderer.render()
            if ffmpeg:
                ffmpeg.stdin.write(rgb.tobytes())
            if index in rows:
                if capture is not None:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                    ok, bgr = capture.read()
                    if not ok:
                        raise ValueError(f"Source video has no frame {index}")
                    real = cv2.resize(np.ascontiguousarray(bgr[..., ::-1]), (width, height))
                tile = Image.fromarray(np.hstack([rgb] if capture is None else [real, rgb, real // 2 + rgb // 2]))
                tile.thumbnail((tile.width * 240 // tile.height, 240))
                ImageDraw.Draw(tile).text((5, 5), f"frame {index}", fill="white", stroke_width=1, stroke_fill="black")
                tiles.append(tile)
    capture is None or capture.release()
    if ffmpeg and (ffmpeg.stdin.close() or ffmpeg.wait()):
        raise RuntimeError("Native video encoding failed")
    if tiles:
        (w, h), columns = tiles[0].size, min(4, len(tiles))
        sheet = Image.new("RGB", (columns * w, -(-len(tiles) // columns) * h))
        for i, tile in enumerate(tiles):
            sheet.paste(tile, (i % columns * w, i // columns * h))
        sheet.save(out / "preview.jpg")
    if video:
        hashes.update(video_sha256=file_digest(out / "mujoco_ext1.mp4"), frames=len(qpos), fps=float(fps))
        (out / "render.json").write_text(json.dumps(hashes, indent=2) + "\n")
