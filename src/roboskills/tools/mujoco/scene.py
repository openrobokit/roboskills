"""Scene parts for a real2sim robot scene (the workflow's pnp step): an object's collision as a floor and thin walls (a receptacle),
a lid's prism or a narrowed body; a backdrop from the source frame (table and wall planes, or MoGe's depth as a mesh),
a container's open cylinder; the MJCF scene of objects (welds, a drawer on a slide joint, a cap screwed on) and two
Pandas with 2F-85s; its native rollout and source-camera renders."""

import subprocess
from pathlib import Path
from typing import Literal

import numpy as np

from roboskills.tools.mujoco.mujoco import FRANKA


FLIP = np.diag([1.0, -1.0, -1.0])  # OpenCV camera axes to MuJoCo's


def lid_parts(parts: list[Path], pose: np.ndarray, floor: float, out: Path) -> list[Path]:
    """A lid's collision: its parts' footprint straight up from world height `floor` (at `pose`) to their top, one
    prism, as out/00.obj: seen sunk into its cup, it rests on the rim, flat under and straight-sided for the jaws."""
    import trimesh

    v = np.concatenate([trimesh.transform_points(trimesh.load(p, force="mesh").vertices, pose) for p in parts])
    prism = np.r_[np.c_[v[:, :2], np.full(len(v), floor)], np.c_[v[:, :2], np.full(len(v), max(v[:, 2].max(), floor + 0.01))]]  # fmt: skip
    out.mkdir(parents=True, exist_ok=True)
    trimesh.Trimesh(trimesh.transform_points(prism, np.linalg.inv(pose))).convex_hull.export(out / "00.obj")
    return [out / "00.obj"]


def squeeze_parts(parts: list[Path], pose: np.ndarray, out: Path, squeeze: float = 0.75) -> list[Path]:
    """Collision parts narrowed to `squeeze` of their width about their centre (world z up at `pose`, their height
    kept), as out/<i>.obj: a soft object a parallel gripper can close on."""
    import trimesh

    meshes = [trimesh.load(p, force="mesh") for p in parts]
    centre = np.concatenate([trimesh.transform_points(m.vertices, pose) for m in meshes]).mean(0)
    out.mkdir(parents=True, exist_ok=True)
    for i, m in enumerate(meshes):
        v = trimesh.transform_points(m.vertices, pose)
        v[:, :2] = centre[:2] + (v[:, :2] - centre[:2]) * squeeze
        trimesh.Trimesh(trimesh.transform_points(v, np.linalg.inv(pose)), m.faces).export(out / f"{i:02d}.obj")
    return [out / f"{i:02d}.obj" for i in range(len(meshes))]


def hollow_parts(mesh, pose: np.ndarray, out: Path, wall: float = 0.006, sides: int = 12,
                 taper: bool = False) -> list[Path]:  # fmt: skip
    """Collision parts for a receptacle (a solid depth hull, or SAM 3D's mesh): a floor slab and `sides` thin walls
    around its footprint (world z up at `pose`), up to its top, in its own frame, as out/<i>.obj; things go in. With
    `taper`, `sides` (at least 16) walls inside its outline instead, from its footprint at its base (its lowest tenth)
    out to its outline at its rim (its top tenth): a cup or a bowl nests in another."""
    import trimesh
    from scipy.spatial import ConvexHull

    v = trimesh.transform_points(mesh.vertices, pose)
    bottom, top = v[:, 2].min(), v[:, 2].max()
    ring = v[ConvexHull(v[:, :2]).vertices, :2]
    ring = ring[np.linspace(0, len(ring), min(sides, len(ring)), endpoint=False).astype(int)]
    floor = np.r_[np.c_[ring, np.full(len(ring), bottom)], np.c_[ring, np.full(len(ring), bottom + wall)]]
    parts = [trimesh.Trimesh(floor).convex_hull]
    if taper:  # each ring's radius by angle about the rim's centre, walls inward of both
        centre, angles = ring.mean(0), np.linspace(0, 2 * np.pi, max(sides, 16), endpoint=False)

        def radii(points: np.ndarray) -> np.ndarray:
            hull = points[ConvexHull(points).vertices] - centre
            turn = np.arctan2(hull[:, 1], hull[:, 0])
            order = np.argsort(turn)
            return np.interp(angles, turn[order], np.linalg.norm(hull, axis=1)[order], period=2 * np.pi)

        low, high = radii(v[v[:, 2] < bottom + 0.1 * (top - bottom), :2]), radii(v[:, :2])
        floor = np.c_[centre + low[:, None] * np.c_[np.cos(angles), np.sin(angles)], np.full(len(angles), bottom)]
        parts = [trimesh.Trimesh(np.r_[floor, floor + [0, 0, wall]]).convex_hull]
        for i, j in zip(range(len(angles)), np.roll(range(len(angles)), -1)):
            corners = []
            for k in (i, j):
                ray = np.array([np.cos(angles[k]), np.sin(angles[k])])
                for r, z in ((low[k], bottom), (high[k], top)):
                    corners += [[*(centre + ray * r), z], [*(centre + ray * (r - wall)), z]]
            parts.append(trimesh.Trimesh(np.array(corners)).convex_hull)
    for a, b in zip(ring, np.roll(ring, -1, 0)) if not taper else ():
        normal = np.array([b[1] - a[1], a[0] - b[0]]) / max(np.linalg.norm(b - a), 1e-6) * wall
        corners = np.array([a, b, b + normal, a + normal])
        parts.append(
            trimesh.Trimesh(np.r_[np.c_[corners, np.full(4, bottom)], np.c_[corners, np.full(4, top)]]).convex_hull
        )
    out.mkdir(parents=True, exist_ok=True)
    paths = [out / f"{i:02d}.obj" for i in range(len(parts))]
    for path, part in zip(paths, parts):
        part.apply_transform(np.linalg.inv(pose)).export(path)
    return paths


def paint_backdrop(out: Path, ref: dict, world: np.ndarray, table: dict, objects: list[dict], step: int = 12,
                   walls: Literal["planes", "panel"] = "planes") -> dict:  # fmt: skip
    """The scene textured from the source's reference frame: out/backdrop.png (the frame at half size, hands, arms and
    objects inpainted), out/table.obj (a grid on the table plane over the table box) and the walls, `planes` (up to
    three upright planes RANSAC-fitted to the rest of the background, out/wall_<i>.obj, each a grid over its inliers)
    or a `panel` (out/wall.obj: that background as one depth mesh, every `step` pixels, torn at depth jumps over 10%;
    a photo behind the scene, which Real2Gym's rules do not accept), all UV-mapped onto it as the reference camera
    sees them; out/frame.png (the frame as is) and out/object_<name>.obj (each object's visual mesh UV-mapped onto it
    at its reference pose, a walled receptacle's (not a lid) open at the top, set as the object's `textured`).
    Returns the texture's path and the meshes'."""
    import cv2
    import trimesh

    h, w = ref["depth"].shape
    small = cv2.resize(np.ascontiguousarray(ref["rgb"][..., ::-1]), (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    hidden = cv2.dilate((cv2.resize(ref["labels"], (w // 2, h // 2), interpolation=cv2.INTER_NEAREST) > 0).astype(np.uint8), np.ones((9, 9), np.uint8))  # fmt: skip
    cv2.imwrite(str(out / "backdrop.png"), cv2.inpaint(small, hidden, 5, cv2.INPAINT_TELEA))
    cv2.imwrite(str(out / "frame.png"), small)
    k, camera = ref["K"], np.linalg.inv(world)

    def write(path: Path, vertices: np.ndarray, uv: np.ndarray, faces: np.ndarray):
        lines = [f"v {a:.5f} {b:.5f} {c:.5f}" for a, b, c in vertices] + [f"vt {a:.5f} {1 - b:.5f}" for a, b in uv]
        path.write_text("\n".join(lines + [f"f {a}/{a} {b}/{b} {c}/{c}" for a, b, c in faces + 1]) + "\n")

    for o in objects:  # each object's visual mesh, its pixels where the reference camera saw it
        mesh = trimesh.load(o["visual"], force="mesh")
        if o["walled"] and "lid" not in o["noun"]:  # open at the top, as its walls are: what goes in stays seen
            up = mesh.face_normals @ o["poses"][o["row"]][2, :3]
            z = trimesh.transform_points(mesh.triangles_center, o["poses"][o["row"]])[:, 2]
            mesh.update_faces(~((up > 0.5) & (z > z.min() + 0.6 * np.ptp(z))))
        seen = trimesh.transform_points(mesh.vertices, o["camera"][o["row"]])
        uv = np.clip((seen[:, :2] / np.maximum(seen[:, 2:], 1e-3) * np.diag(k)[:2] + k[:2, 2]) / [w, h], 0, 1)
        write(out / f"object_{o['name']}.obj", mesh.vertices, uv, mesh.faces)
        o["textured"] = out / f"object_{o['name']}.obj"

    (cx, cy, _), (hx, hy, _) = table["pos"], table["size"]
    gx, gy = np.meshgrid(np.linspace(cx - hx, cx + hx, 64), np.linspace(cy - hy, cy + hy, 64), indexing="ij")
    grid = np.c_[gx.ravel(), gy.ravel(), np.full(gx.size, 0.0005)]
    seen = grid @ camera[:3, :3].T + camera[:3, 3]
    uv = np.clip((seen[:, :2] / np.maximum(seen[:, 2:], 1e-3) * np.diag(k)[:2] + k[:2, 2]) / [w, h], 0, 1)
    quads = (np.arange(63)[:, None] * 64 + np.arange(63)[None]).ravel()
    write(
        out / "table.obj",
        grid,
        uv,
        np.r_[np.c_[quads, quads + 64, quads + 1], np.c_[quads + 1, quads + 64, quads + 65]],
    )
    rows, cols = np.arange(0, h, step), np.arange(0, w, step)
    keep = (ref["valid"] & (ref["labels"] == 0) & ~table["plane"] & (ref["depth"] > 0))[np.ix_(rows, cols)]
    points = table["xyz"][np.ix_(rows, cols)].reshape(-1, 3) @ world[:3, :3].T + world[:3, 3]
    meshes = [out / "table.obj"]
    if walls == "panel":
        depth, n = ref["depth"][np.ix_(rows, cols)], len(cols)
        corner = (np.arange(len(rows) - 1)[:, None] * n + np.arange(n - 1)[None]).ravel()
        faces = np.r_[np.c_[corner, corner + n, corner + 1], np.c_[corner + 1, corner + n, corner + n + 1]]
        flat, ok = depth.ravel(), keep.ravel()
        good = ok[faces].all(1) & (flat[faces].max(1) < 1.1 * flat[faces].min(1))
        pixels = np.stack(np.meshgrid(cols, rows), -1).reshape(-1, 2) / [w, h]
        write(out / "wall.obj", points, pixels, faces[good])
        return dict(texture=out / "backdrop.png", meshes=[*meshes, out / "wall.obj"])
    rest, rng = points[keep.ravel()], np.random.default_rng(0)
    for i in range(3):  # upright planes, largest first: three points, a normal near horizontal, 2 cm inliers
        best = np.zeros(len(rest), bool)
        for a, b, c in rest[rng.integers(0, len(rest), (300, 3))] if len(rest) > 100 else ():
            normal = np.cross(b - a, c - a)
            if np.linalg.norm(normal) > 1e-6 and abs(normal[2]) < 0.3 * np.linalg.norm(normal):
                near = np.abs((rest - a) @ (normal / np.linalg.norm(normal))) < 0.02
                best = near if near.sum() > best.sum() else best
        if best.sum() < 0.1 * keep.sum():
            break
        centre, normal = trimesh.points.plane_fit(rest[best])
        across = np.cross([0, 0, 1], normal) / np.linalg.norm(np.cross([0, 0, 1], normal))
        (u0, z0), (u1, z1) = np.percentile(np.c_[(rest[best] - centre) @ across, rest[best][:, 2]], [2, 98], axis=0)
        gu, gz = np.meshgrid(np.linspace(u0, u1, 32), np.linspace(min(z0, 0), z1, 32), indexing="ij")
        grid = (
            centre
            + np.outer(gu.ravel(), across)
            - np.outer(np.ones(gu.size), [0, 0, centre[2]])
            + np.outer(gz.ravel(), [0, 0, 1])
        )
        seen = grid @ camera[:3, :3].T + camera[:3, 3]
        uv = np.clip((seen[:, :2] / np.maximum(seen[:, 2:], 1e-3) * np.diag(k)[:2] + k[:2, 2]) / [w, h], 0, 1)
        cells = (np.arange(31)[:, None] * 32 + np.arange(31)[None]).ravel()
        write(
            out / f"wall_{i}.obj",
            grid,
            uv,
            np.r_[np.c_[cells, cells + 32, cells + 1], np.c_[cells + 1, cells + 32, cells + 33]],
        )
        meshes.append(out / f"wall_{i}.obj")
        rest = rest[~best]
    return dict(texture=out / "backdrop.png", meshes=meshes)


def own_pixels(o: dict, labels: np.ndarray):
    """Object `o`'s textured mesh (paint_backdrop's) remapped in place so no vertex takes another tracked object's
    pixels (the reference frame's `labels`): each such one the nearest pixel of its own mask. A bottle seen with its
    cap on would otherwise keep the cap's colour once the cap is off."""
    from scipy.ndimage import distance_transform_edt

    lines, own = o["textured"].read_text().splitlines(), int(o["name"])
    h, w = labels.shape
    uv = np.array([[float(x) for x in line.split()[1:]] for line in lines if line.startswith("vt ")])
    px = np.clip((np.c_[uv[:, 0], 1 - uv[:, 1]] * [w, h]).astype(int), 0, [w - 1, h - 1])
    _, (iy, ix) = distance_transform_edt(labels != own, return_indices=True)
    other = (labels[px[:, 1], px[:, 0]] != own) & (labels[px[:, 1], px[:, 0]] > 0)
    px[other] = np.c_[ix[px[other, 1], px[other, 0]], iy[px[other, 1], px[other, 0]]]
    rows = iter(np.c_[(px[:, 0] + 0.5) / w, 1 - (px[:, 1] + 0.5) / h])
    lines = [f"vt {u:.5f} {v:.5f}" if line.startswith("vt ") else line for line in lines for u, v in [next(rows) if line.startswith("vt ") else (0, 0)]]  # fmt: skip
    o["textured"].write_text("\n".join(lines) + "\n")


def build_scene(out: Path, objects: list[dict], table: dict, fovy: float, size: tuple[int, int], bases: dict,
                backdrop: dict | None = None, friction: float = 1.5, density: float = 300, force: float = 1.0,
                ghosts: tuple[str, ...] = (), holds: dict[str, int] = {}, drawers: list[dict] = (),
                screws: dict[str, tuple[str, int]] = {}):  # fmt: skip
    """out/scene.xml and scene.mjb: the table box, each object (free; its visual mesh, convex parts colliding), both
    arms on their `bases` (side: world from base), the `source` camera (posed per frame at render) and, given
    `backdrop` (paint_backdrop), its table and wall meshes textured with the source frame (lit mostly by their own
    emission), the box then colliding unseen. Objects' parts slide with `friction` (condim 4) at `density` (a
    non-grasped one's own `density` first); the fingers squeeze with `force` times the 2F-85's; `ghosts` (sides) touch
    nothing; `holds` (object: frame, -1 never) weld each object where it rests until that frame (see rollout); an
    `inert` object touches nothing; `drawers` (plan_drawer) each a fixed housing and its tray on a slide joint
    (drawer_<i>), the housing's child: they never touch each other."""
    import mujoco

    spec = mujoco.MjSpec()
    spec.compiler.degree = False
    spec.option.timestep, spec.option.impratio = 0.002, 10
    spec.option.integrator, spec.option.cone = mujoco.mjtIntegrator.mjINT_IMPLICITFAST, mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.visual.global_.offwidth, spec.visual.global_.offheight = size
    spec.visual.headlight.ambient, spec.visual.headlight.diffuse = [0.45] * 3, [0.45] * 3
    spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_FLAT,
                     rgb1=table["sky"], rgb2=table["sky"], width=8, height=8)  # fmt: skip
    world = spec.worldbody
    world.add_light(
        pos=[0.3, 0, 2],
        dir=[0, 0, -1],
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
        diffuse=[0.5] * 3,
        castshadow=False,
    )
    world.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_BOX, pos=table["pos"], size=table["size"], rgba=table["rgba"],
                   group=3 if backdrop else 0)  # fmt: skip
    if backdrop:
        spec.add_texture(name="backdrop", type=mujoco.mjtTexture.mjTEXTURE_2D, file=str(backdrop["texture"]))
        material = spec.add_material(name="backdrop", emission=0.6, specular=0, shininess=0)
        material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "backdrop"
        spec.add_texture(
            name="frame", type=mujoco.mjtTexture.mjTEXTURE_2D, file=str(backdrop["texture"].with_name("frame.png"))
        )
        material = spec.add_material(name="frame", emission=0.5, specular=0.1, shininess=0)
        material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "frame"
        for path in backdrop["meshes"]:
            spec.add_mesh(
                name=f"backdrop_{path.stem}", file=str(path), inertia=mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
            )
            world.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"backdrop_{path.stem}", material="backdrop", contype=0,
                           conaffinity=0)  # fmt: skip
    world.add_camera(name="source", fovy=np.degrees(fovy))
    for o in objects:
        name = o["name"]
        spec.add_mesh(name=f"{name}_visual", file=str(o.get("textured", o["visual"])))
        p = o["poses"][0]
        body = world.add_body(name=name, pos=p[:3, 3], quat=mat2quat(p[:3, :3]))
        body.add_freejoint(name=name)
        body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname=f"{name}_visual",
            contype=0,
            conaffinity=0,
            density=0,
            **(dict(material="frame") if "textured" in o else dict(rgba=o["rgba"])),
        )
        for i, part in enumerate(o["parts"]):
            spec.add_mesh(name=f"{name}_{i}", file=str(part))
            body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname=f"{name}_{i}",
                group=3,
                contype=0 if o.get("inert") else 1,
                conaffinity=0 if o.get("inert") else 1,
                density=o.get("density", density),
                friction=[friction, 0.08, 0.002],
                condim=4,
            )
    for i, d in enumerate(drawers):
        house = world.add_body(name=f"housing_{i}", pos=d["pose"][:3, 3], quat=mat2quat(d["pose"][:3, :3]))
        for centre, half in d["housing"]:
            house.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=centre, size=half, rgba=d["rgba"], friction=[friction, 0.08, 0.002], condim=4)  # fmt: skip
        tray = house.add_body(name=f"drawer_{i}", pos=d["tray"])
        tray.add_joint(name=f"drawer_{i}", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[-1, 0, 0], range=[0, d["range"]],
                       damping=2.0, frictionloss=0.2)  # fmt: skip
        for j, (centre, half) in enumerate(d["boxes"]):  # the front see-through, as an organiser's
            rgba = [*d["rgba"][:3], 0.55] if j == 1 else [0.95, 0.95, 0.92, 1] if j == len(d["boxes"]) - 1 else d["rgba"]  # fmt: skip
            tray.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=centre, size=half, rgba=rgba, density=500, friction=[friction, 0.08, 0.002], condim=4)  # fmt: skip
    for side, base in bases.items():
        arm = mujoco.MjSpec.from_file(str(FRANKA))
        for m in arm.meshes:
            m.file = str(FRANKA.parent / m.file)
        for light in list(arm.lights):
            arm.delete(light)
        for body in arm.bodies[1:]:  # held up by its controller, as a Panda's torque loop does: no sag to lag behind
            body.gravcomp = 1
            for geom in body.geoms if body.name.startswith("link") or side in ghosts else ():  # only a grasping
                geom.contype = geom.conaffinity = 0  # gripper touches the scene: links (or an arm that only mimics its
                # hand) on paths planned without obstacles would plough through it
        arm.actuator("fingers_actuator").forcerange *= force
        world.add_frame(pos=base[:3, 3]).attach_body(arm.body("link0"), f"{side}_", "")
    for name, frame in holds.items():  # resting where it is until grasped: no slide or topple from shape errors
        weld = spec.add_equality(type=mujoco.mjtEq.mjEQ_WELD, name1=name, objtype=mujoco.mjtObj.mjOBJ_BODY,
                                 name=f"hold:{name}:{frame}", solref=[0.004, 1], solimp=[0.99, 0.999, 0.001, 0.5, 2])  # fmt: skip
        weld.data = np.r_[np.zeros(3), np.array(weld.data)[3:]]
    for name, (on, frame) in screws.items():  # a cap screwed on its bottle (unscrew) until that frame
        weld = spec.add_equality(type=mujoco.mjtEq.mjEQ_WELD, name1=name, name2=on, objtype=mujoco.mjtObj.mjOBJ_BODY,
                                 name=f"hold:{name}@{on}:{frame}", solref=[0.004, 1], solimp=[0.99, 0.999, 0.001, 0.5, 2])  # fmt: skip
        weld.data = np.r_[np.zeros(3), np.array(weld.data)[3:]]
    model = spec.compile()
    out.mkdir(parents=True, exist_ok=True)
    (out / "scene.xml").write_text(spec.to_xml())
    mujoco.mj_saveModel(model, str(out / "scene.mjb"))
    return model


def mat2quat(m: np.ndarray) -> np.ndarray:
    import mujoco

    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(m, float).ravel())
    return q


def render_video(model, qpos: np.ndarray, cameras: np.ndarray, path: Path, size: tuple[int, int], fps: float):
    """`qpos` per frame through the `source` camera at `cameras` (world from camera, OpenCV axes), to H264 `path`."""
    import mujoco

    data, cid, (width, height) = mujoco.MjData(model), model.camera("source").id, size
    command = f"ffmpeg -y -loglevel error -f rawvideo -pix_fmt rgb24 -s {width}x{height} -r {fps} -i - -an -c:v libx264 -pix_fmt yuv420p"
    ffmpeg = subprocess.Popen([*command.split(), str(path)], stdin=subprocess.PIPE)
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        for q, camera in zip(qpos, cameras):
            model.cam_pos[cid], model.cam_quat[cid] = camera[:3, 3], mat2quat(camera[:3, :3] @ FLIP)
            data.qpos[:] = q
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=cid)
            ffmpeg.stdin.write(renderer.render().tobytes())
    ffmpeg.stdin.close()
    if ffmpeg.wait():
        raise RuntimeError(f"encoding {path} failed")


def separate(model, objects: list[dict], steps: int = 30, cap: float = 0.03, apart: bool = False):
    """Objects that start inside one another (a lid fitted on a cup, two tracked meshes overlapping), as physics
    would throw them apart: at the first frame the upper of each pair is lifted out of the lower where they meet
    from above, else the second pushed out sideways along the contact normal (`apart`, where either is walled: away
    from the first's centre, as thin walls' normals point either way), at most `cap` in all; the shift is applied to
    all of each object's `poses` (in place), before anything is planned on them."""
    import mujoco

    data, adr = (
        mujoco.MjData(model),
        {model.body(o["name"]).id: model.jnt_qposadr[model.joint(o["name"]).id] for o in objects},
    )
    moved, walled = {b: np.zeros(3) for b in adr}, {model.body(o["name"]).id for o in objects if o["walled"]}
    for o in objects:
        data.qpos[adr[model.body(o["name"]).id] : adr[model.body(o["name"]).id] + 3] = o["poses"][0, :3, 3]
        mujoco.mju_mat2Quat(
            data.qpos[adr[model.body(o["name"]).id] + 3 : adr[model.body(o["name"]).id] + 7],
            o["poses"][0, :3, :3].ravel(),
        )
    for _ in range(steps):
        mujoco.mj_forward(model, data)
        shift = {}
        for c in data.contact[: data.ncon]:
            one, two = model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]
            if one in adr and two in adr and one != two and c.dist < (0.0 if apart else -0.002):  # thin walls: touching
                if abs(c.frame[2]) > 0.7:  # one on top of the other: the upper goes up
                    upper = max(one, two, key=lambda b: data.xpos[b][2])
                    shift[upper] = np.maximum(shift.get(upper, 0), [0, 0, -c.dist + 0.001])
                elif apart and walled & {one, two}:  # side by side: the second away from the first, a mm at a time
                    away = data.xpos[two][:2] - data.xpos[one][:2]
                    shift[two] = np.r_[away / max(np.linalg.norm(away), 1e-6) * 0.001, 0]
                else:  # side by side: the second out along the normal, with a millimetre to spare
                    shift[two] = shift.get(two, 0) - np.r_[c.frame[:2] * (c.dist - 0.001), 0]
        shift = {b: v for b, v in shift.items() if np.linalg.norm(moved[b] + v) <= cap}
        if not shift:
            break
        for b, v in shift.items():
            data.qpos[adr[b] : adr[b] + 3] += v
            moved[b] += v
    for o in objects:
        o["poses"][:, :3, 3] += moved[model.body(o["name"]).id]


def seat(model, objects: list[dict], steps: list[dict], step: float = 0.002):
    """pnp: each step going `into` another object ends (in place) where it would rest in it: lowered from 15 cm over
    its end pose, once clear of every other object (each where it is by then, the ones placed before at their ends),
    until its collision parts first meet one (over a millimetre deep), the last clear height; never under the table.
    A nested cup stands as high in the other as its walls let it."""
    import mujoco
    import trimesh

    data, placed = mujoco.MjData(model), {}
    adr = [model.jnt_qposadr[model.joint(o["name"]).id] for o in objects]
    bodies = {model.body(o["name"]).id: i for i, o in enumerate(objects)}
    for s in sorted(steps, key=lambda s: np.inf if s["release"] is None else s["release"]):
        if s["into"] is not None:
            for i, o in enumerate(objects):
                pose = s["end"] if i == s["object"] else placed.get(i, o["poses"][0])
                data.qpos[adr[i] : adr[i] + 3], data.qpos[adr[i] + 3 : adr[i] + 7] = pose[:3, 3], mat2quat(pose[:3, :3])
            base = trimesh.transform_points(objects[s["object"]]["mesh"].vertices, s["end"])[:, 2].min()
            base -= s["end"][2, 3]
            free = None
            for z in np.arange(s["end"][2, 3] + 0.15, -base, -step):
                data.qpos[adr[s["object"]] + 2] = z
                mujoco.mj_forward(model, data)
                touch = {bodies.get(model.geom_bodyid[c.geom1]) for c in data.contact[: data.ncon] if c.dist < -0.001 and bodies.get(model.geom_bodyid[c.geom2]) == s["object"]}  # fmt: skip
                touch |= {bodies.get(model.geom_bodyid[c.geom2]) for c in data.contact[: data.ncon] if c.dist < -0.001 and bodies.get(model.geom_bodyid[c.geom1]) == s["object"]}  # fmt: skip
                if touch - {None, s["object"]}:
                    if free is not None:
                        break
                    continue
                free = z
            if free is not None:
                s["end"][2, 3] = free
        placed[s["object"]] = s["end"]


def rollout(model, start: np.ndarray, ctrl: np.ndarray, fps: float, path: Path, objects: list[str],
            speed: tuple[float, float] = (2.0, 30.0)) -> dict:  # fmt: skip
    """Step `ctrl` (a row per source frame, linearly interpolated between frames) from `start` in CPU MuJoCo; save
    `qpos`, `qvel` and `ctrl` at each source frame to `path`; returns each object's final displacement from its start, the
    deepest penetration and how many steps an object's speed was capped at `speed` (m/s, rad/s): an arm driven into
    a deep overlap would otherwise fling it away. Welds named hold:<object>:<frame> (build_scene) hold until that
    frame (-1: throughout)."""
    import mujoco

    data = mujoco.MjData(model)
    data.qpos[:] = start
    data.ctrl[:] = ctrl[0]
    mujoco.mj_forward(model, data)
    steps, saved, deepest, capped = (
        int(round(1 / fps / model.opt.timestep)),
        [(data.qpos.copy(), data.qvel.copy())],
        0.0,
        0,
    )
    dofs = [model.jnt_dofadr[model.joint(o).id] + k for o in objects for k in (0, 3)]
    holds = {i: int(model.equality(i).name.rsplit(":", 1)[1]) for i in range(model.neq) if model.equality(i).name.startswith("hold:")}  # fmt: skip
    for f, (a, b) in enumerate(zip(ctrl[:-1], ctrl[1:])):
        for i, until in holds.items():  # a held object goes free when its grasp starts
            data.eq_active[i] = until < 0 or f < until
        for s in range(steps):
            data.ctrl[:] = a + (b - a) * (s + 1) / steps
            mujoco.mj_step(model, data)
            deepest = max(deepest, max((-c.dist for c in data.contact), default=0.0))
            for i, d in enumerate(dofs):
                if (norm := np.linalg.norm(data.qvel[d : d + 3])) > speed[i % 2]:
                    data.qvel[d : d + 3] *= speed[i % 2] / norm
                    capped += 1
        saved.append((data.qpos.copy(), data.qvel.copy()))
    qpos = np.array([q for q, _ in saved])
    np.savez(path, qpos=qpos, qvel=np.array([v for _, v in saved]), ctrl=ctrl)
    moved = {o: float(np.linalg.norm(qpos[-1, a : a + 3] - qpos[0, a : a + 3])) for o in objects
             for a in [model.jnt_qposadr[model.joint(o).id]]}  # fmt: skip
    return dict(
        object_moved_m=moved, max_penetration_m=deepest, seconds_simulated=float(data.time), capped_steps=capped
    )


def backdrop_render(model, data, q, size, emission: float) -> np.ndarray:
    """The source camera's image at `q` with the backdrop's emission set to `emission`."""
    import mujoco

    model.mat_emission[model.material("backdrop").id] = emission
    data.qpos[: len(q)] = q
    mujoco.mj_forward(model, data)
    model.vis.global_.offwidth, model.vis.global_.offheight = size[1], size[0]
    with mujoco.Renderer(model, *size) as renderer:
        renderer.update_scene(data, "source")
        return renderer.render().astype(float)


def depth_mesh(
    d: dict, labels: np.ndarray, robot: np.ndarray, k: np.ndarray, world: np.ndarray, table: float, step: int = 3
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(vertices, faces, uv) of the frame-0 depth as a mesh in the world (every `step` px): the robot's and the object's
    pixels take the depth of the background nearest them, points within 1.5 cm of the table go onto it, and triangles
    across a depth step of over 10% are left out."""
    from scipy.ndimage import distance_transform_edt

    z, h, wd = d["depth"][0].astype(float), *labels.shape
    hole = (labels > 0) | robot | ~d["valid"][0] | (z <= 0)
    z = z[tuple(distance_transform_edt(hole, return_distances=False, return_indices=True))]
    v, u = np.mgrid[0 : h + step - 1 : step, 0 : wd + step - 1 : step].clip(max=[[[h - 1]], [[wd - 1]]])
    zz = z[v, u]
    p = np.stack([(u - k[0, 2]) / k[0, 0] * zz, (v - k[1, 2]) / k[1, 1] * zz, zz], -1).reshape(-1, 3)
    p = p @ world[:3, :3].T + world[:3, 3]
    p[np.abs(p[:, 2] - table) < 0.015, 2] = table
    i = np.arange(zz.size).reshape(zz.shape)
    quads = np.stack([i[:-1, :-1], i[:-1, 1:], i[1:, 1:], i[1:, :-1]], -1).reshape(-1, 4)
    faces = np.r_[quads[:, [0, 2, 1]], quads[:, [0, 3, 2]]]
    depth = zz.ravel()[faces]
    faces = faces[depth.max(1) / depth.min(1) < 1.1]
    return p, faces, np.c_[u.ravel() / wd, 1 - v.ravel() / h]


def container_parts(
    found, d: dict, k: np.ndarray, world: np.ndarray, table: float, rgb: np.ndarray, show: bool, sides: int = 16
) -> list[dict]:
    """The task's container as an open cylinder of `sides` wall boxes and a floor on the table: the median of SAM 3's
    container mask (frame 0) in the depth, out to its 90th-percentile radius (3-20 cm), up to its 95th-percentile
    height (at least 2 cm); drawn in its mean colour when `show`. No parts without a container."""
    mask = found["container"] if "container" in found.files else np.zeros(1, bool)
    z, keep = d["depth"][0].astype(float), None
    if mask.sum() < 20:
        return []
    keep = mask & d["valid"][0] & (z > 0)
    v, u = np.nonzero(keep)
    p = np.stack([(u - k[0, 2]) / k[0, 0] * z[keep], (v - k[1, 2]) / k[1, 1] * z[keep], z[keep]], -1)
    p = p @ world[:3, :3].T + world[:3, 3]
    centre = np.median(p[:, :2], 0)
    radius = float(np.clip(np.percentile(np.linalg.norm(p[:, :2] - centre, axis=1), 90), 0.03, 0.2))
    height = max(float(np.percentile(p[:, 2], 95)) - table, 0.02)
    look = dict(rgba=[*rgb[mask].mean(0) / 255, 1], group=0 if show else 3)
    parts = [dict(type="cylinder", pos=[*centre, table + 0.003], size=[radius, 0.003], **look)]
    for a in np.arange(sides) * 2 * np.pi / sides:
        parts.append(dict(type="box", pos=[*(centre + radius * np.array([np.cos(a), np.sin(a)])), table + height / 2],
                          size=[0.004, radius * np.pi / sides * 1.1, height / 2], euler=[0, 0, a], **look))  # fmt: skip
    return parts


def decimate(vertices: np.ndarray, faces: np.ndarray, cell: float) -> tuple[np.ndarray, np.ndarray]:
    """Vertex clustering: vertices merged per `cell` (m) grid cell, collapsed faces dropped."""
    keys, inverse = np.unique(np.floor(vertices / cell).astype(np.int64), axis=0, return_inverse=True)
    merged = np.zeros((len(keys), 3))
    np.add.at(merged, inverse.ravel(), vertices)
    merged /= np.bincount(inverse.ravel())[:, None]
    f = inverse.ravel()[faces]
    f = f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])]
    return merged, f[np.unique(np.sort(f, 1), axis=0, return_index=True)[1]]


def write_obj(path: Path, vertices: np.ndarray, faces: np.ndarray, uv: np.ndarray | None = None):
    lines = [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in vertices]
    if uv is not None:
        lines += [f"vt {a:.5f} {b:.5f}" for a, b in uv]
        lines += [f"f {a}/{a} {b}/{b} {c}/{c}" for a, b, c in faces + 1]
    else:
        lines += [f"f {a} {b} {c}" for a, b, c in faces + 1]
    Path(path).write_text("\n".join(lines) + "\n")


def background(frames: np.ndarray, robot: np.ndarray, still: np.ndarray) -> np.ndarray:
    """The per-pixel median of `frames` where the (dilated) `robot` silhouettes are not; inside `still` (the object at
    frame 0) the last quarter's, so a moved object leaves no ghost; frame 0 where never free."""
    from scipy.ndimage import binary_dilation

    hidden = np.array([binary_dilation(r, iterations=4) for r in robot])
    x = np.where(hidden[..., None], np.nan, frames.astype(np.float32))
    median = np.nanmedian(x, 0)
    late = np.nanmedian(x[-max(len(x) // 4, 1) :], 0)
    median = np.where(still[..., None] & np.isfinite(late), late, median)
    return np.where(np.isfinite(median), median, frames[0]).astype(np.uint8)
