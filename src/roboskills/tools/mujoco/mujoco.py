"""A MuJoCo scene at one source frame from perception outputs: a gravity-aligned world (z up, origin on the ground under
the object), the object (free, convex collision parts) resting on a box on the table plane or on Do as I Do's pedestals,
Do as I Do's MANO hand (mano/, MIT; contacts off) fitted to HaWoR's joints and the source camera; or a DROID robot
(franka/, Menagerie's Panda and Robotiq 2F-85) replaying its joints, the world its base, the camera its calibrated one."""

import json
from pathlib import Path
from typing import Literal

import numpy as np


MANO, FRANKA = Path(__file__).with_name("mano") / "right.xml", Path(__file__).with_name("franka") / "panda_2f85.xml"
FINGERS = [("thumb", "thumb1x"), ("index", "index1y"), ("middle", "middle1y"), ("ring", "ring1y"), ("pinky", "pinky1y")]
KEYPOINTS = [("body", "right_palm")] + [  # HaWoR's 21 joints in OpenPose order: wrist, then each finger base to tip
    point for finger, base in FINGERS for point in [("body", f"right_{base}")] + [("site", f"right_{finger}_{p}") for p in ("pip", "dip", "tip")]
]  # fmt: skip


def fit_hand(joints: np.ndarray, steps: int = 30) -> np.ndarray:
    """The MANO hand's qpos per frame fitted to `joints` (N, 21, 3; world frame, NaN where unseen) by damped least
    squares on MuJoCo's point Jacobians, each frame starting from the previous one (the first from the palm's rigid fit);
    unseen frames hold the nearest seen one."""
    import mujoco
    import trimesh
    from scipy.spatial.transform import Rotation

    model = mujoco.MjModel.from_xml_path(str(MANO))
    data, (low, high), jacobians = mujoco.MjData(model), model.jnt_range.T, np.zeros((len(KEYPOINTS), 3, model.nv))
    points = [
        (model.body(n).id, np.zeros(3)) if kind == "body" else (model.site(n).bodyid[0], model.site(n).pos)
        for kind, n in KEYPOINTS
    ]

    def keypoints(q):  # world positions, Jacobians ready
        data.qpos[:] = q
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        return np.array([data.xpos[b] + data.xmat[b].reshape(3, 3) @ o for b, o in points])

    rest, fitted, q = keypoints(np.zeros(model.nq)), np.full((len(joints), model.nq), np.nan), None
    for i, target in enumerate(joints):
        if not np.isfinite(target).all():
            continue
        if q is None:  # slides then x, y, z hinges on the palm: its origin, then intrinsic XYZ angles
            palm = trimesh.registration.procrustes(
                rest[[0, 1, 5, 9, 13, 17]], target[[0, 1, 5, 9, 13, 17]], scale=False, return_cost=False
            )
            q = np.r_[palm[:3, 3], Rotation.from_matrix(palm[:3, :3]).as_euler("XYZ"), np.zeros(model.nq - 6)]
        for _ in range(steps):
            x = keypoints(q)
            for jacobian, (b, _), point in zip(jacobians, points, x):
                mujoco.mj_jac(model, data, jacobian, None, point, b)
            J = jacobians.reshape(-1, model.nv)
            q = np.clip(q + np.linalg.solve(J.T @ J + 1e-4 * np.eye(model.nv), J.T @ (target - x).ravel()), low, high)
        fitted[i] = q
    seen = np.isfinite(fitted[:, 0])
    held = np.maximum.accumulate(np.where(seen, np.arange(len(fitted)), -1))
    return fitted[np.where(held < 0, seen.argmax(), held)] if seen.any() else np.zeros_like(fitted)


def assemble_scene(video: Path, masks: Path, depth: Path, shape: Path, collision: Path, out: Path, frame: int = 0,
                   gravity: Path | None = None, hands: Path | None = None, track: Path | None = None,
                   support: Literal["table", "pedestal"] = "table", robot: Path | None = None, camera: Path | None = None):  # fmt: skip
    """out/scene.xml, scene.mjb, trajectory.npz (hand or robot qpos, then the object's, per source frame to `frame` or,
    with `track` or `robot`, all), world.json (`world_from_camera`), preview.jpg (source | scene | blend) at `frame` and,
    moving, mujoco_ext1.mp4; from masks/, MoGe depth.npz, shape/ (mesh.ply, pose.json), CoACD collision/ and optional
    GeoCalib gravity.json (else up is the table's alone), HaWoR hands.npz, track's poses.npz, DROID's `robot`, `camera`."""
    import xml.etree.ElementTree as ET

    import mujoco
    import trimesh
    from PIL import Image

    from roboskills.eval.droid.read import read_robot
    from roboskills.tools.mujoco.engine import build, load, render

    out, collision = Path(out).resolve(), Path(collision).resolve()
    out.mkdir(parents=True, exist_ok=True)
    arm, world = read_robot(robot, camera) if robot else (None, None)
    d, down = np.load(depth), json.loads(Path(gravity).read_text())["down"] if gravity else [0, 1, 0]  # or the camera's
    row, up = int(np.searchsorted(d["frames"], frame)), world[2, :3] if robot else -np.array(down)
    z, k = d["depth"][row].astype(float), d["K"]
    (v, u), labels = np.indices(z.shape), np.array(Image.open(Path(masks) / f"{frame:05d}.png"))
    xyz = np.stack([(u - k[0, 2]) / k[0, 0] * z, (v - k[1, 2]) / k[1, 1] * z, z], -1)
    cloud, seen = xyz[d["valid"][row] & (labels == 0)], xyz[d["valid"][row] & (labels == 1)]  # background, object
    placed = np.array(json.loads((Path(shape) / "pose.json").read_text())["camera_from_object"])
    mesh = trimesh.load(Path(shape) / "mesh.ply").apply_transform(placed)  # camera frame
    bottom = np.percentile(seen @ up, 2) if len(seen) else (mesh.vertices @ up).min()  # the object's lowest depth point
    centre = mesh.vertices[np.argmin(mesh.vertices @ up)]  # the pedestals' ground; without gravity, the table's seed
    table = cloud[np.abs(cloud @ up - bottom + 0.06) < 0.09 if gravity or robot else np.linalg.norm(cloud - centre, axis=1) < 0.1]  # a band just under the object, or 10 cm round it  # fmt: skip
    for _ in range(3 if support == "table" else 0):  # the table plane by least squares, refit on its 1 cm inliers; up
        centre, normal = trimesh.points.plane_fit(table)
        up = normal * np.sign(normal @ up)
        table = cloud[np.abs((cloud - centre) @ up) < 0.01]
        if len(table) < 100:
            raise ValueError("no table plane under the object in the depth")
    if not robot:  # world from camera: z up, the origin on the ground under the object
        x, world = np.eye(3)[0] - up[0] * up, np.eye(4)
        world[:3, :3] = np.stack([x / np.linalg.norm(x), np.cross(up, x / np.linalg.norm(x)), up])
        world[:3, 3] = -world[:3, :3] @ (mesh.centroid - (mesh.centroid - centre) @ up * up)
    pose, top = world @ placed, (world[:3, :3] @ centre + world[:3, 3])[2]  # the ground's height
    pose[2, 3] -= mesh.apply_transform(world).bounds[0, 2] - top  # resting on it
    parts = [Path(p).stem for p in json.loads((collision / "parts.json").read_text())["parts"]]
    poses = np.load(track)["camera_from_object"] if track else placed[None].repeat(len(arm) if robot else frame + 1, 0)
    joints = np.load(hands)["joints"][: len(poses)] @ world[:3, :3].T + world[:3, 3] if hands and not robot else None
    actor = ET.parse(FRANKA if robot else MANO).getroot()  # the robot or the MANO hand, its meshes, then the object's
    actor.find("compiler").set("meshdir", str(FRANKA.parent if robot else MANO.parent / "meshes"))
    robot or actor.find("default/geom").set("rgba", "0.85 0.65 0.55 1")  # skin
    for tag in () if robot or hands else ("worldbody", "actuator"):  # no hand: MANO's defaults alone
        actor.remove(actor.find(tag))
    for name, file in {"object": collision / "visual.obj", **{p: collision / f"{p}.obj" for p in parts}}.items():
        ET.SubElement(actor.find("asset"), "mesh", name=name, file=str(file))
    ET.ElementTree(actor).write(out / "assets.xml")
    if support == "table":  # its box spans the plane's inliers
        low, high = np.percentile(trimesh.transform_points(table, world), [2, 98], axis=0)
    hull, ends = [trimesh.load(collision / f"{p}.obj").vertices for p in parts if support == "pedestal"], []
    for f in (0, -1) if support == "pedestal" else ():  # Do as I Do's pedestals, at each end out of the hand's reach
        x = trimesh.transform_points(np.concatenate(hull), world @ poses[f])  # its collision parts there
        base, xy = x[x[:, 2] < x[:, 2].min() + 0.01, :2], x[:, :2]  # its bottom centimetre (all of it, slanted)
        centre, middle = (base.min(0) + base.max(0)) / 2, (xy.min(0) + xy.max(0)) / 2
        if np.linalg.norm(middle - centre) > (radius := np.linalg.norm(base - centre, axis=1).max()):
            centre, radius = middle, np.linalg.norm(xy - middle, axis=1).max()
        held = joints is not None and np.linalg.norm(joints[f][:, None] - x, axis=-1).min() < 0.1  # NaN: unseen
        if not held and all(np.linalg.norm(centre - c) >= min(radius, r) for c, r, _ in ends):  # one per spot
            ends.append((centre, radius, x[:, 2].min()))
    look = dict(type="mesh", contype=0, conaffinity=0, density=0)  # MANO's default class turns contacts off (condim 1)
    solid = dict(contype=1, conaffinity=1, condim=6, friction=[1.5, 0.15, 1e-4])  # on: Do as I Do's ground contact
    camera = dict(name="source", pos=world[:3, 3], xyaxes=[*world[:3, 0], *-world[:3, 1]])
    build(out / "scene.xml", out / "assets.xml", cameras=[dict(camera, fovy=np.degrees(2 * np.arctan(z.shape[0] / 2 / k[1, 1])))], bodies=[
        *([dict(name="table", pos=[*(low + high)[:2] / 2, top - 0.025], parts=[dict(solid, type="box", size=[*(high - low)[:2] / 2, 0.025], rgba=[0.6, 0.55, 0.5, 1])])] if support == "table" else []),
        *(dict(name=f"pedestal{i}", pos=[*c, h - 0.0051], parts=[dict(solid, type="cylinder", size=[r, 0.005], rgba=[0.6, 0.55, 0.5, 1])]) for i, (c, r, h) in enumerate(ends)),
        dict(name="object", free=True, pos=pose[:3, 3], xyaxes=[*pose[:3, 0], *pose[:3, 1]],
             parts=[dict(look, mesh="object", rgba=[*mesh.visual.vertex_colors[:, :3].mean(0) / 255, 1]),
                    *(dict(solid, type="mesh", mesh=p, group=3, density=1000) for p in parts)]),
    ])  # fmt: skip
    model, _ = load(out / "scene.xml")
    mujoco.mj_saveModel(model, str(out / "scene.mjb"))
    qpos, adr = np.repeat(model.qpos0[None], len(poses), 0), model.jnt_qposadr[model.body("object").jntadr[0]]
    qpos[:, : model.nq - 7] = arm[: len(poses)] if robot else 0 if joints is None else fit_hand(joints)  # object last
    for row, p in zip(qpos, world @ poses):  # row f is source frame f
        row[adr : adr + 3] = p[:3, 3] + pose[:3, 3] - (world @ placed)[:3, 3]
        mujoco.mju_mat2Quat(row[adr + 3 : adr + 7], p[:3, :3].flatten())
    np.savez(out / "trajectory.npz", qpos=qpos)
    (out / "world.json").write_text(json.dumps(dict(world_from_camera=world.tolist())) + "\n")
    render(out, camera="source", size=z.shape[::-1], video=bool(track or robot), preview=[frame], source=video)
