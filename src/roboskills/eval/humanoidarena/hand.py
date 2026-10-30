"""The G1's Dex3 hands in the shared hand conventions, and the RoboKit arm model.

Robot facts from Unitree's G1 rev 1.0 description with Dex3 hands (`g1_29dof_with_hand_rev_1_0.urdf`, BSD-3-Clause, as
cuRobo v0.8.0 ships it; the simulator's robot matches it, while upstream HumanoidArena's own copy is an older revision
with the shoulders 1 cm lower and another Dex3): each Dex3 hangs off `<side>_hand_palm_link`, 4.15 cm past the wrist yaw
joint; its index and middle fingers extend along the palm link's +x (bases 7.8 cm out) and curl toward the palm's face
(-y on the left hand, +y on the right), the thumb sits on that face and curls toward +x. The TCP is RAX's
hand centre (`g1_dex3_frames.json`: the midpoint between the finger and thumb tips of the open hand), 9.1 cm along +x and
4.95 cm out of the palm's face: +x out of the palm's face (the approach), +y along the fingers (thumb to fingers), +z =
x cross y. SONIC's hands are binary: a command closer to `closed` than to `open` closes the hand
(HumanoidArena's DEFAULT_HAND_POSE poses, read from the simulator).

`model(src, dst)` writes the RoboKit model: `torso_link` (the arms' mount, which moves with walking, the waist and the
pelvis height) and everything below it, both arms and hands, with the description's meshes made absolute; the torso and
the head visual only.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


SIDES = ("left", "right")
URDF = "curobo/curobo/content/assets/robot/g1/g1_29dof_with_hand_rev_1_0.urdf"  # in the vendor directory
ROOT = "torso_link"
VISUAL_ONLY = ("torso_link", "logo_link", "head_link", "waist_support_link", "imu_link")
TCP = {"left": (0.091, -0.0495, 0.0), "right": (0.091, 0.0495, 0.0)}  # in the palm link


def T_ee_tcp(side):
    """The TCP in `<side>_hand_palm_link`: +x out of the palm's face, +y along the fingers, +z = x cross y."""
    face = np.array([0.0, -1.0 if side == "left" else 1.0, 0.0])
    T = np.eye(4)
    T[:3, 0], T[:3, 1] = face, [1.0, 0.0, 0.0]
    T[:3, 2] = np.cross(T[:3, 0], T[:3, 1])
    T[:3, 3] = TCP[side]
    return T


def closed(u, open_, closed_):
    """SONIC's binary hand command for a 7-joint command `u`: 1 (closed) when it is nearer `closed_` than `open_`."""
    u = np.asarray(u, float).reshape(-1)
    return int(np.linalg.norm(u - closed_) < np.linalg.norm(u - open_))


def _subtree(root, top):
    children = {}
    for j in root.findall("joint"):
        children.setdefault(j.find("parent").get("link"), []).append(j.find("child").get("link"))
    out, todo = [], [top]
    while todo:
        out.append(todo.pop())
        todo += children.get(out[-1], [])
    return set(out)


def model(src, dst, top=ROOT, visual_only=VISUAL_ONLY):
    """The RoboKit model: `top` and everything below it in the G1's URDF `src`, written to `dst`."""
    root = ET.parse(src).getroot()
    for mesh in root.iter("mesh"):
        if not Path(mesh.get("filename")).is_absolute():
            mesh.set("filename", str((Path(src).parent / mesh.get("filename")).resolve()))
    keep = _subtree(root, top)
    out = ET.Element("robot", name="g1_arms")
    for el in root:
        if el.tag == "link" and el.get("name") in keep:
            if el.get("name") in visual_only:
                for c in el.findall("collision"):
                    el.remove(c)
            out.append(el)
        elif el.tag == "joint" and el.find("child").get("link") in keep and el.find("parent").get("link") in keep:
            out.append(el)
        elif el.tag == "material":
            out.append(el)
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(out).write(dst)
    return str(dst)


def press_point(path, side, q_closed, joints):
    """The closed hand's foremost point along the TCP's +x (the fist's front), in the TCP frame, from the model's
    finger meshes; the TCP itself if the model cannot be loaded."""
    try:
        import trimesh
        import yourdfpy

        urdf = yourdfpy.URDF.load(path, load_meshes=True, build_scene_graph=True)
        urdf.update_cfg({n: float(v) for n, v in zip(joints, q_closed)})
        T_tcp = urdf.get_transform(f"{side}_hand_palm_link", urdf.base_link) @ T_ee_tcp(side)
        pts = []
        for link in urdf.robot.links:
            if not link.name.startswith(f"{side}_hand_") or link.name.endswith("palm_link"):
                continue
            for v in link.visuals:
                mesh = trimesh.load(v.geometry.mesh.filename, force="mesh")
                T = urdf.get_transform(link.name, urdf.base_link) @ (v.origin if v.origin is not None else np.eye(4))
                pts.append(mesh.vertices @ T[:3, :3].T + T[:3, 3])
        P = np.vstack(pts) @ np.linalg.inv(T_tcp)[:3, :3].T + np.linalg.inv(T_tcp)[:3, 3]
        front = P[P[:, 0] > P[:, 0].max() - 0.005]
        return np.array([P[:, 0].max(), *np.median(front[:, 1:], axis=0)])
    except Exception:  # noqa: BLE001
        return np.zeros(3)
