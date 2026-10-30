"""R1Pro's parallel gripper (`<side>_gripper_link` with two prismatic fingers) in the shared hand conventions.

Robot facts from BEHAVIOR's robot assets (`omnigibson-robot-assets/models/r1pro/urdf/r1pro.urdf`): each finger joint
travels 0..5 cm along the gripper link's +-y, the flat inner faces (pads) meet at 0 and stand 10 cm apart fully open; the
fingers hang 7.8 cm along the gripper link's -z, padded over their whole length, 1.6 cm wide. The TCP is OmniGibson's
own grasp frame (`<side>_eef_link`, 6 cm below the gripper link, 1.8 cm short of the fingertips), midway between the
pads: +x along the fingers (-z of the gripper link), +y the squeeze axis (+y of the gripper link), +z = x cross y (+x of
the gripper link).

`model(src, dst)` writes the RoboKit model: the torso link both arms hang off as the root, with both arms and grippers
and the head camera link; meshes from the robot assets, each visual mesh also its collision geometry (the URDF's own
collision meshes are not in the assets), the torso and the camera mounts visual only.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


TRAVEL = 0.05  # m, each finger joint from closed (pads touching) to open
WIDTH = 2 * TRAVEL  # m between the pads fully open
TCP_Z = -0.06  # the TCP along the gripper link's z (OmniGibson's eef link); the fingertips end at -0.078
R_EE_TCP = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])  # TCP axes (columns) in the gripper link
T_EE_TCP = np.eye(4)
T_EE_TCP[:3, :3], T_EE_TCP[2, 3] = R_EE_TCP, TCP_Z
ROOT = "torso_link4"
VISUAL_ONLY = ("torso_link4", "zed_link", "left_arm_base_link", "right_arm_base_link")


def finger_q(u):
    """Both finger joints at jaw opening u (0 closed .. 1 open): Hand.model_q."""
    return np.full(2, TRAVEL * float(np.clip(np.asarray(u, float).reshape(-1)[0], 0.0, 1.0)))


def opening(fingers):
    """Measured jaw opening 0..1 from the two finger joints: the pad gap over the gap fully open."""
    return np.array([float(np.clip(np.mean(fingers) / TRAVEL, 0.0, 1.0))])


def _tree(src):
    root = ET.parse(src).getroot()
    for mesh in root.iter("mesh"):
        if not Path(mesh.get("filename")).is_absolute():
            mesh.set("filename", str((Path(src).parent / mesh.get("filename")).resolve()))
    return root


def _subtree(root, top):
    """Names of `top` and every link below it."""
    children = {}
    for j in root.findall("joint"):
        children.setdefault(j.find("parent").get("link"), []).append(j.find("child").get("link"))
    out, todo = [], [top]
    while todo:
        out.append(todo.pop())
        todo += children.get(out[-1], [])
    return out


def _collide_as_seen(link, visual_only=False):
    """The link's visual geometry as its collision geometry too (or no collision geometry)."""
    for c in link.findall("collision"):
        link.remove(c)
    if not visual_only:
        for i, v in enumerate(link.findall("visual")):
            c = ET.fromstring(ET.tostring(v).replace(b"<visual", b"<collision").replace(b"</visual>", b"</collision>"))
            c.set("name", f"{link.get('name')}_collision{i}")  # (unnamed, it takes the link's name and is lost)
            for m in c.findall("material"):
                c.remove(m)
            link.append(c)


def model(src, dst, top=ROOT, visual_only=VISUAL_ONLY):
    """The RoboKit model: `top` and everything below it in R1Pro's URDF `src`, written to `dst`."""
    root = _tree(src)
    keep = set(_subtree(root, top))
    out = ET.Element("robot", name="r1pro_arms")
    for el in root:
        if el.tag == "link" and el.get("name") in keep:
            _collide_as_seen(el, el.get("name") in visual_only)
            out.append(el)
        elif el.tag == "joint" and el.find("child").get("link") in keep and el.find("parent").get("link") in keep:
            out.append(el)
        elif el.tag == "material":
            out.append(el)
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(out).write(dst)
    return str(dst)
