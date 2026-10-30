"""The Panda's parallel gripper and the G1's three-finger hands (Unitree Dex3) in the shared hand conventions, from the
robot model the simulator writes (`sim/server.py`: the robot's USD as a URDF).

Panda (Isaac Lab's `panda_instanceable.usd`): two prismatic fingers on `panda_hand`, each 0..4 cm along the hand's
+-y, the pads 8 cm apart fully open. The TCP is Franka's own (`panda_hand` +z 10.34 cm, between the pads 1 cm short of
the fingertips): +x along the fingers (the hand's +z), +y the squeeze axis (the hand's +y), +z = x cross y.

Dex3 (G1_29DOF_CFG's hands): on `<side>_hand_palm_link`, an index and a middle finger (2 joints each, at palm
x = 7.8 cm, z = +-2.85 cm, pointing +x) and a thumb (3 joints, at x = 2.55 cm, pointing out of the palm). The palm faces
-y on the left hand and +y on the right (`m` below): the fingers curl and the thumb points that way. The TCP sits
4.5 cm out of the palm and 7.5 cm along it, between the thumb and the fingers: +x out of the palm, +y the palm's +x (the
thumb-to-fingers direction they close along), +z = x cross y. Joint signs mirror between the hands (a left curl is
negative, a right one positive; the thumb's the other way), except `thumb_0`'s, which turns about the palm's y.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


PANDA_TRAVEL = 0.04  # m, each finger joint from closed to open
PANDA_WIDTH = 2 * PANDA_TRAVEL  # m between the pads fully open
PANDA_TCP_Z = 0.1034
R_PANDA = np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])  # TCP axes (columns) in panda_hand
DEX_TCP = (0.075, 0.045, 0.0)  # palm frame (y times m): the TCP among the fingers, in front of the palm
CURL = {
    "index_0": 1.4,
    "index_1": 1.5,
    "middle_0": 1.4,
    "middle_1": 1.5,
    "thumb_0": 0.0,
    "thumb_1": 0.6,
    "thumb_2": 1.4,
}  # fmt: skip  (a fist, rad, before the side's signs)


def panda_q(u):
    """Both Panda finger joints at jaw opening u (0 closed .. 1 open): Hand.model_q."""
    return np.full(2, PANDA_TRAVEL * float(np.clip(np.asarray(u, float).reshape(-1)[0], 0.0, 1.0)))


def panda_opening(fingers):
    """Measured jaw opening 0..1 from the two finger joints: the pad gap over the gap fully open."""
    return np.array([float(np.clip(np.sum(fingers) / PANDA_WIDTH, 0.0, 1.0))])


def panda_tcp():
    T = np.eye(4)
    T[:3, :3], T[2, 3] = R_PANDA, PANDA_TCP_Z
    return T


def mirror(side):
    """m: the palm faces m * y of the palm link (left -1, right +1)."""
    return -1.0 if side == "left" else 1.0


def dex_tcp(side):
    """The TCP in `<side>_hand_palm_link`."""
    m = mirror(side)
    T = np.eye(4)
    T[:3, 0], T[:3, 1] = (0.0, m, 0.0), (1.0, 0.0, 0.0)
    T[:3, 2] = np.cross(T[:3, 0], T[:3, 1])
    T[:3, 3] = (DEX_TCP[0], m * DEX_TCP[1], DEX_TCP[2])
    return T


def R_palm(side):
    """The hand model's frame axes (columns) in the palm link: +x out of the palm, +z along the fingers."""
    m = mirror(side)
    x, z = np.array([0.0, m, 0.0]), np.array([1.0, 0.0, 0.0])
    return np.stack([x, np.cross(z, x), z], 1)


def part(name):
    """'left_hand_index_1_joint' -> 'index_1'."""
    return name.split("_hand_", 1)[1].rsplit("_joint", 1)[0]


def dex_pose(side, names, curl, keep=()):
    """Hand joint targets (in `names`' order) curled by `curl` {part: rad}, with the side's signs; parts in `keep`
    straight."""
    m = mirror(side)
    sign = lambda p: 1.0 if p == "thumb_0" else (-m if p.startswith("thumb") else m)  # noqa: E731
    return np.array([0.0 if part(n) in keep else sign(part(n)) * curl[part(n)] for n in names])


def _subtree(root, top):
    children = {}
    for j in root.findall("joint"):
        children.setdefault(j.find("parent").get("link"), []).append(j.find("child").get("link"))
    out, todo = [], [top]
    while todo:
        out.append(todo.pop())
        todo += children.get(out[-1], [])
    return out


def _hand_urdf(src, top, R_top_palm, key, dst):
    """The links of model `src` from `top` down, under a link "palm" at top's origin with axes `R_top_palm`, written to
    `dst`."""
    from roboskills.eval.embodiedswe.sim.server import rpy

    root = ET.parse(src).getroot()
    keep = set(_subtree(root, top))
    out = ET.Element("robot", name=key)
    ET.SubElement(out, "link", name="palm")
    j = ET.SubElement(out, "joint", name="palm_fixed", type="fixed")
    ET.SubElement(j, "parent", link="palm")
    ET.SubElement(j, "child", link=top)
    ET.SubElement(j, "origin", xyz="0 0 0", rpy=" ".join(map(str, rpy(R_top_palm.T))))
    for el in root:
        if el.tag == "link" and el.get("name") in keep:
            out.append(el)
        elif el.tag == "joint" and el.find("child").get("link") in keep and el.find("parent").get("link") in keep:
            out.append(el)
    Path(dst).write_text(ET.tostring(out, encoding="unicode"))


def index_tip(src, side, directory):
    """The pointed index's fingertip in the TCP frame of the G1's `side` Dex3 hand of model `src`, the rest curled: the far
    end of its last link's mesh. `directory` keeps the hand's URDF."""
    from robokit.robo import Robot

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"dex3_{side}.urdf"
    R = R_palm(side)
    _hand_urdf(src, f"{side}_hand_palm_link", R, path.stem, path)
    hand = Robot.load(str(path), load_meshes=True)
    T_hand_tcp = np.eye(4)
    T_hand_tcp[:3, :3] = R.T
    T_hand_tcp = T_hand_tcp @ dex_tcp(side)
    q = dex_pose(side, list(hand.spec.actuated_joint_names), CURL, keep=("index_0", "index_1"))
    T = hand.forward_kinematics_via_matrix_numpy(q[None].astype(np.float32), device="cpu")[0]
    tip = f"{side}_hand_index_1_link"
    V = np.asarray(hand.spec.get_link_mesh(tip, mode="collision", return_mesh=True).vertices, float)
    end = V[V[:, 0] > V[:, 0].max() - 0.004].mean(0)
    P = T[list(hand.link_names).index(tip)] @ np.r_[end, 1.0]  # in the hand frame
    return (np.linalg.inv(T_hand_tcp) @ P)[:3]
