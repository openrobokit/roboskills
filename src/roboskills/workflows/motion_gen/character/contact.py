"""Joint ranges and contact on one character's posed skin.

`joints(character, turns)`: every joint's bend, sideways bend and twist from its rest (degrees, about the joint's
anatomical axes), and how far past a person's range each is. `touch(...)`: each hand bone's skin distance to an
object's surface per frame (mm, negative inside)."""

import re

import numpy as np
import warp as wp
from scipy.spatial.transform import Rotation

from roboskills.workflows.motion_gen.character.character import SPINE, Character, skinned


# A person's range from the rig's T-pose, degrees, about each joint's anatomical axes: bending (toward the side the
# joint flexes to; the palm for the wrist and fingers, toes up for the ankle), sideways (away from the midline, toward
# the thumb at the wrist, up for the collarbone) and twist about the bone (+ turns a T-pose's downward palm forward,
# for the forearm). Clinical active ranges (AAOS; Kapandji) with a margin; the elbow and knee from straight. The
# shoulder's ball joint (Arm) bounds only its twist.
RANGES = {"Spine": ((-30, 65), (-25, 25), (-20, 20)), "Spine1": ((-20, 45), (-25, 25), (-25, 25)),
          "Spine2": ((-20, 45), (-25, 25), (-25, 25)), "Neck": ((-40, 40), (-35, 35), (-50, 50)),
          "Head": ((-40, 35), (-30, 30), (-50, 50)), "Shoulder": ((-45, 45), (-25, 55), (-45, 45)),
          "Arm": ((-180, 180), (-180, 180), (-110, 110)), "ForeArm": ((-10, 155), (-10, 10), (-30, 175)),
          "Hand": ((-75, 85), (-35, 25), (-15, 15)), "UpLeg": ((-30, 130), (-30, 50), (-45, 45)),
          "Leg": ((-10, 155), (-10, 10), (-30, 30)), "Foot": ((-65, 30), (-30, 30), (-35, 35)),
          "ToeBase": ((-40, 70), (-15, 15), (-15, 15)), "Finger1": ((-20, 90), (-25, 25), (-10, 10)),
          "Finger2": ((-5, 110), (-15, 15), (-15, 15)), "Finger3": ((-5, 90), (-15, 15), (-15, 15)),
          "Thumb1": ((-60, 60), (-60, 60), (-60, 60)), "Thumb2": ((-10, 70), (-25, 25), (-25, 25)),
          "Thumb3": ((-15, 90), (-25, 25), (-25, 25))}  # fmt: skip


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def _axes(character: Character) -> dict[int, tuple[str, np.ndarray, np.ndarray, np.ndarray, float, float]]:
    """Per ranged joint: its range key, and in the rest frame the bone's direction, the way its tip moves when it bends
    and when it bends sideways, the rest's own bend (the elbow and knee are measured from straight) and the twist's
    sign."""

    at = lambda role: character.rest_joints[character.bone(role)]  # noqa: E731
    left = _unit(at("LeftUpLeg") - at("RightUpLeg"))
    up = _unit(at("Neck") - at("Hips"))
    up = _unit(up - (up @ left) * left)
    front = np.cross(left, up)
    out = {}
    for role in character.roles:
        side = next((s for s in ("Left", "Right") if role.startswith(s)), "")
        part, outward = role[len(side) :], (left if side == "Left" else -left)
        if side:  # index x pinky leaves the left hand's back, the right one's palm
            palm = _unit(
                np.cross(at(f"{side}HandIndex1") - at(f"{side}Hand"), at(f"{side}HandPinky1") - at(f"{side}Hand"))
            )
            palm, thumb = palm * (-1.0 if side == "Left" else 1.0), _unit(at(f"{side}HandThumb1") - at(f"{side}Hand"))
        finger = re.fullmatch(r"Hand(Thumb|Index|Middle|Ring|Pinky)([123])", part)
        child, key = None, part
        if part in SPINE[1:5]:
            child, bend, aside = SPINE[SPINE.index(part) + 1], front, left
        elif part == "Head":
            bend, aside = front, left
        elif part in ("Shoulder", "Arm", "ForeArm", "UpLeg", "Leg", "Foot"):
            child = side + {"Shoulder": "Arm", "Arm": "ForeArm", "ForeArm": "Hand", "UpLeg": "Leg", "Leg": "Foot", "Foot": "ToeBase"}[part]  # fmt: skip
            bend = {"Leg": -front, "Foot": up}.get(part, front)
            aside = up if part in ("Shoulder", "ForeArm") else outward
        elif part == "Hand":
            child, bend, aside = side + "HandMiddle1", palm, thumb
        elif part == "ToeBase":
            bend, aside = up, outward
        elif finger:
            key = ("Thumb" if finger[1] == "Thumb" else "Finger") + finger[2]
            child, bend, aside = f"{side}Hand{finger[1]}{int(finger[2]) + 1}", palm, thumb
        else:
            continue
        j = character.bone(role)
        if part == "Head":
            along = up
        elif part == "ToeBase":
            along = _unit(at(role) - at(side + "Foot"))
            along = _unit(along - (along @ up) * up)
        else:
            along = _unit(at(child) - character.rest_joints[j])
        bend = _unit(bend - (bend @ along) * along)
        axis = _unit(np.cross(along, bend))  # turning `along` about it moves it toward `bend`
        aside = axis * (1.0 if axis @ aside >= 0 else -1.0)
        sign = (
            1.0 if part != "ForeArm" or np.cross(along, palm) @ front >= 0 else -1.0
        )  # + turns the downward palm forward
        before = _unit(character.rest_joints[j] - character.rest_joints[character.parents[j]])
        rest = np.degrees(np.arctan2(np.cross(before, along) @ axis, before @ along)) if part in ("ForeArm", "Leg") else 0.0  # fmt: skip
        out[j] = (key, along, bend, aside, rest, sign)
    return out


def joints(character: Character, turns: np.ndarray) -> dict[str, dict]:
    """Per ranged joint: ``bend``, ``sideways`` and ``twist`` [T] (degrees from rest about its anatomical axes, the
    bends as how far the bone's tip moves each way), ``past`` [T] (degrees past a person's range, summed) and ``step``
    [T] (degrees it turned in its parent since the previous frame)."""

    out = {}
    for j, (key, along, bend, aside, rest, sign) in _axes(character).items():
        local = turns[:, character.parents[j]].transpose(0, 2, 1) @ turns[:, j]
        moved = local @ along  # where the bone's direction goes; the rest of the turn is its twist about it
        tip = np.arccos(np.clip(moved @ along, -1, 1))
        sideways_of = moved - (moved @ along)[:, None] * along
        norm = np.linalg.norm(sideways_of, axis=1)
        way = np.where(norm[:, None] > 1e-9, sideways_of / np.maximum(norm, 1e-9)[:, None], 0.0)
        b, s = np.degrees(tip * (way @ bend)) + rest, np.degrees(tip * (way @ aside))
        q = Rotation.from_matrix(local).as_quat()  # x y z w; the twist is the part about the bone (swing-twist)
        t = np.degrees(2 * np.arctan2(q[:, :3] @ along, q[:, 3]))
        t = ((t + 180) % 360 - 180) * sign
        (b0, b1), (s0, s1), (t0, t1) = RANGES[key]
        past = sum(
            np.maximum(lo - v, 0) + np.maximum(v - hi, 0) for v, lo, hi in ((b, b0, b1), (s, s0, s1), (t, t0, t1))
        )
        cos = (np.einsum("tij,tij->t", local[1:], local[:-1]) - 1) / 2  # of the turn from one frame to the next
        step = np.r_[0.0, np.degrees(np.arccos(np.clip(cos, -1, 1)))]
        out[character.joint_names[j]] = {"bend": b, "sideways": s, "twist": t, "past": past, "step": step}
    return out


@wp.kernel
def _signed(mesh: wp.uint64, points: wp.array(dtype=wp.vec3), out: wp.array(dtype=float)):
    i = wp.tid()
    q = wp.mesh_query_point_sign_winding_number(mesh, points[i], 1.0e6)
    out[i] = wp.length(points[i] - wp.mesh_eval_position(mesh, q.face, q.u, q.v)) * q.sign


def _distance(vertices: np.ndarray, faces: np.ndarray, points: np.ndarray, device: str) -> np.ndarray:
    """Signed distance (m, negative inside; winding number, so an open mesh works) of points [N, 3] to a mesh."""

    mesh = wp.Mesh(wp.array(vertices, dtype=wp.vec3, device=device), wp.array(faces.ravel(), dtype=int, device=device),
                   support_winding_number=True)  # fmt: skip
    out = wp.zeros(len(points), dtype=float, device=device)
    wp.launch(_signed, len(points), [mesh.id, wp.array(points, dtype=wp.vec3, device=device), out], device=device)
    return out.numpy()


def _hand(character: Character, side: str) -> dict[int, str]:
    """The hand's bones and their roles without the side (``Hand``, ``HandIndex2``...)."""
    return {character.bone(r): r[len(side) :] for r in character.roles if r.startswith(f"{side}Hand")}


def touch(character: Character, turns: np.ndarray, joints_: np.ndarray, side: str, vertices: np.ndarray,
          faces: np.ndarray, poses: np.ndarray, device: str = "cuda:0") -> dict[str, np.ndarray]:  # fmt: skip
    """Per hand bone (``Hand`` the palm, ``HandIndex2``...): its skin's distance to the object's surface [T] (mm,
    negative inside) for turns [T, J, 3, 3], joint positions [T, J, 3] and object poses [T, 4, 4] (world from it)."""

    owner, hand = character.skin_weights.argmax(1), _hand(character, side)
    ids = np.flatnonzero(np.isin(owner, list(hand)))
    skin = skinned(character, turns, joints_)[:, ids]
    inv = np.linalg.inv(poses)
    local = np.einsum("tij,tvj->tvi", inv[:, :3, :3], skin) + inv[:, None, :3, 3]
    d = _distance(np.asarray(vertices, float), np.asarray(faces), local.reshape(-1, 3), device).reshape(len(turns), -1)
    return {hand[b]: d[:, owner[ids] == b].min(1) * 1000 for b in set(owner[ids].tolist())}
