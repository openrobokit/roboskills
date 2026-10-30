"""A skinned Mixamo character (glTF/GLB): its bones and skin.

Convention: metres, +X the character's left, -Y its front, +Z up, the floor at z = 0, every
bone's rest frame aligned with the world axes. Only the role bones below are kept; the skin of
any other bone goes to its nearest kept ancestor. A pose is each bone's world turn from its rest.
"""

import json
import re
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


GLTF_TO_ZUP = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])  # glTF is y-up, facing +z
FINGERS = ("Thumb", "Index", "Middle", "Ring", "Pinky")
SPINE = ("Hips", "Spine", "Spine1", "Spine2", "Neck", "Head")
LIMBS = ("Shoulder", "Arm", "ForeArm", "Hand", "UpLeg", "Leg", "Foot", "ToeBase")
BODY_ROLES = SPINE + tuple(f"{side}{limb}" for side in ("Left", "Right") for limb in LIMBS)
FINGER_ROLES = tuple(f"{side}Hand{finger}{k}" for side in ("Left", "Right") for finger in FINGERS for k in (1, 2, 3, 4))


@dataclass(frozen=True)
class Character:
    """``rest_joints`` [J, 3], ``verts`` [V, 3], ``faces`` [F, 3], ``skin_weights`` [V, J];
    parents precede children; ``roles`` maps a role (``LeftHand``) to its bone."""

    name: str
    joint_names: tuple[str, ...]
    parents: np.ndarray
    rest_joints: np.ndarray
    verts: np.ndarray
    faces: np.ndarray
    skin_weights: np.ndarray
    roles: dict
    uvs: np.ndarray | None = None  # [V, 2] TEXCOORD_0
    face_material: np.ndarray | None = None  # [F] the source glTF's material index
    source: str = ""  # the glTF, for its materials and textures

    @property
    def rest_offsets(self) -> np.ndarray:
        offsets = self.rest_joints - self.rest_joints[np.maximum(self.parents, 0)]
        offsets[self.parents < 0] = 0.0
        return offsets

    def bone(self, role: str) -> int:
        return self.joint_names.index(self.roles[role])


def _gltf(path: Path) -> tuple[dict, list[bytes]]:
    data = path.read_bytes()
    if path.suffix.lower() != ".glb":
        gltf = json.loads(data)
        return gltf, [(path.parent / b["uri"]).read_bytes() for b in gltf["buffers"]]
    chunks, offset = {}, 12
    while offset < len(data):
        size, kind = struct.unpack_from("<II", data, offset)
        chunks[kind] = data[offset + 8 : offset + 8 + size]
        offset += 8 + size
    return json.loads(chunks[0x4E4F534A]), [chunks.get(0x004E4942, b"")]


def _accessor(gltf: dict, buffers: list[bytes], index: int) -> np.ndarray:
    entry = gltf["accessors"][index]
    view = gltf["bufferViews"][entry["bufferView"]]
    dtype = np.dtype({5121: "u1", 5123: "<u2", 5125: "<u4", 5126: "<f4"}[entry["componentType"]])
    width = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}[entry["type"]]
    raw = np.ndarray(
        (entry["count"], width), dtype, buffers[view["buffer"]], view.get("byteOffset", 0) + entry.get("byteOffset", 0),
        (view.get("byteStride", width * dtype.itemsize), dtype.itemsize),
    ).copy()  # fmt: skip
    if entry.get("normalized"):
        raw = raw.astype(np.float32) / np.iinfo(dtype).max
    return raw[:, 0] if width == 1 else raw


def load_character(path: str | Path) -> Character:
    """The first skin of a Mixamo ``.glb``/``.gltf`` and every mesh bound to it, at rest."""

    path = Path(path)
    gltf, buffers = _gltf(path)
    nodes, skin = gltf["nodes"], gltf["skins"][0]
    parent_of = {c: p for p, node in enumerate(nodes) for c in node.get("children", ())}
    world: dict[int, np.ndarray] = {}

    def world_of(i: int) -> np.ndarray:
        if i not in world:
            node, local = nodes[i], np.eye(4)
            if "matrix" in node:
                local = np.asarray(node["matrix"], float).reshape(4, 4).T
            else:
                local[:3, :3] = Rotation.from_quat(node.get("rotation", [0, 0, 0, 1])).as_matrix() * node.get(
                    "scale", [1, 1, 1]
                )
                local[:3, 3] = node.get("translation", [0, 0, 0])
            world[i] = local if i not in parent_of else world_of(parent_of[i]) @ local
        return world[i]

    joints = list(skin["joints"])
    names = [nodes[j].get("name", f"joint_{k}") for k, j in enumerate(joints)]
    stripped = {re.sub(r"^mixamorig\d*[:_]", "", n): n for n in names}
    roles = {r: stripped[r] for r in BODY_ROLES + FINGER_ROLES if r in stripped}
    if missing := [r for r in BODY_ROLES if r not in roles]:
        raise ValueError(f"{path} lacks body bones {missing}")
    # Kept bones in skin order with parents first; each skin joint's owner is its nearest kept ancestor.
    kept = sorted((names.index(b) for b in roles.values()), key=lambda k: len(_chain(joints[k], parent_of)))
    slot = {joints[k]: s for s, k in enumerate(kept)}
    owner = []
    for j in joints:
        at = j
        while at not in slot and at in parent_of:
            at = parent_of[at]
        owner.append(slot.get(at, 0))
    parents = []
    for k in kept:
        at = parent_of.get(joints[k], -1)
        while at >= 0 and at not in slot:
            at = parent_of.get(at, -1)
        parents.append(slot.get(at, -1))
    inverse_bind = _accessor(gltf, buffers, skin["inverseBindMatrices"]).reshape(-1, 4, 4).transpose(0, 2, 1)
    bind = np.stack([world_of(j) @ inverse_bind[k] for k, j in enumerate(joints)])
    verts, faces, weights, uvs, material = [], [], [], [], []
    for node in nodes:
        if "mesh" not in node or node.get("skin", -1) != 0:
            continue
        for primitive in gltf["meshes"][node["mesh"]]["primitives"]:
            a = primitive["attributes"]
            pos = _accessor(gltf, buffers, a["POSITION"]).astype(float)
            idx = _accessor(gltf, buffers, a["JOINTS_0"]).astype(int)
            w = _accessor(gltf, buffers, a["WEIGHTS_0"]).astype(np.float32)
            posed = np.einsum("vk,vkab,vb->va", w, bind[idx], np.c_[pos, np.ones(len(pos))])[:, :3]
            index = _accessor(gltf, buffers, primitive["indices"]) if "indices" in primitive else np.arange(len(pos))
            faces.append(index.astype(int).reshape(-1, 3) + sum(map(len, verts)))
            material.append(np.full(len(faces[-1]), primitive.get("material", -1)))
            uvs.append(
                _accessor(gltf, buffers, a["TEXCOORD_0"]) if "TEXCOORD_0" in a else np.zeros((len(pos), 2), np.float32)
            )
            verts.append(posed / np.maximum(w.sum(1, keepdims=True), 1e-12))
            skin_w = np.zeros((len(pos), len(kept)), np.float32)
            np.add.at(skin_w, (np.arange(len(pos))[:, None].repeat(idx.shape[1], 1), np.asarray(owner)[idx]), w)
            weights.append(skin_w / np.maximum(skin_w.sum(1, keepdims=True), 1e-12))
    verts = np.concatenate(verts) @ GLTF_TO_ZUP.T
    floor = np.array([0.0, 0.0, verts[:, 2].min()])
    return Character(
        path.stem, tuple(names[k] for k in kept), np.asarray(parents),
        np.stack([world_of(joints[k])[:3, 3] for k in kept]) @ GLTF_TO_ZUP.T - floor,
        verts - floor, np.concatenate(faces), np.concatenate(weights), roles,
        np.concatenate(uvs).astype(np.float32), np.concatenate(material), str(path),
    )  # fmt: skip


def _chain(node: int, parent_of: dict) -> list[int]:
    chain = [node]
    while chain[-1] in parent_of:
        chain.append(parent_of[chain[-1]])
    return chain


def positions(character: Character, turns: np.ndarray, root: np.ndarray) -> np.ndarray:
    """Bone positions [T, J, 3] from world turns and the root's path."""

    out = np.empty(turns.shape[:2] + (3,))
    for j, p in enumerate(character.parents):
        out[:, j] = root if p < 0 else out[:, p] + turns[:, p] @ character.rest_offsets[j]
    return out


def skinned(character: Character, turns: np.ndarray, joints: np.ndarray) -> np.ndarray:
    """[T, V, 3] the character's skin posed by world turns and joint positions (linear blend skinning)."""

    blend = np.zeros(turns.shape[:2] + (3, 4), np.float32)  # each bone's rest-to-pose transform, blended by skin weight
    blend[..., :3], blend[..., 3] = turns, joints - np.einsum("tjik,jk->tji", turns, character.rest_joints)
    skin = np.empty((len(turns), len(character.verts), 3), np.float32)
    homogeneous = np.c_[character.verts, np.ones(len(character.verts))].astype(np.float32)
    for k in range(0, len(turns), 64):
        mixed = character.skin_weights.astype(np.float32) @ blend[k : k + 64].transpose(1, 0, 2, 3).reshape(
            len(blend[0]), -1
        )
        skin[k : k + 64] = np.einsum("vtik,vk->tvi", mixed.reshape(len(homogeneous), -1, 3, 4), homogeneous)
    return skin
