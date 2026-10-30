"""A character as a RoboKit `Robot`, for RoboKit's humanoid retargeting (and its other solvers).

`robot(character)`: three revolute joints per bone about its anatomical axes (bend, sideways, twist; `contact`'s),
limited to a person's range (`contact.RANGES`), so an elbow or knee bends only as a hinge; a bone without a range
turns freely; the root is RoboKit's floating base. Each bone is a link of the same name. `turns(character, robot,
qpos)`: the bones' world turns [T, J, 3, 3] for RoboKit's solutions [T, 7 + dofs] (base xyz + wxyz, then the
actuated joints). `SMPLX`: the SMPL-X joint each body bone answers, by role (collars and toes included).
"""

import json

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from roboskills.workflows.motion_gen.character import contact
from roboskills.workflows.motion_gen.character.character import FINGER_ROLES, Character


BASE = "__base__"
SMPLX = {"Hips": "pelvis", "LeftUpLeg": "left_hip", "RightUpLeg": "right_hip", "LeftLeg": "left_knee",
         "RightLeg": "right_knee", "LeftFoot": "left_ankle", "RightFoot": "right_ankle", "LeftToeBase": "left_foot",
         "RightToeBase": "right_foot", "Spine": "spine1", "Spine1": "spine2", "Spine2": "spine3", "Neck": "neck",
         "Head": "head", "LeftShoulder": "left_collar", "RightShoulder": "right_collar", "LeftArm": "left_shoulder",
         "RightArm": "right_shoulder", "LeftForeArm": "left_elbow", "RightForeArm": "right_elbow",
         "LeftHand": "left_wrist", "RightHand": "right_wrist"}  # fmt: skip


def robot(character: Character, fingers: bool = False):
    """The character's `Robot`; its finger bones are held unless ``fingers`` (they triple the joints)."""

    from robokit.robo import Robot, RobotSpec
    from robokit.robo.robot_spec import JointType

    names, parents, count = character.joint_names, character.parents, len(character.joint_names)
    n = 3 * count
    joints = [f"{b}__{a}" for b in names for a in ("bend", "side", "twist")]
    parent_joints = np.empty(n, np.int32)
    for j, p in enumerate(parents):
        parent_joints[3 * j : 3 * j + 3] = (-1 if p < 0 else 3 * p + 2, 3 * j, 3 * j + 1)
    transforms = np.zeros((n, 7), np.float32)
    transforms[:, 3] = 1.0
    transforms[::3, :3] = character.rest_offsets
    axes = np.tile(np.eye(3, dtype=np.float32), (count, 1))
    limits = np.tile(np.float32([-np.pi, np.pi]), (n, 1))
    for j, (key, along, bend, aside, rest, sign) in contact._axes(character).items():
        (b0, b1), (s0, s1), (t0, t1) = contact.RANGES[key]
        axes[3 * j : 3 * j + 3] = (np.cross(along, bend), np.cross(along, aside), along)
        twist = (t0, t1) if sign > 0 else (-t1, -t0)
        limits[3 * j : 3 * j + 3] = np.radians([(b0 - rest, b1 - rest), (s0, s1), twist])
    held = set() if fingers else {character.bone(r) for r in FINGER_ROLES if r in character.roles}
    parented = set(parents[parents >= 0].tolist())
    moves = [parents[j] >= 0 and j not in held and (j in parented or character.skin_weights[:, j].max() > 0)
             for j in range(count)]  # fmt: skip
    live = [3 * j + a for j in range(count) for a in range(3) if moves[j]]
    actuated = np.full(n, -1, np.int32)
    actuated[live] = np.arange(len(live))
    links = [*names, BASE, *(f"__{a}__{b}" for a in ("bend", "side") for b in names)]
    link_parents = np.concatenate((np.arange(2, n, 3), [-1], np.arange(0, n, 3), np.arange(1, n, 3))).astype(np.int32)
    velocity = np.full(n, 20.0, np.float32)
    empty = {link: trimesh.Scene() for link in links}
    description = json.dumps([character.name, names, np.round(axes, 6).tolist(), np.round(limits, 6).tolist(), live])
    return Robot(RobotSpec(
        name=f"character_{character.name}", robot_description=description, format="urdf",
        joint_names=joints, nonfixed_joint_names=joints, actuated_joint_names=[joints[k] for k in live],
        joint_types=np.full(n, JointType.REVOLUTE, np.int32), joint_axes=axes,
        joint_twists=np.concatenate((np.zeros_like(axes), axes), axis=1),
        actuated_joint_indices=actuated, parent_joint_indices=parent_joints, parent_joint_transforms=transforms,
        mimic_actuated_joint_indices=np.full(n, -1, np.int32), mimic_multipliers=np.ones(n, np.float32),
        mimic_offsets=np.zeros(n, np.float32), joint_limits=limits, joint_velocity_limits=velocity,
        actuated_joint_limits=limits[live].copy(), actuated_joint_velocity_limits=velocity[live].copy(),
        link_names=links, link_parent_joint_indices=link_parents, base_link_name=BASE,
        ee_link_names=[b for j, b in enumerate(names) if j not in parented],
        link_visual_geometries=empty, link_collision_geometries=empty,
        local_collision_sphere_centers=np.empty((0, 3), np.float32), collision_sphere_radii=np.empty(0, np.float32),
        collision_spheres_link_indices=np.empty(0, np.int32), self_collision_ignored_pairs=np.empty((0, 2), np.int32),
        link_masses=np.zeros(len(links), np.float32), link_local_com_positions=np.zeros((len(links), 3), np.float32),
    ))  # fmt: skip


def turns(character: Character, robot, qpos: np.ndarray) -> np.ndarray:
    """Each bone's world turn from rest [T, J, 3, 3] for ``qpos`` [T, 7 + dofs]."""

    actuated = robot.spec.actuated_joint_indices
    angles = np.zeros((len(qpos), len(actuated)))
    angles[:, actuated >= 0] = qpos[:, 7 + actuated[actuated >= 0]]
    about = Rotation.from_rotvec((angles[..., None] * robot.spec.joint_axes).reshape(-1, 3))
    local = about.as_matrix().reshape(len(qpos), -1, 3, 3)
    out = np.empty((len(qpos), len(character.joint_names), 3, 3))
    for j, p in enumerate(character.parents):
        bone = local[:, 3 * j] @ local[:, 3 * j + 1] @ local[:, 3 * j + 2]
        out[:, j] = (Rotation.from_quat(qpos[:, [4, 5, 6, 3]]).as_matrix() if p < 0 else out[:, p]) @ bone
    return out
