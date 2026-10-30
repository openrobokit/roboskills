"""GVHMR's motion on a Mixamo character through RoboKit's humanoid retargeting, the recipe in SKILL.md.

`retarget_motion(motion, character, out)`: each bone turns as its SMPL-X joint, down the character's own proportions;
RoboKit fits the character's joint ranges to those targets frame by frame, the feet the person plants held where
they land (`feet`). `check_motion` measures the result.
"""

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from roboskills.workflows.motion_gen.character import contact, feet
from roboskills.workflows.motion_gen.character.character import Character, load_character, positions, skinned
from roboskills.workflows.motion_gen.character.robot import SMPLX, robot, turns


Y_TO_Z = Rotation.from_euler("x", 90, degrees=True).as_matrix()  # SMPL-X's y-up T-pose onto a Mixamo rest
TURN = (0, 0.5)  # a bone followed by its turn alone: its place comes from the bones above it
WEIGHTS = {"Hips": (30, 2), "Spine": TURN, "Spine1": TURN, "Spine2": (0.7, 0.7), "Neck": TURN, "Head": TURN,
           "Shoulder": TURN, "Arm": (1.5, 1), "ForeArm": TURN, "Hand": (2, 1.2), "UpLeg": (1.5, 1), "Leg": TURN,
           "Foot": (30, 2), "ToeBase": (30, 1)}  # role, its side's if it has one -> position, orientation  # fmt: skip
SETTLE = 8  # solves of the first frame before the clip: one, from the T-pose, leaves the arms short of it
LIMB = 0.1  # metres a foot or hand may stray from its target


def person(motion: Path, out: Path) -> tuple[np.ndarray, np.ndarray, float, float, np.ndarray, np.ndarray]:
    """From GVHMR's `motion`: the joints' world turns [T, 55, 3, 3] and the pelvis [T, 3], both z-up over the floor
    under the person (`feet.level`), the person's standing pelvis height, the fps, and the sole joints
    (`feet.SOLE_JOINTS`) [T, 4, 3] in that world and in the camera. Writes out/amass.npz, out/camera.npz and
    out/rest.npz, the files RoboKit's loader reads."""
    from robokit.helpers.humanoid_retargeting.loaders.smplx import load_smplx_file
    from robokit.smplx import SMPLX_JOINT_NAMES

    d = np.load(motion)
    fps, betas = float(d["fps"]), d["world_betas"].mean(0)
    np.savez(out / "amass.npz", root_orient=d["world_global_orient"], pose_body=d["world_body_pose"],
             trans=d["world_transl"], betas=betas, mocap_frame_rate=fps)  # fmt: skip
    np.savez(out / "rest.npz", root_orient=np.zeros((1, 3)), pose_body=np.zeros((1, 63)), trans=np.zeros((1, 3)),
             betas=betas, mocap_frame_rate=fps)  # fmt: skip
    np.savez(out / "camera.npz", root_orient=d["camera_global_orient"], pose_body=d["camera_body_pose"],
             trans=d["camera_transl"], betas=betas, mocap_frame_rate=fps)  # fmt: skip
    clip = load_smplx_file(str(out / "amass.npz"))
    poses, rest = clip.T_world_joint_xyz_wxyz, load_smplx_file(str(out / "rest.npz"))
    world = Rotation.from_quat(poses[..., [4, 5, 6, 3]].reshape(-1, 4)).as_matrix().reshape(*poses.shape[:2], 3, 3)
    standing = float(rest.joints[0, 0, 1] - rest.vertices[0, :, 1].min())
    joints = poses[..., :3] @ Y_TO_Z.T
    joints[..., 2] -= feet.level(clip.vertices @ Y_TO_Z.T, fps)[:, None]
    soles = [list(SMPLX_JOINT_NAMES).index(j) for j in feet.SOLE_JOINTS]
    seen = load_smplx_file(str(out / "camera.npz"), compute_vertices=False).T_world_joint_xyz_wxyz[:, soles, :3]
    return Y_TO_Z @ world @ Y_TO_Z.T, joints[:, 0], standing, fps, joints[:, soles], seen


def targets(character: Character, world: np.ndarray, joint_names: list[str]) -> np.ndarray:
    """Each bone's world turn [T, J, 3, 3]: its SMPL-X joint's in `world` [T, joints, 3, 3] (named `joint_names`), a
    bone without one its parent's."""
    role = {bone: r for r, bone in character.roles.items()}
    out = np.empty((len(world), len(character.joint_names), 3, 3))
    for j, p in enumerate(character.parents):
        name = SMPLX.get(role[character.joint_names[j]])
        out[:, j] = world[:, joint_names.index(name)] if name else out[:, p]
    return out


def camera_side(motion: Path) -> np.ndarray:
    """The horizontal direction from the person toward GVHMR's camera, z-up."""
    d = np.load(motion)
    camera_to_world = (
        Rotation.from_rotvec(d["world_global_orient"]) * Rotation.from_rotvec(d["camera_global_orient"]).inv()
    )
    toward = Y_TO_Z @ camera_to_world.apply(-d["camera_transl"]).mean(0)
    return toward * [1, 1, 0] / np.linalg.norm(toward[:2])


def weights(character: Character) -> dict[str, tuple[float, float]]:
    """`WEIGHTS` by the character's bones."""
    sides = lambda role: (role,) if role in character.roles else (f"Left{role}", f"Right{role}")  # noqa: E731
    return {character.roles[r]: w for role, w in WEIGHTS.items() for r in sides(role) if r in character.roles}


def retarget_motion(motion: Path, character: Path, out: Path):
    """out/motion.npz, motion_studio.py's MOTION: `fps`, `turns` [T, J, 3, 3], `positions` [T, J, 3], `joint_names`,
    `studio_camera_direction`; and for the bake and the check `rest_joints`, `targets` [T, J, 3], the position each
    bone was asked for, `planted` [T, 4] (`feet.down`) and `qpos`. The clip's lowest skin point lies on the floor,
    z = 0. Fingers stay at rest."""
    from robokit.helpers.humanoid_retargeting import (
        HumanoidRetargeterOnline,
        HumanoidRetargeterOnlineConfig,
        LinkMapping,
    )
    from robokit.smplx import SMPLX_JOINT_NAMES

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    c, (world, pelvis, standing, fps, soles, seen) = load_character(character), person(motion, out)
    wanted = targets(c, world, list(SMPLX_JOINT_NAMES))
    root = pelvis * c.rest_joints[c.bone("Hips"), 2] / standing
    planted = feet.down(soles, fps, seen)
    aim = feet.plant(feet.soles(c), wanted, positions(c, wanted, root), planted, fps)

    scored = weights(c)
    config = HumanoidRetargeterOnlineConfig(
        human_root_name=c.roles["Hips"], human_height_assumption=1.0, velocity_limit_dt=1 / fps,
        link_mapping={b: LinkMapping(b, np.zeros(3), np.zeros(4)) for b in scored},
    )  # fmt: skip
    config.task("target").position_weight = {b: w[0] for b, w in scored.items()}
    config.task("target").orientation_weight = {b: w[1] for b, w in scored.items()}
    config.task("rest").rest_q_arg, config.task("rest").weight = None, 0.05
    r = robot(c)
    solver = HumanoidRetargeterOnline(config, robot=r)
    solver.warmup(1)
    bones = [c.joint_names.index(b) for b in solver.human_joint_names]
    wxyz = Rotation.from_matrix(wanted[:, bones].reshape(-1, 3, 3)).as_quat()[:, [3, 0, 1, 2]]
    frames = np.concatenate((aim[:, bones], wxyz.reshape(len(wanted), len(bones), 4)), axis=2)
    solve = lambda t: solver.solve_numpy(frames[t : t + 1], human_heights=np.ones(1))  # noqa: E731
    for _ in range(SETTLE):
        solve(0)
    qpos = np.concatenate([solve(t) for t in range(len(frames))])

    solved = turns(c, r, qpos)
    joints = positions(c, solved, qpos[:, :3])
    floor = np.array([0.0, 0.0, skinned(c, solved, joints)[..., 2].min()])
    np.savez(out / "motion.npz", fps=fps, turns=solved, positions=joints - floor, joint_names=np.array(c.joint_names),
             studio_camera_direction=camera_side(motion), rest_joints=c.rest_joints, targets=aim - floor,
             planted=planted, qpos=qpos)  # fmt: skip


def check_motion(motion: Path, character: Path) -> dict:
    """`ok`, a `reason` and `limb_mm` (the feet's and hands' worst distance from their targets), `past_deg` (the worst
    joint's degrees past a person's range) and `step_deg` (the largest turn of a joint between frames, a pop)."""
    d, c = np.load(motion), load_character(character)
    if not np.isfinite(d["positions"]).all():
        return dict(ok=False, reason="the solved motion is not finite")
    limbs = [c.bone(f"{side}{limb}") for side in ("Left", "Right") for limb in ("Foot", "Hand")]
    limb = np.linalg.norm(d["positions"][:, limbs] - d["targets"][:, limbs], axis=2).max()
    measured = contact.joints(c, d["turns"]).values()
    report = dict(limb_mm=round(float(limb) * 1000, 1), past_deg=round(float(max(j["past"].max() for j in measured)), 1),
                  step_deg=round(float(max(j["step"].max() for j in measured)), 1))  # fmt: skip
    return dict(ok=bool(limb <= LIMB), reason="" if limb <= LIMB else f"a limb strays {limb:.3f} m from its target") | report  # fmt: skip
