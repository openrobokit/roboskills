"""A HaWoR hand's motion on a floating robot hand in a real2sim scene, by RoboKit's offline hand retargeting, and
that scene with the hand for SimMPC (tools/robokit/mpc.py).

    python -m roboskills.tools.robokit retarget --video V --hands hands/hands.npz --scene scene/ --out retarget/
"""

import json
from copy import copy
from dataclasses import replace
from importlib import import_module
from pathlib import Path

import numpy as np


BASE = ("base_x", "base_y", "base_z", "base_rx", "base_ry", "base_rz")  # slides, then hinges: R = Rx Ry Rz
TIPS = np.array([4, 8, 12, 16, 20])  # HaWoR's fingertips (OpenPose order), thumb to pinky


def held(points: np.ndarray, tracked: np.ndarray, mesh) -> tuple[int, int]:
    """The frames (grasp, release) between which the hand holds the object: from the fingertips' closest approach
    (`points`, OpenPose order in the object's frame; the mean of the three nearest to `mesh`) before the `tracked`
    object (world xyz wxyz, z up) is halfway along its motion (if within 5 cm; else from a quarter of the way), to its
    end if it ends lifted 2 cm or more, else to where it comes to rest; (0, 0) if it never moves 2 cm."""
    from scipy.spatial import cKDTree

    surface, moved = cKDTree(mesh.sample(4000, seed=0)), np.linalg.norm(tracked[:, :3] - tracked[:1, :3], axis=1)
    if moved.max() < 0.02:
        return 0, 0
    gap = np.sort(surface.query(points[:, TIPS].reshape(-1, 3))[0].reshape(-1, 5), 1)[:, :3].mean(1)
    half = int(np.argmax(moved >= moved.max() / 2))  # the last of the nearest, as HaWoR's unseen frames repeat
    grasp = half - int(np.argmin(gap[half::-1]))
    grasp = grasp if gap[grasp] < 0.05 else int(np.argmax(moved >= moved.max() / 4))  # never near: as it sets off
    moving = np.flatnonzero(np.linalg.norm(tracked[:, :3] - tracked[-1:, :3], axis=1) > 0.01)
    return grasp, len(tracked) if tracked[-1, 2] - tracked[0, 2] >= 0.02 or not len(moving) else int(moving[-1]) + 1


def rest(objects, moved: np.ndarray, frozen: np.ndarray):
    """Move the `objects` scene's pedestals, under the `moved` object's first or last pose (xyz wxyz), under the
    `frozen` one's, as mujoco.py places them: centred under the bottom centimetre of its collision parts, at its lowest."""
    import trimesh
    from scipy.spatial.transform import Rotation

    parts = [trimesh.load(m.file, force="mesh").vertices for m in objects.meshes if m.name.startswith("part_")]
    parts = np.concatenate(parts)

    def support(pose):
        x = Rotation.from_quat(pose[3:], scalar_first=True).apply(parts) + pose[:3]
        base, xy = x[x[:, 2] < x[:, 2].min() + 0.01, :2], x[:, :2]
        centre, middle = (base.min(0) + base.max(0)) / 2, (xy.min(0) + xy.max(0)) / 2
        if np.linalg.norm(middle - centre) > (radius := np.linalg.norm(base - centre, axis=1).max()):
            centre, radius = middle, np.linalg.norm(xy - middle, axis=1).max()
        return centre, radius, x[:, 2].min()

    ends = [support(moved[0])[0], support(moved[-1])[0]]
    for body in [b for b in objects.bodies if b.name.startswith("pedestal")]:
        centre, radius, top = support(frozen[-int(np.argmin([np.linalg.norm(body.pos[:2] - c) for c in ends]))])
        body.pos, body.geoms[0].size = [*centre, top - 0.0051], [radius, 0.005, 0]


def retarget_hand(video: Path, hands: Path, scene: Path, out: Path, robot: str = "sharpa", lead: int = 15,
                  gain: float = 0.0, armature: float = 0.0, mass: float = 0.0, timestep: float = 0.01,
                  oppose: float = 0.0, mount: float = 1000.0, friction: float = 1.5,
                  ground: bool = True, root: tuple = (0.03, 0.016, 0.005), anchor: float = 50.0, steady: float = 2.0,
                  calm: int = 2, plain: bool = False):  # fmt: skip
    """out/scene.xml and scene.mjb: scene/ with a right `robot` hand (RoboKit's preset and `<robot>_hand` MJCF) for the
    MANO one, floating on position-servoed `BASE` joints of its root link; trajectory.npz: `qpos` per source frame, the
    object as in scene/trajectory.npz and the hand retargeted from HaWoR's joints (OpenPose order, the preset's
    `target_names`; interpolated where unseen; scene/world.json brings them into the world), solved in the object's
    frame clear of its mesh, `packed` (RoboKit's base xyz, wxyz and joints) and `hold`: frames whose hand `held` holds
    still in the object's frame (from `lead` frames before its grasp, the approach sped up to match), moved for its
    three nearest fingertips to touch it and those within 3 cm on its surface, the object resting (its pedestals moved
    to match) outside; mujoco_ext1.mp4, preview.jpg. The base servos take Do as I Do's `mount` gain on all six joints
    (N m or N per unit, armature 1, 200 force limit; 0 for SPIDER's 5000 and 20); `ground`: the track rests until the
    reference frame (input/reference.json beside scene/, where the scene stands it on the ground) and never sinks into
    the ground (z = 0), so the floor is the table. Off by default (0 keeps RoboKit's): the fingers' servo `gain` (N
    m/rad, critically damped, without RoboKit's force limits) and joints' `armature`, Do as I Do's 300 and 1; the
    object at least `mass` kg (its density scaled); `oppose`: the thumb's held target moved that fraction of the way
    to the point across from the index and middle tips; also `timestep` and the hand's sliding `friction` on it. The
    root link keeps to HaWoR's own wrist in the world, held or not (on the hand's schedule), `root` m from it in MANO's
    wrist frame (Sharpa's C_MC: 3.4 cm, from no-hold solves of six clips), at `anchor` weight (RoboKit's 1 let the
    held fingertips drag it 7-28 cm off, as the object's track and HaWoR disagree). The wrist is steadied: `steady`, a
    Gaussian over that many frames on the track (the hand is solved in the object's frame, so the track's noise swings
    it), and `calm`, a median and a Gaussian on the solved root (RoboKit's solve zig-zags 5-15 mm a frame, more
    iterations or its root smoothness do not settle it), the fingers then solved again on that root; 0 for neither.
    `plain`: HaWoR's hand alone, solved in the world with no hold, contact or collision, the object as tracked."""
    import cv2
    import mujoco
    import trimesh
    import warp as wp
    from robokit.geom import MeshGeom, WarpScene
    from robokit.helpers.hand_retargeting import HandRetargeterOffline, refine_config
    from robokit.robo import Robot
    from robokit.robo.robot_state import RobotState
    from robokit.utils.warp_utils import wp_vec7
    from scipy.ndimage import gaussian_filter1d, median_filter
    from scipy.spatial import cKDTree
    from scipy.spatial.transform import Rotation

    from roboskills.tools.mujoco.engine import render

    preset = import_module(f"robokit.helpers.hand_retargeting.presets.{robot}")
    asset = import_module(f"robokit.assets.robots.end_effectors.{robot}_hand")
    saved, scene, out = np.load(hands), Path(scene), Path(out).resolve()
    joints = saved["joints"]
    seen, frames = np.flatnonzero(np.isfinite(joints).all((1, 2))), np.arange(len(joints))
    if saved["side"] != "right" or not len(seen):
        raise ValueError("retargeting needs a right hand that HaWoR saw at least once")
    rows = [np.interp(frames, seen, c) for c in joints[seen].reshape(len(seen), -1).T]
    joints = np.stack(rows, 1).reshape(joints.shape)
    tracked = np.load(scene / "trajectory.npz")["qpos"][:, -7:]  # the object's xyz wxyz per frame, held past its rows
    tracked = tracked[np.minimum(frames, len(tracked) - 1)]
    tracked[:, :3] = median_filter(tracked[:, :3], size=(5, 1), mode="nearest")  # without one- or two-frame outliers
    if steady:
        quat = tracked[:, 3:] * np.where(np.cumsum(np.r_[0, np.sum(tracked[1:, 3:] * tracked[:-1, 3:], 1) < 0]) % 2, -1, 1)[:, None]  # fmt: skip
        quat = gaussian_filter1d(quat, steady, axis=0, mode="nearest")
        tracked[:, 3:] = quat / np.linalg.norm(quat, axis=1, keepdims=True)
        tracked[:, :3] = gaussian_filter1d(tracked[:, :3], steady, axis=0, mode="nearest")
    objects = mujoco.MjSpec.from_file(str(scene / "scene.xml"))
    mesh = trimesh.load(objects.mesh("object").file, force="mesh")
    if ground:  # resting until the reference frame (where the scene stands it on the ground), and never into the
        known = scene.parent / "input/reference.json"  # ground (z = 0), as the track's occluded frames can have it
        still = min(json.loads(known.read_text())["frame"] if known.exists() else 0, len(tracked) - 1)
        tracked[:still] = tracked[still]
        low = np.einsum("fij,vj->fvi", Rotation.from_quat(tracked[:, 3:], scalar_first=True).as_matrix(), mesh.vertices)
        tracked[:, 2] = np.maximum(tracked[:, 2], tracked[:, 2] - (low[..., 2] + tracked[:, 2:3]).min(1))
    world = np.array(json.loads((scene / "world.json").read_text())["world_from_camera"])
    hand, turn = joints @ world[:3, :3].T + world[:3, 3], Rotation.from_quat(tracked[:, 3:], scalar_first=True)
    up = hand[:, 9] - hand[:, 0]  # wrist to middle MCP
    palm = np.cross(up, hand[:, 17] - hand[:, 5])  # a right palm's normal, from index to pinky MCP
    x, y = (-v / np.linalg.norm(v, axis=1, keepdims=True) for v in (up, palm))  # MANO's: fingers -x, palm faces -y
    facing = Rotation.from_matrix(np.stack([x, y, np.cross(x, y)], -1))  # in the world
    hand[:, 0] += facing.apply(root)  # where the robot's root link sits on HaWoR's hand
    path = tracked.copy()
    if plain:  # the world as the frame: the object never moves in it, so `held` finds no hold
        tracked, turn = np.tile([0.0, 0, 0, 1, 0, 0, 0], (len(frames), 1)), Rotation.identity(len(frames))
    points = (hand - tracked[:, None, :3]) @ turn.as_matrix()  # the object's frame
    grasp, release = held(points, tracked, mesh)  # the hand holds still in the object's frame between them
    early = grasp - min(grasp, lead) if release > grasp else grasp  # it gets there `lead` frames early, approaching
    warp = np.minimum(np.rint(np.arange(early) * grasp / max(early, 1)), grasp).astype(int)  # faster, for the
    points[:early], points[early:release] = points[warp], points[grasp]  # simulated fingers to close before it moves
    if release > grasp:  # Do as I Do's freeze: the object rests where the hand takes it (and leaves it)
        moved = tracked.copy()  # at rest at its median pose before the grasp, easing into the track over `lead` frames
        still = int(np.argmin(np.linalg.norm(moved[: grasp + 1, :3] - np.median(moved[: grasp + 1, :3], 0), axis=1)))
        tracked[:grasp], tracked[release:] = moved[still], moved[release - 1]
        ease = np.clip(1 - (frames[grasp:release] - grasp) / max(lead, 1), 0, 1)[:, None]
        tracked[grasp:release, :3] += (moved[still, :3] - moved[grasp, :3]) * ease
        turn = Rotation.from_quat(tracked[:, 3:], scalar_first=True)
        rest(objects, moved, tracked)
        gap, near = (surface := cKDTree(mesh.sample(20000, seed=0))).query(points[grasp, TIPS])
        nearest = np.argsort(gap)[:3]  # the hand's three nearest fingertips onto it, eased in over the approach
        shift = (surface.data[near[nearest]] - points[grasp, TIPS[nearest]]).mean(0)
        ease = np.clip(frames / max(early, 1), 0, 1)[:, None, None]
        points += shift * ease * ease * (3 - 2 * ease)  # smoothstep: it arrives slowly
        gap, near = surface.query(points[grasp, TIPS])  # then fingertips within 3 cm of it touch it while held
        points[early:release, TIPS[gap < 0.03]] = surface.data[near[gap < 0.03]]
        if oppose:  # the thumb across from the index and middle tips: where the inward normal there leaves the object
            grip = points[grasp, TIPS[1:3]].mean(0)
            _, _, face = trimesh.proximity.closest_point(mesh, grip[None])
            normal = mesh.face_normals[face[0]] * np.sign(mesh.face_normals[face[0]] @ (grip - mesh.centroid) or 1)
            ahead = (surface.data - grip) @ -normal
            line = np.linalg.norm(surface.data - grip + ahead[:, None] * normal, axis=1) < 0.01
            far = surface.data[np.flatnonzero(line)[np.argmax(ahead[line])]] if line.any() else grip
            if 0.01 < np.linalg.norm(far - grip) < 0.12:  # across it, within the hand's span
                thumb = points[grasp, TIPS[0]] + oppose * (far - points[grasp, TIPS[0]])
                points[early:release, TIPS[0]] = surface.data[surface.query(thumb)[1]]
    source = np.r_[warp, np.maximum(frames[early:], grasp)]  # the root: HaWoR's, on that schedule, into the object's
    points[:, 0] = np.einsum("fi,fij->fj", hand[source, 0] - tracked[:, :3], turn.as_matrix())  # (resting) frame
    wrist = (turn.inv() * facing[source]).as_quat(scalar_first=True)
    spheres = str(asset.COLLISION_SPHERE_PATH_RIGHT)
    kinematics = Robot.load(str(asset.URDF_PATH_RIGHT), load_collision_spheres=True, collision_spheres_path=spheres)
    identity = wp.array(np.eye(4, dtype=np.float32)[None], dtype=wp.mat44)
    shape = MeshGeom([mesh], np.array([0, 1], np.int32), poses=identity)
    collision = refine_config(preset.offline).task("collision")
    collision.weight = 100.0  # 5x RoboKit's, so fingertip targets inside the object do not pull the hand into it
    terms = [*map(copy, preset.offline.terms), *[collision] * (not plain)]
    config = replace(preset.offline, target_scale=1.0, terms=terms)
    config.task("root_position").weight = anchor  # Do as I Do's scale; copies, so the preset keeps its own weights
    mesh_scene = None if plain else WarpScene(1, wp.get_device()).add(shape)
    solver = HandRetargeterOffline(config, kinematics, preset.spec, scene=mesh_scene)
    local = solver.solve_numpy(points, wrist)  # as in RoboKit's Sharpa SimMPC example
    if calm:
        calmed = lambda x: gaussian_filter1d(median_filter(x, size=(2 * calm + 1, 1), mode="nearest"), calm, axis=0, mode="nearest")  # noqa: E731  # fmt: skip
        quat = local[:, 3:7] * np.where(np.cumsum(np.r_[0, np.sum(local[1:, 3:7] * local[:-1, 3:7], 1) < 0]) % 2, -1, 1)[:, None]  # fmt: skip
        local[:, :3], quat = calmed(local[:, :3]), calmed(quat)
        local[:, 3:7] = quat / np.linalg.norm(quat, axis=1, keepdims=True)
        fixed = HandRetargeterOffline(replace(config, optimize_base=False), kinematics, preset.spec, scene=mesh_scene)
        start = RobotState(kinematics, wp.array(local[None, :, 7:].copy(), dtype=wp.float32),
                           wp.array(local[None, :, :7].copy(), dtype=wp_vec7))  # fmt: skip
        local = fixed.solve_numpy(points, wrist, init_state=start)  # the fingers again, on that root
    base = turn * Rotation.from_quat(local[:, 3:7], scalar_first=True)
    packed = np.c_[tracked[:, :3] + turn.apply(local[:, :3]), base.as_quat(scalar_first=True), local[:, 7:]]
    if plain:
        tracked, turn = path, Rotation.from_quat(path[:, 3:], scalar_first=True)

    spec = mujoco.MjSpec.from_file(str(asset.MJCF_PATH_RIGHT))
    for part in spec.meshes:  # absolute, so the saved XML loads anywhere
        part.file = str(Path(asset.MJCF_PATH_RIGHT).parent / spec.meshdir / part.file)
    root = spec.body(preset.spec.target_link_names[preset.spec.root_target_name])
    for geom in spec.geoms:  # Do as I Do's hand-object contact (stiff, condim 6, friction 1.5), over the object's own
        if geom.contype:
            geom.priority, geom.condim, geom.friction, geom.solref = 1, 6, [friction, 0.15, 1e-4], [0.02, 1]
            geom.solimp = [0.998, 0.998, 0.001, 0.5, 2]
    for servo in spec.actuators if gain else ():  # Do as I Do's finger servos: stiff, heavy, unlimited
        servo.set_to_position(kp=gain, dampratio=1)
        servo.forcelimited = False
    for joint in spec.joints if gain or armature else ():
        joint.actfrclimited = mujoco.mjtLimited.mjLIMITED_FALSE if gain else joint.actfrclimited
        joint.armature = armature or joint.armature
    for i, name in enumerate(BASE):
        kind, span = (mujoco.mjtJoint.mjJNT_SLIDE, 5.0) if i < 3 else (mujoco.mjtJoint.mjJNT_HINGE, 2 * np.pi)
        root.add_joint(name=name, type=kind, axis=np.eye(3)[i % 3], range=[-span, span])
        servo = spec.add_actuator(name=name, target=name, trntype=mujoco.mjtTrn.mjTRN_JOINT, ctrlrange=[-span, span])
        servo.set_to_position(kp=mount or (5000 if i < 3 else 20), dampratio=1)  # SPIDER's, critically damped
        servo.forcelimited, servo.forcerange = True, [-200, 200] if i < 3 or mount else [-20, 20]  # no kilonewton rams
        if mount:
            spec.joint(name).armature = 1.0
    for servo in list(objects.actuators):  # the scene's MANO hand (mujoco/mano/right.xml) gives way to the robot's
        objects.delete(servo)
    objects.delete(objects.body("right_palm"))
    for geom in objects.body("object").geoms:  # the object alone lands on a table plane, under the ground (z = 0) and
        geom.conaffinity |= 2 * (geom.contype > 0)  # under its every pose, if it drops
    low = np.einsum("fij,vj->fvi", turn.as_matrix(), mesh.vertices[::10])[..., 2] + tracked[:, 2:3]
    plane = dict(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 1], contype=2, conaffinity=0, group=3)
    objects.worldbody.add_geom(name="floor", pos=[0, 0, min(0.0, low.min()) - 0.01], **plane)
    for part in objects.meshes:  # absolute too (the scene's meshdir is lost on attach), MANO's unused ones included
        part.file = str(Path(objects.meshdir) / part.file)
    objects.default.name = "scene"  # unprefixed, its unnamed main class would save as an invalid empty one
    spec.attach(objects, prefix="", frame=spec.worldbody.add_frame())
    spec.option.timestep, spec.option.iterations = timestep, 20  # RoboKit's Sharpa SimMPC example; after attach agrees
    model = spec.compile()
    if model.body("object").mass[0] < mass:  # a floor: SAM 3D's light shells (15-40 g) fly off a pushing finger
        for geom in spec.body("object").geoms:
            geom.density *= mass / model.body("object").mass[0]
        model = spec.compile()
    out.mkdir(parents=True, exist_ok=True)
    (out / "scene.xml").write_text(spec.to_xml())
    mujoco.mj_saveModel(model, str(out / "scene.mjb"))

    qpos, adr = np.repeat(model.qpos0[None], len(frames), 0), lambda name: model.jnt_qposadr[model.joint(name).id]
    euler, names = np.unwrap(base.as_euler("XYZ"), axis=0), (*BASE, *kinematics.spec.actuated_joint_names)
    qpos[:, [adr(n) for n in names]] = np.c_[packed[:, :3], euler, local[:, 7:]]
    first = model.jnt_qposadr[model.body("object").jntadr[0]]
    qpos[:, first : first + 7] = tracked
    np.savez(out / "trajectory.npz", qpos=qpos, packed=packed, hold=(frames >= early) & (frames < release))
    capture = cv2.VideoCapture(str(video))
    size = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    render(out, camera="source", fps=capture.get(cv2.CAP_PROP_FPS), size=size, source=video)
