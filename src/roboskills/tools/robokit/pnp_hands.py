"""Pick and place from a clip of human hands: two Panda arms with Robotiq 2F-85s (the MuJoCo tool's franka/,
Menagerie) seen from a camera that follows the source's head motion. The scene textured from the source frame (table
and wall planes, tools/mujoco/scene.py), its objects, and both arms taking each object a hand picks and places through
keyframes timed to the source (plan_pnp, pnp_path) by RoboKit IK, checked and retried in physics (pick_place, against
route's check); recon_ext1.mp4 (set per frame) and mujoco_ext1.mp4 (actuators, contact only).

    python -m roboskills.workflows.real2sim --workspace W --tools pnp
"""

import json
import time
from pathlib import Path
from typing import Literal

import numpy as np

from roboskills.tools.mujoco.mujoco import FRANKA
from roboskills.tools.mujoco.scene import (
    build_scene,
    hollow_parts,
    lid_parts,
    mat2quat,
    own_pixels,
    paint_backdrop,
    render_video,
    rollout,
    seat,
    separate,
    squeeze_parts,
)  # fmt: skip


GRIPPER = np.array([1, 0, 1, -0.96] * 2) * 0.8  # the 2F-85's linkage joints when closed (eval/droid/read.py)


READY = np.array([0, -0.3, 0, -2.2, 0, 2.0, 0.785])  # elbow up, reaching forward; joint 1 turns it to the hand


PARK = dict(
    left=[1.5, -0.6, 0, -2.6, 0, 2.0, 0.785], right=[-1.5, -0.6, 0, -2.6, 0, 2.0, 0.785]
)  # an idle arm, folded aside


SIDE = dict(left=0.35, right=-0.35)  # each arm's base on the table: world y (left); x at the camera's foot


def track_camera(w: Path, out: Path, frame: int = 0, scale: float = 0.5) -> np.ndarray:
    """out/camera.npz: `reference_from_camera` [N, 4, 4] (OpenCV axes) and PnP `inliers` [N], frame by frame out from
    the reference frame: low-contrast SIFT on the background (mask label 0, eroded), matched to the reference frame's
    and the neighbour's, each lifted to 3D by its own aligned depth and posed in the reference camera, through RANSAC
    PnP from the neighbour's pose; under 20 inliers, or a jump of over 5 degrees or 5 cm from it, keeps the
    neighbour's pose. Lightly smoothed."""
    import cv2
    from PIL import Image
    from scipy.ndimage import uniform_filter1d
    from scipy.spatial.transform import Rotation

    d, capture, images = dict(np.load(w / "align/depth.npz")), cv2.VideoCapture(str(w / "input/video.mp4")), []
    while (got := capture.read())[0]:
        images.append(
            cv2.cvtColor(cv2.resize(got[1], None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        )
    k, sift, matcher = d["K"] * [[scale], [scale], [1]], cv2.SIFT_create(6000, contrastThreshold=0.005), cv2.BFMatcher()
    poses, inliers, cache = np.repeat(np.eye(4)[None], len(images), 0), np.zeros(len(images), int), {}

    def features(i: int):  # keypoints (pixels), descriptors and 3D points in the reference camera, once posed
        if i not in cache:
            labels = np.array(Image.open(w / f"masks/{i:05d}.png").resize(images[i].shape[::-1], Image.NEAREST))
            found, described = sift.detectAndCompute(
                images[i], cv2.erode((labels == 0).astype(np.uint8), np.ones((15, 15), np.uint8))
            )
            uv = np.array([p.pt for p in found]).reshape(-1, 2)
            z = d["depth"][i].astype(float)[np.minimum(np.round(uv[:, 1] / scale).astype(int), d["depth"].shape[1] - 1),
                                            np.minimum(np.round(uv[:, 0] / scale).astype(int), d["depth"].shape[2] - 1)]  # fmt: skip
            xyz = np.c_[(uv - k[:2, 2]) / np.diag(k)[:2] * z[:, None], z] @ poses[i][:3, :3].T + poses[i][:3, 3]
            cache[i] = uv, described, np.where((z > 0)[:, None], xyz, np.nan)
        return cache[i]

    for rows in (range(frame + 1, len(images)), range(frame - 1, -1, -1)):
        r, t, before = np.zeros(3), np.zeros(3), frame
        for i in rows:
            uv, described, _ = features(i)
            three, two = [], []
            for j in dict.fromkeys([frame, before]):
                _, theirs, xyz = features(j)
                for m, n in matcher.knnMatch(described, theirs, k=2):
                    if m.distance < 0.8 * n.distance and np.isfinite(xyz[m.trainIdx]).all():
                        three.append(xyz[m.trainIdx]), two.append(uv[m.queryIdx])
            if len(three) >= 20:
                ok, r1, t1, used = cv2.solvePnPRansac(np.array(three), np.array(two), k, None, r.copy(), t.copy(), True,
                                                      300, 2.0, flags=cv2.SOLVEPNP_ITERATIVE)  # fmt: skip
                turn = lambda: np.degrees(
                    np.linalg.norm(cv2.Rodrigues(cv2.Rodrigues(r1)[0] @ cv2.Rodrigues(r)[0].T)[0])
                )
                if ok and used is not None and len(used) >= 20 and turn() < 5 and np.linalg.norm(t1.ravel() - t) < 0.05:
                    r, t, inliers[i] = r1.ravel(), t1.ravel(), len(used)
            camera = np.eye(4)  # camera i from the reference camera
            camera[:3, :3], camera[:3, 3] = cv2.Rodrigues(r)[0], t
            poses[i], before = np.linalg.inv(camera), i
            cache.pop(i)  # recomputed in the reference camera now that it is posed
    ref = frame
    quats = Rotation.from_matrix(poses[:, :3, :3]).as_quat()
    quats *= np.sign(quats @ quats[ref])[:, None]
    poses[:, :3, :3] = Rotation.from_quat(uniform_filter1d(quats, 5, axis=0, mode="nearest")).as_matrix()
    poses[:, :3, 3] = uniform_filter1d(poses[:, :3, 3], 5, axis=0, mode="nearest")
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "camera.npz", reference_from_camera=poses, inliers=inliers)
    return poses


def reference_frame(w: Path, out: Path, frame: int = 0) -> dict:
    """The reference frame: aligned `depth` (m), `valid`, `K`, mask `labels` and `rgb`; the reader's camera/reference.npz
    when present, else align/depth.npz."""
    import cv2
    from PIL import Image

    if (out / "reference.npz").exists():
        r = dict(np.load(out / "reference.npz"))
    else:
        d = np.load(w / "align/depth.npz")
        row = int(np.searchsorted(d["frames"], frame))
        r = dict(depth=d["depth"][row], valid=d["valid"][row], K=d["K"])
    capture = cv2.VideoCapture(str(w / "input/video.mp4"))
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
    r.update(
        depth=r["depth"].astype(float),
        rgb=capture.read()[1][..., ::-1],
        labels=np.array(Image.open(w / f"masks/{frame:05d}.png")),
    )
    return r


def place_world(w: Path, objects: list[dict], ref: dict) -> tuple[np.ndarray, dict]:
    """World from the reference camera: z up (GeoCalib), the origin on the table plane under the camera, x the optical
    axis along the table; and the table box (world centre, half sizes, colour, its plane's `inliers` as a pixel mask)
    and the background colour, both from the reference frame. The table plane is the background fitted just under the
    lowest object."""
    import trimesh

    up, z, k = -np.array(json.loads((w / "gravity/gravity.json").read_text())["down"]), ref["depth"], ref["K"]
    v, u = np.indices(z.shape)
    xyz = np.stack([(u - k[0, 2]) / k[0, 0] * z, (v - k[1, 2]) / k[1, 1] * z, z], -1)
    background = ref["valid"] & (ref["labels"] == 0) & (z > 0)
    bottom = min((trimesh.transform_points(o["mesh"].vertices, o["camera"][o["row"]]) @ up).min() for o in objects)
    cloud, colours = xyz[background], ref["rgb"][background]
    near = np.abs(cloud @ up - bottom + 0.05) < 0.1
    if near.sum() < 1000:  # no support under the objects: the most common height of the lower background
        lower = background[background.shape[0] // 2 :].sum() and (v[background] > z.shape[0] / 2)
        heights = cloud[lower] @ up
        counts, edges = np.histogram(heights, np.arange(heights.min(), heights.max() + 0.01, 0.01))
        near = np.abs(cloud @ up - edges[counts.argmax()] - 0.005) < 0.02
    for _ in range(3):  # least squares on the band, then on its 1 cm inliers
        centre, normal = trimesh.points.plane_fit(cloud[near])
        up = normal * np.sign(normal @ up)
        if (np.abs((cloud - centre) @ up) < 0.01).sum() < 100:
            break
        near = np.abs((cloud - centre) @ up) < 0.01
    origin, x = (centre @ up) * up, np.eye(3)[2] - up[2] * up
    world = np.eye(4)
    world[:3, :3] = np.stack([x / np.linalg.norm(x), np.cross(up, x / np.linalg.norm(x)), up])
    world[:3, 3] = -world[:3, :3] @ origin
    low, high = np.percentile(trimesh.transform_points(cloud[near], world)[:, :2], [1, 99], axis=0)
    low[0] = min(low[0], -0.3)  # under the arms' bases too
    plane = np.zeros(z.shape, bool)
    plane[background] = np.abs((cloud - centre) @ up) < 0.02
    table = dict(pos=[*(low + high) / 2, -0.025], size=[*(high - low) / 2, 0.025], rgba=[*colours[near].mean(0) / 255, 1],
                 sky=(colours[~near].mean(0) / 255).tolist(), plane=plane, xyz=xyz)  # fmt: skip
    return world, table


def read_objects(w: Path, world: np.ndarray, cameras: np.ndarray, frame: int = 0, moved: float = 0.06,
                 taper: bool = False) -> list[dict]:  # fmt: skip
    """Each object's mesh, convex parts and world pose per frame (its tracked camera pose through the moving camera),
    resting on the table at the reference frame and never under it; one that moves under `moved` holds that pose,
    one that moves more holds its first pose until it is `moving` (start, end, how far), then follows its tracked
    path without turning, and holds its last pose after (`track`: its poses as seen). A receptacle (`hollow`) is
    walled (hollow_parts, `taper`ed with 2 mm walls: a cup nests in another about as deep as it does)
    if it is a depth hull or deep (a cup, not a bowl); with `taper` (pnp), a soft one (SOFT) narrowed (squeeze_parts)."""
    import trimesh
    from scipy.ndimage import uniform_filter1d
    from scipy.spatial.transform import Rotation

    found = []
    for o in sorted((w / "objects").glob("*/track/poses.npz")) or [w / "track/poses.npz"]:
        o = o.parents[1]
        mesh, track = trimesh.load(o / "fit/mesh.ply"), np.load(o / "track/poses.npz")
        camera, turned = track["camera_from_object"], bool(track["turned"]) if "turned" in track else False
        if not np.isfinite(camera).all() or mesh.extents.max() < 0.01:  # nothing usable seen of it
            continue
        poses, row = world @ cameras @ camera, frame
        track = poses.copy()  # as seen (pnp)
        lowest = np.array([trimesh.transform_points(mesh.vertices, p)[:, 2].min() for p in poses])
        poses[:, 2, 3] += np.maximum(-lowest[row], -lowest)  # resting at the reference frame, never under the table
        p = poses[:, :3, 3]
        far, moving = np.linalg.norm(p[:, :2] - p[row, :2], axis=1).max(), None  # across the table: things stacked on
        # an object raise its mask's centre, not move it
        if far < moved:
            poses[:] = poses[row]
        else:  # still until 40% of the way out (tracking drifts below that), and once set down
            quats = Rotation.from_matrix(poses[:, :3, :3]).as_quat()
            quats *= np.sign(quats @ quats[row])[:, None]
            poses[:, :3, :3] = Rotation.from_quat(uniform_filter1d(quats, 5, axis=0, mode="nearest")).as_matrix()
            poses[:, :3, 3] = uniform_filter1d(p, 5, axis=0, mode="nearest")
            p = poses[:, :3, 3]
            start = max(int((np.linalg.norm(p - p[0], axis=1) > 0.4 * far).argmax()) - 3, 0)
            peak = start + int(p[start:, 2].argmax())
            calm = [
                e
                for e in range(max(peak, start + 5), len(p))
                if np.linalg.norm(p[e : e + 10] - p[e], axis=1).max() < max(0.03, 0.1 * far)
            ]
            end = calm[0] + 1 if calm else len(p)  # set down: past its highest, then still for 10 frames
            if turned:  # its six-degree track, from its rest pose
                poses[start:end] = poses[0] @ np.linalg.inv(poses[start]) @ poses[start:end]
            else:  # its path from its rest pose, upright as it rests: an untrusted turn (a bowl, a cup) is dropped
                path = poses[start:end, :3, 3] - poses[start, :3, 3]
                poses[start:end, :3, :3], poses[start:end, :3, 3] = poses[0, :3, :3], poses[0, :3, 3] + path
            poses[:start], poses[end:] = poses[0], poses[end - 1]
            moving = (start, end, float(far))
        parts = [o / "collision" / Path(p).name for p in json.loads((o / "collision/parts.json").read_text())["parts"]]
        noun = (o / "name.txt").read_text() if (o / "name.txt").exists() else ""
        hollow = any(word in noun for word in RECEPTACLES)
        extent = np.ptp(trimesh.transform_points(mesh.vertices, poses[row]), axis=0)
        walled = hollow and (not (o / "shape").exists() or extent[2] > 0.8 * extent[:2].min())  # a depth hull is
        if walled:  # solid; SAM 3D sees little inside a cup
            parts = hollow_parts(mesh, poses[row], w / f"cache/hollow{'_taper' * taper}" / o.name, wall=0.002 if taper else 0.006, taper=taper)  # fmt: skip
        elif taper and any(word in noun for word in SOFT):  # pnp: a soft toy gives in the jaws, its collision a
            parts = squeeze_parts(parts, poses[row], w / "cache/soft" / o.name)  # quarter narrower than it looks
        colour = mesh.visual.vertex_colors[:, :3].mean(0) / 255 if hasattr(mesh.visual, "vertex_colors") else [0.7] * 3
        found.append(dict(name=o.name if o != w else "object", mesh=mesh, camera=camera, poses=poses, track=track, parts=parts, row=row, moving=moving,
                          visual=o / "collision/visual.obj", rgba=[*colour, 1], hollow=hollow, walled=walled, noun=noun))  # fmt: skip
    return found


RECEPTACLES = ("bowl", "cup", "mug", "basket", "box", "container", "tupperware", "drawer", "bin", "pot", "jar")


SOFT = ("plush", "stuffed", "toy", "cloth", "towel", "sponge", "pillow", "sock", "rag", "bread", "bun")


def read_hands(w: Path, world: np.ndarray, cameras: np.ndarray) -> dict[str, np.ndarray]:
    """{side: HaWoR's 21 joints per frame in the world}, unseen frames interpolated, ends held; sides never seen left out."""
    hands = {}
    for path in sorted(w.glob("hands*/hands.npz")):
        saved = np.load(path)
        joints = saved["joints"]
        seen, frames = np.flatnonzero(np.isfinite(joints).all((1, 2))), np.arange(len(joints))
        if len(seen) < 2:
            continue
        joints = np.stack([np.interp(frames, seen, c) for c in joints[seen].reshape(len(seen), -1).T], 1).reshape(
            joints.shape
        )
        homogeneous = np.concatenate([joints, np.ones((*joints.shape[:2], 1))], -1)
        hands[str(saved["side"])] = np.einsum("fij,fkj->fki", world @ cameras, homogeneous)[..., :3]
    return hands


def grip_targets(joints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The gripper's pinch pose per frame (4x4) from a hand's joints (OpenPose order): between the thumb and index
    tips, approaching along the wrist to middle knuckle, closing along thumb to index; and its closure (0 open, 1
    closed), the 2F-85's 85 mm stroke matching the thumb-index gap. The approach is tilted halfway to straight down,
    as a parallel gripper would take it, and the pinch kept 3 cm above the table and 0.3 m out from the camera."""
    pinch = (joints[:, 4] + joints[:, 8]) / 2
    approach = joints[:, 9] - joints[:, 0]
    approach = approach / np.linalg.norm(approach, axis=1, keepdims=True) - [0, 0, 1]  # tilted down, clear of the table
    approach /= np.linalg.norm(approach, axis=1, keepdims=True)
    closing = joints[:, 8] - joints[:, 4]
    gap = np.linalg.norm(closing, axis=1)
    closing -= (closing * approach).sum(1, keepdims=True) * approach
    closing /= np.linalg.norm(closing, axis=1, keepdims=True)
    for i in range(1, len(closing)):  # the jaws are symmetric: keep their axis from flipping between frames
        closing[i] *= np.sign(closing[i] @ closing[i - 1]) or 1
    poses = np.repeat(np.eye(4)[None], len(joints), 0)
    poses[:, :3, :3] = np.stack([np.cross(closing, approach), closing, approach], -1)
    poses[:, :3, 3] = pinch
    poses[:, 0, 3] = np.maximum(poses[:, 0, 3], 0.3)  # a hand by the body: the arm would fill the camera's view
    poses[:, 2, 3] = np.maximum(poses[:, 2, 3], 0.03)
    return poses, np.clip(1 - gap / 0.085, 0, 1)


def solve_arm(pinch: np.ndarray, base: np.ndarray, start: np.ndarray | None = None, preset: str = "smooth",
              smooth: float = 1.0, avoid: tuple[np.ndarray, float] | None = None) -> np.ndarray:  # fmt: skip
    """The Panda's seven joints per frame putting the 2F-85's pinch site at `pinch` (world, 4x4), the arm's base at
    `base`: RoboKit's IK (`preset`, its smoothness weight `smooth`) on the arm alone (link7), seeded, leaning to READY
    turned toward the targets, frame by frame from the previous solution (the first from `start`, default that
    posture). The jaws are symmetric, so it solves both jaw orders (the pinch turned half about its axis) as two
    chains and joins them where that moves the arm least (dynamic programming): no wrist flips at joint limits.
    `avoid` (world centre, radius) is a sphere the links keep out of: the camera's near field."""
    import tempfile
    from dataclasses import replace

    import mujoco
    import warp as wp
    from robokit.geom import SphereGeom, WarpScene
    from robokit.helpers.ik import IKContext, IKSolver, presets
    from robokit.robo import Robot
    from robokit.terms.dense.configuration_task import ConfigurationTask
    from robokit.terms.dense.scene_collision_task import SceneCollisionTask
    from robokit.terms.dense.smoothness_task import SmoothnessTask
    from scipy.spatial.transform import Rotation

    from roboskills.tools.robokit.robot_model import write_collision_spheres

    spec = mujoco.MjSpec.from_file(str(FRANKA))
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    link7, site = np.eye(4), np.eye(4)
    link7[:3, :3], link7[:3, 3] = data.body("link7").xmat.reshape(3, 3), data.body("link7").xpos
    site[:3, :3], site[:3, 3] = data.site("pinch").xmat.reshape(3, 3), data.site("pinch").xpos
    for item in [*spec.actuators, *spec.tendons, *spec.equalities, *spec.excludes]:  # the arm alone for RoboKit
        spec.delete(item)
    spec.delete(spec.body("base_mount"))
    with tempfile.TemporaryDirectory() as tmp:
        for m in spec.meshes:
            m.file = str(FRANKA.parent / m.file)
        Path(f"{tmp}/arm.xml").write_text(spec.to_xml())
        robot = Robot.load(f"{tmp}/arm.xml", load_meshes=avoid is not None)
        if avoid is not None:  # its links as spheres, for the scene term
            spheres = write_collision_spheres(robot, f"{tmp}/spheres.yml")
            robot = Robot.load(f"{tmp}/arm.xml", load_collision_spheres=True, collision_spheres_path=str(spheres))
    chosen = getattr(presets, preset)  # RoboKit's IK presets: basic, smooth (default), single_candidate
    terms = [t for t in chosen.terms if not isinstance(t, SmoothnessTask)]
    terms.append(SmoothnessTask(prev_var=IKContext.PREVIOUS_STATE, weight=smooth))
    terms.append(ConfigurationTask(rest_q=IKContext.REST_Q, weight=0.05))  # leaning to READY: elbow up, aside
    if avoid is not None:
        where = np.eye(4, dtype=np.float32)
        where[:3, 3] = np.linalg.inv(base)[:3, :3] @ avoid[0] + np.linalg.inv(base)[:3, 3]
        ball = SphereGeom(
            np.array([avoid[1]], np.float32), np.array([0, 1], np.int32), poses=wp.array(where[None], dtype=wp.mat44)
        )
        terms.append(
            SceneCollisionTask(robot, scene=WarpScene(1, wp.get_device("cuda")).add(ball), weight=100.0, margin=0.02)
        )
    config = replace(chosen, terms=terms, sampler=replace(chosen.sampler, seed=0))  # the same answer every run
    ik, device = IKSolver(config, robot=robot, target_links="link7", device="cuda"), wp.get_device("cuda")
    state = lambda q: robot.state(q=wp.array(np.float32(q)[None], dtype=wp.float32, device=device))
    chains = []
    for turn in (np.eye(4), np.diag([-1.0, -1.0, 1.0, 1.0])):  # either jaw order
        goal = np.linalg.inv(base) @ pinch @ turn @ np.linalg.inv(np.linalg.inv(link7) @ site)
        goal = np.c_[goal[:, :3, 3], Rotation.from_matrix(goal[:, :3, :3]).as_quat(scalar_first=True)].astype(
            np.float32
        )
        rest = np.r_[np.arctan2(*goal[:, 1::-1].mean(0)), READY[1:]]
        last, rest, q = state(rest if start is None else start), state(rest), []
        for target in goal:
            last = ik.solve_numpy(target[None, None], prev_state=last, init_state=last, rest_state=rest)
            q.append(last.q.numpy()[0])
        chains.append(np.array(q))
    chains = np.array(chains)  # [2, frames, 7]
    cost, back = np.zeros(2), []
    for t in range(1, chains.shape[1]):  # the least joint travel (the worst joint each frame), switching allowed
        step = np.abs(chains[None, :, t] - chains[:, None, t - 1]).max(-1) + cost[:, None]  # [from, to]
        back.append(step.argmin(0))
        cost = step.min(0)
    path = [int(cost.argmin())]
    for b in reversed(back):
        path.append(int(b[path[-1]]))
    return chains[path[::-1], np.arange(chains.shape[1])]


def place_base(pinch: np.ndarray | None, y: float, reach: float = 0.8, room: float = 0.0) -> np.ndarray:
    """The arm's base (world, on the table plane at `y`) at the camera's foot, as Real2Gym's, or as little forward
    (to 0.2 m) as keeps every pinch target within `reach` of its shoulder (0.333 m up); with `room`, as little back
    (to 0.2 m) as keeps them all `room` from it, where none is out of reach: a top-down arm folded over targets at its
    foot rises into the camera."""
    base = np.eye(4)
    base[:2, 3] = 0.0, y
    for x in np.arange(0.0, 0.21, 0.05) if pinch is not None else ():
        base[0, 3] = x
        if np.linalg.norm(pinch[:, :3, 3] - [x, y, 0.333], axis=1).max() <= reach:
            break
    for x in -np.arange(0.0, 0.21, 0.05) if pinch is not None and room and base[0, 3] == 0 else ():
        base[0, 3] = x
        far = np.linalg.norm(pinch[:, :3, 3] - [x, y, 0.333], axis=1)
        if far.min() >= room or far.max() > reach:
            base[0, 3] += 0.05 * (far.max() > reach)
            break
    return base


def grasp_pose(o: dict, t: int, near: np.ndarray, turn: float = 0.0, deeper: float = 0.0, rim: bool | None = None,
               others: np.ndarray | None = None, lean: bool = False, lip: np.ndarray | None = None) -> tuple[np.ndarray, float]:  # fmt: skip
    """A top-down pinch on object `o` at frame `t` (in its frame) and the gap it closes to: across its narrow
    horizontal axis at its centre when that is under 7.5 cm (a cup too wide at its rim: at its narrowest height, if
    the open jaws pass all above it), else (or with `rim`) straddling its rim, one jaw inside the wall, at the top
    point nearest the pinch pose `near` whose outer jaw clears `others` (world points of the other objects).
    Variants: the jaws' axis turned by `turn` (degrees), the pinch `deeper` (m); with `lean`, a rim pinch leans in
    as the wall does (a walled one's own tapered wall plate there, hollow_parts', else its flare from a third of its
    height up to its rim), the pads flat on it: a flared cup held so stays upright. With `lip` (a horizontal
    direction), a side pinch on its edge instead (a lid's lip, lid_parts' prism where it has a `floor`)."""
    import trimesh
    from scipy.spatial import cKDTree

    shape = trimesh.load(o["parts"][0], force="mesh") if "floor" in o else o["mesh"]  # a lid's: what the jaws meet
    v = trimesh.transform_points(trimesh.sample.sample_surface(shape, 4000, seed=0)[0], o["poses"][t])  # surface
    low, high = np.percentile(v, [2, 98], axis=0)
    centre, top, bottom = (low + high) / 2, high[2], low[2]
    if lip is not None:  # its edge from the side (a lid's lip), where it overhangs its support most within 107
        v = v[v[:, 2] >= o.get("floor", -np.inf)]  # degrees of `lip` (from its arm): horizontal toward its centre,
        under = np.zeros((0, 3)) if others is None else others[others[:, 2] < o.get("floor", bottom) + 0.01]  # the
        under = under[np.linalg.norm(under[:, :2] - centre[:2], axis=1) < np.ptp(v[:, :2], axis=0).max() / 2]  # jaws
        reach = lambda d: np.percentile((v[:, :2] - centre[:2]) @ -d, 98)  # closing up and down across it, their tips
        hang = lambda d: reach(d) - (((under[:, :2] - centre[:2]) @ -d).max() if len(under) else 0.0)  # (closed) up
        ways = [np.array([np.cos(a), np.sin(a)]) for a in np.linspace(0, 2 * np.pi, 24, endpoint=False)]  # to 13 mm
        d = max((w for w in ways if w @ lip[:2] > -0.3 * np.linalg.norm(lip[:2])), key=hang)  # in, clear of its support
        tip = min(0.013, hang(d) - 0.004) + deeper
        lo, hi = (o["floor"], v[:, 2].max()) if "floor" in o else np.percentile(v[:, 2], [5, 95])  # (lid_parts' prism)
        pose = np.eye(4)
        pose[:3, :3] = np.stack([np.cross([0, 0, 1.0], [*d, 0]), [0, 0, 1.0], [*d, 0]], -1)
        pose[:3, 3] = [*(centre[:2] - (reach(d) + 0.0175 - tip) * d), (lo + hi) / 2]
        return np.linalg.inv(o["poses"][t]) @ pose, hi - lo
    axes = np.linalg.eigh(np.cov((v[:, :2] - centre[:2]).T))[1]  # minor axis first
    width = lambda band: np.subtract(*np.percentile((band[:, :2] - centre[:2]) @ axes, [98, 2], axis=0))
    widths = width(v)
    heights = [
        h for h in np.linspace(bottom + 0.3 * (top - bottom), top - 0.015, 5) if (np.abs(v[:, 2] - h) < 0.01).sum() > 20
    ]
    narrow = min(heights, key=lambda h: width(v[np.abs(v[:, 2] - h) < 0.01])[0], default=None)  # a cup's waist
    if (
        widths[0] > 0.075
        and narrow is not None
        and width(v[np.abs(v[:, 2] - narrow) < 0.01])[0] <= 0.075
        and width(v[v[:, 2] > narrow])[0] <= 0.08  # the open jaws pass the rim
        and rim is None
    ):
        widths, top = width(v[np.abs(v[:, 2] - narrow) < 0.01]), narrow + 0.02  # wrapped there, across its body
    edges = v[v[:, 2] > top - 0.02]
    if (widths[0] <= 0.075 if rim is None else not rim) or not len(edges):
        closing, point, gap = np.r_[axes[:, 0], 0], np.r_[centre[:2], max(top - 0.02, bottom + 0.01)], widths[0]
    else:
        out = (edges[:, :2] - centre[:2]) / np.linalg.norm(edges[:, :2] - centre[:2], axis=1, keepdims=True)
        crowd = [] if others is None else others[others[:, 2] > top - 0.06, :2]  # what the outer jaw could hit
        blocked = cKDTree(crowd).query(edges[:, :2] + 0.03 * out)[0] < 0.03 if len(crowd) else np.zeros(len(edges))
        edge = edges[np.argmin(np.linalg.norm(edges[:, :2] - near[:2, 3], axis=1) + blocked)]
        closing = np.r_[edge[:2] - centre[:2], 0] / np.linalg.norm(edge[:2] - centre[:2])
        if lean:  # out to its outline there (where hollow_parts' tapered wall is), then half a wall in
            ahead = (edges[:, :2] - centre[:2]) @ closing[:2]
            beside = np.abs((edges[:, :2] - centre[:2]) @ [-closing[1], closing[0]]) < 0.01
            edge = np.r_[centre[:2] + closing[:2] * (ahead[beside].max() if beside.any() else ahead.max()), 0]
        point, gap = np.r_[edge[:2] - (0.001 if lean else 0.004) * closing[:2], top - 0.02], 0.01  # the wall's middle
    angle, radial = np.radians(turn), closing.copy()
    closing = np.r_[
        np.cos(angle) * closing[0] - np.sin(angle) * closing[1],
        np.sin(angle) * closing[0] + np.cos(angle) * closing[1],
        0,
    ]
    point = np.r_[point[:2], point[2] - deeper]
    approach = np.array([0.0, 0, -1])
    if lean and gap == 0.01 and o.get("walled") and len(o["parts"]) > 2:  # its own wall there (hollow_parts' plate
        plates = [trimesh.transform_points(trimesh.load(p, force="mesh").vertices, o["poses"][t]) for p in o["parts"][1:]]  # fmt: skip
        plate = min(plates, key=lambda q: np.linalg.norm(q[:, :2].mean(0) - edge[:2]))  # nearest that edge): the pads
        middle, axes = plate.mean(0), np.linalg.eigh(np.cov((plate - plate.mean(0)).T))[1]  # flat on its faces, down
        slope = axes[:, 2] * np.sign(axes[2, 2])  # its slope, a wall's thickness between the jaws: no pivot
        approach, normal = -slope, axes[:, 0] - (axes[:, 0] @ slope) * slope
        point = middle + slope * (top - 0.02 - deeper - middle[2]) / slope[2]
        normal /= np.linalg.norm(normal)
        closing = np.cos(angle) * normal + np.sin(angle) * np.cross(approach, normal)
    elif lean and gap == 0.01:  # down the wall: inward by its flare
        radius = lambda band: np.linalg.norm(band[:, :2] - centre[:2], axis=1).mean()
        third = v[np.abs(v[:, 2] - bottom - (top - bottom) / 3) < 0.01]
        flare = (radius(edges) - radius(third)) / max(top - 0.01 - bottom - (top - bottom) / 3, 0.01) if len(third) else 0.0  # fmt: skip
        approach = approach - np.clip(flare, 0, 1) * radial
        approach /= np.linalg.norm(approach)
        closing = closing - (closing @ approach) * approach
        closing /= np.linalg.norm(closing)
    pose = np.eye(4)
    pose[:3, :3] = np.stack([np.cross(closing, approach), closing, approach], -1)
    pose[:3, 3] = point[0], point[1], max(point[2], 0.012)
    return np.linalg.inv(o["poses"][t]) @ pose, gap


def plan_pnp(w: Path, objects: list[dict], hands: dict, frames: int, moved: float = 0.05, near: float = 0.08,
             whole: float = 0.7) -> list[dict]:  # fmt: skip
    """The source's pick-and-places, for `pnp` mode: [{object, side, grasp, arrive, release, far, start, end, into}]
    in grasp order. Each object's tracked position (`track`, lightly smoothed) where it is seen whole (its mask, W's
    per-frame masks/, at least `whole` of its usual area and no hand touching it: the first five such frames before a
    hand first does, the last five after one last does, else its first and last three) gives where it starts and
    ends; what the objects that end within `moved` of their start (still) appear to move by is the camera's drift,
    taken off every track and hand. One that ends farther is picked and placed by the hand whose finger tips come
    within `near` of it as it leaves and as it arrives (the nearest such): grasped 5 frames before it is half way
    out, it arrives once within 30% of its end (five frames on end each) or, lifted, back within 30% of its lift;
    released where those tips leave it (4 cm off their nearest, three frames on end), else held to the end (dropped
    if it moved under twice `moved`: a hand hides it, its track drifts). It starts upright as it rests (its pose at
    the reference frame, turned; set down if its base is within 3 cm of the table) and ends where it is last seen,
    once released on the table, or `into` the object its centre ends over (centred on it if a receptacle, where that
    one is by then, its base between that one's base and its rim); one seen resting on another (its base over 3 cm
    up, or over 1 cm up and sunk into one under a quarter of its footprint or more: a leg in its socket) starts as
    seen then, its hand within twice `near`; one lifted where it stands (it moves under `moved` across the table:
    steadied, put back) is no step; one out of its hand's arm's reach (0.7 m from its shoulder) goes to the other.
    One object at a time per hand: released 24 frames before the next grasp and no sooner than 10 after its own or
    before its arrival (the next grasp pushed later), else the nearer moving dropped; a grasp under 30 frames after
    the same hand's last goes to the other arm instead where that one is idle then and reaches it (0.7 m from its
    shoulder), grasping no sooner than 6 frames after the first one's release (`hand`: the source's)."""
    import cv2
    import trimesh
    from PIL import Image
    from scipy.ndimage import uniform_filter1d
    from scipy.spatial import cKDTree

    found, paths = json.loads((w / "masks/objects.json").read_text()), sorted((w / "masks").glob("*.png"))[:frames]
    labels = np.stack([np.array(Image.open(p))[::2, ::2] for p in paths]) if len(paths) == frames else None
    if labels is not None:  # a hand's pixels, dilated: what it touches is not seen whole
        hand = np.isin(labels, [int(k) for k, v in found.items() if v["prompt"].split("|")[0] == "hand"])
        touched = np.stack([cv2.dilate(h.astype(np.uint8), np.ones((9, 9), np.uint8)) for h in hand]).astype(bool)
    tracks, ends, handled = [], [], []
    for o in objects:
        p = uniform_filter1d(o["track"][:frames, :3, 3], 5, axis=0, mode="nearest")
        seen, near_hand = np.ones(frames, bool), np.zeros(frames, bool)
        if labels is not None and o["name"].isdigit():
            mask = labels == int(o["name"])
            area, near_hand = mask.sum((1, 2)), (mask & touched).any((1, 2))
            seen = (area >= whole * max(np.percentile(area, 90), 1)) & ~near_hand
        a, b = int(near_hand.argmax()) if near_hand.any() else frames, frames - int(near_hand[::-1].argmax()) if near_hand.any() else 0  # fmt: skip
        first, last = np.flatnonzero(seen[:a])[:5], b + np.flatnonzero(seen[b:])[-5:]
        tracks.append(p)
        ends.append((first if len(first) else np.arange(3), last if len(last) else np.arange(frames - 3, frames)))
        start = np.median(p[ends[-1][0]], 0)  # a hand takes it more than `moved` away mid-clip (and puts it back)
        handled.append(bool(near_hand.any() and np.linalg.norm(p[near_hand] - start, axis=1).max() > moved))
    seen = [trimesh.transform_points(o["mesh"].vertices, o["track"][o["row"]]) for o in objects]  # at the reference
    rests_on = [[j for j, u in enumerate(seen) if j != i and v[:, 2].min() > 0.03 and abs(u[:, 2].max() - v[:, 2].min()) < 0.03  # frame, what
                 and (u[:, :2].min(0) <= v[:, :2].mean(0)).all() and (v[:, :2].mean(0) <= u[:, :2].max(0)).all()] for i, v in enumerate(seen)]  # each sits on  # fmt: skip
    still = [i for i, (a, b) in enumerate(ends) if np.linalg.norm(np.median(tracks[i][b], 0) - np.median(tracks[i][a], 0)) < moved]  # fmt: skip
    # one a hand moves with another resting on it (a bottle its cap is taken off) shows its own motion, not the camera's
    steady = [i for i in still if not (handled[i] and any(i in r for r in rests_on))]
    drift = np.median([tracks[i] - np.median(tracks[i][ends[i][0]], 0) for i in steady], 0) if steady else 0
    steps = []
    for i, o in enumerate(objects):
        p = tracks[i] - drift
        start, end = np.median(p[ends[i][0]], 0), np.median(p[ends[i][1]], 0)
        far = float(np.linalg.norm(end - start))
        if far < moved or not hands:
            continue
        gone = np.convolve(np.linalg.norm(p - start, axis=1) > 0.5 * far, np.ones(5), "valid") == 5
        leave = int(gone.argmax()) if gone.any() else 0
        there = np.convolve(np.linalg.norm(p[leave:] - end, axis=1) < 0.3 * far, np.ones(5), "valid") == 5
        arrive = leave + int(there.argmax()) if there.any() else frames - 1
        peak, top = leave + int(p[leave : leave + 60, 2].argmax()), max(start[2], end[2])
        if p[peak, 2] > top + 0.03:  # lifted: set down once back within 30% of its lift (three frames on end)
            down = np.convolve(p[peak:, 2] < top + 0.3 * (p[peak, 2] - top), np.ones(3), "valid") == 3
            arrive = min(arrive, peak + int(down.argmax()) if down.any() else frames - 1)
        rest = o["poses"][0].copy()  # upright as it rests
        seen = o["track"][o["row"]]  # (as seen at the reference frame if it rests on another there, over 3 cm up,
        v = trimesh.transform_points(o["mesh"].vertices, seen)  # or over 1 cm up and sunk into one under a quarter
        low, high = v[:, :2].min(0), v[:, :2].max(0)  # of its footprint or more: a leg in its socket)
        sunk = lambda u: (np.clip(np.minimum(high, np.percentile(u[:, :2], 98, 0)) - np.maximum(low, np.percentile(u[:, :2], 2, 0)), 0, None).prod() >= 0.25 * np.prod(high - low)) and np.percentile(u[:, 2], 98) > v[:, 2].min()  # noqa: E731  # fmt: skip
        stacked = v[:, 2].min() > 0.03 or (v[:, 2].min() > 0.01 and any(sunk(trimesh.transform_points(x["mesh"].vertices, x["track"][x["row"]])) for x in objects if x is not o))  # fmt: skip
        rest[:3, 3] = seen[:3, 3] if stacked else start
        bottom = trimesh.transform_points(o["mesh"].vertices, rest)[:, 2].min()
        rest[2, 3] -= bottom if bottom < 0.03 and not stacked else 0
        goal = rest.copy()
        goal[:3, 3] += end - start
        start = rest[:3, 3]
        surface, best, side = trimesh.sample.sample_surface(o["mesh"], 300, seed=0)[0], near * (1 + stacked), None
        for s, j in hands.items():  # its tips, drift taken off, nearest it as it leaves and as it arrives
            tips = j[:frames, [4, 8, 12]] - (drift[:, None] if np.ndim(drift) else 0)
            to = lambda pose, t: cKDTree(trimesh.transform_points(surface, pose)).query(tips[t].reshape(-1, 3))[0].min()
            score = max(to(rest, range(max(leave - 10, 0), leave + 4)), to(goal, range(max(arrive - 3, 0), min(arrive + 10, frames))))  # fmt: skip
            if score < best:
                d = np.array([to(goal, [t]) for t in range(arrive, frames)])
                off = np.convolve(d > d[:5].min() + 0.04, np.ones(3), "valid") == 3  # its tips leave it: released
                best, side, release = score, s, arrive + int(off.argmax()) if off.any() else None
        if side is None:
            continue
        if release is None and far < 2 * moved:  # held to the end, hardly moved: the hand hides it, its track drifts
            continue
        if np.linalg.norm((goal - rest)[:2, 3]) < moved:  # lifted where it stands (steadied, put back): nothing moved
            continue
        steps.append(dict(object=i, side=side, grasp=max(leave - 5, 0), arrive=arrive, release=release, far=far,
                          start=rest, end=goal, on=rests_on[i][0] if rests_on[i] else None))  # fmt: skip
    kept, reach = [], lambda side, pose: np.linalg.norm(pose[:3, 3] - [0, SIDE[side], 0.333]) < 0.7
    for s in sorted(steps, key=lambda s: s["grasp"]):  # one object at a time per hand: released 24 frames before
        other = next(side for side in SIDE if side != s["side"])  # the next grasp, no sooner than 10
        if not (reach(s["side"], s["start"]) and reach(s["side"], s["end"])) and reach(other, s["start"]) and reach(other, s["end"]):  # fmt: skip
            s.update(side=other, hand=s["side"])  # (out of its arm's reach: the other's)
            other = next(side for side in SIDE if side != s["side"])
        last = next((k for k in kept[::-1] if k["side"] == s["side"]), None)  # (the other arm takes it where it is
        busy = any(k["side"] == other and (k["release"] or frames) + 10 > s["grasp"] - 30 for k in kept)  # idle and
        if last and s["grasp"] - (last["release"] or last["arrive"]) < 30 and not busy and reach(other, s["start"]) and reach(other, s["end"]):  # fmt: skip
            s.update(side=other, hand=s["side"], grasp=max(s["grasp"], (last["release"] or last["arrive"]) + 6))  # fmt: skip
            last = None  # reaches it, rather than wait for this one's; grasping as this one lets go)
        if last:  # frames after its own grasp or before its arrival (the next pushed later), else the nearer dropped
            floor = max(last["grasp"] + 10, last["arrive"] - 10)
            release = max(min(last["release"] or last["arrive"] + 3, s["grasp"] - 24), floor)
            if max(s["grasp"], release + 24) + 10 > (s["release"] or s["arrive"]):
                if s["far"] < last["far"]:
                    continue
                kept.remove(last)
            else:
                last["release"], s["grasp"] = release, max(s["grasp"], release + 24)
        kept.append(s)
    placed = {}
    for s in kept:  # into what its centre ends over, centred on it where that one is by then; else on the table
        v, s["into"], nearest = trimesh.transform_points(objects[s["object"]]["mesh"].vertices, s["end"]), None, np.inf
        centre = (v[:, :2].min(0) + v[:, :2].max(0)) / 2
        for j, k in enumerate(objects):
            u = trimesh.transform_points(k["mesh"].vertices, placed.get(j, k["poses"][0]))
            low, high = np.percentile(u[:, :2], [2, 98], axis=0)
            if j != s["object"] and s["release"] is not None and (low <= centre).all() and (centre <= high).all() and np.linalg.norm(centre - (low + high) / 2) < nearest:  # fmt: skip
                s["into"], nearest, middle, rim = j, np.linalg.norm(centre - (low + high) / 2), (low + high) / 2, (u[:, 2].min(), u[:, 2].max())  # fmt: skip
        if s["into"] is not None:  # centred in a receptacle; on a flat support (a plate) where it is seen to end
            s["end"][:2, 3] += (middle - centre) if objects[s["into"]]["hollow"] else 0
            s["end"][2, 3] += np.clip(v[:, 2].min(), rim[0] + 0.005, rim[1] + 0.005) - v[:, 2].min()
        else:  # on the table once released, never under it
            s["end"][2, 3] -= v[:, 2].min() if s["release"] is not None else min(v[:, 2].min() - 0.005, 0)
        placed[s["object"]] = s["end"]
    for s in [s for s in kept if s.get("on") is not None and handled[s["on"]] and s["on"] in still]:
        unscrew(s, kept, objects, hands, [t - drift for t in tracks], ends, frames)
    return sorted(kept, key=lambda s: s["grasp"])


def unscrew(s: dict, steps: list[dict], objects: list[dict], hands: dict, tracks: list[np.ndarray], ends: list,
            frames: int, near: float = 0.03, smooth: int = 7):  # fmt: skip
    """plan_pnp: step `s` takes its object (a cap) off what it sits on (`on`, a bottle) that the other hand tilts and
    puts back (it ends where it started): that arm (`steady`, appended to `steps`) pinches the bottle from its side and
    turns it so the cap goes where the cap is tracked, its axis along the bottle's track to the cap's (`motion`: world
    from the bottle's rest, per frame, raised clear of the table, kept in the bottle's dict), until the cap comes off
    (`sep`: the cap leaves the bottle's track by 2.5 cm, three frames on end), then back to rest 6 frames after the
    cap's arrival, and lets go where its hand leaves (4 cm from where it rests then; at the latest 10 frames from the
    end). The cap, screwed on (welded to the bottle until its grasp closes), is grasped
    once its hand's pinch is within `near` of it (three frames on end; `screw`), twisted, pulled off along the axis
    and set down where it is seen to end (arriving once within 6 cm of it and 2 cm of its height); in place."""
    import trimesh
    from scipy.ndimage import uniform_filter1d
    from scipy.spatial.transform import Rotation, Slerp

    k, side = s["on"], s["side"]
    other = next(x for x in SIDE if x != side)
    if other not in hands or side not in hands or any(x["side"] == other or x is not s and x["object"] == k for x in steps):  # fmt: skip
        return
    cap, base = tracks[s["object"]], tracks[k]
    on = lambda ok, a=0: a + int((np.convolve(ok[a:], np.ones(3), "valid") == 3).argmax()) if (np.convolve(ok[a:], np.ones(3), "valid") == 3).any() else None  # noqa: E731  # fmt: skip
    c0 = np.median(cap[ends[s["object"]][0]], 0)
    tilt = on(np.linalg.norm(cap - c0, axis=1) > 0.02)
    pinch = (hands[side][:frames, 4] + hands[side][:frames, 8]) / 2
    grasp = on(np.linalg.norm(pinch - cap, axis=1) < near, tilt or 0)
    if tilt is None or grasp is None:
        return
    grasp += 3
    length = np.linalg.norm(cap - base, axis=1)
    sep = on(length - np.median(length[max(grasp - 5, 0) : grasp + 5]) > 0.025, grasp) or min(grasp + 15, frames - 20)
    sep = max(sep, grasp + 8)
    c1 = np.median(cap[ends[s["object"]][1]], 0)
    arrive = max(on((np.linalg.norm(cap - c1, axis=1) < 0.06) & (np.abs(cap[:, 2] - c1[2]) < 0.02), sep) or sep + 10, sep + 8)  # fmt: skip
    release = max(s["release"] or arrive + 4, arrive + 4)
    back = min(arrive + 6, frames - 16)  # the bottle upright again
    hand = (hands[other][:frames, 4] + hands[other][:frames, 8]) / 2
    let = on(np.linalg.norm(hand - np.median(hand[back : back + 10], 0), axis=1) > 0.04, back + 4)
    let = max(min(let or frames - 10, frames - 10), back + 6)
    hold = max(tilt - 2, 18)  # the bottle's grasp: closed as it starts to turn
    tilt = max(tilt, hold + 2)
    axis = uniform_filter1d(cap - base, smooth, axis=0, mode="nearest")
    axis /= np.linalg.norm(axis, axis=1, keepdims=True)
    at = uniform_filter1d(cap, smooth, axis=0, mode="nearest")
    a0, c0 = np.median(axis[ends[s["object"]][0]], 0), s["start"][:3, 3]  # the cap's rest point turns about itself
    motion = np.repeat(np.eye(4)[None], frames, 0)
    for t in range(tilt, sep + 1):  # the bottle turned so the cap goes where it is seen, its axis toward it
        w = min(1.0, (t - tilt + 1) / 5)
        turn = Rotation.align_vectors([axis[t]], [a0])[0].as_matrix()
        turn = Rotation.from_rotvec(w * Rotation.from_matrix(turn).as_rotvec()).as_matrix()
        motion[t, :3, :3], motion[t, :3, 3] = turn, c0 + w * (at[t] - cap[ends[s["object"]][0]].mean(0)) - turn @ c0
    for t in range(sep + 1, back + 1):  # and back to rest soon after the cap's arrival
        h = (t - sep) / (back - sep)
        h = h * h * (3 - 2 * h)
        motion[t, :3, :3] = Slerp([0, 1], Rotation.from_matrix([motion[sep, :3, :3], np.eye(3)]))(h).as_matrix()
        motion[t, :3, 3] = (1 - h) * motion[sep, :3, 3]
    for t in range(frames):  # clear of the table: the bottle and the cap on it raised
        low = min(trimesh.transform_points(objects[j]["mesh"].vertices[::7], motion[t] @ x)[:, 2].min()
                  for j, x in ((k, objects[k]["poses"][0]), (s["object"], s["start"])) if j == k or t <= sep)  # fmt: skip
        motion[t, 2, 3] += max(0.0, 0.01 - low)
    motion[:, :3, 3] = uniform_filter1d(motion[:, :3, 3], 3, axis=0, mode="nearest")
    objects[k]["motion"] = motion
    s.update(grasp=int(grasp), arrive=int(arrive), release=int(release), screw=dict(on=int(k), sep=int(sep)))
    steps.append(dict(kind="steady", object=int(k), side=other, grasp=int(hold), arrive=int(back), release=int(let), far=0.0,
                      start=objects[k]["poses"][0].copy(), end=objects[k]["poses"][0].copy(), into=None))  # fmt: skip


def plan_drawer(objects: list[dict], steps: list[dict], frames: int, xyz: np.ndarray, background: np.ndarray,
                rgb: np.ndarray, close: int = 6, tilt: float = 15.0) -> list[dict]:  # fmt: skip
    """pnp: a picked object that starts within the footprint of an object named a drawer (not itself picked; where
    it is seen, as tracked) is in a drawer, which opens toward where the picked one moved before its grasp (else
    toward the camera): a housing (fixed boxes: its sides, back and the solid over the drawer up to the background's
    depth points over that footprint, in their colour) and, under the named one's front if the object lies below it
    (else in its place), the drawer: a tray of 4 mm walls with a knob, on a slide joint. Its travel is what the
    picked one moved with it before its grasp (at least what brings its centre 5 cm out of the housing to be grasped),
    timed by when it moved (20% to 80% of that), else just before its grasp. The arm that picks it gets two `slide`
    steps (pnp_path): its knob pinched (the jaws across it, leaning `tilt` degrees back from straight down) and pulled
    open, and after its release pushed shut by the closed jaws; the picked one starts on the tray's floor, its step
    marked with the `drawer` and its `travel`. Steps (in grasp order) are updated in place; returns the drawers for
    build_scene: `pose` (world from the housing: x into it, z up), `housing` and the tray's `boxes` ([centre, half
    sizes] in it), `tray` (its origin there), `range` and `rgba`."""
    import trimesh
    from scipy.ndimage import uniform_filter1d

    drawers, picked = [], {s["object"] for s in steps}
    for s in [s for s in steps if not s.get("kind")]:
        o, g = objects[s["object"]], s["grasp"]
        inside = trimesh.transform_points(o["mesh"].vertices, s["start"])
        for d, holder in enumerate(objects):
            seen = trimesh.transform_points(holder["mesh"].vertices, holder["track"][holder["row"]])
            low, high = np.percentile(seen, 2, axis=0), np.percentile(seen, 98, axis=0)
            if "drawer" not in holder["noun"] or d in picked or not ((low[:2] - 0.02 <= inside.mean(0)[:2]) & (inside.mean(0)[:2] <= high[:2] + 0.02)).all():  # fmt: skip
                continue
            track = uniform_filter1d(o["track"][:frames, :3, 3], 5, axis=0, mode="nearest") - o["track"][:5, :3, 3].mean(0)  # fmt: skip
            moved = np.median(track[max(g - 12, 0) : max(g - 4, 1), :2], axis=0)
            centre = (low[:2] + high[:2]) / 2
            out = moved / np.linalg.norm(moved) if np.linalg.norm(moved) > 0.03 else -centre / np.linalg.norm(centre)
            a = np.r_[-out, 0]
            frame = np.eye(4)
            frame[:3, :3], frame[:2, 3] = np.stack([a, np.cross([0, 0, 1], a), [0, 0, 1]], -1), centre
            local = lambda p: (p - frame[:3, 3]) @ frame[:3, :3]  # noqa: E731
            u, v = local(seen), local(inside)
            front, back = min(np.percentile(u[:, 0], 2), v[:, 0].min() - 0.005), max(np.percentile(u[:, 0], 98), v[:, 0].max() + 0.02)  # fmt: skip
            back = max(back, front + 0.16)
            room = float(np.clip(front + 0.025 - v[:, 0].min(), 0, back - 0.01 - v[:, 0].max()))  # a jaw's room before
            v[:, 0] += room  # it: pushed back in the drawer
            (left, right), (bottom, top) = np.percentile(u[:, 1], [2, 98]), np.percentile(u[:, 2], [2, 98])
            ceiling = max(bottom, v[:, 2].max() + 0.012) if v[:, 2].mean() < bottom else top
            p = local(xyz[background])
            over = (p[:, 0] > front - 0.01) & (p[:, 0] < back + 0.02) & (p[:, 1] > left) & (p[:, 1] < right) & (p[:, 2] > 0.01)  # fmt: skip
            roof = max(np.percentile(p[over, 2], 97) if over.sum() > 50 else 0.0, top, ceiling + 0.02)
            colour = [*(rgb[background][over].mean(0) / 255 if over.sum() > 50 else holder["rgba"][:3]), 1]
            along = track[:g] @ np.r_[out, 0]
            seen_travel = float(np.median(along[max(g - 12, 0) : max(g - 4, 1)]))
            travel = float(np.clip(max(seen_travel, v[:, 0].mean() - front + 0.05), 0.03, back - front - 0.03))
            there = np.convolve(along >= 0.8 * seen_travel, np.ones(3), "valid") == 3 if seen_travel > 0.03 else np.zeros(1, bool)  # fmt: skip
            end = int(there.argmax()) if there.any() and there.argmax() < g - 2 * close - 12 else g - 2 * close - 12
            begin = max([t for t in range(end) if along[t] <= 0.2 * seen_travel] or [end - 15])
            begin = min(begin, end - 6)
            width, wall, mid, height = right - left - 0.008, 0.004, (left + right) / 2, ceiling - 0.002
            depth = back - front - 0.006
            boxes = [((depth / 2, 0, 0.003), (depth / 2, width / 2, wall / 2)),  # floor, a millimetre off the table
                     ((wall / 2, 0, height / 2 + 0.001), (wall / 2, width / 2, height / 2)),
                     ((depth - wall / 2, 0, height / 2 + 0.001), (wall / 2, width / 2, height / 2)),
                     *(((depth / 2, side * (width - wall) / 2, height / 2 + 0.001), (depth / 2, wall / 2, height / 2)) for side in (-1, 1)),  # fmt: skip
                     ((-0.0175, 0, 0.6 * height), (0.0175, 0.015, 0.006))]  # the knob, 3.5 cm out  # fmt: skip
            housing = [((front + depth / 2, mid, (ceiling + roof) / 2 + 0.001), ((back - front) / 2, (right - left) / 2 + 0.006, (roof - ceiling) / 2)),  # fmt: skip
                       *(((front + depth / 2, mid + side * ((right - left) / 2 + 0.003), ceiling / 2), ((back - front) / 2, 0.003, ceiling / 2)) for side in (-1, 1)),  # fmt: skip
                       ((back + 0.003, mid, roof / 2), (0.003, (right - left) / 2 + 0.006, roof / 2))]  # fmt: skip
            drawers.append(dict(pose=frame, housing=housing, boxes=boxes, tray=(front, mid, 0.0), range=back - front - 0.03, rgba=colour))  # fmt: skip
            approach = np.sin(np.radians(tilt)) * a - np.cos(np.radians(tilt)) * np.array([0, 0, 1.0])
            pose = lambda point: np.r_[np.c_[np.stack([np.cross(frame[:3, 1], approach), frame[:3, 1], approach], -1), frame[:3, :3] @ point + frame[:3, 3]], [[0, 0, 0, 1]]]  # noqa: E731  # fmt: skip
            knob = pose(np.array([front - 0.022, mid, 0.6 * height]))
            shut = pose(np.array([front - travel - 0.035 - 0.015, mid, 0.6 * height]))
            shut[:3, 3] -= 0.0175 * approach  # the closed jaws' tips on the knob's face, 1.5 cm short of it
            r = s["release"] if s["release"] is not None else s["arrive"]
            slide = dict(kind="slide", object=d, side=s["side"], start=holder["track"][holder["row"]], into=None, far=travel, gap=0.03, drawer=len(drawers) - 1)  # fmt: skip
            steps += [dict(slide, grasp=begin, arrive=end, release=end, knob=knob.tolist(), move=(-a * travel).tolist(), pull=True),  # fmt: skip
                      dict(slide, grasp=r + 12, arrive=r + 20, release=min(r + 20, frames - 3), knob=shut.tolist(), move=(a * (travel + 0.018)).tolist(), pull=False)]  # fmt: skip
            for k in steps[-2:]:
                k["end"] = k["start"]
            s.update(drawer=len(drawers) - 1, travel=(-a * travel).tolist(), grasp=max(g, end + 2 * close + 12))
            s["start"] = s["start"].copy()
            s["start"][:3, 3] += a * room + [0, 0, 0.005]  # on the tray's floor
            break
    steps.sort(key=lambda s: s["grasp"])
    return drawers


def pnp_path(steps: list[dict], every: list[dict], objects: list[dict], frames: int, grip=None, close: int = 6,
             hover: float = 0.1, speed: float = 0.025) -> tuple[np.ndarray, np.ndarray]:  # fmt: skip
    """One arm's pinch poses (world, 4x4) and closure commands (0 open, 1 closed) per frame through its pick-and-place
    `steps` (plan_pnp, in grasp order; `every` arm's, for where each object is by then): to each object (3 cm a frame),
    from `hover` above it (back along the approach, from over that, for a side pinch) onto its grasp (grasp_pose: its
    `turn`, `deeper`, `rim`, `lip` from its arm's side, leaning; the hand's pinch `grip` at the grasp picks a rim point,
    a rim away from its neighbours for a step its hand did not do; out of a drawer, leaning back as plan_drawer's knob
    pinch), the jaws pre-shaped 3 cm over the gap, settling 2 frames, closing over `close` and holding 3 more to the
    grasp frame; 4 cm up or straight up till its base clears what stands beside it, rising over the first half of the
    way across to 4 cm over what lies between, (settled 3 frames over what it goes `into`), down to its end pose (`drop`
    over it, default 1 cm when released, 2.5 cm by a lip) as it arrives, opening at the release over 3 frames, then 4
    till the jaws are clear and up, or 6 cm up if it has more to do; an arm done before the other's next grasp goes back
    where it began. A `slide` step (plan_drawer) pinches its knob and pulls it, or pushes it with closed jaws, by
    `move`. Keyframes timed to the source, pushed later where closer than the hand can go at `speed` (m a frame; a
    step's own `speed`; faster, to FAST, where that would not release it before the clip ends) while holding it,
    eased between."""
    import trimesh
    from scipy.spatial.transform import Rotation, Slerp

    if len(steps) == 1 and (steps[0].get("kind") == "steady" or "screw" in steps[0]):  # unscrew's two arms
        return held_path(steps[0], objects, frames, steps[0].get("close", close))
    done = {s["object"]: (s["release"] if s["release"] is not None else frames, s["end"]) for s in every}
    at = lambda n, t: done[n][1] if n in done and done[n][0] <= t else objects[n]["poses"][0]
    keys = []  # (frame, pose, closure, least frames after the one before)
    for s in steps:
        o, g, r, c = objects[s["object"]], s["grasp"], s["release"], s.get("close", close)
        if s.get("kind") == "slide":  # a drawer (plan_drawer): its knob pinched and pulled, or pushed by closed jaws
            knob, move, hold = np.array(s["knob"]), np.array(s["move"]), 0.0 if s["pull"] else 1.0
            back = knob.copy()
            back[:3, 3] -= 0.06 * knob[:3, 2]  # 6 cm back along the approach
            moved, after = knob.copy(), back.copy()
            moved[:3, 3] += move
            after[:3, 3] += move
            shape = np.clip(1 - (s["gap"] + 0.03) / 0.085, 0, 1) if s["pull"] else 1.0
            free = int(np.ceil(np.linalg.norm(back[:3, 3] - keys[-1][1][:3, 3]) / 0.03)) if keys else 4
            keys += [(g - c - 8, back, shape, max(4, free)), (g - c - 2, knob, shape, 4)]
            keys += [(g, knob, 1.0, max(4, c)), (g + 2, knob, 1.0, 2)] if s["pull"] else []
            keys += [(r - 2, moved, 1.0, max(2, int(np.ceil(np.linalg.norm(move) / s.get("speed", speed))))),
                     (r, moved, 1.0, 2), (r + 3, moved, hold, 3), (r + 6, after, hold, 3)]  # fmt: skip
            continue
        others = [trimesh.transform_points(x["mesh"].vertices[::3], at(n, g)) for n, x in enumerate(objects) if n != s["object"]]  # fmt: skip
        human = grip[0][min(g, frames - 1)] if grip is not None else s["start"]
        if s.get("hand", s["side"]) != s["side"] and others:  # not its hand's: a rim away from what stands nearest
            v = np.concatenate(others)
            human = s["start"].copy()
            human[:2, 3] += 0.1 * (human[:2, 3] - v[np.linalg.norm(v[:, :2] - human[:2, 3], axis=1).argmin(), :2])
        lip = s["start"][:3, 3] - [0, SIDE[s["side"]], 0] if s.get("lip") else None  # from its arm's side
        local, gap = grasp_pose(dict(o, poses=s["start"][None]), 0, human, s.get("turn", 0.0), s.get("deeper", 0.0),
                                s.get("rim"), np.concatenate(others) if others else None, lean=True, lip=lip)  # fmt: skip
        grasp, place = s["start"] @ local, s["end"] @ local
        if "drawer" in s:  # out of a drawer: leaning back from its housing as the knob's pinch, clear of it
            into = -np.array(s["travel"]) / np.linalg.norm(s["travel"])
            grasp[:3, :3] = Rotation.from_rotvec(np.radians(s.get("lean", 15.0)) * np.cross(into, [0, 0, 1])).as_matrix() @ grasp[:3, :3]  # fmt: skip
        place[2, 3] += s.get("drop", (0.025 if lip is not None else 0.01) if r is not None else 0.0)  # (the lower jaw)
        below = grasp[2, 3] - trimesh.transform_points(o["mesh"].vertices, s["start"])[:, 2].min()  # over its base
        a, b = grasp[:2, 3], place[:2, 3]
        between = [v for v in others if len(v)]
        if between:  # what the carried base passes over, 6 cm either side of its way
            v = np.concatenate(between)
            u = np.clip((v[:, :2] - a) @ (b - a) / max((b - a) @ (b - a), 1e-9), 0, 1)
            v = v[np.linalg.norm(v[:, :2] - (a + u[:, None] * (b - a)), axis=1) < 0.06 + o["mesh"].extents.max() / 2]
        top = v[:, 2].max() if between and len(v) else 0.0
        high = max(grasp[2, 3] + 0.06, place[2, 3] + 0.04, top + below + 0.04)
        beside = [x[np.linalg.norm(x[:, :2] - s["start"][:2, 3], axis=1) < o["mesh"].extents.max() / 2 + 0.03] for x in between]  # fmt: skip
        clear = max((x[:, 2].max() + below + 0.02 for x in beside if len(x)), default=0.0)  # base over its neighbours
        up = lambda pose, z: np.r_[np.c_[pose[:3, :3], [*pose[:2, 3], z]], [[0, 0, 0, 1]]]
        ready = np.clip(1 - (gap + 0.03) / 0.085, 0, 1)
        lift, above = up(grasp, max(grasp[2, 3] + 0.04, clear)), up(place, high)  # off its support and straight up
        middle = up(grasp, high)  # clear of what stands beside it, up over the first half of the way across, along
        middle[:2, 3] = (grasp[:2, 3] + place[:2, 3]) / 2  # the rest
        t, over = min(r, s["arrive"]) if r is not None else s["arrive"], up(grasp, grasp[2, 3] + hover)  # set down
        way = lift[2, 3] - grasp[2, 3] + np.linalg.norm(middle[:3, 3] - lift[:3, 3]) + np.linalg.norm(above[:3, 3] - middle[:3, 3]) + high - place[2, 3]  # fmt: skip
        room = frames - g - 11 - 3 * (s["into"] is not None)  # frames left to carry it and set it down in the clip
        pace = s.get("speed", float(np.clip(way / max(room, 1), speed, FAST)) if r is not None else speed)  # faster
        steady = lambda d: max(2, int(np.ceil(d / pace)))  # (to FAST) where too few: frames to go that far holding it
        if grasp[2, 2] > -0.7:  # as it arrives; from the side: in along its approach, from over the way in
            over = grasp.copy()
            over[:3, 3] -= hover * grasp[:3, 2]
            keys.append((g - c - 14, up(over, over[2, 3] + hover), ready, max(4, int(np.ceil(np.linalg.norm(over[:3, 3] + [0, 0, hover] - keys[-1][1][:3, 3]) / 0.03)) if keys else 4)))  # fmt: skip
        free = int(np.ceil(np.linalg.norm(over[:3, 3] - keys[-1][1][:3, 3]) / 0.03)) if keys else 4  # 3 cm a frame
        keys += [(g - c - 10, over, ready, max(4, free)), (g - c - 4, grasp, ready, 4), (g - c - 2, grasp, ready, 2),
                 (g - 4, grasp, 1.0, max(4, c - 2)), (g, grasp, 1.0, 3), (g + 4, lift, 1.0, steady(lift[2, 3] - grasp[2, 3]) + 1),
                 ((g + 4 + t - 8) // 2, middle, 1.0, steady(np.linalg.norm(middle[:3, 3] - lift[:3, 3]))),
                 (t - 8, above, 1.0, steady(np.linalg.norm(above[:3, 3] - middle[:3, 3]))),
                 *([(t - 6, above, 1.0, 3)] if s["into"] is not None else []),  # over what it goes into, settled
                 (t - 2, place, 1.0, steady(high - place[2, 3]))]  # fmt: skip
        if r is not None:
            keys += [(r, place, 1.0, 2), (r + 4, place, 0.0, 3)]
            keys += [(r + 8, place, 0.0, 4), (r + 14, up(place, place[2, 3] + hover), 0.0, 3)] if s is steps[-1] else [(r + 8, up(place, place[2, 3] + 0.06), 0.0, 3)]  # the jaws clear of it  # fmt: skip
    first = keys[0][1].copy()
    first[2, 3] += 0.1
    keys.insert(0, (0, first, 0.0, 0))
    if steps[-1]["release"] is not None and any(e["side"] != steps[-1]["side"] and e["grasp"] > steps[-1]["release"] for e in every):  # fmt: skip
        keys.append((keys[-1][0] + 10, first, 0.0, 6))  # done before the other arm: back where it began, out of its way
    times = [0]
    for t, _, _, least in keys[1:]:  # in order, at least `least` frames apart
        times.append(max(t, times[-1] + least))
    pinch, command = np.repeat(np.eye(4)[None], frames, 0), np.zeros(frames)
    for (t0, p0, c0, _), (t1, p1, c1, _), a, b in zip(keys, keys[1:], times, times[1:]):
        for t in range(a, min(b, frames)):
            h = (t - a) / (b - a)
            h = h * h * (3 - 2 * h)
            pinch[t, :3, :3] = Slerp([0, 1], Rotation.from_matrix([p0[:3, :3], p1[:3, :3]]))(h).as_matrix()
            pinch[t, :3, 3], command[t] = (1 - h) * p0[:3, 3] + h * p1[:3, 3], c0 + (c1 - c0) * (t - a) / (b - a)
    if times[-1] < frames:
        pinch[times[-1] :], command[times[-1] :] = keys[-1][1], keys[-1][2]
    return pinch, command


def ease(keys: list[tuple], frames: int) -> tuple[np.ndarray, np.ndarray]:
    """Pinch poses and closures per frame through keyframes [(frame, pose, closure)], eased between (each at least a
    frame after the one before), the last held."""
    from scipy.spatial.transform import Rotation, Slerp

    pinch, command, times = np.repeat(np.eye(4)[None], frames, 0), np.zeros(frames), [keys[0][0]]
    for t, *_ in keys[1:]:
        times.append(max(t, times[-1] + 1))
    pinch[: times[0] + 1], command[: times[0] + 1] = keys[0][1], keys[0][2]
    for (_, p0, c0), (_, p1, c1), a, b in zip(keys, keys[1:], times, times[1:]):
        for t in range(a, min(b + 1, frames)):
            h = (t - a) / (b - a)
            h = h * h * (3 - 2 * h)
            pinch[t, :3, :3] = Slerp([0, 1], Rotation.from_matrix([p0[:3, :3], p1[:3, :3]]))(h).as_matrix()
            pinch[t, :3, 3], command[t] = (1 - h) * p0[:3, 3] + h * p1[:3, 3], c0 + (c1 - c0) * (t - a) / (b - a)
    pinch[times[-1] :], command[times[-1] :] = keys[-1][1], keys[-1][2]
    return pinch, command


def held_path(s: dict, objects: list[dict], frames: int, close: int = 6, twist: float = 90.0, pull: float = 0.03,
              hover: float = 0.08) -> tuple[np.ndarray, np.ndarray]:  # fmt: skip
    """unscrew's arms (pnp_path for them): a `steady` step pinches its bottle from its arm's side, level, across its
    middle (half way up), closing as it starts to turn, follows its `motion` and lets go at its release; a
    `screw` step pinches the cap on it top-down as the cap sits (grasp_pose: its `turn`, `deeper`) as it turns,
    closing over `close` frames, twists it `twist` degrees (unscrewing) till it comes off, pulls it `pull` out along
    the bottle's axis and sets it down where it ends, held as it came off (the jaws turned level about their approach:
    a top-down place this near its arm's base folds the arm up into the head camera's view), its lowest point `drop`
    (default 1.5 cm) over the table. Each opens over 4 frames at its release and backs out once the jaws are open
    (5 frames on): sooner, they drag what they held."""
    import trimesh

    o, g = objects[s["object"]], s["grasp"]
    rot = lambda a: np.r_[np.c_[[[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]], np.zeros(3)], [[0, 0, 0, 1]]]  # noqa: E731  # fmt: skip
    back = lambda pose, d, up=0.0: np.r_[np.c_[pose[:3, :3], pose[:3, 3] - d * pose[:3, 2] + [0, 0, up]], [[0, 0, 0, 1]]]  # noqa: E731  # fmt: skip
    if s.get("kind") == "steady":
        rest, motion = o["poses"][0], o["motion"]
        v = np.concatenate([trimesh.transform_points(trimesh.sample.sample_surface(trimesh.load(p, force="mesh"), 2000, seed=0)[0], rest) for p in o["parts"]])  # fmt: skip
        centre, (bottom, top) = (v.min(0) + v.max(0)) / 2, (v[:, 2].min(), v[:, 2].max())
        approach = np.r_[centre[:2] - [0, SIDE[s["side"]]], 0]
        approach /= np.linalg.norm(approach)
        closing = np.cross([0, 0, 1.0], approach)
        height = (bottom + top) / 2
        band = v[np.abs(v[:, 2] - height) < 0.015]
        gap = np.ptp(band @ closing) if len(band) else 0.06
        grip = np.eye(4)
        grip[:3, :3], grip[:3, 3] = np.stack([np.cross(closing, approach), closing, approach], -1), [*centre[:2], height]  # fmt: skip
        ready, r = np.clip(1 - (gap + 0.03) / 0.085, 0, 1), s["release"]
        keys = [(0, back(grip, 0.14, 0.06), 0.0), (g - close - 10, back(grip, hover), ready), (g - close - 3, grip, ready), (g, grip, 1.0)]  # fmt: skip
        pinch, command = ease(keys, frames)
        for t in range(g, min(r, frames)):
            pinch[t], command[t] = motion[t] @ grip, 1.0
        tail, shut = ease([(r, motion[min(r, frames - 1)] @ grip, 1.0), (r + 4, grip, 0.0), (r + 9, grip, 0.0), (r + 15, back(grip, hover), 0.0), (r + 21, back(grip, 0.14, 0.06), 0.0)], frames)  # fmt: skip
        pinch[r:], command[r:] = tail[r:], shut[r:]
        return pinch, command
    motion, sep, r = objects[s["screw"]["on"]]["motion"], s["screw"]["sep"], s["release"]
    local, gap = grasp_pose(dict(o, poses=s["start"][None]), 0, s["start"], s.get("turn", 0.0), s.get("deeper", 0.0), s.get("rim"))  # fmt: skip
    grip = s["start"] @ local
    ready = np.clip(1 - (gap + 0.03) / 0.085, 0, 1)
    turned = lambda t: motion[t] @ grip @ rot(-np.radians(twist) * np.clip((t - g - 1) / max(sep - g - 3, 1), 0, 1))  # noqa: E731  # fmt: skip
    a = g - close - 4
    keys = [(0, back(turned(a), hover, 0.1), 0.0), (a - 6, back(turned(a - 6), hover), ready), (a, turned(a), ready)]
    pinch, command = ease(keys, frames)
    for t in range(a, sep + 1):
        pinch[t] = turned(t)
        command[t] = ready if t < g - close else min(1.0, ready + (1 - ready) * (t - g + close + 1) / close)
    out = turned(sep)
    out[:3, 3] -= pull * out[:3, 2]  # off along the bottle's axis
    place, level = out.copy(), np.cross(out[:3, 2], [0, 0, 1.0])
    if np.linalg.norm(level) > 0.3:  # the jaws level, the nearer way
        level *= np.sign(level @ out[:3, 1]) / np.linalg.norm(level)
        place[:3, 1], place[:3, 0] = level, np.cross(level, out[:3, 2])
    held = trimesh.transform_points(o["mesh"].vertices, place @ np.linalg.inv(local))
    place[:3, 3] += [*(s["end"][:2, 3] - (held[:, :2].min(0) + held[:, :2].max(0)) / 2), s.get("drop", 0.015) - held[:, 2].min()]  # fmt: skip
    rest = [(sep, turned(sep), 1.0), (sep + 4, out, 1.0), (s["arrive"] - 3, back(place, 0.0, 0.04), 1.0), (s["arrive"], place, 1.0), (r, place, 1.0),
            (r + 4, place, 0.0), (r + 9, place, 0.0), (r + 14, back(place, 0.06), 0.0), (r + 20, back(place, hover, 0.04), 0.0)]  # fmt: skip
    tail, shut = ease(rest, frames)
    pinch[sep:], command[sep:] = tail[sep:], shut[sep:]
    return pinch, command


FAST = 0.04  # m a frame: the fastest carry (PNP's faster one; pnp_path's to release it before the clip ends)


PNP = [{}, dict(turn=90.0), dict(deeper=0.015), dict(turn=90.0, deeper=0.015), dict(speed=FAST, close=4),
       dict(close=12), dict(drop=0.03), dict(turn=45.0), dict(deeper=-0.01)]  # fmt: skip


def pick_place(model, steps: list[dict], objects: list[dict], grips: dict, bases: dict, frames: int, fps: float,
               out: Path, solve, budget: float = 90.0, tolerance: float = 0.03, check=None) -> tuple[np.ndarray, np.ndarray, dict]:  # fmt: skip
    """`pnp` mode's motion and its checks: each arm through its steps (pnp_path, joints by `solve(side, pinch,
    grasping)`), its gripper closed by command, driven through the actuators (2 frames ahead) in native rollouts; each
    step, in grasp order, retried over PNP's variants (a lid's lip pinches first, a rim grasp first for a receptacle,
    then each turned, deeper, a faster carry, a slower close, a higher drop) until it passes, else the best kept;
    past `budget` seconds only the first. One going `into` another is first re-centred on where that one (or the
    last put in it) is in the rollout kept so far. It passes
    by the routing check `check(native qpos)` (route.score_rollout against the source's masks: its object's score,
    1 - off / moved, at least route's THRESHOLD) where the source shows some tracked object moved, else by its object,
    lifted 2 cm on the way, ending within `tolerance` of its end pose (10 frames after its release, else at the clip's
    end); a `slide` step by its drawer's joint within 30% of its travel of open (or shut) 4 frames after the jaws let
    go. Returns qpos (the arms' targets), ctrl and each step's [variant, error, route score] tries."""
    import copy

    from roboskills.tools.mujoco.route import MOVED, STILL, THRESHOLD

    names, began, tried, solved = [o["name"] for o in objects], time.monotonic(), {}, {}

    def drive(steps: list[dict]) -> tuple[np.ndarray, np.ndarray]:
        qpos, ctrl = np.repeat(model.qpos0[None], frames, 0), np.zeros((frames, model.nu))
        for side, base in bases.items():
            mine = [s for s in steps if s["side"] == side]
            key = json.dumps([{k: v for k, v in s.items() if k not in ("start", "end")} for s in mine], sort_keys=True)
            if (side, key) not in solved:
                if mine:
                    pinch, command = pnp_path(mine, steps, objects, frames, grips.get(side))
                    solved[side, key] = solve(side, pinch, command > 0), command
                else:
                    solved[side, key] = np.repeat([PARK[side]], frames, 0), np.zeros(frames)
            q, command = solved[side, key]
            adr = model.jnt_qposadr[model.joint(f"{side}_joint1").id]
            qpos[:, adr : adr + 15] = np.c_[q, np.outer(command, GRIPPER)]
            ctrl[:, [model.actuator(f"{side}_actuator{j}").id for j in range(1, 8)]] = q[np.minimum(np.arange(frames) + 2, frames - 1)]  # fmt: skip
            ctrl[:, model.actuator(f"{side}_fingers_actuator").id] = 255 * command
        for o in objects:
            adr = model.jnt_qposadr[model.joint(o["name"]).id]
            qpos[:, adr : adr + 3], qpos[:, adr + 3 : adr + 7] = o["poses"][0, :3, 3], mat2quat(o["poses"][0, :3, :3])
        return qpos, ctrl

    def error(native: np.ndarray, s: dict) -> float:  # how far its object ends from its end pose; 1 m more unlifted
        a = model.jnt_qposadr[model.joint(names[s["object"]]).id]
        t = min(s["release"] + 10, frames - 1) if s["release"] is not None else frames - 1
        lifted = native[s["grasp"] : t + 1, a + 2].max() > s["start"][2, 3] + 0.02 or s.get("kind") == "steady"
        return float(np.linalg.norm(native[t, a : a + 3] - s["end"][:3, 3])) + (0 if lifted else 1.0)

    def judge(native: np.ndarray, s: dict) -> tuple[bool, float | None, float]:  # passes, route's score, error
        if s.get("kind") == "slide":  # a drawer: within 30% of its travel of open (or shut) 4 frames after release
            q = native[min(s["release"] + 4, frames - 1), model.jnt_qposadr[model.joint(f"drawer_{s['drawer']}").id]]
            e = float(abs(q - (s["far"] if s["pull"] else 0.0)))
            return e < 0.3 * s["far"], None, e
        e, found = error(native, s), check(native) if check is not None else {}
        record = next((o for o in found.get("objects", []) if o.get("body") == names[s["object"]] and "off" in o), None)
        if record is None or not any(o.get("moved", 0) >= MOVED for o in found["objects"]):  # no evidence: the pose
            return e < tolerance, None, e
        score = max(0.0, 1 - record["off"] / record["moved"]) if record["moved"] >= MOVED else float(record["off"] <= STILL)  # fmt: skip
        return score >= THRESHOLD, score, e

    native = None
    for j, s in enumerate(steps):
        options = [{}] if s.get("kind") else [dict(rim=True), *PNP] if objects[s["object"]]["hollow"] else PNP
        options = [dict(lip=True), dict(lip=True, deeper=0.005), dict(lip=True, deeper=-0.004), *options] if "lid" in objects[s["object"]]["noun"] else options  # fmt: skip
        stack = [k for k in steps[:j] if s["into"] is not None and k["into"] == s["into"] and k["release"] is not None]
        if native is not None and s["into"] is not None:  # onto where what it goes into (the last put in it) is by then
            top = max(stack, key=lambda k: k["release"]) if stack else None
            a = model.jnt_qposadr[model.joint(names[top["object"] if top else s["into"]]).id]
            seen = native[min(s["arrive"], s["release"] or frames, frames - 1), a : a + 3]
            s["end"][:3, 3] += seen - (top["end"] if top else objects[s["into"]]["poses"][0])[:3, 3]
        scores, runs = [], []
        for variant in options[: len(options) if time.monotonic() - began < budget else 1]:
            trial = copy.deepcopy(steps)
            trial[j].update(variant)
            qpos, ctrl = drive(trial)
            rollout(model, qpos[0], ctrl, fps, out / "trial.npz", names)
            runs.append(np.load(out / "trial.npz")["qpos"])
            scores.append((*judge(runs[-1], trial[j]), variant))
            if scores[-1][0]:
                break
        best = min(range(len(scores)), key=lambda i: (not scores[i][0], -(scores[i][1] or 0), scores[i][2]))
        s.update(scores[best][3])
        native = runs[best]
        tried[f"{s['side']}{j}"] = [[v, round(e, 4), None if r is None else round(r, 3)] for _, r, e, v in scores]
    (out / "trial.npz").unlink(missing_ok=True)
    qpos, ctrl = drive(steps)
    return qpos, ctrl, tried


def robot(w: Path, out: Path, frame: int = 0, size: tuple[int, int] = (960, 540), friction: float = 1.5,
          density: float = 300, force: float = 1.0, walls: Literal["planes", "panel"] = "planes",
          ik: str = "smooth", nearby: float = 0.16, hold: bool = True, budget: float = 90.0) -> dict:  # fmt: skip
    """out/: scene.xml, scene.mjb (the source's reference frame as backdrop, the objects, two arms); trajectory.npz
    (`qpos`, `ctrl` per source frame) and mujoco_ext1.mp4, the arms taking each object plan_pnp finds picked and
    placed through keyframes timed to the source (pnp_path), each object welded where it starts until its grasp
    closes, checked and retried in physics for up to `budget` seconds (pick_place; the routing check against the
    source's masks), objects moved by contact only; recon.npz and recon_ext1.mp4 (set per frame) unless already in
    `out`; world.json and report.json (counts, object names, bases, steps, physics checks, route and seconds per
    step). `friction`, `density` and finger `force` go to build_scene, `walls` to paint_backdrop, `ik` (a RoboKit IK
    preset) to solve_arm; the arms' links keep out of a ball of radius `nearby` centred that far ahead of the camera
    (an arm it keeps from a grasp goes without it); with `hold`, each object is welded where it rests until its
    grasp starts. An object picked out of a drawer gets its housing and sliding drawer, opened and shut by the arm
    (plan_drawer); one resting on another (over 3 cm up as seen) and not picked stays where it is seen; a lid over
    another object rests on its top as lid_parts' prism; one going into another ends where it would rest in it
    (seat). A clip with no pick and place raises."""
    import cv2
    import mujoco
    import trimesh

    from roboskills.tools.mujoco.route import score_rollout

    times, began = {}, time.monotonic()
    capture = cv2.VideoCapture(str(w / "input/video.mp4"))
    fps, frames = capture.get(cv2.CAP_PROP_FPS), int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    reference = (
        np.load(out / "camera.npz")["reference_from_camera"]
        if (out / "camera.npz").exists()
        else track_camera(w, out, frame)
    )
    times["camera"] = time.monotonic() - began
    ref = reference_frame(w, out, frame)
    world, table = place_world(w, read_objects(w, np.eye(4), reference, frame), ref)
    objects, cameras = read_objects(w, world, reference, frame, taper=True), world @ reference
    hands = read_hands(w, world, reference)
    backdrop = paint_backdrop(out, ref, world, table, objects, walls=walls)
    times["read"] = time.monotonic() - began - times["camera"]
    grips = {side: grip_targets(hands[side][:frames]) for side in SIDE if side in hands}
    steps, hand = plan_pnp(w, objects, hands, frames), dict(grips)
    if not steps:
        raise ValueError("plan_pnp finds no object picked and placed")
    background = ref["valid"] & (ref["labels"] == 0) & (ref["depth"] > 0)
    drawers = plan_drawer(objects, steps, frames, table["xyz"] @ world[:3, :3].T + world[:3, 3], background, ref["rgb"])  # fmt: skip
    for s in steps:  # a drawer's named front stays as seen, touching nothing: the housing stands for it
        objects[s["object"]]["inert"] = s.get("kind") == "slide"
    held = {s["object"] for s in steps if s.get("kind") != "slide"}
    for i, o in enumerate(objects):  # only a grasped object moves; the rest hold their first pose, and weigh more
        if i not in held:  # (one resting on another, its base over 3 cm up as seen, stays where it is seen)
            up = (o["mesh"].vertices @ o["track"][o["row"]][2, :3] + o["track"][o["row"]][2, 3]).min() > 0.03  # fmt: skip
            o["poses"][:], o["density"] = o["track"][o["row"]] if up else o["poses"][0], 2000
    for s in steps:  # pnp: each picked object starts where it is first seen whole
        objects[s["object"]]["poses"][:] = s["start"]
    for o in objects:  # a lid over another object rests on its top, not sunk into it
        here = trimesh.transform_points(o["mesh"].vertices, o["poses"][0])
        under = [trimesh.transform_points(x["mesh"].vertices, x["poses"][0]) for x in objects if x is not o]
        tops = [u[:, 2].max() for u in under if (here[:, :2].min(0) <= u[:, :2].mean(0)).all() and (u[:, :2].mean(0) <= here[:, :2].max(0)).all() and here[:, 2].min() < u[:, 2].max() < here[:, 2].max()]  # fmt: skip
        if "lid" in o["noun"] and tops:
            o["floor"] = max(tops) - 0.003
            o["parts"] = lid_parts(o["parts"], o["poses"][0], o["floor"], w / "cache/lid" / o["name"])
    for s in (s for s in steps if s.get("kind") == "steady"):  # unscrew's bottle: narrowed to what the jaws close on,
        o = objects[s["object"]]  # its texture its own pixels (not its cap's)
        if "textured" in o and o["name"].isdigit():
            own_pixels(o, ref["labels"])
        width = np.ptp(trimesh.transform_points(o["mesh"].vertices, o["poses"][0])[:, :2], axis=0).min()
        o["parts"] = squeeze_parts(o["parts"], o["poses"][0], w / "cache/steady" / o["name"], 0.068 / width) if width > 0.07 else o["parts"]  # fmt: skip
    travel = {side: np.ptp(grip[0][:, :3, 3], axis=0).max() for side, grip in grips.items()}
    mimic = [max(travel, key=travel.get)] if travel and max(travel.values()) > 0.15 else []
    grips = {side: grip for side, grip in grips.items() if side in mimic}  # the rest park out of view
    paths = {side: pnp_path([s for s in steps if s["side"] == side], steps, objects, frames, hand.get(side))[0] for side in {s["side"] for s in steps}}  # fmt: skip
    bases = {side: place_base(paths[side], y, room=0.45) if side in paths else place_base(grips[side][0] if side in grips else None, y) for side, y in SIDE.items()}  # fmt: skip
    k = ref["K"]
    grasped = {s["object"]: s["grasp"] for s in steps if s.get("kind") != "slide"}
    screws = {objects[s["object"]]["name"]: (objects[s["screw"]["on"]]["name"], s["grasp"]) for s in steps if "screw" in s}  # fmt: skip
    for s in (s for s in steps if "screw" in s):  # a cap screwed on: welded to its bottle (not in place) till grasped
        grasped[s["object"]] = 0
    for s in steps:  # in a drawer: free as the drawer is pinched, to slide out with it
        if "drawer" in s and not s.get("kind"):
            grasped[s["object"]] = next(k["grasp"] for k in steps if k.get("kind") and k["drawer"] == s["drawer"])
    holds = {o["name"]: grasped.get(i, -1) for i, o in enumerate(objects)} if hold else {}
    model = build_scene(out, objects, table, 2 * np.arctan(k[1, 2] / k[1, 1]), size, bases, backdrop, friction, density, force, (), holds, drawers, screws if hold else {})  # fmt: skip
    first = [o["poses"][0, :3, 3].copy() for o in objects]
    separate(model, objects, steps=60, apart=True)
    for s in steps:  # where separate left them (out with its drawer, if in one), and what they go into
        s["start"] = objects[s["object"]]["poses"][0].copy()
        s["start"][:3, 3] += s.get("travel", 0)
        if s["into"] is not None:
            s["end"][:3, 3] += objects[s["into"]]["poses"][0, :3, 3] - first[s["into"]]
    seat(model, objects, steps)
    for i in range(model.neq):  # each hold where its object now starts (separate may have moved it)
        if model.equality(i).name.startswith("hold:"):  # (cap@bottle, a cap screwed on: its bottle seen from the cap)
            one, *two = model.equality(i).name.split(":")[1].split("@")
            pose = lambda n: next(o["poses"][0] for o in objects if o["name"] == n)  # noqa: E731
            relative = np.linalg.inv(pose(one)) @ pose(two[0]) if two else None
            model.eq_data[i, 3:6] = relative[:3, 3] if two else -pose(one)[:3, :3].T @ pose(one)[:3, 3]
            model.eq_data[i, 6:10] = mat2quat(relative[:3, :3] if two else pose(one)[:3, :3].T)
    mujoco.mj_saveModel(model, str(out / "scene.mjb"))
    names = {o["name"]: (w / "objects" / o["name"] / "name.txt").read_text() for o in objects if (w / "objects" / o["name"] / "name.txt").exists()}  # fmt: skip
    report = dict(fps=fps, frames=frames, hands=sorted(hands), objects=[o["name"] for o in objects], names=names)
    eye = cameras[frame]  # the camera's near field: a ball ahead of it the links keep out of
    near_field = (eye[:3, 3] + eye[:3, 2] * nearby, nearby) if nearby else None

    def miss(side: str, q: np.ndarray, pinch: np.ndarray, grasping: np.ndarray) -> float:  # IK's worst miss there
        data, adr, worst = mujoco.MjData(model), model.jnt_qposadr[model.joint(f"{side}_joint1").id], 0.0
        for t in np.flatnonzero(grasping):
            data.qpos[adr : adr + 7] = q[t]
            mujoco.mj_kinematics(model, data)
            worst = max(worst, float(np.linalg.norm(data.site(f"{side}_pinch").xpos - pinch[t, :3, 3])))
        return worst

    def solve(side: str, pinch: np.ndarray, grasping: np.ndarray) -> np.ndarray:  # pnp's arm joints
        q = solve_arm(pinch, bases[side], preset=ik, avoid=near_field)
        return solve_arm(pinch, bases[side], preset=ik) if near_field and miss(side, q, pinch, grasping) > 0.02 else q

    check = (lambda native: score_rollout(model, native, cameras, w / "masks")) if len(list((w / "masks").glob("*.png"))) > 1 else None  # fmt: skip
    qpos, ctrl, tried = pick_place(model, steps, objects, hand, bases, frames, fps, out, solve, budget, check=check)
    report["mpc"], report["mode"] = {}, "pnp"
    for side, base in bases.items():
        report[side] = dict(base=base[:3, 3].round(2).tolist(), steps=[{k: v.tolist() if k in ("start", "end") else v for k, v in s.items()} for s in steps if s["side"] == side])  # fmt: skip
    report["repair"] = tried
    times["retarget"] = time.monotonic() - began - sum(times.values())
    if not (out / "recon_ext1.mp4").exists():
        np.savez(out / "recon.npz", qpos=qpos)
        render_video(model, qpos, cameras, out / "recon_ext1.mp4", size, fps)
    times["recon_render"] = time.monotonic() - began - sum(times.values())
    report["physics"] = rollout(model, qpos[0], ctrl, fps, out / "trajectory.npz", [o["name"] for o in objects])
    if check is not None:  # the routing check on what is delivered
        found = check(np.load(out / "trajectory.npz")["qpos"])
        report["route"] = dict(passed=found["pass"], score=found["score"], reasons=found["reasons"])
    times["physics"] = time.monotonic() - began - sum(times.values())
    render_video(model, np.load(out / "trajectory.npz")["qpos"], cameras, out / "mujoco_ext1.mp4", size, fps)
    times["physics_render"] = time.monotonic() - began - sum(times.values())
    table = {k: v for k, v in table.items() if k not in ("plane", "xyz")}
    (out / "world.json").write_text(json.dumps(dict(world_from_reference=world.tolist(), table=table)) + "\n")
    (out / "report.json").write_text(
        json.dumps(dict(report, seconds={k: round(v, 1) for k, v in times.items()}), indent=2) + "\n"
    )
    return report
