"""The action layer: `Pose`, one action target (where the TCP goes and the pre-pose it is approached from, the hand
commands around it, or a base or body target), and the two targets the agent does not build itself from the points
it passes in: `grasp_poses` (the learned backend, best first) and `place_pose`; whether one is reachable is
the planner's question, not theirs. The verb tables at the end say how the planner moves each verb, by name: a Pose
the agent builds itself (move, lift, release, home, press, pinch, drive, posture: agent_as_policy/SKILL.md) moves the
same way."""

import importlib
import os
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from roboskills.tools.robokit.geom import near


@dataclass(frozen=True)
class Pose:
    """One target of one verb, world frame. Lists of them are best first."""

    verb: str
    """The verb that made it: grasp, pinch, press, place, move, lift, release, twist, home, drive or posture."""
    arm: str | None
    """The hand it moves; None for drive and posture."""
    obj: str | None = None
    """The object id it acts on, the agent's own label; None: none."""
    tcp: np.ndarray | None = None
    """(4, 4) the TCP target; None for drive and posture."""
    ee_pre: np.ndarray | None = None
    """(4, 4) the pre-pose: planned to around obstacles, then a straight line to `tcp`; None: planned straight to `tcp`."""
    hand_pre: np.ndarray | None = None
    """Hand command on the way out; None: unchanged."""
    hand: np.ndarray | None = None
    """Hand command once at `tcp`; None: unchanged."""
    base: np.ndarray | None = None
    """(3,) the base target [x, y, yaw] (yaw unwrapped from now), drive only."""
    height: float | None = None
    """The pelvis height target above the floor, m, posture only."""
    rank: int | str = 0
    """Where it came from: the backend's order (0 best), "contact", "spin<deg>", "place", ..."""
    via: list | None = None
    """(4, 4) TCP poses passed on the way to `tcp`, in order (a twist's arc); None: none."""
    speed: float | None = None
    """Speed scale of a twist's arc (1.0: a move's); None: SLOW["twist"]."""


PRE_GRASP = 0.08  # pre-grasp stand-off along -approach, m: where every grasp is approached from
PUSH = 0.002  # m, how far past the expected contact a placement lowers what it holds (the arm stops on contact)


# ---- grasp: the agent's own TCP pose and hand command, or the learned policy and ranker a `$GRASP_BACKEND` module
# wraps, one ranked list to plan and try


def heuristic_grasp(model, arm, tcp, hand=None):
    """The agent's own grasp of `arm`: the TCP at `tcp` (4, 4), approached straight along its +x from `PRE_GRASP`
    back, the hand open on the way in and at `hand` once there (one value per hand DOF, the hand's command order;
    None: closed). ValueError for a wrong shape."""
    h, T = model.robot.arms[arm].hand, np.asarray(tcp, float)
    u = np.asarray(h.closed if hand is None else hand, float).reshape(-1)
    if T.shape != (4, 4) or len(u) != len(h.closed):
        raise ValueError(f"heuristic_grasp needs tcp (4, 4) and hand ({len(h.closed)},), not {T.shape} and {u.shape}")
    pre = T.copy()
    pre[:3, 3] -= PRE_GRASP * T[:3, 0]
    return Pose("grasp", arm, tcp=T, ee_pre=pre, hand_pre=np.asarray(h.open, float), hand=u, rank="agent")


def _approach_cone(spec, toward=None):
    """(axis, cone_deg): None (any way not from under the table), "top" (down), "side" (horizontal, along `toward`,
    the arm's reach direction) or a direction vector (the hand's travel). Strings are told apart before any array
    exists: `array == "top"` is elementwise."""
    if spec is None:
        return None, 30.0
    if not isinstance(spec, str):
        return (a := np.asarray(spec, float)) / np.linalg.norm(a), 30.0
    if spec == "top":
        return np.array([0, 0, -1.0]), 35.0
    if spec != "side":
        raise ValueError(f"approach must be None, 'top', 'side' or [x, y, z], not {spec!r}")
    d = np.r_[np.asarray(toward if toward is not None else [1.0, 0], float)[:2], 0.0]
    return d / np.linalg.norm(d), 50.0


def grasp_backend():
    """The learned grasp backend this process runs, as its settings: the $GRASP_* environment ({"GRASP_BACKEND":
    module, ...}), or None without a $GRASP_BACKEND."""
    if not os.environ.get("GRASP_BACKEND"):
        return None
    return {k: v for k, v in sorted(os.environ.items()) if k.startswith("GRASP_")}


def grasp_poses(model, arm, P, scene, table_z, approach=None, avoid=None, variant=0, out=None, tries=8):
    """The grasps for `arm` (a `RobotModel`'s) to try on object points P (N, 3), best first: the backend's best `tries`
    in its order, clear of the rest of the observed `scene` points (M, 3; None: none) and of `avoid` (K, 3; kept clear
    even where it touches the object: the other hands and what they hold), the list rotated by `variant` (a retry
    starts elsewhere in it). `approach`: None (any way not from under the table), "top" (down), "side" (horizontal,
    from the arm's base toward the object) or a direction vector (the hand's travel). $GRASP_BACKEND names an
    importable module whose `candidates(S, arm, points, table_z, obstacles, target, avoid, axis, cone, seed)` (called
    positionally, S an episode view: robot, kin the RobotModel, out where it may log; target always None here)
    returns grasp `Pose`s. ValueError for no points or no backend (then build the grasp with `heuristic_grasp`)."""
    P = np.asarray(P, float).reshape(-1, 3)
    if not len(P):
        raise ValueError("no object points to grasp: segment the object in a fresh observe first")
    scene = np.zeros((0, 3)) if scene is None else np.asarray(scene, float).reshape(-1, 3)
    avoid = np.zeros((0, 3)) if avoid is None else np.asarray(avoid, float).reshape(-1, 3)
    obstacles = scene[~near(scene, P, 0.015) & (scene[:, 2] > table_z + 0.008)] if len(scene) else scene
    axis, cone = _approach_cone(approach, P.mean(0) - model.info[arm].M.root.T_world_base[:3, 3])
    if (backend := grasp_backend()) is None:
        raise ValueError("no grasp backend here: build the grasp yourself with heuristic_grasp(arm, tcp, hand)")
    name = backend["GRASP_BACKEND"]
    S = SimpleNamespace(robot=model.robot, kin=model, out=out)
    backend = importlib.import_module(name).candidates(S, arm, P, table_z, obstacles, None, avoid, axis, cone, variant)
    cands = list(backend)[:tries]
    k = variant % len(cands) if cands else 0
    return cands[k:] + cands[:k]


# ---- place: a held object set upright at a spot and heading, the hand opening there after hovering above it


def _upright(T, T_tcp_obj, P_tcp, xy, z, heading):
    """The TCP contact pose that, with the hand at T (4, 4) holding an object posed T_tcp_obj in the TCP frame (its
    points P_tcp (N, 3) there), turns the object's frame upright to `heading` (deg) about the TCP, puts its origin at
    `xy` [x, y] and its lowest points at height `z`."""
    from robokit.xform.numpy import euler_angles_to_matrix, rot_tl_to_tf_mat, transform_points

    X, p = T @ T_tcp_obj, T[:3, 3]
    G = (
        euler_angles_to_matrix(np.radians([0.0, 0.0, heading]), "sxyz") @ X[:3, :3].T
    )  # the turn, world frame, that sets the object upright
    # c, low: the object about the TCP, turned
    c, low = G @ (X[:3, 3] - p), np.percentile((transform_points(P_tcp, T) - p) @ G[2], 0.5)  # fmt: skip
    return rot_tl_to_tf_mat(G @ T[:3, :3], np.r_[np.asarray(xy, float)[:2] - c[:2], float(z) - low])


def place_pose(model, arm, T_tcp, T_tcp_obj, P, pos, yaw=None, hover=0.04, table_z=None):
    """One `Pose`: the object `arm` holds (posed `T_tcp_obj` in the TCP frame, its world points P (N, 3) now, the TCP
    at `T_tcp`) set upright, its centre at `pos` [x, y] with its bottom on `table_z` (or on `pos[2]`, a surface
    height), turned to long-axis `yaw` (deg, None: as it is): the TCP there (pressed `PUSH` past contact), hovering
    `hover` m straight above first (at least 3 cm clear of its top: where the open hand backs off to), the hand opening
    there. ValueError for too few points or no height."""
    from robokit.xform.numpy import transform_points

    if len(P := np.asarray(P, float).reshape(-1, 3)) < 20:
        raise ValueError("the held object's shape is unknown (under 20 points): observe it in the hand first")
    pos, T = np.asarray(pos, float), np.asarray(T_tcp, float)
    z = float(pos[2]) if len(pos) > 2 else table_z
    if z is None:
        raise ValueError("no surface height: give table_z or pos [x, y, surface z]")
    X = T @ T_tcp_obj
    heading = np.degrees(np.arctan2(X[1, 0], X[0, 0]))  # the object's own heading now
    if yaw is not None:  # the directed heading nearest long-axis yaw (mod 180): a slip across 90 deg is no half-turn
        heading += (yaw - heading + 90.0) % 180.0 - 90.0
    contact = _upright(T, T_tcp_obj, transform_points(P, np.linalg.inv(T)), pos[:2], z, heading)
    # the hover is also where the open hand backs off to: clear of the object's top by 3 cm
    hover = max(hover, float(transform_points(P, contact @ np.linalg.inv(T))[:, 2].max() - contact[2, 3]) + 0.03)
    hand, down, up = model.robot.arms[arm].hand, contact.copy(), contact.copy()
    down[2, 3] -= PUSH
    up[2, 3] += hover
    return Pose("place", arm, tcp=down, ee_pre=up,
                hand=np.asarray(hand.open, float), rank="place")  # fmt: skip


# ---- twist: what a hand holds turned about an axis


TWIST_STEP = 10.0  # deg between a twist's arc waypoints (each solved onto the arc, joined by a short straight line)


def twist_pose(T_tcp, axis, point, deg, arm, hand=None, speed=None):
    """One `Pose` turning the TCP (now at `T_tcp`, 4 x 4) `deg` degrees about the world `axis` through `point` (a
    lid's axis through its centre; right-handed: positive turns counterclockwise looking down -axis), on the arc: a
    waypoint every TWIST_STEP deg, the hand at `hand` (None: as it is), in contact throughout, at `speed` (None:
    slowly, SLOW["twist"], for a lid; up to 1.0, a move's, for a tilt such as a pour)."""
    k = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    T0, c, n = np.asarray(T_tcp, float), np.asarray(point, float), max(1, int(np.ceil(abs(deg) / TWIST_STEP)))
    arc = []
    for a in np.radians(deg) * np.arange(1, n + 1) / n:
        R = np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R @ T0[:3, :3], c + R @ (T0[:3, 3] - c)
        arc.append(T)
    return Pose("twist", arm, tcp=arc[-1], via=arc[:-1], hand=None if hand is None else np.asarray(hand, float),
                rank="twist", speed=speed)  # fmt: skip


# ---- the verb tables: the planner moves each verb by its name

LINE = ("lift", "release")  # verbs whose motion is a short straight joint line, not collision-checked (it starts in contact)  # fmt: skip
LIFT = dict(grasp=0.1, pinch=0.1)  # m a pick must also lift straight up from its closure (planned, not followed)
BACK = ("press", "place")  # verbs whose hand comes straight back up to the pre-pose once it has pressed or let go
SLOW = dict(place=0.3, twist=0.2)  # speed scale of the straight leg into contact: a placement sets its object down
#                                    gently; a twist turns what it holds, all of it in contact
