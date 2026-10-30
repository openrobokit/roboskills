"""What the cameras say about objects and holds, as pure functions of a `View` and the points the agent keeps: the
table (`table_height`, `facing_up`), an object's points in a mask and fused across cameras (`mask_points`,
`merge_masks`, `merge`), whether points are where the cameras see them (`seen`), whether a lift took its object along
(`lift_check`) and how far a press stopped short (`press_short`).
Nothing here remembers anything: the agent keeps its objects and holds itself."""

from types import SimpleNamespace

import numpy as np

from roboskills.tools.robokit.geom import clusters, flat, near, voxel_ids


def press_short(model, plan):
    """How far (m) an executed press's press point stopped above its planned bottom, at the lowest of the plan's
    trajectory ends."""
    a, pp = plan.pose.arm, np.r_[model.robot.arms[plan.pose.arm].hand.press_point, 1.0]
    z = [(model.tcp(a, model.fq(a, SimpleNamespace(q=q, hand=plan.end_hand))) @ pp)[2] for q in plan.ends_q]
    return round(float(min(z) - (plan.pose.tcp @ pp)[2]), 4)


# ---- what the cameras say about a hold


def seen(view, X, tol=0.015):
    """Do the cameras of `view` see points X (N, 3) where they are? Per point and camera the depth there matches (seen),
    is farther (seen through: not there), or nearer (occluded: no vote); pixels NaN in the clouds (the robot) do not
    vote. Returns (seen share, votes)."""
    from robokit.xform.numpy import transform_points

    yes = no = 0
    X = np.atleast_2d(np.asarray(X, float))
    for name, P in view.clouds.items():
        cam = view.cams[name]
        C, K = transform_points(X, np.linalg.inv(cam.T_world_cam)), cam.K
        z = C[:, 2]  # each point's depth, and its pixel
        u, v = np.round(C[:, :2] / np.maximum(z[:, None], 1e-6) * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]).astype(int).T
        D = np.asarray(cam.depth, float)
        ok = (z > 0.05) & (u >= 0) & (u < D.shape[1]) & (v >= 0) & (v < D.shape[0])
        d, z = D[v[ok], u[ok]], z[ok]
        good = np.isfinite(d) & (d > 0) & np.isfinite(P[v[ok], u[ok], 0])
        yes, no = yes + int((np.abs(d - z) < tol)[good].sum()), no + int((d > z + tol)[good].sum())
    return yes / max(yes + no, 1), yes + no


def lift_check(view, P, tz, T_closed, T_now, votes=30, gripping=None):
    """Did the hand take the object along? Its points P as grasped, carried with the hand (TCP T_closed -> T_now):
    does a camera of `view` see them there (`in_hand`), and still see it where it stood (`left_behind`)? The clouds
    are robot-free, so fingers do not count as the object. `gripping` (the caller's reading of the hand after the lift:
    do the jaws stand apart on something?): the
    fingers held apart by something say held unless the cameras see it left behind and not in the hand (a tall or
    hollow object lifted less than its height still shows where it stood); closed all the way, not held (the cameras
    see little of an object in the hand: it shifts as the fingers close, and they hide it). lifted: True, False, or
    None (cannot tell: look)."""
    from robokit.xform.numpy import transform_points

    moved = transform_points(P, T_now @ np.linalg.inv(T_closed))
    in_hand, n = seen(view, moved)
    old = (old := P[P[:, 2] > tz + 0.004])[
        ~near(old, moved, 0.015)
    ]  # where it stood, clear of the table and of where it went
    tol = float(np.clip(0.5 * (np.percentile(P[:, 2], 98) - tz), 0.004, 0.03))  # nudged is not gone; flat: table near
    stayed, m = seen(view, old, tol) if len(old) >= 20 else (0.0, 0)
    left = m >= votes and stayed > 0.6
    lifted = True if n >= votes and in_hand > 0.5 else False if left or (n >= votes and in_hand < 0.1) else None
    return dict(lifted=lifted if gripping is None else gripping and (lifted is True or not left), in_hand=round(float(in_hand), 2),
                left_behind=round(float(stayed), 2))  # fmt: skip


def facing_up(P, cos=0.9):
    """(H, W) bool: where an organized cloud (H, W, 3) shows a surface facing up (its normal within 25 deg of +z)."""
    n = np.cross(P[1:-1, 2:] - P[1:-1, :-2], P[2:, 1:-1] - P[:-2, 1:-1])
    up = np.zeros(P.shape[:2], bool)
    with np.errstate(invalid="ignore", divide="ignore"):
        up[1:-1, 1:-1] = np.abs(n[..., 2]) > cos * np.linalg.norm(n, axis=-1)
    return up


def table_height(clouds, band=None, min_area=0.25, cell=0.02):
    """Height of the table top: among the horizontal levels (2 mm height bins of surfaces facing up) within `band`
    (lo, hi) (where the hands work; the floor is below it), the one covering the largest plan area (at least
    `min_area` m^2, else the densest); lo when the cameras see no surface there. Walls seen edge-on at a camera's
    height are no level."""
    P = flat({c: np.where(facing_up(C)[..., None], C, np.nan) for c, C in clouds.items()})[::7]
    if band is not None:
        if ((P[:, 2] > band[0]) & (P[:, 2] < band[1])).sum() < 100:
            return float(band[0])
        P = P[(P[:, 2] > band[0]) & (P[:, 2] < band[1])]
    if len(P) < 100:
        raise ValueError("too few points to find the table")
    z = P[:, 2]
    edges = np.arange(z.min(), z.max() + 0.004, 0.002)
    h = np.histogram(z, bins=edges)[0]
    best = (0.0, np.abs(z - edges[h.argmax()] - 0.001) < 0.004)
    for i in np.flatnonzero(h >= 0.005 * len(z)):  # every well-filled level
        level = np.abs(z - edges[i] - 0.001) < 0.004
        if (area := len(voxel_ids(P[level, :2], cell)) * cell**2) >= min_area and area > best[0]:
            best = (area, level)
    return float(np.median(z[best[1]]))


def merge(dets, overlap=0.3, radius=0.012):
    """Group per-camera detections [(label, score, points, cam)] of one object: same label, other camera, and at
    least `overlap` of the smaller cloud within `radius` of the larger. Returns lists of indices."""
    parent = list(range(len(dets)))
    root = lambda i: i if parent[i] == i else root(parent[i])
    for i, (label, _, P, cam) in enumerate(dets):
        for j in range(i):
            if (
                dets[j][0] == label
                and dets[j][3] != cam
                and near(*sorted((P, dets[j][2]), key=len), radius).mean() >= overlap
            ):
                parent[root(i)] = root(j)
    groups = {}
    for i in range(len(dets)):
        groups.setdefault(root(i), []).append(i)
    return list(groups.values())


def mask_points(view, masks, cam, k, tz, min_pts=30):
    """(N, 3) the points of mask k of camera `cam` in `masks` (segment's): its pixels' cloud in `view`, the table dropped
    unless the object is flat, only its big pieces (stray pixels on the background or a neighbour are no part of it);
    empty below `min_pts`."""
    Q = view.clouds[cam][masks[cam][k][2]]
    Q = Q[np.isfinite(Q).all(1)]
    Q = Q[above] if (above := Q[:, 2] > tz + 0.004).sum() >= max(min_pts, 0.3 * len(Q)) else Q
    parts = clusters(Q, cell=0.01, min_pts=10) if len(Q) >= min_pts else []
    return Q[np.concatenate([p for p in parts if len(p) > 0.3 * len(parts[0])])] if parts else Q[:0]


def merge_masks(view, masks, tz, min_pts=30):
    """The objects in `masks` (segment's {cam: [(label, score, mask)]}), fused across cameras: [(label, confidence,
    points (N, 3))], best first. A mask's table pixels are dropped unless the object is flat; its small pieces too."""
    dets = [(label, score, Q, cam) for cam, ms in masks.items() for k, (label, score, _) in enumerate(ms)
            if len(Q := mask_points(view, masks, cam, k, tz, min_pts))]  # fmt: skip
    objs = [(dets[g[0]][0], round(float(max(dets[i][1] for i in g)), 3), np.concatenate([dets[i][2] for i in g]))
            for g in merge(dets)]  # fmt: skip
    return sorted(objs, key=lambda o: (-o[1], -len(o[2])))
