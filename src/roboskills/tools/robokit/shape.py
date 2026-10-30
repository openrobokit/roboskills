"""An object's shape from its observed points: its outline, size and yaw, hidden bottom face, and rigid registration of
a remembered shape to a new view. World metres; no simulator information."""

from dataclasses import dataclass
from itertools import product

import numpy as np
from scipy.spatial import cKDTree

from roboskills.tools.robokit.geom import voxel_ids


@dataclass
class Box:
    """An object's oriented box from its world points, metres."""

    center: np.ndarray
    """[x, y, z] its centre: of the outline in xy, halfway between bottom and top in z."""
    size: np.ndarray
    """[length, width, height]: length along the yaw axis, width across it."""
    yaw: float
    """Heading (deg, mod 180) of the length axis about the vertical."""
    bottom: float
    """Height it rests at: the table when it stands on it (the cameras never see its underside)."""


def describe(P, tz):
    """The `Box` of world points P (N, 3) of one object standing over (or on) the table at height `tz`. Its outline is
    the least-area rectangle, in 0.5 deg steps, of the 1st-99th percentiles of one xy sample per 2 mm (so pixel density
    weighs nothing and stray points do not stretch it)."""
    top, low = float(np.percentile(P[:, 2], 98)), float(np.percentile(P[:, 2], 2))
    bottom = tz if low < tz + 0.02 else low  # resting on the table: the hidden bottom is the table
    xy = P[np.sort(voxel_ids(P[:, :2], 0.002)), :2]
    a = np.radians(np.arange(0, 90, 0.5))
    axes = np.stack([np.c_[np.cos(a), np.sin(a)], np.c_[-np.sin(a), np.cos(a)]], 1)  # (K, 2, 2): rows the box axes
    lo, hi = np.percentile(np.einsum("nd,kad->kna", xy, axes), [1, 99], axis=1)
    k = int(np.argmin(np.prod(hi - lo, axis=1)))
    A, lo, hi = (axes[k][::-1], lo[k][::-1], hi[k][::-1]) if hi[k, 1] - lo[k, 1] > hi[k, 0] - lo[k, 0] else (axes[k], lo[k], hi[k])  # fmt: skip
    yaw = float(np.degrees(np.arctan2(A[0, 1], A[0, 0])) % 180)  # the long axis, modulo 180
    return Box(center=np.r_[(lo + hi) / 2 @ A, (bottom + top) / 2], size=np.r_[hi - lo, top - bottom], yaw=yaw,
               bottom=bottom)  # fmt: skip


def fit_rigid(P, observed, max_turn_deg=25.0):
    """Rigid fit of shape P (N, 3) onto `observed` points (M, 3), and whether to trust it: (R, t, quality).
    quality: residual_m, matched, shift_m (centroid), turn_deg, accepted: at least 40 matched points, a residual under
    4 mm, a shift of at most 3 cm and a turn of at most `max_turn_deg`."""
    from robokit.xform.numpy import matrix_to_rotation_angle

    P, observed = np.asarray(P, float), np.asarray(observed, float)
    R, t, residual, n = register_rigid(P, observed)
    q = dict(residual_m=residual, matched=n, shift_m=float(np.linalg.norm((P @ R.T + t).mean(0) - P.mean(0))), turn_deg=float(np.degrees(matrix_to_rotation_angle(R))))  # fmt: skip
    ok = n >= 40 and residual < 0.004 and q["shift_m"] <= 0.03 and q["turn_deg"] <= max_turn_deg
    return R, t, dict(q, accepted=bool(ok))


def register_rigid(P, observed, iters=100, cell=0.002, near=0.02):
    """Rigid ICP with uniform spatial sampling (one point per `cell`), including roll/pitch slip in the jaws, refined
    against the unsampled model (a voxel grid alone permits one-cell translation aliases): (R, t, median residual m,
    matched points). No simulator information."""
    from robokit.xform.numpy import rot_tl_to_tf_mat, transform_points

    raw, sample = np.asarray(P, float), lambda A: A[np.sort(voxel_ids(A, cell))]
    O = sample(np.asarray(observed, float))
    # nearest model point to each observed one: the model's tree, built once, queried with O in the model's frame
    # (a rigid motion keeps distances), not a tree of the moved model every round
    query = lambda tree, T: tree.query(transform_points(O, np.linalg.inv(T)), distance_upper_bound=near)

    def align(model, tree, T, rounds):
        for _ in range(rounds):
            d, ids = query(tree, T)
            if (keep := np.isfinite(d)).sum() < 10:
                break
            A, B = transform_points(model[ids[keep]], T), O[keep]
            ca, cb = A.mean(0), B.mean(0)
            U, _, V = np.linalg.svd((A - ca).T @ (B - cb))
            R = V.T @ np.diag([1, 1, np.linalg.det(V.T @ U.T)]) @ U.T
            T = (D := rot_tl_to_tf_mat(R, cb - R @ ca)) @ T
            if np.linalg.norm(D - np.eye(4)) < 1e-7:
                break
        return T

    coarse, best = align(S := sample(raw), cKDTree(S), np.eye(4), iters), (np.inf, np.eye(4), np.inf, 0)
    whole = cKDTree(raw)
    # Missing edges can move the coarse fit by two cells: refine from the neighbouring basins too, against the raw model.
    for dx, dy in product((0, -cell, cell, -2 * cell, 2 * cell), repeat=2):
        T = align(raw, whole, rot_tl_to_tf_mat(tl=np.array([dx, dy, 0.0])) @ coarse, min(iters, 40))
        d = query(whole, T)[0]
        if (finite := np.isfinite(d)).sum() >= 10 and (score := float(np.minimum(d, near).mean())) < best[0]:
            best = (score, T, float(np.median(d[finite])), int(finite.sum()))
    _, T, res, n = best
    return T[:3, :3], T[:3, 3], res, n
