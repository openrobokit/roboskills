"""Point-set helpers the RoboKit tools share: neighbourhoods, voxels, flattened clouds and connected components, metres,
world frame. Transforms and rotations come from `robokit.xform.numpy` directly."""

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree


def voxel_ids(P, cell):
    """Index of one point of P (N, D) per `cell`-sized voxel, in voxel order."""
    return np.unique(np.floor(np.asarray(P, float) / cell).astype(np.int64), axis=0, return_index=True)[1]


def near(P, X, radius):
    """(N,) bool: is each point of P (N, 3) within `radius` m of a point of X (M, 3)?"""
    return np.isfinite(cKDTree(X).query(P, distance_upper_bound=radius)[0])


def flat(clouds):
    """All finite points of per-camera clouds {cam: (H, W, 3)} as one (N, 3) array."""
    P = np.concatenate([c.reshape(-1, 3) for c in clouds.values()] or [np.zeros((0, 3))])
    return P[np.isfinite(P).all(1)]


def clusters(P, cell=0.01, min_pts=20):
    """3-D connected components of points (`cell` voxels, 26-connected), largest first: index arrays."""
    if len(P) == 0:
        return []
    ijk = (ijk := np.floor(P / cell).astype(np.int64)) - ijk.min(0)
    if np.prod(ijk.max(0) + 1) > 4e7:  # too spread out for a dense grid: coarser cells
        return clusters(P, cell * 2, min_pts)
    grid = np.zeros(ijk.max(0) + 1, bool)
    grid[tuple(ijk.T)] = True
    ids = ndimage.label(grid, structure=np.ones((3, 3, 3)))[0][tuple(ijk.T)]
    return sorted([c for k in range(1, ids.max() + 1) if len(c := np.flatnonzero(ids == k)) >= min_pts], key=len, reverse=True)  # fmt: skip
