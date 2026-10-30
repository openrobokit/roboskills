"""Cameras -> world points: `view` is the one way an observation comes in, as a `View` of what the cameras show;
`state.merge_masks` fuses per-camera SAM 3 masks (roboskills.tools.sam3) into objects."""

from dataclasses import dataclass

import numpy as np

from roboskills.tools.robokit.geom import flat, near


def cloud(cam, edge=0.015):
    """World xyz of every pixel (H, W, 3); NaN where depth is missing or the pixel lies more than `edge` m (scaled with
    range) behind a neighbour: the far side of a depth jump (flying pixels at outlines). The near side is kept."""
    from robokit.xform.numpy import transform_points

    bad = ~np.isfinite(d := np.asarray(cam.depth, float)) | (d <= 0)
    d, far = np.where(bad, 0.0, d), np.where(bad, np.inf, d)
    with np.errstate(invalid="ignore"):  # how far each pixel lies behind its nearest neighbour
        dv, dh = (np.nan_to_num(np.diff(far, axis=k), posinf=0, neginf=0) for k in (0, 1))
    behind = np.zeros_like(d)
    for sl, x in (((slice(1, None),), dv), ((slice(None, -1),), -dv), ((slice(None), slice(1, None)), dh),
                  ((slice(None), slice(None, -1)), -dh)):  # fmt: skip
        behind[sl] = np.fmax(behind[sl], x)
    (v, u), K = np.mgrid[0 : d.shape[0], 0 : d.shape[1]], cam.K
    P = transform_points(np.stack([(u - K[0, 2]) / K[0, 0] * d, (v - K[1, 2]) / K[1, 1] * d, d], -1), cam.T_world_cam)
    P[bad | (behind > edge * np.maximum(d, 0.5))] = np.nan
    return P


@dataclass(frozen=True)
class View:
    """What one camera observation shows, both {cam: ...} over the cameras with depth: `cams` the `Camera`s (rgb,
    depth, K, OpenCV T_world_cam) and `clouds` world xyz of every pixel (H, W, 3), NaN on the robot and missing depth.
    Robot readings come from `Tools.status`."""

    cams: dict
    clouds: dict

    @property
    def points(self):
        """(N, 3) the finite points of every cloud, one in three."""
        return flat(self.clouds)[::3]


def view(model, obs, clearance=0.012):
    """The `View` of observation `obs` of the robot `model` is of: the runtime's `robot_mask`, else the robot's joints
    (its points within `clearance` m), mask the robot out of every cloud."""
    robot, robot_pts, clouds = model.robot, model.robot_points(obs), {}
    for name, cam in obs.cams.items():
        if cam.depth is None:
            continue
        P = cloud(cam)
        if (m := robot.robot_mask(name, obs) if hasattr(robot, "robot_mask") else None) is not None:
            P[np.asarray(m, bool)] = np.nan
        elif robot_pts is not None and len(robot_pts):
            flat, ok = P.reshape(-1, 3), np.flatnonzero(np.isfinite(P).all(2).ravel())
            flat[ok[near(flat[ok], robot_pts, clearance)]] = np.nan
        clouds[name] = P
    return View({c: obs.cams[c] for c in clouds}, clouds)
