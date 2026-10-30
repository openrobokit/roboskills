"""Object shape and registration: the oriented box and the rigid fits the tools use."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from roboskills.tools.robokit import shape
from roboskills.tools.robokit.shape import register_rigid


def test_registration_recovers_roll_pitch_and_ignores_pixel_density():
    x, y = np.mgrid[-0.04:0.041:0.002, -0.04:0.021:0.002]
    keep = (y >= 0) | (np.abs(x) < 0.011)
    top = np.c_[x[keep], y[keep], np.full(keep.sum(), 0.015)]
    P = np.concatenate([top, top - [0, 0, 0.015]]) + [0.1, -0.2, 0.85]
    R = Rotation.from_euler("xyz", [3, -5, 12], degrees=True).as_matrix()
    c = P.mean(0)
    O = (P - c) @ R.T + c + [0.003, -0.002, 0.001]
    R1, t1, res, n = register_rigid(P, O)
    assert np.degrees(Rotation.from_matrix(R1 @ R.T).magnitude()) < 0.5
    assert np.linalg.norm(P @ R1.T + t1 - O, axis=1).mean() < 0.0005 and res < 0.001 and n > 40
    repeated = np.concatenate([O, np.repeat(O[:150], 50, axis=0)])
    R2, t2, _, _ = register_rigid(P, repeated)
    assert np.allclose(R1, R2) and np.allclose(t1, t2)  # a close camera cannot dominate just by duplicating pixels


@pytest.mark.parametrize("yaw", [3.2, 31.41, 89.6, 119.738, 176.2])
def test_describe_yaw_ignores_pixel_density_and_stray_points(yaw):
    x, y = np.mgrid[-0.04:0.041:0.002, -0.02:0.021:0.002]
    c, s = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
    P = np.c_[np.c_[x.ravel(), y.ravel()] @ np.array([[c, s], [-s, c]]), np.full(x.size, 0.77)] + [0.3, -0.1, 0]
    box = shape.describe(P, 0.75)
    assert abs((box.yaw - yaw + 90) % 180 - 90) <= 0.25
    assert box.size[:2] == pytest.approx([0.08, 0.04], abs=0.003)
    assert box.center[:2] == pytest.approx([0.3, -0.1], abs=0.001)
    repeated = np.concatenate([P, np.repeat(P[:100], 50, axis=0)])  # a close camera's many pixels of one corner
    stray = np.concatenate([P, [[0.6, 0.2, 0.77], [0.0, 0.2, 0.77], [0.6, -0.4, 0.77], [0.0, -0.4, 0.77]]])
    for Q in (repeated, stray):
        other = shape.describe(Q, 0.75)
        assert other.yaw == box.yaw and np.allclose(other.size[:2], box.size[:2], atol=0.002)


def test_describe_sizes_a_line():
    t = np.linspace(0, 0.1, 50)
    P = np.c_[t * np.cos(np.radians(30)), t * np.sin(np.radians(30)), np.full(50, 0.8)]  # a line: no hull
    d = shape.describe(P, 0.75)
    assert abs(d.yaw - 30) <= 1 and d.size[0] > 0.09 and d.size[1] < 0.005
    assert np.allclose(d.center[:2] - P[:, :2].mean(0), 0, atol=0.002)  # arrays: the agent does arithmetic on them
