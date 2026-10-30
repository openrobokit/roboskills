import numpy as np
from scipy.spatial.transform import Rotation

from roboskills.tools.robokit.geom import flat, near, voxel_ids
from roboskills.tools.robokit.shape import fit_rigid


def test_point_set_helpers():
    P = np.array([[0.0, 0, 0], [0.001, 0, 0], [0.05, 0, 0], [np.nan, 0, 0]])
    assert len(flat({"cam": P.reshape(2, 2, 3)})) == 3
    assert voxel_ids(P[:3], 0.01).tolist() == [0, 2]  # the first two share a voxel
    assert near(P[:3], [[0.0, 0, 0]], 0.01).tolist() == [True, True, False]


def test_fit_rigid_recovers_a_small_turn_and_rejects_a_large_one():
    rng = np.random.default_rng(0)
    P = rng.uniform([-0.05, -0.03, 0.0], [0.05, 0.03, 0.04], (2000, 3))
    R0 = Rotation.from_euler("z", 6, degrees=True).as_matrix()
    R, t, quality = fit_rigid(P, P @ R0.T + [0.01, 0, 0], max_turn_deg=15)
    assert quality["accepted"] and np.isclose(quality["turn_deg"], 6, atol=0.5)
    _, _, strict = fit_rigid(P, P @ R0.T + [0.01, 0, 0], max_turn_deg=3)
    assert not strict["accepted"]
