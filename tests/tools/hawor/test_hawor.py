"""HaWoR's root path smoothing: Do as I Do's jump-and-back removal, then Savitzky-Golay."""

import numpy as np
from scipy.spatial.transform import Rotation

from roboskills.tools.hawor.hawor import smooth_path


def test_smooth_path_drops_a_spike_and_keeps_a_smooth_path():
    t = np.linspace(0, 1, 30)
    trans, rotvec = np.stack([t, t**2, 0.5 + 0 * t], 1), np.stack([0 * t, 0 * t, t], 1)
    noisy = trans.copy()
    noisy[12] += [0, 0, 0.3]  # out and back in one frame
    smoothed, angles = smooth_path(noisy, rotvec)
    assert np.abs(smoothed - trans).max() < 1e-3
    assert np.allclose(Rotation.from_rotvec(angles).as_rotvec(), rotvec, atol=1e-6)


def test_smooth_path_takes_short_runs():
    for n in (1, 2, 3, 5):
        trans, rotvec = np.random.default_rng(n).normal(size=(2, n, 3)) * 0.1
        assert smooth_path(trans, rotvec)[0].shape == (n, 3)
