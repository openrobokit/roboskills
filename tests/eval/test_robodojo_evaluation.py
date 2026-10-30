"""Saved RGB evidence cannot silently gain simulator geometry or official layouts."""

from types import SimpleNamespace

import numpy as np
import pytest

from roboskills.eval.robodojo.evaluation import check_rgb_evidence, rgb_contract
from roboskills.eval.robodojo.layouts import development_layouts


@pytest.fixture
def rgb():
    robot = SimpleNamespace(
        official=True,
        cams={name: dict(rgb=np.zeros((2, 3, 3), dtype=np.uint8)) for name in ("head", "left", "right")},
        rgbd=SimpleNamespace(mono=SimpleNamespace(runtime=dict(torch="recorded runtime"))),
    )
    return robot, dict(cam_head_rgb=robot.cams["head"]["rgb"])


def test_rgb_contract_accepts_estimated_geometry_without_exposing_native_geometry(rgb):
    robot, arrays = rgb
    contract = rgb_contract(robot)
    assert contract["depth_runtime"] == dict(torch="recorded runtime")
    assert len(contract["depth_worker_sha256"]) == 64
    check_rgb_evidence(contract, arrays)


@pytest.mark.parametrize("field", ["depth", "K", "T"])
def test_live_rgb_contract_rejects_private_geometry(rgb, field):
    robot, _ = rgb
    robot.cams["left"][field] = []
    with pytest.raises(RuntimeError, match="forbidden simulator"):
        rgb_contract(robot)


@pytest.mark.parametrize("change", ["depth", "K", "T", "unverified_contract", "missing_rgb", "wrong_dtype"])
def test_saved_evidence_rejects_information_leaks_and_unverified_rgb(rgb, change):
    robot, arrays = rgb
    contract = rgb_contract(robot)
    if change in {"depth", "K", "T"}:
        arrays["cam_head_" + change] = np.zeros((2, 3))
    elif change == "unverified_contract":
        del contract["simulator_object_poses"]
    elif change == "missing_rgb":
        arrays.clear()
    else:
        arrays["cam_head_rgb"] = arrays["cam_head_rgb"].astype(float)
    with pytest.raises(ValueError):
        check_rgb_evidence(contract, arrays)


def test_declared_development_extension_preserves_official_exclusion():
    assert development_layouts("35,36,38,40,41", stop=55) == [35, 36, 38, 40, 41]
    assert development_layouts([10, 29]) == [10, 29]
    for layouts in ([0, 4], [10, 10], [55], [], [True]):
        with pytest.raises(ValueError):
            development_layouts(layouts, stop=55)
