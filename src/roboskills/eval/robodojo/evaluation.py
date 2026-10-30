"""RoboDojo RGB evidence shared by evaluation runners, without starting a simulator.

Campaigns own policies, cohorts, deadlines and scores. These helpers retain the
native RGB/proprioception boundary and provenance of geometry inferred from it.
"""

import hashlib

import numpy as np

from roboskills.eval.robodojo.rgbd import CAMS_JSON


def rgb_contract(robot):
    """Describe inferred geometry only after verifying native cameras contain RGB alone."""
    if not robot.official or any(set(camera) != {"rgb"} for camera in robot.cams.values()):
        raise RuntimeError("RGB episode received forbidden simulator camera geometry")
    return dict(
        input="RGB + robot proprioception",
        simulator_depth=False,
        simulator_camera_calibration=False,
        simulator_object_poses=False,
        depth_estimator="MoGe-2 ViT-L; affine scale fitted to the robot-base table plane",
        calibration="Offline self-calibration from development RGB and robot joints; wrist forward kinematics",
        calibration_sha256=hashlib.sha256(CAMS_JSON.read_bytes()).hexdigest(),
        depth_worker_sha256=hashlib.sha256(CAMS_JSON.with_name("depth_worker.py").read_bytes()).hexdigest(),
        depth_runtime=robot.rgbd.mono.runtime,
        estimates_log="rgbd.jsonl",
    )


def check_rgb_evidence(contract, arrays):
    """Reject disclosed native geometry or malformed RGB in an already saved NPZ mapping.

    Runtime version pins and campaign-specific files remain the caller's checks.
    Does not repair evidence or infer a task outcome from camera contents.
    """
    private = ("simulator_depth", "simulator_camera_calibration", "simulator_object_poses")
    if any(contract.get(name) is not False for name in private):
        raise ValueError("RGB contract permits or does not verify simulator geometry")
    if "cam_head_rgb" not in arrays or any(name.endswith(("_depth", "_K", "_T")) for name in arrays):
        raise ValueError("Native input was not RGB-only")
    rgb = np.asarray(arrays["cam_head_rgb"])
    if rgb.ndim != 3 or rgb.shape[-1] != 3 or min(rgb.shape[:2]) < 1 or rgb.dtype != np.uint8:
        raise ValueError("Native head camera must contain uint8 RGB pixels")
