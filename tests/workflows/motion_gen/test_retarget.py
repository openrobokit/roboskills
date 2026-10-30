"""The person's turns onto a character (roboskills.workflows.motion_gen.character.retarget), without a solve."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation


pytest.importorskip("warp")
pytest.importorskip("trimesh")

from roboskills.workflows.motion_gen.character.character import Character  # noqa: E402
from roboskills.workflows.motion_gen.character.retarget import Y_TO_Z, camera_side, targets, weights  # noqa: E402


def tiny() -> Character:
    names = ("m:Hips", "m:Spine", "m:LeftHandIndex1", "m:LeftFoot")
    roles = {"Hips": names[0], "Spine": names[1], "LeftHandIndex1": names[2], "LeftFoot": names[3]}
    rest = np.array([[0, 0, 1.0], [0, 0, 1.2], [0.5, 0, 1.2], [0.1, 0, 0.1]])
    return Character("tiny", names, np.array([-1, 0, 1, 0]), rest, np.zeros((0, 3)), np.zeros((0, 3), int), np.zeros((0, 4)), roles)  # fmt: skip


def test_the_person_turns_about_up_after_the_change_of_basis():
    about_y = Rotation.from_euler("y", 30, degrees=True).as_matrix()  # GVHMR's up
    expected = Rotation.from_euler("z", 30, degrees=True).as_matrix()
    assert np.allclose(Y_TO_Z @ about_y @ Y_TO_Z.T, expected)
    assert np.allclose(Y_TO_Z @ [0, 0, 1.0], [0, -1, 0])  # a T-pose's front is a Mixamo rest's


def test_a_bone_turns_as_its_joint_or_else_as_its_parent():
    names = ["pelvis", "spine1", "left_ankle"]
    world = Rotation.from_euler("xyz", np.arange(9).reshape(3, 3), degrees=True).as_matrix()[None]
    out = targets(tiny(), world, names)
    assert np.allclose(out[0, 0], world[0, 0]) and np.allclose(out[0, 1], world[0, 1])
    assert np.allclose(out[0, 3], world[0, 2]) and np.allclose(out[0, 2], out[0, 1])  # the finger follows the spine


def test_weights_cover_a_bones_both_sides_by_full_name():
    c = tiny()
    c.roles.update(RightFoot="m:RightFoot", Spine2="m:Spine2", **{f"{s}{r}": f"m:{s}{r}" for s in ("Left", "Right") for r in ("UpLeg", "ToeBase", "Arm", "Hand")})  # fmt: skip
    found = weights(c)
    assert found["m:Hips"] == (30, 2) and found["m:LeftFoot"] == found["m:RightFoot"] == (30, 2)
    assert len(found) == 12


def test_camera_side_is_where_the_camera_stands_from_the_person(tmp_path):
    # The person faces the camera in the picture (turned half about y) and the world's +z: the camera stands at +z.
    n = 3
    np.savez(tmp_path / "m.npz", world_global_orient=np.zeros((n, 3)), camera_transl=np.tile([0.1, 0, 5.0], (n, 1)),
             camera_global_orient=np.tile([0, np.pi, 0], (n, 1)))  # fmt: skip
    assert np.allclose(camera_side(tmp_path / "m.npz"), Y_TO_Z @ [0, 0, 1.0], atol=0.03)
