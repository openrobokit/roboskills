"""state's pure functions on a rendered top-down camera: the table, an object's fused points, whether points are
seen, whether a lift took its object along, how far a press stopped short."""

from types import SimpleNamespace

import numpy as np
import pytest
from robokit.xform.numpy import rot_tl_to_tf_mat

from roboskills.tools.robokit import perception
from roboskills.tools.robokit.shape import fit_rigid
from roboskills.tools.robokit.state import lift_check, merge_masks, press_short, seen, table_height


K = np.array([[300.0, 0, 160], [0, 300.0, 120], [0, 0, 1]])
T_CAM = rot_tl_to_tf_mat(np.diag([1.0, -1.0, -1.0]), np.array([0.0, 0.0, 1.0]))  # looking straight down from 1 m


def render(boxes, what=("block",)):
    """(View, masks) of boxes [(x0, x1, y0, y1, bottom, top)] over a table at z = 0, a mask per box when `what` is given."""
    v, u = np.mgrid[0:240, 0:320]
    depth, masks = np.ones((240, 320)), []
    for x0, x1, y0, y1, _, top in boxes:
        d = 1.0 - top
        x, y = (u - 160) / 300 * d, -(v - 120) / 300 * d
        hit = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1) & (d < depth)
        depth[hit] = d
        masks.append(hit)
    cam = SimpleNamespace(rgb=np.zeros((240, 320, 3), np.uint8), depth=depth, K=K, T_world_cam=T_CAM)
    m = {"top": [(what[0], 0.9, h) for h in masks]} if what else {}
    return perception.View({"top": cam}, {"top": perception.cloud(cam)}), m


BLOCK = (-0.05, 0.05, -0.03, 0.03, 0.0, 0.015)  # flat: a top-down camera sees no sides
UP = (*BLOCK[:4], 0.1, 0.115)  # the block lifted 10 cm


def block_points():
    """The block's fused points on the rendered table, and the table height."""
    view, masks = render([BLOCK])
    tz = table_height(view.clouds, (-0.5, 1.0))
    [(label, score, P)] = merge_masks(view, masks, tz)
    assert label == "block" and score == pytest.approx(0.9)
    return P, tz


def test_table_and_fused_object_points():
    P, tz = block_points()
    assert tz == pytest.approx(0.0, abs=0.003) and len(P) > 100
    assert P[:, :2].min(0) == pytest.approx([-0.05, -0.03], abs=0.005)
    assert P[:, 2].mean() == pytest.approx(0.015, abs=0.003)
    share, votes = seen(render([BLOCK])[0], P)
    assert share > 0.9 and votes > 100
    assert seen(render([], what=())[0], P)[0] < 0.1  # gone: the camera sees the table through where it was


def test_lift_check_sees_the_object_in_the_hand_or_left_behind():
    P, tz = block_points()
    closed, up = rot_tl_to_tf_mat(tl=np.array([0, 0, 0.03])), rot_tl_to_tf_mat(tl=np.array([0, 0, 0.13]))
    assert lift_check(render([UP])[0], P, tz, closed, up)["lifted"] is True
    out = lift_check(render([BLOCK])[0], P, tz, closed, up)
    assert out["lifted"] is False and out["left_behind"] > 0.6
    assert lift_check(render([UP])[0], P, tz, closed, up, gripping=False)["lifted"] is False  # jaws closed on nothing


def test_prompt_crop_and_register_fit_a_held_object_to_what_the_cameras_see():
    P, tz = block_points()
    around = lambda S: S[
        ((S > P.min(0) - 0.03) & (S < P.max(0) + 0.03)).all(1) & (S[:, 2] > tz + 0.005)
    ]  # agent_as_policy/SKILL.md
    R, t, q = fit_rigid(P, around(render([(BLOCK[0] + 0.003, BLOCK[1] + 0.003, *BLOCK[2:])])[0].points))
    assert q["accepted"] and (P @ R.T + t).mean(0)[:2] == pytest.approx(P.mean(0)[:2] + [0.003, 0], abs=0.002)
    assert not fit_rigid(P, around(render([], what=())[0].points))[2]["accepted"]  # nothing above the table near it


def test_press_short_measures_how_far_the_stroke_stopped_above_its_bottom():
    hand = SimpleNamespace(press_point=np.zeros(3))  # the TCP is the press point
    model = SimpleNamespace(robot=SimpleNamespace(arms={"right": SimpleNamespace(hand=hand)}),
                            fq=lambda a, obs: np.r_[obs.q[a], obs.hand[a]], tcp=lambda a, f: rot_tl_to_tf_mat(tl=f[:3]))  # fmt: skip
    plan = SimpleNamespace(pose=SimpleNamespace(arm="right", tcp=rot_tl_to_tf_mat(tl=np.zeros(3))), end_hand={"right": np.zeros(1)},
                           ends_q=[{"right": np.array([0.0, 0, 0.004])}, {"right": np.array([0.0, 0, 0.05])}])  # down, back up  # fmt: skip
    assert press_short(model, plan) == pytest.approx(0.004)
