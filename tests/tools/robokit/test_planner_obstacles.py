"""The planner on fake models: the table it keeps clear of, the segments of each verb, one RoboKit call per arm and
shape, and the Trajectories it hands the robot."""

from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from roboskills.tools.robokit.action import Pose
from roboskills.tools.robokit.planner import HOME, VOXEL, Planner, segment_traj


def now_of(arms, hand):
    """A live measurement (`Obs`) of fixed `arms` at joint 0 whose hands read `hand`."""
    return SimpleNamespace(q={a: np.zeros(1) for a in arms}, hand={a: np.asarray(hand, float) for a in arms},
                           base=None, body=None)  # fmt: skip


def at(z):
    """A TCP pose at height z."""
    return np.c_[np.eye(4)[:, :3], [0, 0, z, 1]]


def tcp_at_joint(a, fq):
    """One joint per arm: its TCP sits at the joint's value along z."""
    fq = np.asarray(fq, float)
    T = np.broadcast_to(np.eye(4), fq.shape[:-1] + (4, 4)).copy()
    T[..., 2, 3] = fq[..., 0]
    return T


def model_of(arms=("left",)):
    hand = SimpleNamespace(kind="jaw", closed=np.zeros(1), open=np.ones(1), T_ee_tcp=np.eye(4))
    return SimpleNamespace(robot=SimpleNamespace(arms={a: SimpleNamespace(hand=hand, vmax=1.0, home=np.full(1, 0.3), ee_link="ee")
                                                       for a in arms}),
                           info={a: SimpleNamespace(idx=[0]) for a in arms}, tcp=tcp_at_joint,
                           set_arm=lambda a, fq, q: np.asarray(q, float).reshape(np.shape(fq)),
                           fq=lambda a, obs: np.asarray(obs.q[a], float))  # fmt: skip


def faked(model, blocked=()):
    """A Planner whose RoboKit path runs through each waypoint in one frame; a straight segment to a height in
    `blocked` fails. Its calls: (arm, rows, locked flags)."""
    planner, calls = Planner(model), []

    def path(a, now, rows, table, scene):
        calls.append((a, len(rows), tuple(s.locked for s in rows[0])))
        z = lambda s: 0.3 if s.goal is HOME else s.goal[2, 3]
        q = np.array([[[now.q[a][0]], *[[z(s)] for s in S]] for S in rows])
        ok = np.array([not any(s.locked and z(s) in blocked for s in S) for S in rows])
        return q, tuple(range(len(rows[0]) + 1)), ok

    planner._path = path
    return planner, calls


def test_each_verb_is_its_segments():
    o, c = np.ones(1), np.zeros(1)
    grasp = segment_traj(Pose("grasp", "left", tcp=at(0.1), ee_pre=at(0.2), hand_pre=o, hand=c))
    assert [(s.locked, s.check, s.follow) for s in grasp] == [
        (False, True, True),
        (True, False, True),
        (True, False, False),
    ]
    assert grasp[1].then.tolist() == [0.0] and grasp[2].goal[2, 3] == 0.1 + 0.1  # closes there, then a lift is checked
    place = segment_traj(Pose("place", "left", tcp=at(0.1), ee_pre=at(0.2), hand=o))
    assert [s.hand for s in place[:2]] == [None, None] and place[1].speed == 0.3 and place[1].then.tolist() == [1.0]
    assert place[2].goal[2, 3] == 0.2 and place[2].hand.tolist() == [1.0] and place[2].then is None  # back up, open
    press = segment_traj(Pose("press", "left", tcp=at(0.1), ee_pre=at(0.2), hand_pre=c))
    assert press[1].then is None and press[2].then.tolist() == [0.0] and press[2].hand.tolist() == [0.0]  # waits up
    release = segment_traj(Pose("release", "left", tcp=at(0.15), hand_pre=o))
    assert len(release) == 1 and release[0].locked and not release[0].check and release[0].before.tolist() == [1.0]
    (move,) = segment_traj(Pose("move", "left", tcp=at(0.15)))
    assert not move.locked and move.check and move.hand is None and move.before is None
    (push,) = segment_traj(Pose("move", "left", tcp=at(0.15), rank="line"))
    assert push.locked and not push.check
    (home,) = segment_traj(Pose("home", "left", tcp=at(0.15)))
    assert home.goal is HOME and home.check


def test_one_robokit_call_per_arm_and_shape_and_a_result_for_every_pose():
    planner, calls = faked(model_of(("left", "right")), blocked={0.1})
    grasp = lambda a, z: Pose("grasp", a, tcp=at(z), ee_pre=at(z + 0.1), hand_pre=np.ones(1), hand=np.zeros(1), rank=z)
    plans = planner.plan([grasp("left", 0.1), grasp("left", 0.2), grasp("right", 0.2), Pose("move", "left", tcp=at(0.4))],
                         now_of(("left", "right"), np.ones(1)))  # fmt: skip
    assert plans[0] is None and plans[1].pose.rank == 0.2 and plans[2].pose.arm == "right" and plans[3] is not None
    assert [c[:2] for c in calls] == [("left", 2), ("right", 1), ("left", 1)]  # left grasps, right grasps, left move
    assert plans[1].q_start["left"].tolist() == [0.0] and plans[1].q_start["right"].tolist() == [0.0]


def test_the_paths_a_robot_follows():
    planner, calls = faked(model_of(), blocked={0.05, 0.06})
    now = now_of(["left"], np.ones(1))
    (press,) = planner.plan([Pose("press", "left", tcp=at(0.1), ee_pre=at(0.2), hand_pre=np.zeros(1))], now)
    moves = [p for p in press.paths if p.q]
    assert [p.q["left"][-1, 0] for p in moves] == [0.1, 0.2]  # in through the hover and down (a follow ends there), up
    assert 0.2 in moves[0].q["left"][:, 0] and np.all(np.diff(moves[0].t) > 0)  # one Trajectory through the pre-pose
    assert not press.paths[-1].q and press.paths[-1].hand["left"][0, 0] == 0.0 and press.tcp_end[2, 3] == 0.2  # waits
    (place,) = planner.plan([Pose("place", "left", tcp=at(0.1), ee_pre=at(0.2), hand=np.ones(1))], now)
    down, opens, back = place.paths  # a robot leads its hand: the opening waits for the arm, in place
    assert down.q and not down.hand and not opens.q and opens.hand["left"][0, 0] == 1.0 and back.q["left"][-1, 0] == 0.2
    assert all(p.t[0] > 0 for p in place.paths)
    (fast,) = planner.plan([Pose("move", "left", tcp=at(0.1))], now)  # 0.1 at full speed, the place's at 0.3 of it
    slow = down.t[-1] - down.t[np.argmax(down.q["left"][:, 0])]  # its 0.1 down from the pre-pose
    assert slow > fast.paths[0].t[-1]
    (grasp,) = planner.plan([Pose("grasp", "left", tcp=at(0.1), ee_pre=at(0.2), hand=np.zeros(1))], now)
    assert grasp.paths[-1].hand["left"][-1, 0] == 0.0 and grasp.paths[-2].q["left"][-1, 0] == 0.1  # the lift: not run
    calls.clear()
    (release,) = planner.plan([Pose("release", "left", tcp=at(0.05), hand_pre=np.ones(1))], now)
    assert [c[2] for c in calls] == [(True,), (False,)]  # out of straight reach: planned around instead
    opens, up = release.paths  # the hand opens in place, then the arm backs off
    assert not opens.q and opens.hand["left"][0, 0] == 1.0 and up.q["left"][-1, 0] == 0.05
    calls.clear()
    assert planner.plan([Pose("move", "left", tcp=at(0.06), rank="line")], now) == [None] and len(calls) == 1  # a push
    (home,) = planner.plan([Pose("home", "left", tcp=at(0.9))], now)
    assert home.paths[0].q["left"][-1, 0] == 0.3  # to the arm's home joints, not the TCP pose they give


def test_a_pose_without_a_hand_command_sends_none():
    """The robot keeps its hand as last commanded (the Robot.follow contract), so a move carrying a held object sends
    no hand command at all; the plan keeps the joints it was planned from."""
    planner, _ = faked(model_of())
    (p,) = planner.plan([Pose("move", "left", tcp=at(0.2), rank="move")], now_of(["left"], [0.3]))
    assert [path.hand for path in p.paths] == [{}] and p.q_start["left"].tolist() == [0.0]


def test_a_waypoint_off_its_pose_fails_and_the_grid_holds_the_table_and_the_scene_in_the_model_frame():
    """`_path` checks every pose waypoint against its goal; the grid gets the table top over its footprint and the
    observed points less those at a hand, in the model's frame (a base turned about the vertical keeps the table's
    height for the spheres it cuts, a tilted one drops it)."""
    model = model_of()
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    T[:3, 3] = [1.0, 0.0, 0.3]
    model.info["left"].M = SimpleNamespace(root=SimpleNamespace(T_world_base=T))
    model.device = "cpu"
    seen = []

    class RoboKit:
        def solve_offline_numpy(self, W, segment_locked, segment_check_collision):
            goals = [W[j]["ee"][:, 2, 3] + T[2, 3] for j in (1, 2)]  # the goals come in the model frame
            q = np.stack([W[0][:, 0], goals[0], goals[1] + np.r_[0.0, 0.006]], 1)[..., None]
            return SimpleNamespace(q_traj=q, waypoint_indices=(0, 1, 2))

        def compute_metrics_numpy(self, q, T_world_target=None, target_q=None):
            assert (q[:, 2] == q[:, 1]).all()  # the unchecked segment held at the checked one's end
            assert len(q) == len(T_world_target) == 16  # padded: RoboKit's evaluator keeps the batch size it first saw
            return SimpleNamespace(success=np.ones(len(q), bool))

    planner, grid = Planner(model), []
    volume = SimpleNamespace(update=lambda points, reset: grid.append(points))
    planner._planner = lambda a, size, z, fq: seen.append(z) or (RoboKit(), volume)
    rows = [segment_traj(Pose("place", "left", tcp=at(z), ee_pre=at(z + 0.1)))[:2] for z in (0.1, 0.2)]
    square = np.array([[1.0, 0, -1.2], [-1.0, 0, 0.8], [0, 1.0, -0.2], [0, -1.0, -0.2]])  # 0.8 <= x <= 1.2, |y| <= 0.2
    scene = np.array([[0.0, 0.0, 0.05], [1.0, 0.0, 0.6], [np.nan, 0, 0]])  # by the hand (its TCP at 0), apart, invalid
    _, _, ok = planner._path("left", now_of(["left"], [0.3]), rows, (0.8, square), scene)
    assert ok.tolist() == [True, False] and np.isclose(seen[-1], 0.5)  # 6 mm off its tcp
    world = grid[-1] @ T[:3, :3].T + T[:3, 3]
    top = np.isclose(world[:, 2], 0.8 - VOXEL)  # a cell under the top: the grid surface stands a cell over its points
    table, rest = world[top], world[~top]
    assert np.allclose(rest, [[1.0, 0.0, 0.6]]) and 40 * 40 <= len(table) <= 41 * 41  # the hand's point is its own
    assert table[:, 0].min() > 0.8 - 1e-6 and table[:, 0].max() < 1.2 + 1e-6 and np.abs(table[:, 1]).max() < 0.2 + 1e-6
    T[:3, :3] = Rotation.from_euler("x", 5, degrees=True).as_matrix()
    planner._path("left", now_of(["left"], [0.3]), rows, (0.8, square))
    assert seen[-1] is None


def test_table_is_an_up_facing_level_not_a_wall_seen_at_eye_level():
    from roboskills.tools.robokit.state import facing_up, table_height

    u, v = np.meshgrid(np.linspace(0, 1, 60), np.linspace(0, 1, 60))
    table = np.stack([u, v, np.full_like(u, 0.8)], -1)  # a 1 m^2 table top
    wall = np.stack([u, np.full_like(u, 2.0), 1.0 + 0.4 * v], -1)  # a wall facing the camera
    wall[:, :, 2] = np.where(np.abs(wall[:, :, 2] - 1.4) < 0.02, 1.4, wall[:, :, 2])  # dense at the camera's height
    assert facing_up(table)[1:-1, 1:-1].all() and not facing_up(wall).any()
    assert np.isclose(table_height({"a": table, "b": wall}, band=(0.5, 1.9)), 0.8, atol=0.002)
    floor = table.copy()
    floor[..., 2] = 0.0
    assert table_height({"a": floor}, band=(0.15, 1.9)) == 0.15  # no surface in reach is in view: not the floor


def test_planning_spheres_drop_those_under_the_table_at_home():
    from dataclasses import make_dataclass

    Spec = make_dataclass(
        "Spec", ["collision_spheres_link_indices", "local_collision_sphere_centers", "collision_sphere_radii"]
    )
    spec = Spec(np.array([0, 1, 2]), np.array([[0, 0, -0.05], [0, 0, 0.2], [0, 0, 0.3]]), np.array([0.02, 0.02, 0.02]))
    rk = SimpleNamespace(link_names=["base", "arm", "hand"], spec=spec,
                         forward_kinematics_via_matrix_numpy=lambda q, device: np.repeat(np.eye(4)[None, None], 3, 1))  # fmt: skip
    planner = Planner(SimpleNamespace(info={"left": SimpleNamespace(M=SimpleNamespace(root=SimpleNamespace(T_world_base=np.eye(4)), rk=rk), idx=[0])},
                                      robot=SimpleNamespace(arms={"left": SimpleNamespace(home=np.zeros(1))}),
                                      set_arm=lambda a, f, q: f))  # fmt: skip
    assert planner._robot("left", np.zeros(1), 0.0).spec.collision_spheres_link_indices.tolist() == [
        1,
        2,
    ]  # the base's is under


def test_the_planner_passes_robokit_the_keywords_it_takes():
    """`_planner`'s MotionPlanner(...) keywords are RoboKit's (a rename once sent state= and every plan raised before
    the robot moved; the fakes here never reach RoboKit)."""
    import inspect
    import re

    import pytest

    mp = pytest.importorskip("robokit.helpers.motion_plan")
    call = inspect.getsource(Planner._planner).split("MotionPlanner(", 1)[1].split("\n", 1)[0]
    used = set(re.findall(r"(\w+)=", call))
    assert used and used <= set(inspect.signature(mp.MotionPlanner.__init__).parameters), used
