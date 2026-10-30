"""Planning on a `RobotModel`, on RoboKit's motion planner: `Planner.plan(poses, robot_now, table)` gives each `Pose`
its `Plan`, the robot `Trajectory`s reaching it, or None where RoboKit finds no path. An arm pose is a few segments
(`segment_traj`): collision-free to `pose.ee_pre`, a straight joint line in to `pose.tcp`, a hand change held still,
the way back, a lift planned but not run; poses of one arm and one shape are one RoboKit batch, all their segments in
one call, timed here at the arm's speed (`_rows`: RoboKit gives the path). A base or body pose is a smooth start-stop
ride. It starts from the live measurement `robot_now` (an `Obs`) and keeps clear of the robot itself and of one grid
holding the table and the observed points it is given (`_obstacles`), not yet of what a hand holds. Nothing here moves the robot or knows about objects:
`Tools.execute` follows a Plan's `paths` and writes the `Follow`s back (`Plan.record`).

Every trajectory is sampled every 0.02 s from its start (t excludes 0)."""

from dataclasses import dataclass, replace
from os import PathLike
from typing import Optional

import numpy as np

from roboskills.eval.interface import Trajectory
from roboskills.tools.robokit.action import BACK, LIFT, LINE, SLOW, Pose


DWELL = 0.5  # s a hand command change is held still for
SIZES = (16,)  # the planner batch size: RoboKit allocates per size (0.2 GB at 16, 0.8 GB at 64, 12 GB at 1024) and solves 16 as fast as 4 (0.1 s warm), so every batch is padded to 16 and longer lists go in chunks  # fmt: skip
HOME = "home"  # a segment goal: the arm's home joints, not a TCP pose
VOXEL = 0.01  # m, a cell of the obstacle grid: at 2 cm a 4 cm bottle grew enough to fail half the grasps around it
REACH = 0.9  # m the grid spans each way from the arm's model frame
CARVE = 0.1  # m: observed points this near a hand are its own (the hand, what it holds, what it touches)


def _timing(rows, vmax, ramp=0.2):
    """Seconds (M - 1,) from now to each of rows[1:] (joint rows, row 0 = now): a trapezoidal speed profile along the
    path, ramping up and down over at most `ramp` s, cruising at joint speed `vmax` (scalar or per joint)."""
    s = np.r_[0.0, np.cumsum((np.abs(np.diff(rows, axis=0)) / vmax).max(1))]  # path in seconds at full speed
    if s[-1] < 1e-9:
        return np.linspace(0.0, 0.1, len(rows))[1:]
    ta, end = min(ramp, s[-1]), s[-1]
    ramps = np.where(s < ta / 2, np.sqrt(2 * ta * s), end + ta - np.sqrt(2 * ta * np.maximum(end - s, 0.0)))
    return np.where((s >= ta / 2) & (s <= end - ta / 2), s + ta / 2, ramps)[1:].clip(1e-3)


def _rows(Q, vmax, dur=0.0):
    """(t, q): joint waypoints Q (row 0 = the start; one row: standing still) on a trapezoidal speed profile at `vmax`,
    at least `dur` long, sampled every 0.02 s after the start."""
    t = np.r_[0.0, _timing(Q, vmax)] if len(Q) > 1 else np.zeros(1)
    t[-1] = max(t[-1], dur, 0.02)
    tt = np.linspace(0, t[-1], max(2, int(np.ceil(t[-1] / 0.02))) + 1)[1:]
    return tt, np.stack([np.interp(tt, t, Q[:, j]) for j in range(Q.shape[1])], 1)


@dataclass
class Plan:
    """A chosen `Pose`, the robot contract's `Trajectory`s reaching it, in order, and, once executed, what came of it
    (None until then). An arm plan has one Trajectory per motion segment and one per hand change or wait held still: a
    robot commands its hand ahead of the path, so a closure must not start before the arm arrives."""

    pose: "Pose"
    paths: list  # [Trajectory], times from each one's start
    tcp_end: Optional[np.ndarray] = None  # (4, 4) world TCP where an arm plan ends (a press or place: its pre-pose)
    q_start: Optional[dict] = None  # {arm: joints} measured when it was planned: execute refuses it once they moved
    steps: Optional[int] = None
    end_q: Optional[dict] = None
    ends_q: Optional[list] = None  # each trajectory's end joints, in order
    end_hand: Optional[dict] = None
    end_base: Optional[np.ndarray] = None
    end_height: Optional[float] = None
    ended: Optional[bool] = None
    stopped: Optional[bool] = None
    done: Optional[bool] = None

    def record(self, follows, ended):
        """Write what `robot.follow` returned for each trajectory run (`Follow`s, in order) and whether the episode has
        `ended` since into this plan."""
        o = follows[-1].obs
        self.steps, self.done = sum(f.steps for f in follows), follows[-1].done
        self.stopped, self.ended = any(f.stopped for f in follows), bool(ended)
        self.end_q, self.end_hand, self.end_base, self.end_height = dict(o.q), dict(o.hand), o.base, o.body
        self.ends_q = [dict(f.obs.q) for f in follows]


@dataclass(frozen=True)
class Segment:
    """One RoboKit segment of an arm pose, and what is held still around it."""

    goal: object  # (4, 4) world TCP pose, or HOME
    locked: bool  # a straight joint line
    check: bool  # kept clear of the table and the robot (the others start or end in contact)
    hand: Optional[np.ndarray] = None  # hand command on the way; None: none sent, the robot keeps its last
    speed: float = 1.0  # share of the arm's speed
    before: Optional[np.ndarray] = None  # hand command set and held still DWELL s before it moves
    then: Optional[np.ndarray] = None  # hand command set (or kept) and held still DWELL s once there
    follow: bool = True  # False: planned, not run (the lift that makes a pick a pick)


def segment_traj(p):
    """The segments of arm Pose `p`. With an ee_pre: collision-free there, then straight in; without: collision-free
    to the tcp (straight for LINE verbs and rank "line"); `hand_pre` on the way (set first, held still, without an
    ee_pre), `hand` once at the tcp; press and place come straight back up to the ee_pre (a press then waits there: a
    spring button counts the next stroke only once fully back up), grasp and pinch check a lift."""
    pre = None if p.hand_pre is None else np.asarray(p.hand_pre, float)
    end = pre if p.hand is None or (pre is not None and np.array_equal(p.hand, pre)) else np.asarray(p.hand, float)
    if p.ee_pre is not None:
        out = [Segment(p.ee_pre, False, True, pre), Segment(p.tcp, True, False, pre, SLOW.get(p.verb, 1.0))]
    else:
        straight = p.verb in LINE or p.rank == "line"
        out = [Segment(HOME if p.verb == "home" else p.tcp, straight, not straight, pre, before=pre)]
    out[-1] = replace(out[-1], then=None if end is pre else end)
    if p.verb in BACK:
        out.append(Segment(p.ee_pre, True, False, end, then=end if p.verb == "press" else None))
    if p.verb in LIFT:
        up = np.array(p.tcp, float)
        up[2, 3] += LIFT[p.verb]  # straight up, the rotation unchanged
        out.append(Segment(up, True, False, end, follow=False))
    return out


def shape(segments):
    """What one RoboKit batch shares: per segment, a joint goal or a pose, straight or not, checked or not."""
    return tuple((isinstance(s.goal, str), s.locked, s.check) for s in segments)


class Planner:
    _planners = {}  # RoboKit's planners by (description, arm, size, table or not), shared by every Planner in the process

    def __init__(self, model):
        self.model = model

    def plan(self, poses, robot_now, table=None, scene=None):
        """The `Plan` of each of `poses`, in order, reached whole from `robot_now` (the live measurement: an `Obs` with
        q, hand, base, body); None where RoboKit's planner finds no path. `table`: (z, footprint equations or None);
        `scene`: observed world points (N, 3) to keep clear of on the way (None: none).
        Poses of one arm and `shape` are planned as one RoboKit batch; a lift or release out of straight reach is
        planned around instead; a base or body pose is always reached."""
        poses = list(poses)
        q_start = {a: np.array(q, float) for a, q in robot_now.q.items()}
        out, groups = [None] * len(poses), {}
        for k, p in enumerate(poses):
            if p.arm is None:
                out[k] = self._ride(p, robot_now)
            else:
                S = segment_traj(p)
                groups.setdefault((p.arm, shape(S)), []).append((k, S))
        for (a, _), rows in groups.items():
            for (k, _), plan in zip(rows, self._solve(a, [S for _, S in rows], robot_now, table, scene)):
                out[k] = plan
            again = [(k, [replace(S[0], locked=False, check=True), *S[1:]]) for k, S in rows
                     if out[k] is None and poses[k].verb in LINE and poses[k].ee_pre is None]  # fmt: skip
            for (k, _), plan in zip(
                again, self._solve(a, [S for _, S in again], robot_now, table, scene) if again else []
            ):
                out[k] = plan
        return [x and replace(x, pose=poses[k], q_start=q_start) for k, x in enumerate(out)]

    def _ride(self, pose, robot_now):
        """The Plan of a drive (robot_now.base to pose.base [x, y, yaw], yaw unwrapped, a straight line turning on the
        way, within the base's speed limits) or a posture (robot_now.body to pose.height within the `Body` range): one
        Trajectory, a start-stop profile 3s^2 - 2s^3 that peaks at 1.5 times its mean speed, so it takes 1.5 times the
        duration at full speed. Checks nothing around it; the arms keep their joints, riding along. ValueError for a
        robot without the base or legs it needs."""
        r = self.model.robot
        if pose.base is not None:
            if robot_now.base is None:
                raise ValueError("this robot's arms are fixed: it has no base to drive")
            now, goal = np.asarray(robot_now.base, float), np.asarray(pose.base, float)
            dur = max(np.linalg.norm(goal[:2] - now[:2]) / r.base.vmax, abs(goal[2] - now[2]) / r.base.wmax)
        else:
            if robot_now.body is None or pose.height is None:
                raise ValueError("a base Pose needs base= on a mobile robot, height= on a legged one")
            now, goal = float(robot_now.body), float(np.clip(pose.height, r.body.lo, r.body.hi))
            dur = abs(goal - now) / r.body.vmax
        dur = max(1.5 * dur, 0.2)
        t = np.linspace(0, dur, max(2, int(np.ceil(dur / 0.02))) + 1)[1:]
        s = 3 * (t / dur) ** 2 - 2 * (t / dur) ** 3
        path = np.asarray(now) + np.multiply.outer(s, np.asarray(goal) - now)
        traj = dict(base=path) if pose.base is not None else dict(body=path)
        return Plan(pose, [Trajectory(t=t, q={}, hand={}, T={}, **traj)])

    def _solve(self, a, rows, robot_now, table, scene):
        """The Plan (its pose not yet set) of each of `rows`, segment lists of one shape, or None where RoboKit finds no
        path: `_path` plans 16 at a time, `_assemble` times and splits each path it found."""
        out = []
        for c in range(0, len(rows), SIZES[-1]):
            chunk = rows[c : c + SIZES[-1]]
            q, wi, ok = self._path(a, robot_now, chunk, table, scene)
            out += [self._assemble(a, q[k], wi, S) if ok[k] else None for k, S in enumerate(chunk)]
        return out

    def _path(self, a, robot_now, rows, table, scene=None):
        """RoboKit's paths of `rows` (at most 16 segment lists of one shape) from arm `a`'s live joints: (q (n, F, model
        joints), waypoint frame indices, ok (n,)). The obstacle grid rewritten from `table` and `scene`
        (`_obstacles`), then one `solve_offline_numpy` through [now, each segment's goal] (RoboKit pads it to the batch of
        16); ok where every pose waypoint is within 5 mm / 0.05 rad (a home within 0.05 rad) and the collision-checked
        first segment keeps out of the grid and the robot itself by over 5 mm."""
        from robokit.xform.numpy import tf_mat_to_pose7

        m, arm, i, n = self.model, self.model.robot.arms[a], self.model.info[a], len(rows)
        fq = m.fq(a, robot_now)
        T_base = i.M.root.T_world_base
        level = table is not None and table[1] is not None and np.allclose(T_base[2, :3], [0.0, 0.0, 1.0], atol=1e-6)
        z = table[0] - T_base[2, 3] if level else None  # the table top in the model's frame (a mobile base turns it)
        start = np.repeat(np.asarray(fq, np.float32)[None], n, 0)
        goals = [
            None if isinstance(s.goal, str) else np.stack([S[j].goal for S in rows]) for j, s in enumerate(rows[0])
        ]
        W = [start] + [m.set_arm(a, start, arm.home) if G is None else {arm.ee_link: (np.linalg.inv(T_base) @ G @ np.linalg.inv(arm.hand.T_ee_tcp)).astype(np.float32)}
                       for G in goals]  # fmt: skip
        planner, volume = self._planner(a, SIZES[-1], z, fq)
        self._obstacles(a, volume, table, scene, robot_now)
        res = planner.solve_offline_numpy(W, segment_locked=[s.locked for s in rows[0]],
                                          segment_check_collision=[s.check for s in rows[0]])  # fmt: skip
        q, wi = res.q_traj, tuple(res.waypoint_indices)
        ok = np.ones(n, bool)
        for j, G in enumerate(goals):
            f = q[:, wi[j + 1]]
            if G is None:
                ok &= np.abs(f[:, i.idx] - np.asarray(arm.home, float)).max(1) < 0.05
            else:
                T = m.tcp(a, f)
                ok &= np.linalg.norm(T[:, :3, 3] - G[:, :3, 3], axis=1) < 0.005
                cos = (np.einsum("bij,bij->b", T[:, :3, :3], G[:, :3, :3]) - 1) / 2  # of the rotation between them
                ok &= np.arccos(np.clip(cos, -1.0, 1.0)) < 0.05
        if rows[0][0].check:  # only the first segment is ever checked: the frames after it held at its end
            # padded to the planner's batch: RoboKit's evaluator keeps the batch size it first saw
            pad = lambda x: np.concatenate([x, np.repeat(x[-1:], SIZES[-1] - n, 0)])
            held = pad(q)
            held[:, wi[1] :] = held[:, wi[1] : wi[1] + 1]
            target = None if goals[0] is None else pad(tf_mat_to_pose7(W[1][arm.ee_link]).astype(np.float32))
            metrics = planner.compute_metrics_numpy(
                held, T_world_target=target, target_q=pad(W[1]) if goals[0] is None else None
            )
            ok &= metrics.success[:n]
        return q, wi, ok

    def _assemble(self, a, q, wi, segments):
        """The Plan (its pose not yet set) of one RoboKit path `q` (F, model joints) through waypoint frames `wi`: per
        followed segment, its frames timed at the arm's speed (`_rows`), then a hand-only Trajectory held DWELL s for its
        `then` (and one before the first, for a hand_pre without an ee_pre). A straight segment in continues its free
        segment's Trajectory under the same hand command, so the robot runs through the pre-pose instead of settling
        there as at the end of a follow; a press's way back stays its own, its follow ending at the bottom."""
        m, arm, i = self.model, self.model.robot.arms[a], self.model.info[a]
        T_tcp_ee = np.linalg.inv(arm.hand.T_ee_tcp)
        t_still = np.linspace(0, DWELL, int(round(DWELL / 0.02)) + 1)[1:]
        paths, here = [], q[0]
        still = lambda u: paths.append(Trajectory(t=t_still, q={}, hand={a: np.repeat(u[None], len(t_still), 0)}, T={}))
        if segments[0].before is not None:
            still(segments[0].before)
        for j, s in enumerate(segments):
            if not s.follow:
                break
            t, Q = _rows(q[wi[j] : wi[j + 1] + 1][:, i.idx].astype(float), arm.vmax * s.speed)
            full = m.set_arm(a, np.repeat(q[:1], len(Q), 0), Q)
            hand = {} if s.hand is None else {a: np.repeat(s.hand[None], len(t), 0)}
            path = Trajectory(t=t, q={a: Q}, hand=hand, T={a: m.tcp(a, full) @ T_tcp_ee})
            prev = segments[j - 1] if j else None
            if (
                prev is not None
                and not prev.locked
                and s.locked
                and prev.then is None
                and np.array_equal(prev.hand, s.hand)
            ):
                last = paths.pop()
                path = Trajectory(t=np.r_[last.t, last.t[-1] + t], q={a: np.r_[last.q[a], Q]}, T={a: np.r_[last.T[a], path.T[a]]},
                                  hand={k: np.r_[last.hand[k], v] for k, v in hand.items()})  # fmt: skip
            paths.append(path)
            here = full[-1]
            if s.then is not None:
                still(s.then)
        return Plan(None, paths, tcp_end=m.tcp(a, here))

    def _obstacles(self, a, volume, table, scene, robot_now):
        """Rewrite arm `a`'s obstacle grid `volume`, in its model's frame: the observed world points `scene` ((N, 3),
        NaN where the robot or no depth is; None: none) less those within CARVE of either hand, and the table top `table`
        (z, footprint hull equations) over its footprint, VOXEL apart, seen by the cameras or not (no footprint: none)."""
        m, T = self.model, self.model.info[a].M.root.T_world_base
        P = np.zeros((0, 3)) if scene is None else np.asarray(scene, float).reshape(-1, 3)
        P = P[np.isfinite(P).all(1)]
        # shortcut: a ball, not the hand and its load: a held object longer than CARVE stays an obstacle (leave its points
        # out of `scene`); upgrade when RoboKit attaches a load to the robot model
        for b in m.robot.arms:
            P = P[np.linalg.norm(P - m.tcp(b, m.fq(b, robot_now))[:3, 3], axis=1) > CARVE]
        if table is not None and table[1] is not None:
            g = np.arange(-np.sqrt(2) * REACH, np.sqrt(2) * REACH + VOXEL, VOXEL)  # the grid's footprint, turned or not
            xy = np.stack(np.meshgrid(g, g), -1).reshape(-1, 2) + T[:2, 3]
            xy = xy[(xy @ table[1][:, :2].T + table[1][:, 2] <= 0).all(1)]
            # a cell lower: the grid's surface stands up to a cell over the points, which pushed a grasp's or place's tcp
            # 5-6 mm off a pose 4-6 cm over the table (2 of 481 replayed calls; none once lowered)
            P = np.concatenate([P, np.c_[xy, np.full(len(xy), table[0] - VOXEL)]])
        volume.update(points=((P - T[:3, 3]) @ T[:3, :3]).astype(np.float32), reset=True)

    # ---- RoboKit underneath: the planning model and the planner on an arm

    def _robot(self, a, fq, z):
        """RoboKit's Robot of arm `a`'s model for planning: the model's spec with its collision spheres less those under
        the table top `z` (in the model's frame; None: no table) at home (a base or shoulder mounted that low); `fq` the
        model's full joints, the other arms' at home."""
        from robokit.robo import Robot as RK

        m, i, arm = self.model, self.model.info[a], self.model.robot.arms[a]
        spec = i.M.rk.spec
        link, local, radii = (
            spec.collision_spheres_link_indices,
            spec.local_collision_sphere_centers,
            spec.collision_sphere_radii,
        )
        T = i.M.rk.forward_kinematics_via_matrix_numpy(m.set_arm(a, fq, arm.home)[None], device="cpu")[0, link]
        keep = np.ones(len(link), bool)
        if z is not None:
            keep = np.einsum("sij,sj->si", T[:, :3, :3], local)[:, 2] + T[:, 2, 3] - radii >= z
        link, local, radii = link[keep], local[keep], radii[keep]
        return RK(spec=replace(spec, collision_spheres_link_indices=link, local_collision_sphere_centers=local, collision_sphere_radii=radii))  # fmt: skip

    def _planner(self, a, size, z, fq=None):
        """(RoboKit's planner for arm `a` on the `_robot`, for batches of `size`, its obstacle grid): one pair in the
        process per (robot description, arm, size, with a table under the arm or not), since the envs of a process plan
        one at a time on the same robot (the grid is rewritten before every solve). Its scene is the grid alone, a
        signed-distance volume over the model frame +- REACH, VOXEL fine, updated in place, so its CUDA graphs stay
        valid: RoboKit's swept collision cost takes primitive scenes only, so a table slab beside it is refused and the
        table is written into the grid instead, every segment checked against it frame by frame. Its goal IK keeps no
        clearance from the scene: the preset's 2.5 cm pushed a grasp's TCP up to 5 mm off a pose near the table. No retry
        round: the preset's twenty re-plan every unreachable goal, and reach queries are mostly unreachable."""
        import warp as wp
        from robokit.geom import SdfVolume, VolumeGeom, WarpScene
        from robokit.helpers.motion_plan import MotionPlanner, presets

        i, arm, device = self.model.info[a], self.model.robot.arms[a], self.model.device
        desc = i.M.desc if isinstance(i.M.desc, (str, PathLike)) else id(i.M.desc)
        key = (desc, a, size, z is not None)
        if key not in Planner._planners:
            scene = WarpScene(1, device)
            volume = SdfVolume.from_points(np.zeros((0, 3), np.float32), np.full(3, -REACH), np.full(3, REACH), VOXEL, device=device)  # fmt: skip
            scene.add(VolumeGeom([volume], np.array([0, 1], np.int32)))
            cuda = wp.get_device(device).is_cuda
            c = presets.offline_pose_lbfgs.goal_ik.solver.num_candidates  # [samples, refined, goals]
            ik = presets.build_offline_ik_config(
                c[-1], "full" if cuda else "none", num_samples=c[0], collision_margin=0.0
            )
            base = presets.offline_pose_lbfgs
            # the trajectory stage holds a pose goal at its goal-IK answer, not where the target cost pulls it
            stages = (replace(base.stages[0], lock_frames=(0, -1)), *base.stages[1:])
            config = replace(base, stages=stages, max_retries=0, stage_retries=0, goal_ik=ik,
                             cuda_graph="stages" if cuda else "none")  # fmt: skip
            planner = MotionPlanner(config, self._robot(a, fq, z), arm.ee_link, scene=scene, device=device)
            # before the first solve: the collision Jacobians keep this arm's columns, and smaller batches pad to `size`
            planner.set_active_joint_mask(np.isin(np.arange(len(i.M.names)), i.idx).tolist())
            planner.warmup(size)
            Planner._planners[key] = planner, volume
        return Planner._planners[key]
