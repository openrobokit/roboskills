"""What the LLM's code can call: `Tools`, stateless tools over the robot, the cameras and the planner. The agent's own
code keeps what it knows (the objects, what each hand holds) and chains them:

    observe -> segment -> table / fuse -> describe / heuristic_grasp / place_pose -> plan -> execute -> status

No state is kept here: a pose without a hand command sends none, and the robot keeps its hand as last commanded. The
robot is touched only by `status`, `observe`, `plan` (the live joints it starts from) and `execute`; on a robot that
queues plans (`queue`, `queued`, `flush`), `plan` starts where the queue ends."""

import dataclasses

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from roboskills.tools.robokit import action, perception, shape, state
from roboskills.tools.robokit.planner import Planner
from roboskills.tools.robokit.robot_model import RobotModel
from roboskills.tools.sam3.sam3 import MIN_SCORE


ENDED = "the episode is over (the simulator ended it, or the steps ran out): nothing moves any more. Call finish()"
NO_QUEUE = "this robot runs one plan at a time: use execute"
STALE = 0.02  # rad an arm may have moved since its plan was made before the plan is refused
STEP = 0.02  # m `align` moves the hand at most per correction


class Tools:
    Pose = action.Pose  # a target for `plan`: built by heuristic_grasp / place_pose, or by the agent itself

    def __init__(self, robot, sam, out=None):
        """`sam`: a shared `roboskills.tools.sam3.sam3.Sam` created and closed by the caller; `out`: the episode's directory (None:
        none)."""
        self.robot, self.sam, self.out, self.model = robot, sam, out, RobotModel(robot)
        self.planner = Planner(self.model)
        self.grasp_backend = action.grasp_backend()  # its $GRASP_* settings, None without one
        if self.grasp_backend is None:  # the API listing then says so, rather than leaving the agent to guess

            def grasp_poses(arm, points, table_z, scene=None, approach=None, avoid=None, variant=0):
                """Not in this run: it has no learned grasp backend. Build each grasp with `heuristic_grasp`."""
                raise ValueError(
                    "no grasp backend in this run: build the grasp yourself with heuristic_grasp(arm, tcp, hand)"
                )

            self.grasp_poses = grasp_poses

    def status(self):
        """The robot now: steps, step_limit, instruction, ended, q {arm: joints}, hand {arm: reading, jaw 0 closed ..
        1 open}, tcp {arm: (4, 4) world TCP}, base ([x, y, yaw] or None), body (pelvis height or None)."""
        o = self.robot.obs(cams=False)
        return dict(steps=o.steps, step_limit=o.step_limit, instruction=o.instruction,
                    ended=bool(self.robot.ended), q=dict(o.q), hand=dict(o.hand),
                    tcp={a: self.model.tcp(a, self.model.fq(a, o)) for a in self.robot.arms}, base=o.base, body=o.body)  # fmt: skip

    def observe(self):
        """What the cameras see now, a View: cams {camera: Camera (rgb, depth, K, T_world_cam)}, clouds {camera: world
        xyz per pixel, NaN on the robot and where depth is invalid} and `points` (N, 3)."""
        return perception.view(self.model, self.robot.obs())

    def segment(self, view, prompts, min_score=MIN_SCORE, cams=None):
        """SAM 3 masks of `prompts` (a noun phrase or a list) in every camera of `view` (or only `cams`): {cam: [(label,
        score, (H, W) bool)]}, best first."""
        prompts = [prompts] if isinstance(prompts, str) else list(prompts)
        out = {}
        for cam, c in view.cams.items():
            if cams is not None and cam not in cams:
                continue
            hits = self.sam.detect(c.rgb, prompts, threshold=min_score)
            out[cam] = sorted(((label, float(h["score"]), np.asarray(h["mask"], bool)) for label, hs in hits.items()
                               for h in hs), key=lambda m: -m[1])  # fmt: skip
        return out

    def table(self, view):
        """(table_z, floor): the table top's height and its footprint as hull equations a x + b y + c <= 0 (K, 3), or
        None when it has none; the planner's `table`."""
        z, P = state.table_height(view.clouds, self.model.reach_heights()), view.points
        try:
            floor = ConvexHull(P[np.abs(P[:, 2] - z) < 0.006][::2, :2]).equations
        except (ValueError, QhullError, IndexError):  # too few or degenerate points: no footprint
            floor = None
        return float(z), floor

    def fuse(self, view, masks, table_z):
        """The objects in `masks` (segment's), fused across cameras: [(label, score, points (N, 3))], best first."""
        return state.merge_masks(view, masks, table_z)

    def describe(self, points, table_z):
        """The `Box` (center, size [length, width, height], yaw deg mod 180, bottom) of one object's points."""
        return shape.describe(np.asarray(points, float), table_z)

    def register(self, shape_points, observed):
        """Rigid fit of `shape_points` onto `observed` (turned at most 25 deg): (R, t, quality {accepted, ...}); on
        accepted the shape is at shape_points @ R.T + t."""
        return shape.fit_rigid(shape_points, observed)

    def carried(self, view, points, T_closed, T_now, table_z, gripping=None):
        """Did the hand take the object along? Its points as grasped, TCP at closure T_closed and now T_now:
        {lifted (True, False, None: cannot tell), in_hand, left_behind (the fractions 0-1 of its points the cameras
        see carried with the hand, and still where it stood)}; `gripping` your judgement of the jaw reading."""
        return state.lift_check(view, points, table_z, T_closed, T_now, gripping=gripping)

    def heuristic_grasp(self, arm, tcp, hand=None):
        """Your grasp `Pose` of `arm`: TCP at `tcp` (4, 4), approached along its +x, the hand open on the way in and
        at `hand` (one value per hand DOF; None: closed) once there."""
        return action.heuristic_grasp(self.model, arm, tcp, hand)

    def grasp_poses(self, arm, points, table_z, scene=None, approach=None, avoid=None, variant=0):
        """A learned grasp policy's `Pose`s of `arm` on object `points` (N, 3), best first, available in this run:
        pass them to `plan` before building your own with `heuristic_grasp`. `scene` the observed points to keep clear
        of (view.points), `approach` None / "top" / "side" / [x, y, z], `avoid` more points to keep clear of (None: the
        other arms as they stand), `variant` another seed / rank offset."""
        if avoid is None:
            o, own = self.robot.obs(cams=False), self.model.info[arm].M
            bodies = [self.model.points(b, self.model.fq(b, o)) for b in self.robot.arms if self.model.info[b].M is not own]  # fmt: skip
            avoid = np.concatenate([np.zeros((0, 3)), *bodies])
        return action.grasp_poses(self.model, arm, np.asarray(points, float), scene, table_z, approach, avoid,
                                  variant, self.out)  # fmt: skip

    def place_pose(self, arm, T_tcp_obj, obj_points, pos, yaw=None, hover=0.04, table_z=None):
        """The `Pose` setting what `arm` holds (its frame `T_tcp_obj` in the TCP frame, its world points now) upright
        with its centre at `pos` [x, y(, surface z)], turned to long-axis `yaw` deg; the hand opens there."""
        T = self.model.tcp(arm, self.model.fq(arm, self.robot.obs(cams=False)))
        return action.place_pose(self.model, arm, T, T_tcp_obj, np.asarray(obj_points, float), pos, yaw, hover, table_z)

    def twist_pose(self, arm, axis, point, deg, tcp=None, hand=None, speed=None):
        """The `Pose` turning what `arm` holds `deg` degrees about the world `axis` through `point` (a lid's axis
        through its centre; positive: counterclockwise looking down -axis), its TCP from `tcp` (None: where it is now;
        queued after a grasp: bot.queued()["end"]["tcp"][arm]) along the arc, slowly and in contact, the hand at `hand`
        (None: as it is), at `speed` (None: slowly, for a lid; 1.0: a move's, for a tilt such as a pour). Plan it and
        queue it with holding=arm; a wrist turns only so far, so turn far in strokes: twist,
        open the hand, turn it back, grasp again."""
        T = (
            self.model.tcp(arm, self.model.fq(arm, self.robot.obs(cams=False)))
            if tcp is None
            else np.asarray(tcp, float)
        )
        return action.twist_pose(T, axis, point, deg, arm, hand, speed)

    def plan(self, poses, table=None, scene=None):
        """The first of `poses` the arm (or base, or body) reaches whole from the live joints (on a robot that queues,
        from where its queue ends), as a Plan; None when none is reachable. `table`: table()'s (z, floor), a collision
        surface; `scene`: observed points (N, 3) to keep clear of on the way (view.points, less what a hand holds;
        points within 0.1 m of a hand are taken as its own). When it returns None it prints why: blocked by `scene`
        (the poses plan without it) or out of reach. The Plan's `index` is its pose's place in `poses`."""
        poses, start = list(poses), self._start()
        k, p = next(((k, p) for k, p in enumerate(self.planner.plan(poses, start, table, scene)) if p is not None),
                    (None, None))  # fmt: skip
        if p is not None:
            p.index = k
        if p is None and poses:
            blocked = scene is not None and any(x is not None for x in self.planner.plan(poses, start, table, None))
            print("plan: no pose reachable: " + ("blocked by `scene` (they plan without it): leave out points that are no "
                  "obstacle, such as what the hand will touch or holds" if blocked else "out of reach, even with no "
                  "`scene`: another arm, approach, tilt or height"))  # fmt: skip
        return p

    def reach(self, poses, table=None, scene=None):
        """The indices of the `poses` the arm reaches whole from where `plan` starts, in order: a search over many
        candidates (tilts, spins, heights, arms) in one call, planned 16 at a time on the GPU, as fast as 1 each (a
        `plan` per pose in a loop costs 16 times the work). `table`, `scene` as `plan`'s; plan one of them after."""
        return [k for k, p in enumerate(self.planner.plan(list(poses), self._start(), table, scene)) if p is not None]

    def _start(self):
        """Where the next plan starts: the live state, or on a robot that queues, where its queue ends."""
        o = self.robot.obs(cams=False)
        if not hasattr(self.robot, "queued"):
            return o
        end = self.robot.queued()["end"]
        return dataclasses.replace(o, q={a: np.asarray(v, float) for a, v in end["q"].items()},
                                   hand={a: np.asarray(v, float) for a, v in end["hand"].items()})  # fmt: skip

    def nudge(self, arm, d, turn=0.0, frame="tcp"):
        """A Plan moving `arm`'s TCP straight by `d` [dx, dy, dz] m (frame "tcp": x the approach, y the squeeze axis;
        "world": world axes) and turning it `turn` deg about its approach axis, from where it is (or the queue ends).
        Not collision-checked: for small corrections, a few cm. None when the arm cannot reach it."""
        from robokit.xform.numpy import axis_angle_to_matrix

        T = self.model.tcp(arm, self.model.fq(arm, self._start()))
        T2 = T.copy()
        T2[:3, 3] += T[:3, :3] @ np.asarray(d, float) if frame == "tcp" else np.asarray(d, float)
        if turn:
            T2[:3, :3] = np.asarray(axis_angle_to_matrix(T[:3, 0] * np.radians(turn)), float) @ T[:3, :3]
        return self.plan([action.Pose("move", arm, tcp=T2, rank="line")])

    def offset(self, view, arm, points):
        """Where an object lies from `arm`'s hand when `view` was taken, in its TCP frame (x ahead along the approach,
        y along the squeeze axis, z): {center (3,) of `points` (N, 3), its world points from one camera of `view`,
        ahead (m to its nearest point along x), n}. From the camera on that hand, the hand and the object are placed by
        the same camera: the arm's and the cameras' calibration cancel, but for the camera's mount."""
        T = self.model.tcp(arm, self.model.fq(arm, view.robot))
        P = (np.asarray(points, float).reshape(-1, 3) - T[:3, 3]) @ T[:3, :3]
        P = P[np.isfinite(P).all(1)]
        if not len(P):
            raise ValueError("no points: the object is not seen")
        return dict(center=np.median(P, 0), ahead=float(np.percentile(P[:, 0], 5)), n=len(P))

    def align(self, arm, prompt, cam, tol=0.005, tries=4):
        """Close the loop before going in: look with camera `cam` (the one on `arm`'s hand), find `prompt` (SAM 3's best
        mask), and move the hand across its approach (y, z of the TCP, at most 2 cm a move) until the object's centre
        lies on the approach axis within `tol` m, at most `tries` moves. With nothing queued; each move runs at once.
        Returns {aligned, error_m, moves, offset (the last `offset`: its ahead is how far to go in), why}."""
        moves, off, err = 0, None, None
        for _ in range(tries + 1):
            view = self.observe()
            masks = self.segment(view, prompt, cams=[cam]).get(cam) or []
            if not masks:
                return dict(aligned=False, error_m=err, moves=moves, offset=off, why=f"{prompt!r} not seen by {cam}")
            off = self.offset(view, arm, view.clouds[cam][masks[0][2]])
            e = off["center"][1:]
            err = float(np.linalg.norm(e))
            if err <= tol or moves == tries:
                return dict(aligned=err <= tol, error_m=err, moves=moves, offset=off, why=None)
            move = self.nudge(arm, [0.0, *(e * min(1.0, STEP / err))])
            if move is None:
                return dict(aligned=False, error_m=err, moves=moves, offset=off, why="the correction is out of reach")
            r = self.execute(move)
            moves += 1
            if r["refused"] or r["stopped"]:
                return dict(aligned=False, error_m=err, moves=moves, offset=off, why=r["refused"] or "the move stopped")
        return dict(aligned=False, error_m=err, moves=moves, offset=off, why=None)

    def queue(self, plan, holding=None):
        """Queue `plan` to run after what is queued and return its id at once, while the robot moves: plan the next
        one meanwhile (plan starts where the queue ends). `holding`: arms (a name or a list) whose jaws must hold
        something when it starts, else it and everything after it is dropped. The person at the bench approves each
        queued plan before it runs. A refused or stopped plan drops everything queued after it."""
        if not hasattr(self.robot, "enqueue"):
            raise ValueError(NO_QUEUE)
        if self.robot.ended:
            raise RuntimeError(ENDED)
        if plan is None:
            raise ValueError("nothing to queue: plan(...) returned None, no pose was reachable; try another arm, argument or verb")  # fmt: skip
        if plan.steps is not None or getattr(plan, "queued", None):
            raise ValueError("this plan has run or is queued already: plan again")
        end = self.robot.queued()["end"]["q"]
        for a, q in (plan.q_start or {}).items():
            if np.abs(np.asarray(end[a], float) - q).max() > STALE:
                raise ValueError(f"the queue changed since this plan was made (the {a} arm ends elsewhere): plan again")
        holding = [holding] if isinstance(holding, str) else list(holding or [])
        plan.queued = self.robot.enqueue(plan.paths, holding, f"{plan.pose.verb} {plan.pose.arm or ''}".strip())
        return plan.queued

    def queued(self, wait=0.0):
        """The queue: {entries: [{id, label, status (waiting: for approval; approved, running, done, failed,
        dropped), result {steps, why (failed or dropped: why), q, hand, tcp}}], running, end {q, hand, tcp}: where it
        ends}. `wait` s (at most 60): first wait until an entry finishes, fails or is dropped."""
        if not hasattr(self.robot, "queued"):
            raise ValueError(NO_QUEUE)
        r, o = self.robot.queued(wait), self.robot.obs(cams=False)
        for d in [r["end"]] + [e["result"] for e in r["entries"] if e["result"] and "q" in e["result"]]:
            at = dataclasses.replace(o, q=d["q"], hand=d["hand"])
            d["tcp"] = {a: self.model.tcp(a, self.model.fq(a, at)) for a in self.robot.arms}
        return r

    def flush(self):
        """Drop every queued plan not yet running (the running one ends first); returns their ids."""
        if not hasattr(self.robot, "flush"):
            raise ValueError(NO_QUEUE)
        return self.robot.flush()

    def execute(self, plan):
        """Run `plan`: {steps, ended, stopped, refused (why the robot would not run it: a safety limit it breaks; nothing
        moved), q, hand, tcp {arm: 4x4}, tcp_error_m / tcp_error_deg (to the plan's last TCP), short (m a press stopped
        above its depth)}. A Plan runs once, and only from where it was planned."""
        from robokit.xform.numpy import matrix_to_rotation_angle

        if self.robot.ended:
            raise RuntimeError(ENDED)
        if plan is None:
            raise ValueError("nothing to execute: plan(...) returned None, no pose was reachable; try another arm, argument or verb")  # fmt: skip
        if plan.steps is not None:
            raise ValueError("this plan has run already: plan again")
        now = self.robot.obs(cams=False).q if plan.q_start else None
        for a, q in (plan.q_start or {}).items():
            if np.abs(np.asarray(now[a], float) - q).max() > STALE:
                raise ValueError(f"the {a} arm moved since this plan was made: plan again")
        follows = []
        for path in plan.paths:  # a hand change waits for the arm: one follow per segment, hand change or wait
            follows.append(f := self.robot.follow(path))
            if f.done or self.robot.ended or getattr(f, "refused", None):
                break
        plan.record(follows, self.robot.ended or f.done)  # the env can end on its own terminal, not only the budget
        o, a = follows[-1].obs, plan.pose.arm
        tcp = {b: self.model.tcp(b, self.model.fq(b, o)) for b in self.robot.arms}
        err_m = err_deg = None
        if (
            a is not None and plan.tcp_end is not None
        ):  # to where the plan ends (a press or place: back at its pre-pose)
            end = plan.tcp_end
            err_m = float(np.linalg.norm(tcp[a][:3, 3] - end[:3, 3]))
            err_deg = float(np.degrees(matrix_to_rotation_angle(tcp[a][:3, :3] @ end[:3, :3].T)))
        return dict(steps=plan.steps, ended=plan.ended, stopped=plan.stopped, refused=getattr(plan, "refused", None),
                    q=dict(o.q), hand=dict(o.hand), tcp=tcp,
                    tcp_error_m=err_m, tcp_error_deg=err_deg,
                    short=state.press_short(self.model, plan) if plan.pose.verb == "press" and plan.ends_q else None)  # fmt: skip
