"""What the LLM's code can call: `Tools`, stateless tools over the robot, the cameras and the planner. The agent's own
code keeps what it knows (the objects, what each hand holds) and chains them:

    observe -> segment -> table / fuse -> describe / heuristic_grasp / place_pose -> plan -> execute -> status

No state is kept here: a pose without a hand command sends none, and the robot keeps its hand as last commanded. The
robot is touched only by `status`, `observe`, `plan` (the live joints it starts from) and `execute`."""

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from roboskills.tools.robokit import action, perception, shape, state
from roboskills.tools.robokit.planner import Planner
from roboskills.tools.robokit.robot_model import RobotModel
from roboskills.tools.sam3.sam3 import MIN_SCORE


ENDED = "the episode is over (the simulator ended it, or the steps ran out): nothing moves any more. Call finish()"
STALE = 0.02  # rad an arm may have moved since its plan was made before the plan is refused


class Tools:
    Pose = action.Pose  # a target for `plan`: built by heuristic_grasp / place_pose, or by the agent itself

    def __init__(self, robot, sam, out=None):
        """`sam`: a shared `roboskills.tools.sam3.sam3.Sam` created and closed by the caller; `out`: the episode's directory (None:
        none)."""
        self.robot, self.sam, self.out, self.model = robot, sam, out, RobotModel(robot)
        self.planner = Planner(self.model)

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

    def segment(self, view, prompts, min_score=MIN_SCORE):
        """SAM 3 masks of `prompts` (a noun phrase or a list) in every camera of `view`: {cam: [(label, score, (H, W)
        bool)]}, best first."""
        prompts = [prompts] if isinstance(prompts, str) else list(prompts)
        out = {}
        for cam, c in view.cams.items():
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
        {lifted (True, False, None: cannot tell), in_hand, left_behind}; `gripping` your judgement of the jaw reading."""
        return state.lift_check(view, points, table_z, T_closed, T_now, gripping=gripping)

    def heuristic_grasp(self, arm, tcp, hand=None):
        """Your grasp `Pose` of `arm`: TCP at `tcp` (4, 4), approached along its +x, the hand open on the way in and
        at `hand` (one value per hand DOF; None: closed) once there."""
        return action.heuristic_grasp(self.model, arm, tcp, hand)

    def grasp_poses(self, arm, points, table_z, scene=None, approach=None, avoid=None, variant=0):
        """Grasp `Pose`s of `arm` on object `points` (N, 3), best first; `scene` the observed points to keep clear of
        (view.points), `approach` None / "top" / "side" / [x, y, z], `avoid` more points to keep clear of (None: the
        other arms as they stand), `variant` another seed / rank offset. Only with a learned $GRASP_BACKEND."""
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

    def plan(self, poses, table=None, scene=None):
        """The first of `poses` the arm (or base, or body) reaches whole from the live joints, as a Plan; None when none
        is reachable. `table`: table()'s (z, floor), a collision surface; `scene`: observed points (N, 3) to keep clear
        of on the way (view.points, less what a hand holds; points within 0.1 m of a hand are taken as its own)."""
        plans = self.planner.plan(poses, self.robot.obs(cams=False), table, scene)
        return next((p for p in plans if p is not None), None)

    def execute(self, plan):
        """Run `plan`: {steps, ended, stopped, q, hand, tcp {arm: 4x4}, tcp_error_m / tcp_error_deg (to the
        plan's last TCP), short (m a press stopped above its depth)}. A Plan runs once, and only from where it was planned."""
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
            if f.done or self.robot.ended:
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
        return dict(steps=plan.steps, ended=plan.ended, stopped=plan.stopped, q=dict(o.q), hand=dict(o.hand), tcp=tcp,
                    tcp_error_m=err_m, tcp_error_deg=err_deg,
                    short=state.press_short(self.model, plan) if plan.pose.verb == "press" and plan.ends_q else None)  # fmt: skip
