# RoboKit tools

Perception, grasp proposals and motion planning on an adapter's `Robot`
([contract](../../eval/interface.py)), plus `Tools`, the stateless `bot` tools over them.
They plan and measure; only `Tools.execute` moves the robot, through `robot.follow`.
Frames are world metres unless named. The `Tools` docstrings (`bot.py`) are the reference the
agent_as_policy launcher shows the model.

These APIs assume arguments follow their documented types, shapes and ranges.
They convert inputs for computation without separate validity checks. Sensor-data
filtering, planning failures, collision checks and measured execution feedback remain part of the tools.

Construct `Tools(robot, sam, out=None)` with an entrypoint-owned `roboskills.tools.sam3.sam3.Sam`.
The entrypoint shares it across episodes and calls `sam.close()` when finished. `finish` is not a bot method: the
launcher calls `robot.finish`.

## Tools: no state

`Tools` remembers no objects, holds or hand commands: the agent's code keeps them, and a pose without a hand command
sends none, so every runtime's robot keeps its hand as last commanded (the `Robot.follow` contract). An example is the find -> grasp -> lift ->
place -> confirm block in [agent_as_policy/SKILL.md](../../workflows/agent_as_policy/SKILL.md).

| Tool | In -> out | Over |
|---|---|---|
| `status()` | the robot now: steps, step_limit, instruction, ended, q, hand, tcp {arm: 4x4}, base, body | `robot.obs(cams=False)`, `RobotModel.tcp` |
| `observe()` | the cameras now -> `View` (`cams` {cam: Camera}, robot-masked `clouds`, `.points`) | `perception.view` |
| `segment(view, prompts, min_score)` | {cam: [(label, score, mask)]} | [`sam3.Sam`](../sam3/SKILL.md) |
| `table(view)` | (table_z, floor hull equations or None): the planner's `table` | `state.table_height` + hull |
| `fuse(view, masks, table_z)` | masks fused across cameras [(label, score, points)] | `state.merge_masks` |
| `describe`, `register` | `Box`; rigid fit (R, t, quality) | `shape` |
| `carried(view, points, T_closed, T_now, table_z, gripping)` | {lifted, in_hand, left_behind} | `state.lift_check` |
| `heuristic_grasp(arm, tcp, hand)` | the agent's grasp `Pose` | `action.heuristic_grasp` |
| `grasp_poses(arm, points, table_z, scene, approach, avoid, variant)` | `[Pose]`, best first (a `$GRASP_BACKEND` only) | `action.grasp_poses` |
| `place_pose(arm, T_tcp_obj, obj_points, pos, yaw, hover, table_z)` | `Pose` | `action.place_pose` |
| `plan(poses, table=None, scene=None)` | the first `Plan` that plans whole from the live joints, clear of `table` and the observed points `scene`, else None | `Planner.plan` (one result per pose) |
| `execute(plan)` | {steps, ended, stopped, q, hand, tcp, tcp_error_m, tcp_error_deg, short}; refuses a plan that ran or whose start is over 0.02 rad from the live joints | `robot.follow` |

`plan`: collision-free (RoboKit, clear of the robot itself and of one signed-distance grid holding the `table` top over its
footprint and the `scene` points, those within 0.1 m of a hand left out; not yet of a held load; one planner and grid
per arm and batch size shared by the process, the grid rewritten before each solve) to `ee_pre`, then a straight joint line to `tcp`, every
segment of every candidate of one arm and shape in one RoboKit call (`planner.segment_traj`), timed at the arm's speed
here; lift and
release are straight lines (planned around when a line is out of reach), a grasp or pinch must also lift 10 cm
straight up, press and place come straight back up to `ee_pre`, a place lowers at 0.3 of the arm's speed; a drive or
posture is a smooth base or pelvis-height trajectory within its speed limits. A `Plan`'s `paths` are the robot's
`Trajectory`s: one per segment, one per hand change or wait held still. The planner moves a `Pose` by its verb name
(`action`'s `LINE`, `LIFT`, `BACK`, `SLOW`); the agent builds move, lift, release, home, press, pinch, drive and posture
poses itself.

`heuristic_grasp`: the agent's own grasp, its 6-DoF TCP and a hand command of the hand's DOF, approached `action.PRE_GRASP`
back along the TCP's +x with the hand open. `grasp_poses`: the best `tries` (8) of a learned backend (`GRASP_BACKEND`
names a module whose `candidates(S, arm, points, table_z, obstacles, target, avoid, axis, cone, seed)` returns grasp
`Pose`s; without one it raises), `variant` rotating the list. `place_pose`: the held object set upright, its centre at `pos` [x, y(, surface z)], turned to the
long-axis heading nearest `yaw`, pressed `action.PUSH` past contact, the hand opening there.

Under them: `robot_model` (`RobotModel`: TCP poses and surface points, nothing about the scene), `shape` (boxes, hidden
bases, rigid registration) and `geom` (poses, rotations, projection, point sets).

The table is the largest horizontal level of up-facing surfaces within `RobotModel.reach_heights()`. Its lower bound
is at most 0.6 m under the lowest shoulder, so a table under an arm on a pedestal is found; a mobile robot's is
0.15 m over its floor (`Base.floor_z`), so a low table counts and the floor does not.

Call `Sam.close()` after its final request, before reusing its GPU ([SAM 3](../sam3/SKILL.md)).

Tests: `python -m pytest tests/tools/robokit tests/workflows/agent_as_policy`.

## Reference

RoboKit has no paper; cite the project:
[openrobokit/robokit](https://github.com/openrobokit/robokit).
