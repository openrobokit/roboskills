"""General manipulation tools on an adapter's `Robot` (roboskills.eval.interface, duck-typed) over RoboKit, in layers
with one data structure each: perception (the cameras -> `View`), state (pure functions on views: the table, masks to
points, the lift and hand checks; shape and geom under them), action (the `Pose`, grasp_poses and place_pose), planner
(RoboKit motion planning: `Pose` -> `Plan`) and robot_model (kinematics). `bot.Tools` exposes them, with SAM 3
(roboskills.tools.sam3) for masks, as stateless tools; only its `execute` moves the robot, through `robot.follow`."""
