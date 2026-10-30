# Panda arms with parallel grippers (CaP-X Robosuite tasks, MuJoCo)

World coordinates are metres, with +z up. Each arm is a seven-joint Franka Panda
on a pedestal, its base a little above the table top. Single-arm tasks expose
`right`. The two-arm tasks expose `left` and `right`: the arms stand at opposite
ends of the table, facing each other, named by the side they appear on in
`agentview`. Camera observations supply object locations in observation mode.
Explicitly privileged scripted generation may also read physical collision
surfaces and poses through `scene()`; native scores remain evaluator-owned.

The hand is the Panda parallel gripper. Command `[1.0]` opens it to 7.9 cm
between the pads, `[0.0]` closes it. The native gripper moves at a fixed rate
toward open or closed by the command's side of 0.5, so it does not stop at a
partial opening; its fingers take 10–30 steps to close or open fully. The
measured opening is the gap between the pads over 7.9 cm. It stays well above
zero on a grasped object and reads about 0.01 when the gripper closes on
nothing.

In `spill_wipe` a sponge with a 12 × 5 cm wiping face replaces the gripper. It
has no hand command: its commands and measured state are empty.

The TCP lies between the pads, at the gripper's grip site (the sponge: the
centre of its wiping face). +x is the approach direction, +y the squeeze
direction, and +z their cross product. Robot kinematics provide each arm's base
pose, joint limits and hand geometry.

`agentview` is the scene camera. In the two-arm tasks it is CaP-X's raised view
of the whole table from beside it. Each arm also has a wrist camera: `wrist`,
or `wrist_left` and `wrist_right`. Every camera supplies 512 × 512 RGB, metric
depth and calibration.

One control step is 0.05 seconds (20 Hz). Each step sends absolute joint
targets to the native joint-position controller, and every motion, hand change
and settling step counts against the budget. Step limits follow CaP-X's task
horizons: 1500 steps for the cube tasks, 4000 for `spill_wipe`, 1000 for
`nut_assembly` and 5000 for the two-arm tasks. The episode ends only at the step
limit.

`home` folds each arm back over its pedestal. For the single-arm tasks this
clears `agentview`'s view of the table. In the two-arm tasks the folded arms stay
at the edges of `agentview`, beside the table.

The native task checker owns success, and the agent does not receive its score.
`finish` holds the robot still for ten steps, scores the state it is left in and
writes `native_result.json`, `actions.npy` and `video.mp4`.
