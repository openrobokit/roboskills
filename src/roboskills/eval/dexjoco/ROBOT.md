# Panda arms with Allegro hands (DexJoCo, MuJoCo)

World coordinates are metres, with +z up. Each arm has seven Panda joints and
sixteen Allegro finger joints. Single-arm tasks expose `right`; bimanual tasks
expose `right` and `left`. Camera observations supply all object locations.

The hand's model frame is its Allegro palm. The shared TCP lies among the
fingers: +x is the approach, +y is the squeeze direction, +z is their cross
product. Robot kinematics provide each arm's base pose and joint limits.
Right-hand fingers are ordered index/middle/ring/thumb, four joints each;
left-hand native commands use ring/middle/index/thumb. The grasp recipe mirrors
poses, finger order and abduction signs together.

`open`/`flat` straighten the fingers; `closed` forms a fist. `point` keeps the
index finger straight. Closure alone does not establish an object hold; use
visual feedback. `hand_geometry` reports the robot's own surface samples.

Single-arm cameras are `front`, `ego_left`, `ego_right`, and `wrist`.
`water_plant` supplies `front` and `wrist`; bimanual tasks supply `ego`,
`wrist_left`, and `wrist_right`. RGB, metric depth and native camera calibration
are available. Task target-region annotations are hidden from policy cameras.

One native control step is 0.02 seconds. Every motion and settling action spends
this budget. Native task horizons are 1000–1500 steps depending on the task.
The operational-space controller follows the shared planner's palm poses and
finger targets, with measured-pose holding after a stopped motion.

The native task checker owns success. The agent does not receive its score
during execution. `finish` preserves the first native termination result and
writes `native_result.json`, `actions.npy`, and `video.mp4`.
