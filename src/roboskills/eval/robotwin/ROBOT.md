# Robot: RoboTwin 2.0, aloha-agilex (two 6-joint arms with parallel grippers, SAPIEN)

## World and arms

World frame: metres, +z up. The robot stands behind the table and faces +y: +y is away from the robot (across
the table), +x is to the robot's right. The table top is at about z = 0.74 (`table_z` in the `update` summary is
measured).

- Arm `"left"` has its shoulder at (-0.30, -0.42, 0.78), arm `"right"` at (0.31, -0.42, 0.78). Each arm works
  best on its own side (left: x < 0.1, right: x > -0.1); both reach the middle. The reach is short: roughly
  up to y = 0.15 across the table and x = +-0.45 to the sides, at table height.
- Each arm has 6 joints, so not every hand orientation is reachable at every point: top-down grasps and
  grasps with the hand pointing forward (+y) and tilted down work best.
- Both arms start (and `home` returns) with the hands raised in front of the shoulders, fingers pointing +y.

## Hands

Each arm carries a parallel-jaw gripper, 9 cm between the pads fully open. Grasp, pinch and release open
or close it. Closed squeezes past touching, so a closed jaw holds what is
between the pads. The fingers are about 5 cm long.

Tool frame (TCP): between the finger pads. `approach` = along the fingers (out of the gripper), `squeeze` =
the axis the jaws close along. The press point (`press`) is the closed fingertips.

The opening is measured: a closed jaw that stops short of closed is holding something (`held` in the
`update` summary).

## Cameras

Every camera gives RGB and depth at 320x240; `observe()` has them with their poses (`T_world_cam`).

- `head_camera`: high behind the robot (0, -0.45, 1.35), looking down and forward over the whole table.
- `front_camera`: low behind the table edge (0, -0.45, 0.85), looking forward across the table top.
- `left_camera`, `right_camera`: on the wrists, looking out past the fingers; they move with the hands (move
  a hand to look closer at something).

## Control and budget

One budget step is one joint-space waypoint: the simulator moves both arms to it at its own speed. A verb
sends a motion as the few waypoints that keep it on its planned path: a free move costs a few steps, a straight
approach or lift ~5-10, a grasp with lift or a place ~15-30. The step limit (400-1500) is in every reply.
