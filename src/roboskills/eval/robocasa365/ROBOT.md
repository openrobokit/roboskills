# Robot: a Panda arm on a mobile base (RoboCasa365 kitchens, MuJoCo)

World coordinates are metres, with +z up and the floor at z = 0. Each episode is
one of RoboCasa's kitchens: counters, cabinets, drawers, a sink and appliances
along the walls, sometimes an island or a dining counter with stools. The
robot is RoboCasa's PandaOmron: one seven-joint Franka Panda arm, `right`, on a
torso lift (held at its lowest) on a holonomic base. Camera observations supply
all object and fixture locations.

The hand is the Panda parallel gripper. Command `[1.0]` opens it to 7.9 cm
between the pads, `[0.0]` closes it. The gripper moves at a fixed rate toward
open or closed by the command's side of 0.5, so it does not stop at a partial
opening; its fingers take 10–30 steps to close or open fully. The measured
opening is the gap between the pads over 7.9 cm. It stays well above zero on a
grasped object and reads about 0.01 when the gripper closes on nothing. The TCP
lies between the pads: +x is the approach direction, +y the squeeze direction.

## The base

`bot.status()["base"]` is `[x, y, yaw]`: `[x, y]` is the point the base
turns about, the middle of its footprint; `yaw` is its heading in radians and
`[cos(yaw), sin(yaw)]` the way it faces. The footprint is a 0.70 × 0.50 m box: it reaches
0.36 m ahead of `pos`, 0.34 m behind and 0.25 m to each side, so a base turning
in place sweeps a circle of 0.44 m radius. The arm's base, its first joint's
vertical axis (called the shoulder below), stands 0.21 m ahead of `pos`, 0.70 m
above the floor. The base's front edge is 0.15 m ahead of the shoulder and
below 0.38 m.

A `bot.Pose("drive", None, base=[x, y, yaw], rank="drive")` (yaw in radians,
unwrapped from the current one) drives the base to a world position and heading;
shift or turn relative to `status()["base"]` yourself. It moves on a
straight line while turning, at up to 0.5 m/s and 1 rad/s: about 60 steps per
metre or 50 per quarter turn, whichever is longer, plus a few steps settling
within a few millimetres. Nothing is planned around obstacles. A base that runs
into a counter, wall, stool or appliance stops there (more than 10 cm or 10°
behind its path for 10 steps) and the executed plan reports `stopped` with the
base where it stopped. The arm rides along holding its joint angles and
whatever the hand holds; fold it with a home pose before driving near
fixtures.

The arm reaches, measured from the shoulder along the base's heading:

| TCP height (m) | 0.3 | 0.5 | 0.7 | 0.9 | 1.1 | 1.3 | 1.5 | 1.7 |
|---|---|---|---|---|---|---|---|---|
| top-down hand, reach ahead (m) | 0.50 | 0.74 | 0.81 | 0.84 | 0.81 | 0.69 | 0.55 | — |
| forward-facing hand, reach ahead (m) | — | — | 0.89 | 0.93 | 0.93 | 0.90 | 0.84 | 0.69 |

These are kinematic limits; collisions with the counter's edge and the base
shorten them, and a hand far out has little room to turn. RoboCasa starts each
task with the robot parked at its work fixture, facing the counter that fixture
stands on, with the shoulder about 0.20 m from that counter's front (0.30 m
from a fridge or tall cabinet): the base's front edge almost touches it, and
the hand reaches about 0.6 m over a counter top. Objects or fixtures more than
about 0.8 m from the shoulder need the base moved.

Heights in these kitchens: counter tops 0.92 m (some kitchens also have a
raised 1.18 m bar or dining counter); dining counters 0.92 or 1.18 m; sink
basin floors 0.70–0.83 m below a rim at about 0.9 m; microwave start buttons
1.15–1.25 m on a counter-top microwave and 1.49–1.55 m on one built in above
the counter; coffee machines about 1.27 m tall, their cup on the drip tray at
about 1.0 m; the top drawers' fronts span 0.63–0.89 m.

## Cameras

`robot0_agentview_left`, `robot0_agentview_center` and `robot0_agentview_right`
ride on the torso behind and above the shoulder (0.5–0.6 m behind it, 1.75–1.85
m above the floor) and look forward and down over the arm; left and right sit
0.35 m to each side with a 60° field of view, center has 45°.
`robot0_eye_in_hand` is on the wrist, looking along the fingers. Every camera
supplies 512 × 512 RGB, metric depth and calibration at its current pose. All
of them move with the robot: they see only what lies ahead of the base, so to
look at a fixture elsewhere, turn or drive the base to face it, then `observe`
again. The arm in front of the shoulder hides part of the near counter;
`home` folds it up over the shoulder, where it hides the least.

## Control and budget

One control step is 0.05 seconds (20 Hz). Each step sends absolute joint
targets to the arm's joint-position controller, the gripper command, base
velocities tracking the drive path, and the torso's height. Every motion,
drive, hand change and settling step counts against the budget: 2000 steps for
`NavigateKitchen` and `PickPlaceCounterToSink`, 3000 for `DeliverBrewedCoffee`,
1500 for `CloseDrawer` and `TurnOnMicrowave`. The episode ends only at the step
limit.

## Success

RoboCasa's own check judges the state the robot is left in; the agent does not
receive its score. `finish` holds the robot still for ten steps, then checks:

- `NavigateKitchen`: the shoulder within 0.20 m of the spot RoboCasa parks the
  robot at to work on the named fixture (centred on it, facing the counter it
  stands on, the shoulder 0.20 m from that counter's front, 0.30 m for a fridge
  or tall cabinet), and the heading within about 11° of that spot's.
- `DeliverBrewedCoffee`: the mug and the breakfast plate both resting on the
  dining counter, their centres at most 0.30 m apart, and the TCP more than
  0.25 m from the mug.
- `PickPlaceCounterToSink`: the named object's centre inside the sink basin and
  the TCP more than 0.25 m from it.
- `CloseDrawer`: the open drawer pushed in to at most 5 % of how far RoboCasa
  opens drawers (within about 1.5 cm of closed).
- `TurnOnMicrowave`: the gripper has touched the start button while the door
  was closed, the door is still closed, and the TCP ends more than 0.15 m from
  the start button.
