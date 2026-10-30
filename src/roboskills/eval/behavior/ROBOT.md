# Robot: R1Pro mobile manipulator (BEHAVIOR-1K, OmniGibson)

A Galaxea R1Pro in a furnished home or office (the 2026 BEHAVIOR Challenge's scenes): a three-wheel holonomic base, a
torso, two 7-joint arms with parallel grippers, a head camera and a camera on each wrist.

## World and base

World frame: metres, +z up, the floor at z = 0. Only the task's rooms are loaded, and they span several metres: what
the task names may be out of reach, or out of view, from where the robot starts.

- `status()["base"]`: the base's position [x, y] and heading `yaw` (radians); it faces `[cos(yaw), sin(yaw)]`. "In front of the robot" is along `forward`; the left arm is on its left.
- `Pose("drive", None, base=[x, y, yaw], rank="drive")` (yaw in radians, unwrapped from now) drives in a straight
  line to a world position and heading. It plans no path around furniture, and it never pushes: the base
  stops the moment the robot touches something it was not touching when the drive began (a chair, a table edge, a
  wall; also with an arm or what a hand holds), short of its target (execute's `stopped`). Go round obstacles in several
  straight legs over open floor, and turn where nothing is within 0.55 m of the base's centre.
- The robot can lean and fall over (its base is free to tilt, as the challenge's is): any motion stops when it
  starts to lean.
- The base is 0.63 m long and 0.67 m wide, and its origin (the `pos` above) is not its centre: the front edge is
  0.24 m ahead of it, the back 0.39 m behind, the sides 0.34 m either side.
- Driving starts and stops smoothly, at up to 0.5 m/s and 46 degrees/s: about 90 steps per metre or 1 step per degree
  of turn, whichever is longer (0.3 m: 27 steps; a 45 degree turn: 45; 2 m: about 180).

## Arms and reach

- The torso does not move: both shoulders are 1.22 m above the floor, 0.17 m either side of the centre line and
  0.18 m ahead of the base origin. Each arm reaches about 0.8 m from its shoulder.
- In front of the base a hand reaches from about 0.53 m up (pointing down) to 1.5 m. Typical surfaces in these
  scenes: coffee tables 0.42 m, dining, office and bathroom counter tops 0.75-0.8 m, kitchen counters 1.08 m, low
  cabinets 0.84 m, waste bins 0.42-0.74 m. Table and counter tops are in reach out to about 0.7 m ahead of the base
  origin; something on a coffee table only from above and close; the floor and low shelves not at all.
- So drive up to what you work on: its near edge about 0.3-0.35 m ahead of the base origin (the base's front edge
  stops 0.24 m ahead), facing it. Each arm works best on its own side and in front.
- `home`: the arms hang straight down beside the torso, the hands about 0.45 m up, out of the cameras' view and of
  the way while the base drives.

## Hands

Each gripper has two fingers: up to 10 cm between the pads (open; closed squeezes; grasp, pinch and release set it). The
fingers are 7.8 cm long and padded along their whole inner face, 1.6 cm wide. The TCP is 1.8 cm short of the
fingertips, between the pads.

- Grasping is the simulator's assisted grasp: a closing gripper whose pads have touched an object for 0.3 s holds it
  fast; opening the gripper fully lets go. A hand that closes on nothing reads an opening near 0.
- `press` uses the closed fingertips.

## Cameras

All are 720 x 720 RGB with metric depth and calibration; `observe()` has them.

| name | where | looks |
| --- | --- | --- |
| `head` | on top of the torso, 1.4 m up | forward and down, a wide view (100 degrees across) |
| `left_wrist`, `right_wrist` | on each gripper | along the fingers |

The cameras move with the base: turn or drive to bring something into view, then `observe` it again.

## Control and budget

- One step is one control step of the simulator, 1/30 s. Every arm, hand and base motion and every settling step
  counts. The step limit (`status()`) is the challenge's: 1.5 times the mean time its human demonstrators took.
- The episode ends as soon as the goal holds (the challenge's rule) or at the step limit; then nothing moves.

## The goal

The task is a BEHAVIOR goal (BDDL): facts about objects that must all hold at the same moment. The instruction states
them; what they mean in the simulator:

- *inside* A B: A lies within B's volume: dropped into a bin, placed in a cabinet (open its door first).
- *on top of* A B: A rests on B, touching it.
- *next to* A B: A and B are close side by side, not touching necessarily.
- *switched on* A: a device's power button was pressed: fingers touching the button for about 5 steps switch it,
  touching it again switches it back. A device may show no change when it is on.
- *open* / *closed*: a door, drawer or lid is open past a fraction of its range, or shut.

Objects named by category ("the ashcan", "every water glass") are the task's own objects of that kind; "a cabinet"
means any one of them.
