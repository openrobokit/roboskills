# Robot: EmbodiedSWE-Bench (RoboBench on Isaac Sim)

World coordinates are metres, with +z up. Each episode is one RoboBench task: a work table in a kitchen or a factory
hall, the objects the task names, and one robot, either a Franka Panda arm or a Unitree G1 humanoid. The task's own
description is the instruction; the sections below say what the robot is and what the task counts as done. Camera
observations supply every object's location.

## Control and budget (every task)

- One step is one control step: 1/60 s. Every arm, hand and walking motion and every settling step counts.
- The step limit (`status()`) is 45 s of simulated time per sub-goal of the task (a pick-and-place, a drawer pulled or
  pushed, a press, a walk) plus 30 s; the first 30 steps are spent before the first observation, letting the objects
  settle.
- Nothing ends the episode early: the task is judged once, on the state when you finish (after a short hold). Objects
  must be at rest where they belong at that moment; an object still rolling, or knocked back out, does not count.

## The Panda (clear_organic_objects, tool_packing, coffee)

One Franka Panda arm, `right`, stands on the work table itself, its base at the table top (0.55 m above the room's
floor). Its shoulder (joint 2) is 0.33 m above the base; the hand reaches about 0.85 m from it. Things close to the
base are awkward to grasp from above (the arm folds onto itself): work them about 0.3–0.6 m out.

The hand is the Panda's parallel gripper: `[1.0]` opens it to 8 cm between the pads, `[0.0]` closes it; the pads are
2 cm wide, on fingers 5 cm long. The TCP lies between the pads 1 cm short of the fingertips: +x is the approach,
+y the squeeze direction. The fingers are position-servoed and take about 15 steps to cross their travel; they squeeze
what they close on (a held object reads an opening above 0). `press` pushes with the closed fingertips. `home` is the
pose the episode starts in: the arm upright, the hand high over the workspace pointing down.

Cameras (640 x 480 RGB, metric depth, calibration): `front`, fixed, across the table at the robot's front right;
`shoulder`, fixed, behind the robot looking over its shoulder; `wrist`, on the hand beside the fingers, looking along
them (about 90 degrees across). The arm hides part of the table from the fixed cameras.

### packing.clear_organic_objects (budget 15300 steps)

Fruit and vegetables lie among non-food clutter (bottles, jugs, a serving bowl, a pen holder) on the table in front of
the robot; a blue plastic crate (about 42 x 25 cm, 17 cm deep) is fixed to the table to the robot's right. The
description names every organic the task may hold; each episode puts 4 or 5 of them on the table: clear the ones you
see. Done when every organic on the table rests inside the crate (its centre within the crate's walls and below the
rim, a small heap allowed) and no non-food item is in it. Dropping produce from a few centimetres above the crate is
fine; it must stay in.

### packing.tool_packing (budget 26100 steps)

A black chest of four drawers stands on the table ahead of the robot, its drawer fronts and pull handles facing the
robot; a stapler, scissors and a utility knife lie on the table to the robot's right. The drawers slide out toward the
robot. Done when the stapler rests in the top drawer, the scissors in the second and the knife in the third (each
inside its own drawer's tray), every drawer is pushed shut (within 1.5 cm) and nothing is moving. Open one drawer at a
time: an open drawer covers the one below.

### puzzle.coffee (budget 20700 steps)

A capsule coffee machine on the table ahead of the robot, its spout and cup platform facing the robot; a serving tray
to the right holds a mug and a capsule. The description gives the machine's rules. Grasping here locks: closing the
gripper across the cover's grip ridge, the capsule or the mug's rim holds it fast, and opening fully lets it go. The
brew lasts about 5 s (300 steps) once START is accepted. Done when the mug, filled by a completed brew, rests upright
and still on the tray.

## The G1 (classify_objects, fruit_delivery)

A Unitree G1 humanoid with two 7-joint arms, `left` and `right`, and a three-finger Dex3 hand on each. Its waist holds
still, so the torso is the arms' fixed mount; the shoulders are 0.2 m apart, about 1.0 m above the floor.
The arms are short: a hand reaches about 0.48 m from its shoulder, the table top (0.70 m) only about 0.35 m ahead of
the shoulders. Each arm works best on its own side.

The hand has a thumb and two fingers (index and middle), seven joints, commanded in the order `bot` lists them.
Presets: `open` (all straight: the fingers along the palm, the thumb standing out of it), `closed` (a fist), `point`
(index straight, the rest curled), `flat` (= open). The TCP is 4.5 cm out of the palm and 7.5 cm along it, between
the thumb and the fingers: +x out of the palm, +y along the fingers. The fingers are soft position servos: they take
about 25 steps to close, and they hold small, light objects (the blocks and the lemon) better than large or round
ones. `press` uses the pointing index fingertip.

Cameras: `head`, the G1's own head camera (where its RealSense sits), looking ahead and down at the table and turning
with the torso; `front`, fixed, across the room.

### puzzle.classify_objects (budget 9900 steps)

The pelvis is fixed: the G1 stands at the table's near edge and cannot step. Three 45 mm cubes (red, blue, green) lie on
the table in front of it, and three coloured square zones (red, blue, green) are marked a little further out. Done when
each block rests flat, still, within 5.5 cm of the centre of the zone of its own colour.

### locomanip.fruit_delivery (budget 18000 steps)

The G1 stands and walks. A wooden table (1.40 x 1.00 m, top at 0.70 m) holds a lemon and non-fruit items on the
robot's side and a round plate (30 cm across) on the far side, out of reach from where the robot starts: carry the
lemon around the table and set it on the plate from the far side. The floor is at z = 0. Done when the lemon rests
still on the plate (within 13 cm of its centre) and nothing else is on it.

Walking: `Pose("drive", None, base=[x, y, yaw], rank="drive")` (yaw in radians) walks the pelvis (`status()["base"]`) to a world position and heading on a
straight line, turning on the way. A frozen walking policy steps the legs; the robot keeps its arms' joints and what
the hands hold. It walks slowly, at up to 0.1 m/s and 17 degrees/s: 0.4 m takes about 450 steps, a 45 degree turn
about 240. Sideways steps are its steadiest; it cannot turn on the spot without stepping, so a pure turn steps side to
side while it turns. Drives end 1–10 cm from their goal (walking forward and back is the least precise) and within a
couple of degrees; so a longer walk may stop short of it: check the remaining
distance and go on, or drive again. Standing while
the arms move, it drifts a couple of centimetres. It cannot crouch: its pelvis stays 0.72 m up. A drive that falls
0.3 m behind its path (walking into the table, say) stops there. The robot leans up to about 17 degrees while walking;
a motion stops if it leans 20 degrees more than when it began.

### puzzle.stack_blocks (budget 12600 steps; the cohort's spare for the walking task)

The pelvis is fixed, as in classify_objects. Four 45 mm cubes lie flat on the table in front of the G1 beside a marked
square pad. Done when all four stand in one tower on the pad, each centred over the pad (within 3 cm), face level, and
still.
