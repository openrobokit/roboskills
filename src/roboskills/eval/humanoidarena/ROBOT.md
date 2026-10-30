# Robot: Unitree G1 humanoid with Dex3 hands (HumanoidArena, Isaac Sim)

A Unitree G1 humanoid (1.3 m tall, 29 joints) with two three-finger Dex3 hands, in one of HumanoidArena's scenes. A
learned whole-body controller (GEAR-SONIC) keeps it balanced: it walks, turns, squats and moves its arms, and it can
fall. A fall ends the episode as a failure.

## World, walking and posture

World frame: metres, +z up, the floor at z = 0.

- `status()["base"]` is the pelvis: [x, y] and heading `yaw` (radians); it faces `forward` = [cos(yaw), sin(yaw)].
  "In front of the robot" is along `forward`; the left arm is on its left. The feet stand about 0.1 m either side of
  [x, y]; the body is about 0.45 m wide at the shoulders and 0.3 m deep.
- `Pose("drive", None, base=[x, y, yaw], rank="drive")` walks there (yaw in radians, unwrapped from now). The robot
  steps: forward, backward, sideways and turning on the spot. A walk takes about 1 s to get going, moves at up to
  0.35 m/s and 34 degrees/s, and spends 1-3 s stopping and correcting: 0.5 m or a 60 degree turn costs about 250
  steps (5 s), 1 m about 400. It ends 1-13 cm (typically 5) and 0-6 degrees from the target (a turn on the spot
  shifts the pelvis most); about two walks in three end within 8 cm and 5 degrees. Standing
  still afterwards the feet may shuffle a few cm. The arms ride along holding their joints; walk with them `home`
  unless they carry something.
- Nothing is planned around obstacles. A walk stops (the robot stands where it is, short of the target: execute's `stopped`) at the step after
  any part of the robot other than its feet touches something it was not touching when the walk began, when the
  pelvis leans 20 degrees more than when it began, or when it makes no progress for 2 s. Walk around boxes and
  furniture in straight legs over open floor; keep 0.3 m between the pelvis and anything at leg height.
- `Pose("posture", None, height=h, rank="posture")` raises or lowers the pelvis (`status()["body"]`: its height above
  the floor) between `bot.model.robot.body.lo` (0.5 m, a deep squat) and `.hi` (standing, about 0.79 m) on the spot, ending within about 3 cm; down to 0.6 m and
  back up costs about 270 steps. Squatting leans the pelvis forward about 15 degrees. Squatting over a seat lowers the
  pelvis onto it: that is sitting. Walking stands the robot up first.
- Balance is the controller's: pushing hard on something heavy, reaching far out or leaning far over can still
  topple it. The robot counts as standing while its pelvis is above 0.45 m and upright within about 50 degrees.

## Arms and reach

- Two 7-joint arms hang off the torso, whose top (`torso_link`) moves with the pelvis, the waist and walking: the
  shoulders are about 1.08 m above the floor when standing, 0.1 m either side of the centre line. Each arm reaches
  about 0.5 m from its shoulder (to the TCP); a hand reaches from about 0.6 m to 1.5 m above the floor in front of a
  standing robot, lower when squatting.
- `home`: the arms hang beside the body, hands at hip height, out of the camera's view and the way while walking.
- The controller holds each arm joint a few degrees off its target; the adapter corrects for it at the end of every
  arm motion and keeps correcting while the arm holds still (to within 0.03-0.08 rad), so a hand lands within a few
  cm of where it was sent: approach things with a few cm of clearance and check with the camera.

## Hands

Each Dex3 has a thumb and two fingers and only two states: `"open"` (straight, spread) and `"closed"` (a fist). Any
other command closes the hand if it is nearer the fist than the open hand. The TCP is the space between the thumb and
the fingers of the open hand: +x out of the palm's face, +y along the fingers. The hands are small (fingers 10 cm):
grasping is unreliable; push, press and punch with them. `press` uses the closed fist's knuckles.

## Camera

`cam_front`: 640 x 480 RGB with metric depth and calibration, on the head (about 1.27 m up when standing), pitched 48
degrees down; it sees about 105 degrees across and 90 up and down, from the floor at the robot's feet to just below
the horizon: things above 1.2 m only from a few metres away. It moves with the torso: turn or walk to bring something into view, then `observe`
it again. There is no wrist camera; the hands are in view when held in front of the chest.

## Control and budget

One step is one control step, 0.02 s (50 Hz). Every arm, hand, walking and posture step counts, and standing still
counts too. The step limit is the benchmark's: 900 steps (18 s) for the boxing bag, 1800 (36 s) for the door and the
warehouse, 2000 (40 s) for the sofa and the football. That is little time: walk the shortest path, and do not wait.
The episode ends at success, a fall, or the step limit; nothing moves after.

## The tasks

Success is checked every step by the benchmark; the episode ends the moment it holds.

- **warehouse_navigation** (1800 steps). You stand at one end of a long warehouse aisle facing down it toward the
  destination: a square marked with yellow safety tape on the floor, with a yellow "wet floor" sign in it, about 3-4 m
  away. Six red-pink boxes stand in the aisle between: three long low ones and three tall ones. Success: both ankles
  inside the taped square while standing upright.
- **boxing_bag** (900 steps). Two punching bags hang ahead, a black one on your left and a red one on your right; a
  small bright green ball is on the side of one bag facing you. Success: the centre of either palm within about 0.15 m
  of the green ball's centre (the wrist does not count; speed does not matter).
- **opendoor** (1800 steps). A wall crosses your path with a door in it; the leaf is hinged on its left edge, swings
  away from you, and has a lever handle toward its free edge. Pressing the lever down by about 20 degrees releases the
  latch for the rest of the episode; the leaf is heavily damped, moves slowly while pushed and drifts back when
  released. Success: standing, the latch released, and the leaf open at least 60 degrees.
- **football_single** (2000 steps). A turf pitch with a goal straight ahead; a football rests between you and the
  goal. Success: the ball passes between the posts and below the crossbar into the goal. Kick or push it with a foot,
  or walk into it with the body; it rolls.
- **sit_sofa** (2000 steps). A living room: a dark grey sofa with a backrest ahead and to your right, its seat facing
  you, a low black tea table in front of it, and two cardboard boxes on the floor between. Success: a body of the hips,
  pelvis, waist or torso over the seat cushion's area, from 10 cm below the cushion to 35 cm above its top, touching it
  (5 N), or two such bodies there.

(Task descriptions adapted from RAX's HumanoidArena task rules, Apache-2.0.)
