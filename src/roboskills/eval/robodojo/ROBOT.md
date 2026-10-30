# Robot: two ARX X5 arms with parallel grippers (RoboDojo, Isaac Sim)

World frame: metres, +z up, +y away from the robots (across the table), +x to the right as seen from
behind the robots. The table top is at z ≈ 0.765 (`table_z` in the `update` summary).

## Arms

| arm | base (x, y, z) | reaches |
| --- | --- | --- |
| `"left"` | (-0.30, -0.45, 0.765) | x from about -0.6 to +0.1 |
| `"right"` | (+0.30, -0.45, 0.765) | x from about -0.1 to +0.6 |

- Both arms stand at the near edge of the table and reach forward (+y). Each is a 6-joint arm: its reach is
  small. With the hand pointing straight down it reaches about 0.45 m from its base (up to y ≈ 0.0 near the
  table); leaning the hand forward (approach tilted 45° toward +y) reaches about 0.1 m further; with the hand
  level (approach along +y) at a height of 0.95 m or more it reaches to y ≈ +0.2 (+0.3 straight in front of
  its base). Low objects beyond y ≈ +0.15 are out of reach.
- Only the middle strip (x between about -0.1 and +0.1, y below about 0.0) is reachable by both arms: to move
  something from one side to the other, put it down there with one arm and pick it up with the other.
- An arm's own side only: the left arm cannot work at x > +0.1, the right arm not at x < -0.1.
- Wrist roll: the last joint turns the hand about its approach axis within ±180° of its home angle, no
  further. A `move` with `rot` about the approach axis past that limit plans to None; for a long turn (a screw, a
  knob), turn the empty hand the other way first, then grip and turn: up to about 340° per grip.
- `home` posture: the arm folded, the gripper pointing forward (+y), level, at about (±0.30, -0.21, 0.92).
  Arms left out in the workspace block the head camera's view and the other arm: send an idle arm `home`.

## Hands

Each arm has a parallel gripper (two fingers closing along the TCP's `squeeze` axis).
- Opening: up to 8.8 cm between the finger pads (open = 8.8 cm, closed squeezes); grasp, pinch and release set it.
- The finger pads are about 1.8 cm tall (along the TCP z axis) and 1.3 cm of finger reaches past the TCP.
  The fingers are about 8.6 cm long; the wrist behind them is wide (about 17 cm across the squeeze axis).
- `bot.status()["hand"][arm]` is the measured opening (0 closed .. 1 open): a closed gripper that stops wide holds something.
- `press` uses the closed fingertips.

## Cameras

| name | where | looks |
| --- | --- | --- |
| `cam_head` | fixed, above and behind the robots, at (0, -0.41, 1.31) | forward and down over the whole table |
| `cam_left_wrist` | on the left gripper, above the fingers | along the fingers |
| `cam_right_wrist` | on the right gripper, above the fingers | along the fingers |

All are 640 x 480 RGB-D. The wrist cameras move with the grippers: point a gripper at something (with `move`)
and `observe` again to see it up close.

With RoboDojo's official observation (RGB only), depth and camera poses are estimated from the images and the
joints: the head camera's points are good to about 1 cm on objects on the table (worse on tall ones). A camera
without an estimate gives an image only (`show` it; its points are NaN): detection and the checks then rely
on the others.

## Control and budget

- Joint position control at 25 Hz: one step is 0.04 s of motion. The episode's step limit is in `status()`.
- Rough costs in steps (past episodes; with RGB-only observations the hand-camera grasp and the press take
  longer): a grasp with its lift 45-55 (RGB 50-75; a failed one costs about as much); a place 55-70; `release`
  15-20; `home` about 20; a `move` 10-30 (a wrist turn 15-25); a press about 30 (RGB about 45). Add up the plan
  before the first motion and leave room for one failure.
