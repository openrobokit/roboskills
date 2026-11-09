# Two real I2RT YAM arms with parallel jaws (a real bench, not a simulator)

World coordinates are metres with +z up. The table top is z = 0: calibration set the
world on the table plane, its origin midway between the two arm bases, +y toward
the left arm. The arms are six-joint YAM arms standing on the table side by side, named `left` and `right` as the `front` camera sees them. A run may
use only one of them.

The hand is I2RT's linear gripper. Command `[1.0]` opens it to about 9.5 cm
between the fingertips, `[0.0]` closes it. The measured opening is read from the
gripper motor: it stays well above zero on a grasped object and reads near zero
when the jaws close on nothing. Commands between 0 and 1 set the opening in proportion
(`h.width` m at 1). Close on an object just inside its width, not with `[0.0]`: driven
fully closed against something wide, the gripper motor stalls on it (i2rt then caps the
squeeze near 50 N, but holding a stall strains the motor). For a grasp across a
width `w`, pass `hand=[float(np.clip((w - 0.01) / h.width, 0, 1))]` (a 5.7 cm cube:
about 0.5), so the fingers press about 5 mm on each side. With a command `c`, the jaws
hold something when the measured opening stays above `c + 0.03`; near zero they closed on
nothing. Near `c` they may hold nothing, or a soft object squeezed to `c` (a plastic cup
gives way): lift it a few centimetres and check with `bot.carried` and the cameras. The TCP lies between the fingertip facets, 14.5 cm
out from the wrist flange. +x is the approach direction, +y the squeeze
direction, and +z their cross product.

Four RealSense cameras give 640 × 480 RGB, metric depth aligned to colour, and
calibration. `front` (a D435i behind the arms) looks over both grippers at the
table. `top` (a D435i) looks down at the table from above. `wrist_left` and
`wrist_right` (D405s) ride the grippers and see the fingertips and what lies past
them. Calibration is good to a few millimetres. Depth is missing
or noisy on transparent, shiny and very dark surfaces and at object edges.

One budget step is one 0.02 s command tick (50 Hz). Only motion counts: observing,
segmenting and thinking cost no steps. The default limit is 9000 steps, three
minutes of motion. The joints move at most 0.5 rad/s.

This is the real world: there is no reset, objects slide and topple, and nothing
is simulated. Safety limits guard every motion, and `execute` reports what they
did:

- A path that breaks a limit is refused whole. Nothing moves, and `refused` says
  why. The limits are joint limits; the TCP staying inside the workspace box;
  every part of the arm staying 5 mm above the table; the two arms staying 3 cm
  apart; and a plan starting within 0.1 rad of where the arm stands (otherwise,
  plan again).
- Contact waypoints may take the hand 1 cm into the table, and they run
  compliant, at a third of the arm's stiffness. These are a grasp's approach in,
  a lift, a release, and `rank="line"` moves. The arm yields on contact instead
  of forcing through, so a press or push can stop short: check `tcp_error_m` and
  `short`.
- A motion whose joints fall more than 0.15 rad behind their targets stops there
  with `stopped` (0.35 rad on contact waypoints), the arm holding where it is.
  This catches a collision or a blocked arm.
- The person at the bench can stop the episode at any time. Then it has ended.
- The person at the bench reviews every plan before it runs. Nothing moves while
  they look; `execute` returns when they have answered. A plan they reject comes
  back `refused`, and nothing moved: they said no, so plan something different.
  They may instead move the arms to rest (`refused` says so): observe and plan
  again. You cannot message them; `execute` is how you ask.

## Queued execution: keep planning while the arms move

This bench queues whole plans: `bot.queue(p)` returns at once while the arms run
them back to back, and `bot.plan` always plans from where the queue ends. Move the
arms with `bot.queue`, not `execute`, and queue a chunk of several plans in one
turn (a grasp, its lift and the carry over the target), each planned from where the
one before it ends, so the arms keep moving while you think. The person at the
bench approves each queued plan ahead in the panel; the arms wait only at a plan
not yet approved.

```python
g = bot.queue(bot.plan(grasps, table=table, scene=scene))  # the grasp, closing on the object
T = bot.queued()["end"]["tcp"][arm].copy()
T[2, 3] += 0.10  # where the grasp ends, 10 cm up
bot.queue(bot.plan([bot.Pose("lift", arm, tcp=T, rank="lift")], table=table), holding=arm)
T2 = T.copy()
T2[:2, 3] = target_xy  # over the target, as high
bot.queue(bot.plan([bot.Pose("move", arm, tcp=T2, rank="move")], table=table), holding=arm)
r = bot.queued(wait=30)  # meanwhile: think, then check what ran
```
- Queue what follows from a step only behind a condition the bench can check:
  `bot.queue(lift_plan, holding=arm)` runs the lift only if that jaw is closing
  and not shut all the way when it starts (a jaw at its command may hold a soft
  object, or nothing: confirm with the cameras once it is up). A plan that fails (refused, stopped, a jaw holding
  nothing, rejected) drops every plan queued after it; the arms hold.
- `bot.queued()` shows each plan's status and result (`why` it failed or was
  dropped, the end `q`, `hand`, `tcp`) and `end`, where the queue ends;
  `bot.queued(wait=10)` waits until a plan finishes, fails or is dropped.
- Check what you need before acting on it: after a grasp is `done`, look (observe,
  `show`) and read its result before queuing a carry that depends on it, or queue
  it with `holding`. `bot.flush()` drops what has not started (the running plan
  ends first): do it when what you see makes the queued plans wrong.
- `execute` still runs one plan and returns when it is done, but only with nothing
  queued. Observe only once the plans that move what you look at are done
  (`bot.queued(wait=...)` until `running` is False).

## Learned grasps

When the launcher runs with a grasp backend (see SKILL.md), `bot.grasp_poses(arm,
points, table_z, scene=view.points)` returns grasps a learned policy proposes for
the object's points, best first, each already checked against the table, the scene
and the object for the open hand's way in, with a hold command just inside the
object's width. Pass the list to `bot.plan`; the planner picks the first reachable.
Without a backend it raises: build the grasp with `heuristic_grasp`.

## Correcting at close range: align and nudge

The cameras can place an object 1 to 3 cm off where it is. `wrist_left` and
`wrist_right` ride their hands: `bot.offset` from that camera gives where the
object lies from the hand itself, and most of that error cancels. Before going
in to grasp (or to set something down), stop about 8 cm back with the hand
pointing at it, then `r = bot.align(arm, "the object", f"wrist_{arm}")` moves the
hand across until the object is on its approach axis (small moves, which the
person at the bench does not need to approve). Then go straight in by
`r["offset"]["ahead"]` plus how deep the jaws should reach, with
`bot.nudge(arm, [dx, 0, 0])`, close and lift. `bot.nudge` makes any small relative
move from where the hand is.

`home` raises each arm to the bench's home posture, clear of the table. A person
judges success after `finish()`. The agent never sees the verdict.

With the hand pointing straight down, the TCP reaches only about 15 cm above the
table at 40 cm out from an arm's base, and less further out. To carry something
higher or further, tilt the hand 20 to 30 degrees from vertical. `plan` returning
None with no `scene` at all means the poses are out of reach, not blocked.
