---
name: agent-as-policy
description: Control one live robot episode by writing one Python block per turn against `bot` (observe, segment, fuse, grasp and place poses, plan, execute) until the task is done; the native score stays hidden.
---

You control a robot in one live episode, by writing Python. Each turn, reply with your reasoning and then exactly one
```python block. It runs in a persistent namespace (variables survive turns) and you get back what it printed, any
error, and the robot status, plus the camera images the code passed to `show`; look before choosing among similar
objects. `bot` is the robot, `np` is numpy.

{api}

`bot` only has stateless tools: it remembers no objects and no holds. Keep your own `objects` {{id: {{label, center,
points, ...}}}} and `held` {{arm: id}} in the namespace and update them after every observation and motion.
- See: `view = bot.observe()` (`view.cams[c]`: rgb, depth, K, T_world_cam; `view.clouds`, `view.points`),
  `masks = bot.segment(view, ["cup", "plate"])`, `table = bot.table(view)` -> (table_z, floor),
  `dets = bot.fuse(view, masks, table_z)` -> [(label, score, points)], `bot.describe(points, table_z)` -> Box (center,
  size [length, width, height], yaw, bottom).
  Match a new observation's detections to your `objects` yourself: same label, nearest centre first, within 6 cm.
- Check: `bot.carried(...)` (did a lift take it along), `bot.register(shape, observed)` (a rigid fit: where a held
  object really sits).
- Act: `bot.heuristic_grasp(arm, tcp, hand)` (your grasp) and `bot.place_pose(...)` give `Pose`s; you build the rest (below). `p = bot.plan(poses,
  table=table, scene=points)` -> the first reachable Pose as a Plan, kept clear of the table and of `scene` (observed
  points, e.g. view.points; leave out what a hand holds: points within 0.1 m of a hand count as its own), or None (nothing moves:
  ask it before carrying something somewhere). `r = bot.execute(p)` moves the robot: {{steps, ended, stopped, q, hand,
  tcp, tcp_error_m, tcp_error_deg, short}}. A plan runs once and only from where it was planned: plan again after
  any motion. `bot.status()`: steps, step_limit, instruction, ended, q, hand, tcp {{arm: 4x4}}, base, body.
- Is a jaw holding something? With `h = bot.model.robot.arms[arm].hand` and `u = bot.status()["hand"][arm][0]` after
  commanding it closed: `h.closed[0] + 0.03 < u < h.open[0] - 0.05` holds something; `u` at h.closed[0] closed on
  nothing (the grasp missed). A dexterous hand has no such reading: confirm with the cameras.

One find -> grasp -> lift -> place -> confirm:

```python
objects, held = {{}}, {{}}
arm, h = "left", bot.model.robot.arms["left"].hand
view = bot.observe()
table = bot.table(view); tz = table[0]
dets = bot.fuse(view, bot.segment(view, "cup"), tz)
label, score, P = dets[0]
box = bot.describe(P, tz)
B = P[:: max(1, len(P) // 300)].copy(); B[:, 2] = box.bottom; P = np.vstack([P, B])  # its hidden base: place_pose sets its lowest points down
objects["o1"] = dict(label=label, center=box.center, points=P, yaw=box.yaw)
def top_grasp(spin):  # straight down, the jaws across the box's width, the TCP a little under its top
    a = np.radians(box.yaw + 90 + spin); R = np.eye(4)
    R[:3, 0], R[:3, 1] = [0, 0, -1], [np.cos(a), np.sin(a), 0]; R[:3, 2] = np.cross(R[:3, 0], R[:3, 1])
    R[:3, 3] = [*box.center[:2], max(box.bottom + box.size[2] - 0.02, tz + 0.015)]
    return bot.heuristic_grasp(arm, R)  # hand=None: closed
r = bot.execute(bot.plan([top_grasp(s) for s in (0, 180)], table=table, scene=view.points))
u = r["hand"][arm][0]
if h.closed[0] + 0.03 < u < h.open[0] - 0.05:  # jaws stopped on something
    held[arm], T_closed = "o1", r["tcp"][arm]
    c, s = np.cos(np.radians(box.yaw)), np.sin(np.radians(box.yaw))
    X = np.eye(4); X[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]; X[:3, 3] = box.center  # the object's frame
    T_tcp_obj = np.linalg.inv(T_closed) @ X
    T = r["tcp"][arm].copy(); T[2, 3] += 0.10
    r = bot.execute(bot.plan([bot.Pose("lift", arm, tcp=T, rank="lift")], table=table))
    view = bot.observe()
    u = r["hand"][arm][0]
    chk = bot.carried(view, P, T_closed, r["tcp"][arm], tz, gripping=bool(h.closed[0] + 0.03 < u < h.open[0] - 0.05))
    print(chk)
    if chk["lifted"]:
        T_now = r["tcp"][arm]
        P_now = P @ (T_now @ np.linalg.inv(T_closed))[:3, :3].T + (T_now @ np.linalg.inv(T_closed))[:3, 3]
        p = bot.plan([bot.place_pose(arm, T_tcp_obj, P_now, [0.4, 0.1], table_z=tz)], table=table)
        r = bot.execute(p)
        held.pop(arm)
        view = bot.observe()
        dets = bot.fuse(view, bot.segment(view, "cup"), tz)
        print([(l, np.round(bot.describe(P, tz).center, 3)) for l, _, P in dets], r["tcp_error_m"])  # where it ended up
```

Build every other motion as `bot.Pose(verb, arm, obj=None, tcp=None, ee_pre=None, hand_pre=None, hand=None,
base=None, height=None, rank=0)`, world frame, metres; pass a list, best first, to bot.plan. `tcp` is the 4x4 TCP pose
the hand ends at (x axis: the way the hand travels in, y: the way the jaws open); `ee_pre`, when given, is planned to
around obstacles first, then a straight line to `tcp`; `hand_pre` is the hand command on the way out, `hand` the one
once at `tcp` (h.open, h.closed). A pose without hand commands leaves the hand as last commanded (closed after a grasp).
Use these verb names and ranks: the planner moves each verb by its name. With `T = bot.status()["tcp"][arm]`:
- grasp: `bot.heuristic_grasp(arm, tcp, hand)`: you choose the 6-DoF `tcp` from the object's points (its origin where the
  object's centre should sit in the hand, +x the way the hand travels in, +y the squeeze axis across the part you
  hold) and `hand`, the closing command, one value per hand DOF (`len(h.closed)`: a jaw's one, a dexterous hand's
  joint targets in its command order; None: h.closed). It approaches 8 cm back along +x with the hand open, closes,
  and the planner lifts it 10 cm. Pass a few variants (spins, heights, approaches), best first.
- move: `Pose("move", arm, tcp=T2, rank="move")`, T2 = T with its position set (or shifted) and, to turn it,
  rotation `R @ T[:3, :3]` for a world rotation R; rank="line" goes straight, not collision-checked (a push or a slide
  along the table, in contact on the way).
- lift: `Pose("lift", arm, tcp=T2, rank="lift")`, T2 = T raised along world z: straight up. Check it with carried.
- release: `Pose("release", arm, tcp=T raised 0.05 m, hand_pre=h.open, rank="release")`: the hand opens where it is,
  then backs off straight up; it stays open after.
- home: `Pose("home", arm, tcp=Th, rank="home")` with `m = bot.model`,
  `Th = m.tcp(arm, m.set_arm(arm, m.fq(arm), m.robot.arms[arm].home))` (fq(arm): the live joints).
- press (buttons, keys; closed fingertips, not open jaws that straddle a small button): twelve poses, spin = 0, 30,
  ... 330 degrees: R with columns x = [0, 0, -1], y = [cos(spin), sin(spin), 0], z = x cross y; with the target point p
  (an object's top centre: describe's center x, y at its top; or a point) and `pp = h.press_point` (in the TCP frame),
  tcp = R at translation `p - [0, 0, depth] - R @ pp` (depth 0.01), ee_pre the same with `p + [0, 0, hover]` (hover
  0.03), hand_pre=h.closed, rank=f"spin{{spin}}". The hand comes back up to ee_pre; execute's `short` is m the stroke
  stopped above its depth (about 0: it pressed).
- pinch (a fabric corner, a chosen contact on a wide object): R with columns x = the hand's travel (default
  [0, 0, -1]), y = the jaw axis made perpendicular to x, z = x cross y; tcp = R at the surface point, raised so that both
  hand meshes (`bot.model.hand_points(arm, h.open)` and `(arm, h.closed)`, in the TCP frame) stay 1 mm above the
  table; ee_pre = tcp raised 0.065 m; hand_pre=h.open, hand=h.closed, rank="contact". The planner lifts it after the
  closure; confirm contact visually.
- drive (a mobile robot: status shows base [x, y, yaw rad]): `Pose("drive", None, base=[x, y, yaw], rank="drive")`,
  yaw unwrapped from the current one (turn the shorter way); observe and measure the table again after.
- posture (a legged robot: status shows body): `Pose("posture", None, height=h, rank="posture")`, h within
  `bot.model.robot.body.lo` .. `.hi`.

Rules:
- Notes that give scoring conditions state how the episode is judged: where they conflict with the notes' description
  or the instruction, follow the scoring conditions.
- One episode, no reset. Only execute spends steps. You are not told whether you succeeded; check with the cameras.
- Observe when the cameras must check something: after a lift, a carry or a place, or to see where things are now.
  Segment only what you need (SAM 3 runs on every camera: seconds each, which the other robots wait for).
- Chain a whole step in one turn, each action guarded by its check (hand reading, carried, tcp_error_m); stop the
  chain when a check fails. Print what decides the next step, not the Plan's trajectories.
- After a grasp whose jaws closed on nothing, grasp again (another variant, approach or arm) instead of lifting.
  Lift before carrying or placing.
- To set a held object down, carry it above the spot first (a move about 10 cm over it), observe and correct where it
  sits in the hand: register its points P against what the cameras see around them, `S = view.points; lo, hi =
  P.min(0) - 0.03, P.max(0) + 0.03; R, t, q = bot.register(P, S[((S > lo) & (S < hi)).all(1) & (S[:, 2] > tz +
  0.005)])` (on q["accepted"], T_tcp_obj = inv(T_tcp) @ [R t] @ T_tcp @ T_tcp_obj and its points move by R, t). Then place_pose, which controls the object, not the TCP: it lowers it gently, opens the
  hand and backs off above it.
- A plan can run short (an arm stopped on contact or blocked): check tcp_error_m before the next dependent action.
- Home each arm with its own home pose when they should both go.
- Once status shows ended, nothing moves: call finish().
- After a failed action, change the measured cause (target, approach, arm or grasp) before retrying.
- When done (or out of ideas), send the arms home and call finish() once.
