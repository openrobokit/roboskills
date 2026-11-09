---
name: yam
description: Run agent-as-policy episodes on the real I2RT YAM bench (two YAM arms, four RealSense cameras) behind the shared Robot contract, with safety limits and an operator panel; the cameras' poses come from your own calibration.
---

# Real YAM bench

The runtime for real I2RT YAM arms and RealSense cameras. `launch.make(instruction, rig=...)` starts an episode
server on this machine (it owns the CAN buses and the cameras) and returns its `Robot`; the agent loop runs
unchanged. An episode is one instruction, one reset scene, one operator verdict.

| Module | Role |
|---|---|
| `launch.py` | `make(instruction, rig=None, out=None, step_limit="9000", vmax="0.5", arms=None, sim="false", port="8080")`: the launcher's entrypoint; `episode(...)` builds `Yam` in the server. |
| `robot.py` | `Yam`: the real bench (`eval/real/robot.py`: the contract, command streaming at `rate` Hz, the recorder, parking at `rest` on close, `replay(actions)` of a recorded `actions.npy`) with YAM's arms. |
| `arm.py` | i2rt, the arm model, TCP and jaw mapping: the only code that sends to the motors. |
| `../real/` | What every real bench shares: `safety.py` (every safety check: `Limits.check` before a path moves, `Limits.tracking` while it runs), `cameras.py` (RealSense capture only: RGB, metric depth aligned to colour, device intrinsics, frames newer than the last motion), `queued.py` (approve-ahead execution: the agent's queued plans run back to back while it keeps planning; the executor checks each before it runs, the reviewer shows the next one waiting for approval), `panel.py` (the operator's web panel, viser, http://localhost:PORT: Start, Execute, Stop, the verdict and its note, Release). |
| `ROBOT.md` | The bench, for the agent. |

## Prepare

```bash
uv sync --locked --group dev --group yam      # i2rt (git), pyrealsense2, imageio (H.264 video)
ip -br link | grep can                         # can_left and can_right UP
```

The rig lives outside the repository, one per machine: `~/.config/roboskills/<bench>.yaml` (or `$ROBOSKILLS_YAM_RIG`).

```yaml
rate: 50                     # Hz: command ticks, one budget step each
table_z: 0.0                 # world z of the table top
workspace: {lo: [-0.1, -0.6, -0.02], hi: [0.7, 0.6, 0.7]}   # the TCP stays inside, world metres
contact_kp: 0.3              # arm stiffness on contact waypoints
hand_speed: 0.5              # jaw command per s (0 closed .. 1 open): it closes over 2 s, never snaps shut
review_under: 0.025          # m: a plan whose TCP stays this near its start, jaws unchanged, runs unreviewed (0: review all)
arms:
  left:  {channel: can_left,  arm: yam_ultra, gripper: linear_4310, T_world_base: <4x4>, home: [6 rad], rest: [6 rad]}
  right: {channel: can_right, arm: yam_ultra, gripper: linear_4310, T_world_base: <4x4>, home: [6 rad]}
cameras:
  front:       {serial: "317422075675", T_world_cam: <4x4>, depth_scale: 1.0}   # a fixed camera's calibrated pose; depth_scale (default 1) if its depth reads short
  top:         {serial: "250222073512", T_world_cam: <4x4>}
  wrist_left:  {serial: "218622278225", arm: left, T_ee_cam: <4x4>}   # a wrist camera's pose on its arm's flange
  wrist_right: {serial: "130322273320", arm: right, T_ee_cam: <4x4>}
record: [front, top]         # one video_<camera>.mp4 each (one name or a list)
```

`home` should hold the hands clear of the table and out of the fixed cameras' view of the workspace. `rest` (default
all zeros) is where the arms go before their motors turn off. Check which D405 and which CAN channel belong to which
arm: move one arm by hand and watch.

On a new or rewired station, check once before the first run, with a hand under the arms: every RealSense streams
colour and depth; pressing the hardware e-stop stops both arms' motor chains and releasing it lets them reconnect and
hold; Ctrl+C or a SIGTERM partway through a slow move holds the arms and then folds them to `rest`; a path past a joint
limit is refused before anything moves.

## Calibration

The runtime takes the rig's poses as they are: `T_world_base` for each arm, `T_world_cam` for each fixed camera and
`T_ee_cam` for each wrist camera, in a world whose z = `table_z` is the table top. Calibrate them with your own tools
to a few millimetres; the planner and the grasps trust them. Recalibrate after anything is bumped.

## Run

```bash
python -m roboskills.workflows.agent_as_policy.run --adapter roboskills.eval.yam.launch \
  --out runs/agent_as_policy/yam/put_cup-001 -- rig=~/.config/roboskills/yam.yaml instruction="put the red cup on the plate"
```

Open http://localhost:8080. The arms connect holding where they stand. Reset the scene and press **Start**, and the
agent starts. The 3D view shows the arms where they stand and the fixed cameras' clouds; **Agent turns**, a column on the left, shows each
turn's reasoning and code as it arrives and what came back (the slider picks an earlier one). Every plan then waits
for review (`review=true`, the default): its arms in orange (scrub them, or Play) and its TCP path; **Execute** (enabled only while a plan waits) runs it,
**Reject** refuses it (the agent sees `refused`) and nothing moves, **Rest** opens the jaws, then takes the arms to
`rest` instead (also before Start). `review=false` runs plans unreviewed. The agent can queue whole plans
(`bot.queue`, eval/real/queued.py) and plan the next one from where the queue ends while the arms move; each queued plan
waits in the panel for your approval (Execute) before the executor reaches it, and the list under Plan review shows the
queue. Reject drops that plan and everything queued after it. **Stop** ends the episode at any time, holding the arms. After the agent's last turn, give the
verdict: Success, Failure, or Void (the episode does not count, e.g. someone bumped the scene), with a note. A verdict given while the agent is still
working ends the episode there, as Stop does: the arms hold and the agent is told to finish. The
arms go home, and when the run closes they fold to `rest` and their motors turn off. Keep a hand on the hardware
e-stop for any first run.

To quit early, press **Stop** or Ctrl+C the launcher: either way the arms hold, then fold to `rest` on a checked path
at half speed before the motors turn off (the launcher waits for it). If that path is refused, or a second Ctrl+C
interrupts it, the arms hold where they are until you support them and press **Release** (motors off where they
stand; it is greyed out until then, so an earlier press never counts). Only a killed server (`kill -9`), a crash inside i2rt or a lost CAN or power link can drop them: that is
what the e-stop is for.

Outputs (`--out`): the launcher's `summary.json`, plus `env0/log.jsonl` and `result.json`. `env0/finish/` holds
`native_result.json` (verdict, note, steps), `video_<camera>.mp4` (each `record` camera), `actions.npy` (each tick: each arm's 6 joints
and jaw), `rig.yaml` and `faults.jsonl` (every refused path, guard trip and operator stop).

Without hardware, `sim=true` runs i2rt's MuJoCo arms behind the same panel. `tests/eval/yam/` runs them with no one
at the bench (`tests/conftest.py`'s `unattended`).

## Safety

Every path is checked whole before it moves: joint limits; the TCP inside `workspace`; every arm surface 5 mm above
the table (1 cm into it for the hand on contact waypoints); the arms 3 cm apart; and the start within 0.1 rad of
the measured joints. Faster paths are slowed to `vmax`. While a path streams, a joint lagging 0.15 rad (0.35 on
compliant contact waypoints), a stopped motor chain, or a tick over 60 ms late stops it, holding the arm where it is
measured. Every fault ends in a hold, never in zero torque: the motors turn off only at `rest`, or on Release.
