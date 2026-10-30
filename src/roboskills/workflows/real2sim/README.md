# Real2sim

Demo image or video in, simulated 3D scene out, in two parts: reconstruction, then physics. The agent
([SKILL.md](SKILL.md), the `real2sim` skill) runs this code in the background, reconstructs the scene from the
video itself meanwhile, then delivers one rollout that looks like the source and takes its physics from the code's.
The code also runs alone, agent-free.

## Reconstruction

Reconstruct everything from the input: poses, the robot or hand, the objects and the scene.

| Tool                                      | Call                                                                                        | Writes       |
| ----------------------------------------- | ------------------------------------------------------------------------------------------- | ------------ |
| [SAM 3](../../tools/sam3/SKILL.md)        | `segment_video`: object and hand masks per frame                                            | `masks/`     |
| [MoGe-3](../../tools/moge3/SKILL.md)      | `estimate_depth`: metric depth and intrinsics per frame                                     | `depth/`     |
| [GeoCalib](../../tools/geocalib/SKILL.md) | `estimate_gravity`: gravity in the camera frame                                             | `gravity/`   |
| [SAM 3D](../../tools/sam3d/SKILL.md)      | `reconstruct_object`: object mesh and camera pose                                           | `shape/`     |
| [HaWoR](../../tools/hawor/SKILL.md)       | `reconstruct_hands`: MANO hand per frame                                                    | `hands/`     |
| [`align.py`](align.py)                    | `align`: one metric scale for depth and object from the hand's known size                   | `align/`     |
| [RoboKit](../../tools/robokit/SKILL.md)   | `fit_shape`: object pose and size refit to its mask at the reference frame                  | `fit/`       |
| [`track.py`](track.py)                    | `track`: the object's pose per frame, ICP to the depth, then (optionally) refit to the mask | `track/`     |
| [CoACD](../../tools/coacd/SKILL.md)       | `decompose_convex`: convex collision parts                                                  | `collision/` |
| [MuJoCo](../../tools/mujoco/SKILL.md)     | `assemble_scene`: table or pedestals, object as tracked, MANO hand fit to HaWoR, camera     | `scene/`     |
| [`events.py`](events.py)                  | `events`: grasp and release frames from the hand-object distance                            | `events/`    |
| [Blender](../../tools/blender/SKILL.md)   | `render_scene`: the scene in Cycles; source, Blender and MuJoCo side by side                | `blender/`   |
| [RoboKit](../../tools/robokit/SKILL.md)   | `retarget_hand`: the hand on a floating robot hand (`--robot`, Sharpa)                      | `retarget/`  |
| [RoboKit](../../tools/robokit/SKILL.md)   | `track_with_mpc`: SimMPC moves the object with it; `rollout` delivers it                    | `mpc/`       |
| [Pick and place](pnp.py)                  | `pick_and_place`: Panda arms replay the clip's robot, or do what its hands pick and place  | `pnp/`       |

```sh
# The agent in this repo (the real2sim skill): interactive, or one instruction
claude  # then: /real2sim simulate banana.mp4 with a Sharpa hand.
claude -p "/real2sim simulate banana.mp4 with a Sharpa hand."
codex exec '$real2sim simulate banana.mp4 with a Sharpa hand.'
# The agent anywhere
claude -p "Follow the real2sim skill in github.com/openrobokit/roboskills to simulate banana.mp4 with a Sharpa hand."
```

```sh
# Human video to a simulated hand (Sharpa)
python -m roboskills.workflows.real2sim --workspace runs/real2sim/banana --video banana.mp4 --object banana --tools sam3 moge3 sam3d coacd mujoco  # static object
python -m roboskills.workflows.real2sim --workspace runs/real2sim/bowl --video bowl.mp4 --object bowl --tools sam3 moge3 geocalib sam3d hawor align track coacd mujoco --support pedestal  # tracked
python -m roboskills.workflows.real2sim --workspace runs/real2sim/drill --video drill.mp4 --object "power drill" --tools sam3 moge3 geocalib sam3d hawor align track coacd mujoco retarget mpc --support pedestal  # + physics
python -m roboskills.workflows.real2sim --workspace runs/real2sim/can --video can.mp4 --object "tomato soup can" --tools sam3 moge3 geocalib sam3d hawor align track coacd mujoco retarget mpc --support pedestal --retry  # + retries
python -m roboskills.workflows.real2sim --workspace runs/real2sim/can --video can.mp4 --object "tomato soup can" --tools sam3 moge3 geocalib sam3d hawor align track coacd mujoco agent retarget mpc --support pedestal --retry  # + Codex fixes, the cheap one
# Human video to robot arms (EgoDex; workspace from python -m roboskills.eval.egodex.read prepare --w W --video V --task T)
python -m roboskills.workflows.real2sim --workspace runs/real2sim/egodex-cups --tools pnp
python -m roboskills.workflows.real2sim --workspace runs/real2sim/egodex-cups --tools pnp --retry
# Robot video to a robot (DROID; workspace from python -m roboskills.eval.droid.read --w W)
python -m roboskills.workflows.real2sim --workspace runs/real2sim/droid-marker --tools pnp
python -m roboskills.workflows.real2sim --workspace runs/real2sim/droid-marker --tools pnp --retry
```

## Physics

Replay the keyframes through the actuators; MPC fixes what replay misses.

| Tool                                    | Call                                                                            |
| --------------------------------------- | ------------------------------------------------------------------------------- |
| [MuJoCo](../../tools/mujoco/SKILL.md)   | `build`, per-step replay with contact feedback (`rollout`), `measure`, `render` |
| [RoboKit](../../tools/robokit/SKILL.md) | IK, and SimMPC for contact prefixes that replay misses                          |

```sh
# The agent in this repo (the real2sim skill): interactive, or one instruction
claude  # then: /real2sim move the drill in runs/real2sim/drill with a Sharpa hand.
claude -p "/real2sim move the drill in runs/real2sim/drill with a Sharpa hand."
codex exec '$real2sim move the drill in runs/real2sim/drill with a Sharpa hand.'
# The agent anywhere
claude -p "Follow the real2sim skill in github.com/openrobokit/roboskills to move the drill in runs/real2sim/drill with a Sharpa hand."
```

```sh
# The hand on Sharpa, then SimMPC moving the object through it (in a workspace)
python -m roboskills.tools.robokit retarget --video input/video.mp4 --hands hands/hands.npz --scene scene --out retarget
python -m roboskills.tools.robokit mpc --video input/video.mp4 --retarget retarget --out mpc
# Another noise seed
python -m roboskills.tools.robokit mpc --video input/video.mp4 --retarget retarget --out mpc-seed1 --seed 1
```
