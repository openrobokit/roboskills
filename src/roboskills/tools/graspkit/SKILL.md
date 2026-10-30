---
name: graspkit
description: Train a grasp generator and a separate grasp success ranker in GraspKit from supplied scenes and robot assets, then use both checkpoints to select executable grasps.
---

# GraspKit

**Runs in** its own process and venv: `$GRASP_POLICY_PYTHON`, a venv with
[GraspKit](https://github.com/openrobokit/graspkit) 0.1.0 or later; no image yet.

**Call:** `bot.grasp_poses` (`roboskills.tools.robokit.action.grasp_poses`) imports the module `$GRASP_BACKEND` names
and calls its `candidates(S, arm, points, table_z, obstacles, target, avoid, axis, cone, seed)`. RoboDojo's
backend (`GRASP_BACKEND=roboskills.eval.robodojo.grasp_backend`) starts `graspkit.policies.arx_x5_server` with
`GRASP_POLICY_CKPT` and `GRASP_POLICY_RANKER` (default: the published checkpoints from GraspKit's dataset),
`GRASP_POLICY_RANK` (default `ranker_reach`) and `GRASP_POLICY_N` (128 samples) and reaches it over a unix socket;
`GRASP_POLICY_SOCKET` shares one running server.

**Returns** grasp `Pose`s best first (world TCP and pre-grasp (4, 4) in metres, jaw commands), kept when the open hand's
straight way in clears the table, obstacles, avoid points and object; each call appends a line to
`<run>/grasp_backend.jsonl`.

**GPU memory:** the policy server's generator and ranker on the episode's GPU; not separately recorded.

**Known failures:** no `$GRASP_BACKEND` raises "no grasp backend here"; no heuristic fallback; reachability is the
planner's check on the returned list.

## Train and use a grasp generator and ranker

Use [GraspKit](https://github.com/openrobokit/graspkit) to turn a scene
and robot into two learned components:

| Component | Training input | Inference role |
|---|---|---|
| Grasp generator | Scene observations paired with successful grasps | Sample multiple candidate grasps |
| Success ranker | Observations and candidate grasps paired with success/failure labels from rollouts in the target deployment simulator | Score candidates and put likely successes first |

The deliverable is a generator checkpoint and a ranker checkpoint with matching
robot, action and observation conventions. Reuse GraspKit's models, optimizers,
renderers, evaluators and trainers. Keep task-specific adapters and recipes out of
`src/`, and generated data/checkpoints in ignored `runs/`.

### Prepare the scene and robot

Set `GRASPKIT_VENDOR` to the source checkout. Read its `AGENTS.md`,
`docs/guide/pipeline.md`, `docs/guide/training.md`, and the experiment matching
the requested robot. Reuse a compatible environment; for a source installation
the full pipeline uses:

```bash
cd "$GRASPKIT_VENDOR"
uv sync --python 3.10 --extra maniskill --extra spconv
```

Use a free CUDA GPU and the selected simulator's assets. Keep simulator and
training environments separate when their dependencies differ. Check the actual
checkout's APIs rather than assuming a historical experiment or pretrained
checkpoint is required.

Read [scene-and-robot.md](references/scene-and-robot.md) to encode scene geometry,
target actors, camera calibration and robot configuration. Match world frames,
metres, joint order and root-to-TCP transforms across data and execution. Split
objects/scenes before augmentation and reserve held-out evaluation inputs.

### Train the two checkpoints

Read [training-and-policy.md](references/training-and-policy.md) for commands,
data contracts and joint inference.

1. Render observations and generate candidate grasps for the supplied scenes.
   Evaluate candidates in the selected physics backend; retain every attempted
   candidate's outcome.
2. Join observations with successful grasps and train the **generator**. Save
   its checkpoint and effective model/robot configuration.
3. Freeze that generator and sample candidates on training scenes in the
   **specific simulator where the policy will be deployed**. Roll out each
   candidate using the deployment robot, controller and execution pipeline;
   label it with that simulator's observed success/failure. Use these labels to
   train the **ranker**, retaining both outcomes. Optimizer samples can augment
   this data after the same target-simulator rollout and labelling.
4. Validate ranking on separate scenes, then evaluate both checkpoints together
   by executing the selected grasp on held-out scenes. Start with a small
   integration run before increasing the dataset and training budget.

```text
scene + robot + cameras
    → observations + physically labelled candidate grasps
        ├─ successful examples → train generator → generator checkpoint
        └─ candidate rollouts in the target deployment simulator
               → observed successes + failures, including generator samples
                                  → train ranker → ranker checkpoint

new observation → generator → candidates → ranker scores
    → feasible candidates in score order → motion planning → execution
```

### Use and hand off the pair

Load each checkpoint with its matching architecture and feature preprocessing.
Generate candidates, score those same candidates, reject grasps violating
collision/reachability constraints, and execute a feasible candidate in score
order. The ranker orders proposals; the robot's planner checks and executes them.

Return both checkpoint paths, their effective configurations, dataset splits,
and an example of joint inference using the selected runtime. Record the
candidate budget and physical evaluation results. Generator loss and ranker AUC
are component metrics; the end-to-end measure is successful execution of the
selected grasp. A different robot or action representation requires a compatible
ranker and feature adapter as well as a compatible generator.
