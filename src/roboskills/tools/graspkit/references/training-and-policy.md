# Train the two models and use them together

Choose the registered GraspKit experiment for the supplied robot. The experiment
sets the model, robot, trajectory builder and physical evaluator; scene paths,
camera calibration, training budget and output paths come from the current task.
Use the same effective configuration across stages.
Configure `grasp_optim.evaluator` for physical labelling and
`trainer.evaluator` for checking predicted grasps with the selected robot.

## 1. Generate data and train the generator

Set `GRASPKIT_VENDOR`, `GK_RUN` (absolute ignored output directory), `GK_GPU`
(available physical GPU index), `GK_EXPERIMENT` (registered experiment), and
`GK_RUN_NAME` (a fresh run name). Supply `scenes.msgpack`, `camera_pose.json` and
`camera_intrinsic.json` in `GK_RUN`, following
[scene-and-robot.md](scene-and-robot.md).

For an experiment using calibrated world-frame point clouds:

```bash
cd "$GRASPKIT_VENDOR"
gk_exp="$GK_EXPERIMENT"
gk_name="${gk_exp}__${GK_RUN_NAME}"
gk_args=(
  --name "$gk_name"
  --scene_description_path "$GK_RUN/scenes.msgpack"
  --grasp_output_dir "$GK_RUN/grasps"
  --rendered_data_path "$GK_RUN/render"
  --processed_data_path "$GK_RUN/generator_data.h5"
  --grasp_optim.save_labels
  --renderer.camera_poses_path "$GK_RUN/camera_pose.json"
  --renderer.camera_intrinsic_path "$GK_RUN/camera_intrinsic.json"
  --renderer.camera_frame world
  --renderer.point_cloud_frame world
)
uv run python -c 'import graspkit; graspkit.link()'
mkdir -p logs/train "$GK_RUN/generator_train"
ln -s "$GK_RUN/generator_train" "logs/train/$gk_name"

uv run scripts/launch.py "$gk_exp" "${gk_args[@]}" --gpus "$GK_GPU" \
  --steps render_data optimize_grasps process_sim_data
CUDA_VISIBLE_DEVICES="$GK_GPU" uv run scripts/stages/train/train.py \
  "$gk_exp" "${gk_args[@]}"
```

Use a fresh log destination; do not replace an existing run. Adjust the renderer
for other camera setups. Set the batch size no larger than the retained training
scene count per rank, and choose iterations, save frequency and evaluation
frequency for the actual budget. Pass these overrides explicitly to training.

The launcher's training subprocess in the inspected implementation forwards
only the experiment name, losing custom paths and settings. The commands above
run data stages through the launcher and training directly with the same
overrides. Recheck `scripts/launch.py:run_train` in the selected checkout before
using it for custom training. Completion markers can skip changed inputs; use
fresh run paths after changing data or configuration.

| Stage | Check before continuing |
|---|---|
| Rendering | Target points are visible, segmented and in the label frame |
| Grasp generation/evaluation | Successful grasp configurations and outcomes of all attempted candidates are retained |
| Processing | `data/<scene_id>/{obs,actions}` joins the correct scenes; actions are `(K,3,D)` pre/grasp/post configurations |
| Generator training | Native `checkpoints/iter_*.pt` and its effective experiment/model configuration are saved |

The generator learns from successful grasps. Keep unsuccessful candidate records
too; they can seed candidate collection for the ranker. If these labels came
from another simulator or execution protocol, roll out the candidates again in
the target deployment simulator before using them as ranker supervision.
Inspect how many scenes remain in each split after the join.

Use `GK_GENERATOR_CKPT` for the selected native checkpoint path. It need not have
a particular filename. Conditional generators use their corresponding trainer
and loader; do not substitute a generic model architecture when loading them.

## 2. Collect ranking labels and train the ranker

**Ranker labels must come from rollouts in the specific target deployment
simulator.** The generator may have been trained with a different simulator;
its optimizer acceptance flags or source-simulator successes are not target
deployment labels.

Freeze the selected generator while collecting labels. For each **training
scene** in the target simulator:

1. Capture the pre-action observation through the deployment observation path
   and sample candidates from the generator.
2. Restore the same initial scene/robot state for each candidate and roll out
   the deployment pipeline: arm planning, approach, closing and lift/hold, using
   the actual deployment robot assets, controller and physics settings.
3. Read the target simulator's success predicate after the rollout and save the
   pre-action observation, candidate and success/failure label together. Keep
   both successful and failed executions. Simulator/setup errors are incomplete
   samples, not physical failure labels.

Apply the same rollout protocol to any added optimizer-generated candidates.
Record simulator/version, robot/controller configuration and success definition
with the dataset, so the ranker learns which proposals work in its deployment
environment. Collect validation labels there too, on separate scenes/objects.

Each ranker example contains:

| Field | Purpose |
|---|---|
| Scene/observation ID and split | Join the correct point cloud; prevent split leakage |
| Candidate grasp/action | Pose and hand configuration, or trajectory representation required by the ranker |
| Physical label | Observed success or failure from the target deployment simulator's rollout |
| Candidate source | Generator or optimizer; retain useful failures from both |

Fit the ranker to predict grasp success, normally with binary cross-entropy on
success logits. Match feature construction at training and inference: point
sampling, coordinate transforms, units, table/context features and any hand
configuration inputs. Hold out scenes or objects for validation; use AUC and
per-scene top-ranked success to inspect the scorer. Keep a separate final set
untouched by both model training and checkpoint selection.

Find the existing ranker implementation for the selected robot in the checkout.
For the current standalone point-cloud ranker, the entrypoint is
`scripts/diverse_grasps/train_ranker.py`, with `bank`, `pool`/`clicks`, and `train`
subcommands. Its `bank` and fitting commands accept explicit data paths:

Export the pre-action **target-simulator observations** to
`GK_RUN/deploy_observations` in the bank reader's native observation format
(point clouds, segmentation and camera poses keyed by scene ID). Match them to
the candidate rollout labels. Use the earlier rendered observations only when
they already represent this same deployment observation path and scene state.

```bash
uv run scripts/diverse_grasps/train_ranker.py bank \
  --render "$GK_RUN/deploy_observations" --out "$GK_RUN/ranker_bank.npz"
CUDA_VISIBLE_DEVICES="$GK_GPU" uv run scripts/diverse_grasps/train_ranker.py train \
  --dirs "$GK_RUN/ranker_labels" \
  --bank "$GK_RUN/ranker_bank.npz" --scenes "$GK_RUN/scenes.msgpack" \
  --out "$GK_RUN/ranker.pt"
```

Prepare `ranker_labels` from the target-simulator rollouts before fitting.
This trainer accepts `labels.h5` with `scene_id`, `grasp` and `isaac_success`, or
`index.h5` plus evaluated `out_<shard>.h5` files. Its source defines the exact
shapes and bank schema. The checkpoint contains `ranker` weights and a `report`.

This existing standalone implementation has a particular gripper pose encoding
and assumes table Z=0 during fitting. Its candidate-collection commands also
have experiment-specific defaults. For another robot/frame, adapt the data and
feature interface in task artifacts, reusing the existing model/trainer where
compatible. Do not silently feed different joint dimensions, TCP poses or table
frames into it. A policy's integrated evaluator uses a different checkpoint/API
from a separately saved ranker.

## 3. Load both checkpoints and select a grasp

Use the generator's native loader to restore its network and preprocessing, and
the ranker's native loader to restore its scoring network. For generic current
GraspKit policies the generator loader is
`graspkit.models.checkpoint.load_policy`; the standalone ranker uses its
matching `load_ranker` / `score_grasps` helper. Confirm the signatures in the
selected checkout. A generator checkpoint and a ranker checkpoint are different
types and must not be loaded interchangeably.

The joint algorithm is the following pseudocode; the pose/feature adapter is
the one used to build that ranker's training examples:

```text
generator = load generator checkpoint with its saved configuration
ranker = load ranker checkpoint with its saved feature configuration
observation = preprocess the new scene as during training
candidates = generator.sample(observation, candidate_budget)
scores = ranker.score(observation, encode_candidates(candidates))
for candidate in descending score order:
    trajectory = plan candidate for the selected arm and current obstacles
    if trajectory satisfies collision, reachability and control constraints:
        execute trajectory
        return observed grasp outcome
return no feasible candidate
```

For a point-cloud `GraspPolicy`, candidate generation is
`policy.get_action(obs, num_samples_per_scene=N)`, normally returning
`(B,N,3,D)`. Score the same candidates without losing their indices. Convert
floating root poses to the ranker's expected frame once, then apply the robot's
root-to-TCP and joint-command mapping when building executable trajectories.
Keep observations and scene poses fixed while ranking one candidate set.

If using an existing adapter that exposes the `GRASP_POLICY_*` environment
interface, configure **both** models and explicitly select learned ranking:

```bash
export GRASP_POLICY_CKPT="$GK_GENERATOR_CKPT"
export GRASP_POLICY_RANKER="$GK_RUN/ranker.pt"
export GRASP_POLICY_RANK=ranker
export GRASP_POLICY_N="$GK_CANDIDATE_BUDGET"
```

These are adapter settings, not a universal GraspKit API. Reuse the adapter for
its supported robot and check its policy environment and model compatibility.
Robot-specific reachability ranking is optional; the robot's planner still
needs to validate the actual selected arm. Verify both models are exercised by
a proposal call, since some runtimes load the ranker lazily.

## 4. Evaluate and deliver

Evaluate the pair on held-out scenes with a declared candidate budget and one
selected grasp execution per scene. Report the physical success rate of the
selected grasp; ranker validation metrics and generator training loss do not
replace this test. Deliver both checkpoint paths, effective model/feature/robot
configurations, training/validation splits, candidate budget and the joint
inference entrypoint. Update ranker training data if the generator's proposal
distribution changes substantially.
