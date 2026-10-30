---
name: video2world
description: Evaluate frozen reconstruction submissions on Video2World with official scores and synchronized comparison videos. Use for matched benchmark comparisons, evaluator recovery, or reporting existing runs.
---

# Video2World evaluation

The shared real2sim workflow stays dataset-, simulator- and embodiment-independent.

## Matched comparison

- Register case IDs, dataset/code revisions, methods, source budgets and scoring
  scope before reconstruction. Charge shared perception to every method; disclose
  common actor/export adaptations. A development subset is not the full benchmark.
- Keep reconstruction inputs source-only. Seal complete candidate packages and
  end source jobs before releasing references. Use the workflow's `freeze` and
  `verify`; retain source tool logs for access review. Prompt/log checks do not
  establish operating-system isolation.
- Execute each sealed package through the pinned native evaluator. Preserve
  its metric contracts, controller settings, clocks and family-balanced score.
  Missing primary measurements block the aggregate; keep every registered case.
- Separate infrastructure errors from candidate failures. Repair setup, then
  retry the same sealed package; preserve every attempt and select by completion,
  never score. Resume CPU metrics from a completed recording when possible.
- Report official V2WScore, task-success counts and native hand validity
  separately. Disclose benchmark attachments/grasp adjustments; a build or replay
  does not establish task success, and a partial score is not a success rate.
- Produce source/native comparison videos on elapsed seconds. Use the recorded
  execution clock; hold a shorter clip's last frame. Missing clocks or media stay
  explicit. Do not stretch trajectories or use source-inferred object playback
  as evidence of physical execution.

The adapter
requires an explicit allocated GPU and runtime configuration; it does not choose
hardware. Publishing a report needs the user's authorization and the applicable
publication workflow. Finish by preserving evidence and releasing only the run's
owned workers, containers and temporary environments.
