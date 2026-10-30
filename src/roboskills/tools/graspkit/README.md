# GraspKit

GraspKit trains a grasp generator and a separate grasp success ranker from scenes and robot assets. Here
`bot.grasp_poses` reaches it through the module `$GRASP_BACKEND` names (RoboDojo's `grasp_backend.py` runs GraspKit's
ARX X5 policy server) and gets grasp poses best first; [SKILL.md](SKILL.md) also covers training both checkpoints.

## Reference

GraspKit has no paper; cite the project:
[openrobokit/graspkit](https://github.com/openrobokit/graspkit).
