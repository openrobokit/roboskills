# RoboDojo

RoboDojo is a unified sim-and-real benchmark for evaluating generalist robot manipulation policies, its simulation
tasks on Isaac Sim. Here `launch.make_vector(task, layouts, gpu, out)` runs its two-arm ARX X5 episodes in the
simulator container, one `Robot` per env behind the shared contract, with its eval suites. See [SKILL.md](SKILL.md).

## Reference

Tianxing Chen et al., "RoboDojo: A Unified Sim-and-Real Benchmark for Comprehensive Evaluation of Generalist Robot
Manipulation Policies", arXiv 2026. [Paper](https://arxiv.org/abs/2607.04434) ·
[Code](https://github.com/RoboDojo-Benchmark/RoboDojo) · [Project](https://robodojo-benchmark.com/)

```bibtex
@article{chen2026robodojo,
  title={{RoboDojo}: A Unified Sim-and-Real Benchmark for Comprehensive Evaluation of Generalist Robot Manipulation Policies},
  author={Chen, Tianxing and Chen, Yue and Li, Zixuan and Tang, Junyuan and Su, Kailun and Wan, Weijie and Chen, Baijun and Lu, Haoran and Yan, Haowen and Su, Honghao and others},
  journal={arXiv preprint arXiv:2607.04434},
  year={2026}
}
```
