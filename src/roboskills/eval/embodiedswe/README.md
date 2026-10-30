# EmbodiedSWE-Bench

EmbodiedSWE evaluates coding agents on long-horizon dexterous robotics tasks, its RoboBench presets running on
Isaac Lab. Here `launch.make(task, seed, out, gpu)` runs a preset (a Franka Panda, a fixed G1 with three-finger hands
or a walking G1) behind the shared `Robot` contract with cameras and robot state only. See [SKILL.md](SKILL.md).

## Reference

Zeyu Shen et al., "EmbodiedSWE: Coding Agents for Long Horizon Dexterous Robotics", arXiv 2026.
[Paper](https://arxiv.org/abs/2609.27308) · [Code](https://github.com/EmbodiedSWE/EmbodiedSWE)

```bibtex
@misc{embodiedswe2026,
  title         = {EmbodiedSWE: Coding Agents for Long Horizon Dexterous Robotics},
  author        = {Shen, Zeyu and You, Haoxiang and Liu, Yilang and Zheng, Zhicheng and Zha, Lihan and
                   Yamazaki, Kashu and Zhang, Mingtong and Huang, Suning and Sun, Jiankai and
                   Chen, Qianzhong and He, Lucy and Liu, Kaiyuan and Chang, Haoran and Fragkiadaki, Katerina and
                   Shah, Dhruv and Schwager, Mac and Henderson, Peter and Abraham, Ian and Xu, Canwen},
  year          = {2026},
  eprint        = {2609.27308},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  url           = {https://arxiv.org/abs/2609.27308},
}
```
