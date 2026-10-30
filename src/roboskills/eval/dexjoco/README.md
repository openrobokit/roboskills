# DexJoCo

DexJoCo is a benchmark and toolkit for task-oriented dexterous manipulation in MuJoCo. Here
`launch.make(task, seed, out, ...)` runs its native single-arm and bimanual RGB-D episodes (Panda arms with Allegro
hands) behind the shared `Robot` contract, scored by the benchmark's own checks. See [SKILL.md](SKILL.md).

## Reference

Hanwen Wang et al., "DexJoCo: A Benchmark and Toolkit for Task-Oriented Dexterous Manipulation on MuJoCo",
arXiv 2026. [Paper](https://arxiv.org/abs/2605.16257) · [Code](https://github.com/brave-eai/dexjoco)

```bibtex
@misc{wang2026dexjocobenchmarktoolkittaskoriented,
      title={DexJoCo: A Benchmark and Toolkit for Task-Oriented Dexterous Manipulation on MuJoCo},
      author={Hanwen Wang and Weizhi Zhao and Xiangyu Wang and Siyuan Huang and He Lin and Boyuan Zheng and Rongtao Xu and Gang Wang and Yao Mu and He Wang and Lue Fan and Hongsheng Li and Zhaoxiang Zhang and Tieniu Tan},
      year={2026},
      eprint={2605.16257},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2605.16257},
}
```
