# Robosuite (CaP-X tasks)

CaP-X benchmarks coding agents for robot manipulation; seven of its tasks run on robosuite, the modular MuJoCo
simulation framework for robot learning. Here `launch.make(task, seed, out)` rebuilds those tasks' scenes, resets, step
limits and success checks on CaP-X's robosuite fork behind the shared `Robot` contract, running no CaP-X code.
See [SKILL.md](SKILL.md).

## Reference

Letian Fu et al., "CaP-X: A Framework for Benchmarking and Improving Coding Agents for Robot Manipulation", arXiv 2026.
[Paper](https://arxiv.org/abs/2603.22435) · [Code](https://github.com/capgym/cap-x). The BibTeX is arXiv's; the
upstream README's omits an author and gives the year as 2025.

```bibtex
@misc{fu2026capxframeworkbenchmarkingimproving,
      title={CaP-X: A Framework for Benchmarking and Improving Coding Agents for Robot Manipulation},
      author={Letian Fu and Justin Yu and Karim El-Refai and Ethan Kou and Haoru Xue and Huang Huang and Wenli Xiao and Guanzhi Wang and Dantong Niu and Fei-Fei Li and Guanya Shi and Jiajun Wu and Shankar Sastry and Yuke Zhu and Ken Goldberg and Linxi "Jim" Fan},
      year={2026},
      eprint={2603.22435},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2603.22435},
}
```

Yuke Zhu et al., "robosuite: A Modular Simulation Framework and Benchmark for Robot Learning", arXiv 2020.
[Paper](https://arxiv.org/abs/2009.12293) · [Code](https://github.com/ARISE-Initiative/robosuite) ·
[Project](https://robosuite.ai)

```bibtex
@inproceedings{robosuite2020,
  title={robosuite: A Modular Simulation Framework and Benchmark for Robot Learning},
  author={Yuke Zhu and Josiah Wong and Ajay Mandlekar and Roberto Mart\'{i}n-Mart\'{i}n and Abhishek Joshi and Soroush Nasiriany and Yifeng Zhu and Kevin Lin},
  booktitle={arXiv preprint arXiv:2009.12293},
  year={2020}
}
```
