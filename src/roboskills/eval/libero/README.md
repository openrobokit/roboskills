# LIBERO and LIBERO-Pro

LIBERO is a benchmark of language-specified manipulation tasks for lifelong robot learning, one Panda arm in MuJoCo;
LIBERO-Pro re-evaluates its suites with perturbed objects, positions, instructions, tasks and environments to test
robustness beyond memorization. Here `launch.make(suite, task, seed, out, ...)` runs any LIBERO-Pro suite and task
natively behind the shared `Robot` contract, scored by LIBERO's predicates. See [SKILL.md](SKILL.md).

## Reference

Bo Liu et al., "LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning", arXiv 2023.
[Paper](https://arxiv.org/abs/2306.03310) · [Code](https://github.com/Lifelong-Robot-Learning/LIBERO)

```bibtex
@article{liu2023libero,
  title={LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning},
  author={Liu, Bo and Zhu, Yifeng and Gao, Chongkai and Feng, Yihao and Liu, Qiang and Zhu, Yuke and Stone, Peter},
  journal={arXiv preprint arXiv:2306.03310},
  year={2023}
}
```

Xueyang Zhou et al., "LIBERO-PRO: Towards Robust and Fair Evaluation of Vision-Language-Action Models Beyond
Memorization", arXiv 2025. [Paper](https://arxiv.org/abs/2510.03827) · [Code](https://github.com/Zxy-MLlab/LIBERO-PRO)
(this runtime pins the fork `uynitsuj/LIBERO-PRO`). The BibTeX is arXiv's; the upstream README's has placeholder fields.

```bibtex
@misc{zhou2026liberoprorobustfairevaluation,
      title={LIBERO-PRO: Towards Robust and Fair Evaluation of Vision-Language-Action Models Beyond Memorization},
      author={Xueyang Zhou and Yangming Xu and Guiyao Tie and Yongchao Chen and Guowen Zhang and Duanfeng Chu and Pan Zhou and Lichao Sun},
      year={2026},
      eprint={2510.03827},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2510.03827},
}
```
