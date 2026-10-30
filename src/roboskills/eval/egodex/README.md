# EgoDex

EgoDex is a large-scale egocentric video dataset of tabletop dexterous manipulation recorded on Apple Vision Pro,
with 3D head, upper-body and hand poses per frame (CC BY-NC-ND). Here
`python -m roboskills.eval.egodex.read CLIP.hdf5 [--hand left] --out JOINTS.npz` reads one hand's 21 keypoints in the
camera frame and the camera's field of view, for real2sim's `fit --joints`. See [read.py](read.py).

## Reference

Ryan Hoque et al., "EgoDex: Learning Dexterous Manipulation from Large-Scale Egocentric Video", arXiv 2025
(ICLR 2026). [Paper](https://arxiv.org/abs/2505.11709) · [Code](https://github.com/apple/ml-egodex)

```bibtex
@misc{egodex,
      title={EgoDex: Learning Dexterous Manipulation from Large-Scale Egocentric Video},
      author={Ryan Hoque and Peide Huang and David J. Yoon and Mouli Sivapurapu and Jian Zhang},
      year={2025},
      eprint={2505.11709},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2505.11709},
}
```
