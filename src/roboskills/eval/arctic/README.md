# ARCTIC

ARCTIC is a dataset of two hands dexterously manipulating articulated objects, with motion-captured MANO hands and
object poses, filmed from eight third-person views and one egocentric view. Here
`python -m roboskills.eval.arctic.read ROOT s01/laptop_use_01 --view 4 --frames 80 175 --out DIR` reads one
sequence's ground truth from one camera into a real2sim track: both hands' keypoints and the hinged object.
See [read.py](read.py).

## Reference

Zicong Fan et al., "ARCTIC: A Dataset for Dexterous Bimanual Hand-Object Manipulation", CVPR 2023.
[Paper](https://arxiv.org/abs/2204.13662) · [Code](https://github.com/zc-alexfan/arctic) ·
[Project](https://arctic.is.tue.mpg.de)

```bibtex
@inproceedings{fan2023arctic,
  title = {{ARCTIC}: A Dataset for Dexterous Bimanual Hand-Object Manipulation},
  author = {Fan, Zicong and Taheri, Omid and Tzionas, Dimitrios and Kocabas, Muhammed and Kaufmann, Manuel and Black, Michael J. and Hilliges, Otmar},
  booktitle = {Proceedings IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2023}
}
```
