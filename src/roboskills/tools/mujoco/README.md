# MuJoCo

MuJoCo is a physics engine for model-based control and robotics. Here `engine.py` builds scenes around an articulated
MJCF actor, replays actuator controls under physics with contact feedback and renders saved states, and
`python -m roboskills.tools.mujoco` builds a scene from perception outputs. See [SKILL.md](SKILL.md).

## Reference

Emanuel Todorov et al., "MuJoCo: A physics engine for model-based control", IROS 2012.
[Paper](https://doi.org/10.1109/IROS.2012.6386109) · [Code](https://github.com/google-deepmind/mujoco) ·
[Docs](https://mujoco.readthedocs.io/)

```bibtex
@inproceedings{todorov2012mujoco,
  title={MuJoCo: A physics engine for model-based control},
  author={Todorov, Emanuel and Erez, Tom and Tassa, Yuval},
  booktitle={2012 IEEE/RSJ International Conference on Intelligent Robots and Systems},
  pages={5026--5033},
  year={2012},
  organization={IEEE},
  doi={10.1109/IROS.2012.6386109}
}
```
