"""LIBERO / LIBERO-Pro entrypoint for the shared launcher: --adapter roboskills.eval.libero.launch."""

import math
from pathlib import Path

from roboskills.eval.serve import close, episode_dir, remote  # noqa: F401  (close: the launcher's, at exit)


IMAGE = "ghcr.io/openrobokit/roboskills-libero:0.2.0"


def make(suite, task="0", seed="1", out=None, profile="matched-v3", runtime=None, image=None, gpu=None):
    """One native RGB-D episode of `suite`'s task `task`, procedurally reset with `seed`, its simulator in a server of
    its own (`runtime`: native, docker or apptainer of `image`; see eval/serve.py). The default profile is the
    reference-matched protocol (seeds 1-50 are the evaluation seeds)."""
    out = episode_dir(out, f"runs/libero/{suite}-{task}-s{seed}")  # a new one each episode unless given
    kw = dict(suite=suite, task=task, seed=seed, out=out, profile=profile)
    return remote(__name__, kw, out=out, image=image or IMAGE, runtime=runtime, gpu=gpu)


def episode(suite, task, seed, out, profile):
    """The episode in the server; simulator imports occur here, not at import."""
    from roboskills.eval.libero.robot import Libero

    robot = Libero(suite, Path(out) / "robot", task=int(task), profile=profile)
    robot.reset(int(seed), deadline=math.inf, evaluation=True)
    return robot
