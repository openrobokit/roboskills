"""RoboCasa365 entrypoint for the shared launcher: --adapter roboskills.eval.robocasa365.launch."""

from pathlib import Path

from roboskills.eval.robocasa365 import TASKS
from roboskills.eval.serve import close, episode_dir, remote  # noqa: F401  (close: the launcher's, at exit)


IMAGE = "ghcr.io/openrobokit/roboskills-robocasa365:0.2.0"


def make(task, seed="1", out=None, runtime=None, image=None, gpu=None):
    """One episode, its simulator in a server of its own (`runtime`: native, docker or apptainer of `image`; see
    eval/serve.py); `seed` picks RoboCasa's kitchen layout and style, the robot's start and the objects."""
    if task not in TASKS:
        raise ValueError(f"Unsupported RoboCasa365 task: {task}; choose from {tuple(TASKS)}")
    out = episode_dir(out, f"runs/robocasa365/{task}-s{seed}")  # a new one each episode unless given
    kw = dict(task=task, seed=seed, out=out)
    return remote(__name__, kw, out=out, image=image or IMAGE, runtime=runtime, gpu=gpu)


def episode(task, seed, out):
    """The episode in the server, its files in `out`/robot; simulator imports occur here, not at import."""
    from roboskills.eval.robocasa365.robot import RoboCasa

    return RoboCasa(task, int(seed), Path(out) / "robot")
