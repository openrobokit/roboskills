"""CaP-X Robosuite entrypoint for the shared launcher: --adapter roboskills.eval.robosuite.launch."""

from roboskills.eval.robosuite import TASKS
from roboskills.eval.serve import close, episode_dir, remote  # noqa: F401  (close: the launcher's, at exit)


IMAGE = "ghcr.io/openrobokit/roboskills-robosuite:0.2.0"


def make(task, seed="1", out=None, runtime=None, image=None, gpu=None, information="observations",
         python=None):  # fmt: skip
    """One episode; `seed` fixes the layout. Its simulator runs in a server of its own (`runtime`: native, docker or
    apptainer of `image`; see eval/serve.py)."""
    if task not in TASKS:
        raise ValueError(f"Unsupported Robosuite task: {task}; choose from {tuple(TASKS)}")
    out = episode_dir(out, f"runs/robosuite/{task}-s{seed}")  # a new one each episode unless given
    kw = dict(task=task, seed=seed, out=out, information=information)
    return remote(__name__, kw, out=out, image=image or IMAGE, runtime=runtime, gpu=gpu, python=python)


def episode(task, seed, out, information="observations"):
    """The episode in the server; simulator imports occur here, not at import."""
    from roboskills.eval.robosuite.robot import Robosuite

    return Robosuite(task, int(seed), out, information=information)
