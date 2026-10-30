"""RoboTwin 2.0 entrypoint for the shared launcher: --adapter roboskills.eval.robotwin.launch."""

from roboskills.eval.serve import close, episode_dir, remote  # noqa: F401  (close: the launcher's, at exit)


IMAGE = "ghcr.io/openrobokit/roboskills-robotwin:0.2.0"


def make(task, seed, information="observations", out=None, runtime=None, image=None, gpu=None, python=None):
    """One episode of `task` as RoboTwin's eval starts it with `seed`, its simulator in a server of its own
    (`runtime`: native, docker or apptainer of `image`; see eval/serve.py)."""
    out = episode_dir(out, f"runs/robotwin/{task}-s{seed}")  # a new one each episode unless given
    kw = dict(task=task, seed=seed, information=information)
    return remote(__name__, kw, out=out, image=image or IMAGE, runtime=runtime, gpu=gpu, python=python)


def episode(task, seed, information="observations"):
    """The episode in the server; simulator imports occur here, not at import."""
    from roboskills.eval.robotwin.robot import RoboTwin

    r = RoboTwin(task, information=information)
    r.reset(int(seed))
    return r
