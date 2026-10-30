"""Native DexJoCo entrypoint for the shared launcher: --adapter roboskills.eval.dexjoco.launch."""

from pathlib import Path

from roboskills.eval.dexjoco import TASKS
from roboskills.eval.serve import close, episode_dir, remote  # noqa: F401  (close: the launcher's, at exit)


IMAGE = "ghcr.io/openrobokit/roboskills-dexjoco:0.2.0"


def make(task, seed="20000", out=None, vendor=None, runtime=None, image=None, gpu=None):
    """One native RGB-D episode, its simulator in a server of its own (`runtime`: native, docker or apptainer of
    `image`; see eval/serve.py). `vendor` is the prepared checkout as the server sees it (default $DEXJOCO_VENDOR)."""
    if task not in TASKS:
        raise ValueError(f"Unsupported DexJoCo task: {task}; choose from {TASKS}")
    out = episode_dir(out, f"runs/dexjoco/{task}-s{seed}")  # a new one each episode unless given
    kw = dict(task=task, seed=seed, out=out, vendor=vendor and Path(vendor).resolve())
    return remote(__name__, kw, out=out, image=image or IMAGE, runtime=runtime, gpu=gpu)


def episode(task, seed, out, vendor=None):
    """The episode in the server; simulator imports occur here, not at import."""
    from roboskills.eval.dexjoco.robot import DexJoCo

    return DexJoCo(task, int(seed), out, vendor=vendor)
