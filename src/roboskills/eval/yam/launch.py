"""Real YAM arms for the shared launcher: --adapter roboskills.eval.yam.launch -- instruction="..." rig=PATH.

The hardware runs in an episode server of its own (eval/serve.py, natively: the CAN bus and the cameras are this
machine's), so a crashed agent process never leaves an arm mid-command: the server holds it.
"""

import os
from pathlib import Path

from roboskills.eval.serve import close, episode_dir, remote  # noqa: F401  (close: the launcher's, at exit)


def make_vector(out, **kw):
    """The launcher's one episode, its server in out/env0/server and the agent's turns (out/env0) on the panel."""
    env = Path(out) / "env0"
    return [make(**{"out": str(env / "server"), "turns": str(env), **kw})]


def make(instruction, rig=None, out=None, step_limit="9000", vmax="0.5", arms=None, sim="false", port="8080",
         review="true", turns=None, runtime=None, gpu=None, python=None):  # fmt: skip
    """One episode on the rig (`rig`: its YAML, else $ROBOSKILLS_YAM_RIG). `sim=true`: i2rt's MuJoCo arms instead of
    the CAN bus (the operator's panel either way). `review=true`:
    the operator approves every plan in the panel before it moves. `turns`: the agent's folder (core.agent's
    log.jsonl), shown on the panel."""
    rig = rig or os.environ.get("ROBOSKILLS_YAM_RIG")
    if not rig:
        raise ValueError("rig=PATH (the bench's YAML, see eval/yam/SKILL.md) or $ROBOSKILLS_YAM_RIG is required")
    if runtime not in (None, "native"):
        raise ValueError("the YAM runtime is native only: the arms and cameras are this machine's")
    out = episode_dir(out, "runs/yam/episode")
    kw = dict(instruction=instruction, rig=os.path.expanduser(rig), out=out, step_limit=step_limit, vmax=vmax,
              arms=arms, sim=sim, port=port, review=review, turns=turns)  # fmt: skip
    # the operator may take a while to reset the scene and press Start
    # closing waits for the server to park the arms (or for the operator's Release), never kills it mid-move
    return remote(__name__, kw, out=out, image=None, runtime="native", gpu=gpu, python=python, boot_s=3600, stop_s=None)


def episode(instruction, rig, out, step_limit="9000", vmax="0.5", arms=None, sim="false", port="8080", review="true",
            turns=None):  # fmt: skip
    """The episode in the server; i2rt and RealSense are imported here, not at import."""
    from roboskills.eval.yam.robot import Yam

    return Yam(instruction, rig, out, int(step_limit), float(vmax), arms, str(sim).lower() == "true", int(port),
               str(review).lower() == "true", turns)  # fmt: skip
