"""Replay a finished `vector.py` batch from its event log (`--events`): the same batch is built, then every physics
tick's commands, rendered observation and episode end are applied in the order they happened. Writes each env's
head-camera frames (`env{i}_frames.npz`; `launch.replay` makes them videos) and `replay.json`: each episode's recorded
and replayed outcome, and how far its final joints and object positions moved apart.

    /isaac-sim/python.sh sim/replay.py --events RUN/sim_events.jsonl --out RUN/replay --vendor VENDOR/RoboDojo
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


sys.path.append(str(Path(__file__).resolve().parents[4]))  # the directory holding `roboskills` (mounted)
from roboskills.eval.robodojo.sim.vector import ending, start  # noqa: E402


def replay(env, events):
    """Apply the logged events after the batch line to `env`, a fresh build of that batch. Returns {env: comparison}
    for the episodes that finished and {env: [head frames]}, one per rendered observation."""
    results, frames = {}, {}
    for event in events:
        op = event["op"]
        if op == "act":
            env.joint_actions({int(e): action for e, action in event["envs"].items()})
        elif op == "observe":
            rgb = env.observe(event["env"], True)["cams"]["cam_head"]["rgb"]
            frames.setdefault(event["env"], []).append(np.asarray(rgb)[::2, ::2])  # as robot.py's video
        elif op in ("finish", "disconnect"):
            e = event["env"]
            if op == "finish" and env.dev:
                env.truth(e)  # as serve: the dev record reads the native checks before the final one
            env.terminal(e, final=True)
            if op == "finish":
                results[e] = compare(event, ending(env, e))
    return results, frames


def compare(logged, replayed):
    """One episode's logged and replayed ending: outcomes, and the largest joint (rad) and object position (m) gaps."""
    joints = max(
        (float(np.max(np.abs(np.asarray(v, float) - np.asarray(replayed["state"][k], float))))
         for k, v in logged["state"].items() if k.endswith("_joint_state")),
        default=0.0,
    )  # fmt: skip
    a, b = logged["poses"], replayed["poses"]
    objects = None
    if "error" not in a and "error" not in b:
        objects = max(
            (float(np.linalg.norm(np.subtract(a[k][:3], b[k][:3]))) for k in a.keys() & b.keys()), default=0.0
        )
    return dict(
        success=[logged["success"], replayed["success"]],
        score=[logged["score"], replayed["score"]],
        same_outcome=logged["success"] == replayed["success"] and abs(logged["score"] - replayed["score"]) < 1e-6,
        joints_rad=joints,
        objects_m=objects,
        unmatched_objects=sorted(a.keys() ^ b.keys()) if objects is not None else None,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--events", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--vendor", type=Path, required=True)
    a = p.parse_args()
    with open(a.events) as f:
        events = [json.loads(line) for line in f]
    batch = events[0]
    if batch["op"] != "batch":
        raise ValueError(f"{a.events} does not start with its batch")
    env, _ = start(batch["task"], batch["mode"], batch["suite"], batch["layouts"], batch["seed"], a.vendor,
                   batch["official"])  # fmt: skip
    if sorted(env.fixes) != sorted(batch["fixes"]):
        raise RuntimeError(f"the batch ran with fixes {batch['fixes']}, this build has {env.fixes}")
    env.step_lim = batch["step_limit"]
    results, frames = replay(env, events[1:])
    a.out.mkdir(parents=True, exist_ok=True)
    for e, rgb in frames.items():
        np.savez_compressed(a.out / f"env{e}_frames.npz", rgb=np.stack(rgb))
    episodes = {f"env{e}": r for e, r in sorted(results.items())}
    (a.out / "replay.json").write_text(json.dumps(dict(batch=batch, episodes=episodes), indent=1))
    print("replayed", json.dumps({e: r["same_outcome"] for e, r in sorted(results.items())}), flush=True)
    os._exit(0)  # Kit's shutdown can hang; everything is written


if __name__ == "__main__":
    main()
