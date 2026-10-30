"""Check a benchmark's runtime without a model: a no-op episode fails, an oracle succeeds, and the runtimes agree.

    python -m roboskills.eval.check roboskills.eval.libero.launch --oracles ORACLES.json \\
        --runtimes native,docker --out runs/check/libero [--only NAME] [-- k=v ...]

The manifest lists episodes, {"oracles": [{"kw": {...}, "actions": "libero/x.npy"}, ...]}: each `actions`, a path
beside the manifest, is a successful episode's recorded native actions (.npy) or the requests its adapter sent
(.jsonl); `k=v` after `--` (a GPU, say) go to every episode. For each episode and runtime, two episodes of the
adapter (the first robot of `make_vector(**kw, out=, runtime=)`, or `make`'s): one finished at once (the no-op, which
must not succeed), and one that hands the oracle to the adapter's `replay` before finishing (which must succeed); an
entry without `actions` runs the no-op only. The oracle's final proprioception must agree across runtimes to `--tol`.
An episode that raises fails its check. Writes `check.json` to `--out`, prints one line per episode and exits 1 when a
check fails.
"""

import argparse
import importlib
import json
import sys
import traceback
from pathlib import Path

import numpy as np


def episode(mod, kw, runtime, out, actions=None):
    out.mkdir(parents=True, exist_ok=True)  # a container mounts it
    try:
        if hasattr(mod, "make_vector"):  # as the launcher picks it
            robot = mod.make_vector(**kw, out=str(out), runtime=runtime)[0]
        else:
            robot = mod.make(**kw, out=str(out), runtime=runtime)
        obs = robot.obs(cams=False) if actions is None else robot.replay(actions)
        result = robot.finish(out / "result")
    finally:
        mod.close()
    q = {k: np.asarray(v).tolist() for k, v in obs.q.items()}
    return dict(success=bool(result["success"]), steps=int(obs.steps), q=q)


def attempt(mod, kw, runtime, out, actions=None):
    """`episode`, or its error (as a failed episode)."""
    try:
        return episode(mod, kw, runtime, out, actions)
    except Exception as e:  # noqa: BLE001  (the report carries it)
        traceback.print_exc()
        return dict(success=False, steps=None, q=None, error=f"{type(e).__name__}: {e}")


def recorded(path):
    """An oracle: native actions (.npy), or the requests an adapter sent (.jsonl, one JSON object per line)."""
    path = Path(path)
    return np.load(path) if path.suffix == ".npy" else [json.loads(line) for line in path.read_text().splitlines()]


def check(module, oracles, runtimes, out, tol):
    """`oracles`: [(name, kw, actions path or None)]. Returns the report, with its failures."""
    mod = importlib.import_module(module)
    report = dict(module=module, runtimes=runtimes, episodes={}, failures=[])
    for name, kw, path in oracles:
        rows = report["episodes"][name] = dict(kw=kw, oracle=str(path) if path else None)
        for runtime in runtimes:
            noop = attempt(mod, kw, runtime, out / name / runtime / "noop")
            rows[runtime] = dict(noop=noop)
            if "error" in noop:
                report["failures"].append(f"{name} {runtime}: the no-op raised {noop['error']}")
            elif noop["success"]:
                report["failures"].append(f"{name} {runtime}: the no-op succeeded")
            if path:
                solved = rows[runtime]["oracle"] = attempt(
                    mod, kw, runtime, out / name / runtime / "oracle", recorded(path)
                )
                if not solved["success"]:
                    report["failures"].append(f"{name} {runtime}: the oracle failed {solved.get('error', '')}".strip())
        first, *rest = runtimes
        for runtime in rest if path else ():
            a, b = (rows[r]["oracle"] for r in (first, runtime))
            if a["q"] is None or b["q"] is None:  # one raised: already a failure
                continue
            gap = max(float(np.max(np.abs(np.subtract(a["q"][k], b["q"][k])))) for k in a["q"])
            rows[f"gap_{first}_{runtime}"] = gap
            if gap > tol or a["steps"] != b["steps"] or a["success"] != b["success"]:
                steps = f"{a['steps']}/{b['steps']}"
                report["failures"].append(f"{name}: {first} and {runtime} disagree (steps {steps}, gap {gap:.2e})")
    return report


def load(manifest, only=None, extra=None):
    """The manifest's episodes as [(name, kw, actions path or None)], `extra` added to every kw."""
    manifest, oracles = Path(manifest), []
    for entry in json.loads(manifest.read_text())["oracles"]:
        name = "-".join(str(v) for v in entry["kw"].values())
        if not only or only in name:
            kw = {k: str(v) for k, v in {**entry["kw"], **(extra or {})}.items()}
            oracles.append((name, kw, manifest.parent / entry["actions"] if entry.get("actions") else None))
    return oracles


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("module", help="the adapter's launch module")
    p.add_argument("--oracles", type=Path, required=True, help="the manifest of recorded successful episodes")
    p.add_argument("--only", help="only the episodes whose name contains this")
    p.add_argument("--runtimes", default="native", help="comma list of native, docker, apptainer")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--tol", type=float, default=1e-4, help="the largest joint gap between runtimes, rad or m")
    argv = sys.argv[1:]
    cut = argv.index("--") if "--" in argv else len(argv)
    a = p.parse_args(argv[:cut])
    extra, runtimes = dict(x.split("=", 1) for x in argv[cut + 1 :]), a.runtimes.split(",")
    report = check(a.module, load(a.oracles, a.only, extra), runtimes, a.out.resolve(), a.tol)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "check.json").write_text(json.dumps(report, indent=1))
    for name, rows in report["episodes"].items():
        cells = (
            f"| {r}: no-op {rows[r]['noop']['success']}, oracle {rows[r].get('oracle', {}).get('success')}"
            for r in runtimes
        )
        print(name, *cells)
    print(json.dumps(dict(episodes=len(report["episodes"]), failures=report["failures"]), indent=1))
    sys.exit(1 if report["failures"] else 0)


if __name__ == "__main__":
    main()
