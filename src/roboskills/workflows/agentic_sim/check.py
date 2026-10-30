"""The build stage's checks on a task module (build.md gives its contract, README.md lists the checks), or with
``--solution`` the solve stage's on a solution (solve.md), run with the simulators' interpreter (setup.sh):

    python -m roboskills.workflows.agentic_sim.check task.py OUT [--seeds 8] [--solution solution.py]
"""

import argparse
import importlib
import importlib.util
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from PIL import Image


REST_TOLERANCE = 0.02  # rad or m
FRAMES = 150  # about how many frames a video keeps of an episode


def rest_pose(modules):
    """{joint: rest position} from every *REST_Q dict the robokit modules define; the first definition wins."""
    found = [getattr(module, key) for module in map(importlib.import_module, modules)
             for key in dir(module) if key.endswith("REST_Q") and isinstance(getattr(module, key), dict)]  # fmt: skip
    return {joint: position for rest in reversed(found) for joint, position in rest.items()}


def frame(image):
    """``render()``'s image, checked, and its statistics: shape, pixel spread and the share of its commonest colour."""
    image = np.asarray(image)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"render() must return HxWx3 uint8, got {image.dtype} {image.shape}")
    _, counts = np.unique(image.reshape(-1, 3), axis=0, return_counts=True)
    return image, dict(shape=list(image.shape), std=round(float(image.std()), 1),
                       backdrop=round(float(counts.max() / counts.sum()), 3))  # fmt: skip


def write_video(path, frames):
    if frames:
        import imageio.v2 as imageio  # the simulators' interpreter has it (setup.sh)

        imageio.mimwrite(path, frames, fps=30, macro_block_size=1)


def hold_still(module, seed, out, video):
    """Seed ``seed`` held still: its start (object poses, the robot against its rest pose, the start frame, saved to
    ``out``) and the step at which it succeeded, if it did."""
    started = time.monotonic()
    (env := module.make(seed)).reset()
    observed = env.observe()
    row = dict(seed=seed, objects={k: [round(float(x), 4) for x in v] for k, v in observed["objects"].items()})
    q, rest = {k: float(v) for k, v in observed["robot_q"].items()}, rest_pose(env.robot)
    matched = [j for j in q if j in rest]
    row["rest"] = dict(robot=list(env.robot), joints=len(q), matched=len(matched),
                       off={j: round(q[j] - rest[j], 4) for j in matched if abs(q[j] - rest[j]) > REST_TOLERANCE})  # fmt: skip
    image, row["frame"] = frame(env.render())
    Image.fromarray(image).save(out / f"start-{seed}.png")
    steps, row["held_success_step"] = int(env.max_steps), None
    for step in range(steps):
        env.step(None)
        if env.success():
            row["held_success_step"] = step
            break
        if video is not None and step % max(1, steps // FRAMES) == 0:
            video.append(frame(env.render())[0])
    row["seconds"] = round(time.monotonic() - started, 1)
    getattr(env, "close", lambda: None)()
    return row


class Done(BaseException):
    """Ends a solution's episode; not an Exception, so the solution's own handlers let it through."""


def run_solution(module, solution, seed, video):
    """One fresh episode of ``solution.solve(env)``, its steps counted and the task's success read after each."""
    started = time.monotonic()
    (env := module.make(seed)).reset()
    limit, act, state = int(env.max_steps), env.step, dict(solved=False, steps=0)  # the receipt row's order

    def step(action=None):
        if state["steps"] >= limit:
            raise Done
        act(action)
        state["steps"] += 1
        if video is not None and state["steps"] % max(1, limit // FRAMES) == 0:
            video.append(frame(env.render())[0])
        if env.success():
            state["solved"] = True
            raise Done

    env.step, row = step, dict(seed=seed)
    try:
        solution.solve(env)
    except Done:
        pass
    except Exception:  # noqa: BLE001  (the agent reads the traceback)
        row["error"] = traceback.format_exc()[-3000:]
    row.update(state, max_steps=limit, seconds=round(time.monotonic() - started, 1))
    getattr(env, "close", lambda: None)()
    return row


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    spec.loader.exec_module(module := importlib.util.module_from_spec(spec))
    return module


def run_seeds(task, solution, seeds, path, module=None):
    """``solution``'s ``solve(env)`` on every seed of ``task`` (``module`` if loaded): (the rows, an error or None)."""
    rows, video, error = [], [], None
    try:
        sys.path[:0] = [str(solution.parent), str(task.parent)]
        module, policy = module or load(task, "task"), load(solution, "solution")
        for seed in range(seeds):
            rows.append(run_solution(module, policy, seed, video if seed == 0 else None))
    except Exception:  # noqa: BLE001
        error = traceback.format_exc()[-4000:]
    write_video(path, video)
    return rows, error


def solve(task, solution, out, seeds=8):
    """The solve stage's check: ``solution`` solves every seed of ``task``."""
    task, solution, out = Path(task).resolve(), Path(solution).resolve(), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows, error = run_seeds(task, solution, seeds, out / "solve.mp4")
    solved = sum(r["solved"] for r in rows)
    receipt = dict(task=str(task), solution=str(solution), seeds=seeds, **({"error": error} if error else {}),
                   passed=solved == seeds and not error, solved=solved, episodes=rows)  # fmt: skip
    (out / "receipt.json").write_text(json.dumps(receipt, indent=1))
    return receipt


def check(task, out, seeds=8):
    """The build stage's checks on ``task``, with its demo.py."""
    task, out = Path(task).resolve(), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    receipt, rows, video = dict(task=str(task), seeds=seeds), [], []
    try:
        sys.path.insert(0, str(task.parent))
        module = load(task, "task")
        for seed in range(seeds):
            rows.append(hold_still(module, seed, out, video if seed == 0 else None))
    except Exception:  # noqa: BLE001  (the agent reads the traceback and fixes its task)
        receipt["error"] = traceback.format_exc()[-4000:]
    write_video(out / "hold.mp4", video)
    checks = dict(builds=len(rows) == seeds and "error" not in receipt,
                  starts_differ=len(rows) == len({json.dumps(r["objects"], sort_keys=True) for r in rows}) == seeds,
                  holding_still_fails=all(r["held_success_step"] is None for r in rows),
                  robot_at_rest=all(r["rest"]["matched"] > 0 and not r["rest"]["off"] for r in rows),
                  scene_in_frame=all(r["frame"]["std"] > 10 and r["frame"]["backdrop"] < 0.5 for r in rows))  # fmt: skip
    if not (task.parent / "demo.py").exists():
        receipt["demo"] = dict(error="no demo.py beside task.py")
    elif rows:
        demo, error = run_seeds(task, task.parent / "demo.py", seeds, out / "demo.mp4", module)
        receipt["demo"] = dict(solved=sum(r["solved"] for r in demo), episodes=demo,
                               **({"error": error} if error else {}))  # fmt: skip
    checks["demo_solves"] = receipt.get("demo", {}).get("solved", 0) > 0
    receipt.update(passed=bool(rows) and all(checks.values()), checks=checks, episodes=rows)
    (out / "receipt.json").write_text(json.dumps(receipt, indent=1))
    return receipt


def main():
    parser = argparse.ArgumentParser(description="Check a built task module, or a solution to one.")
    parser.add_argument("task", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--solution", type=Path, help="a module defining solve(env): check that it solves every seed")
    args = parser.parse_args()
    if args.solution:
        receipt = solve(args.task, args.solution, args.out, args.seeds)
        brief = dict(passed=receipt["passed"], solved=f"{receipt['solved']}/{args.seeds}", error=receipt.get("error"))
    else:
        receipt = check(args.task, args.out, args.seeds)
        demo = receipt.get("demo", {})
        brief = dict(passed=receipt["passed"], checks=receipt["checks"], error=receipt.get("error"),
                     demo_solved=f"{demo.get('solved', 0)}/{args.seeds}", demo_error=demo.get("error"))  # fmt: skip
    print(json.dumps(brief))


if __name__ == "__main__":
    main()
