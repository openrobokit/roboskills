"""A retargeted hand moving the tracked object in MuJoCo: RoboKit's SimMPC plans on MuJoCo Warp and executes in CPU
MuJoCo, and the engine replays the executed controls into a delivery.

    python -m roboskills.tools.robokit mpc --video V --retarget retarget/ --out mpc/ [--frames 12 --samples 64]
"""

import json
import shutil
from pathlib import Path

import numpy as np


LIMITS = dict(pos_m=0.05, final_m=0.05, rot_rad=0.35, lift_ratio=0.5, launch_mps=1.5, contact=0.5)  # check_rollout's


def track_with_mpc(video: Path, retarget: Path, out: Path, frames: int | None = None, samples: int = 1024,
                   iterations: int = 16, squeeze: float = 0.6, firm: float = 10.0, touch: float = 0.3,
                   horizon: float = 2.0, seed: int = 0, penetration: float = 0.0,
                   focus: float = 1.0) -> dict:  # fmt: skip
    """out/: `engine.rollout`'s delivery (scene.mjb; trajectory.npz with `qpos`, `qvel`, `ctrl`, `time` and
    `source_frame_indices` per source frame; diagnostics.json; mujoco_ext1.mp4, render.json and preview.jpg against
    `video`; result.json: the object's error from its reference and the solve time) of the controls SimMPC chose to
    track retarget/'s `qpos` (the first `frames` source frames): the object most (`focus` per m, 0.3 `focus` per rad),
    the hand's base (0.1, 0.03) and joints (0.003) less; while retarget/ `hold`s the object, the base `firm` times
    more, the fingers' flexion controls `squeeze` rad past it, and `touch` for each of the thumb and the index or middle
    finger not on it.
    Settings, cost weights and noise are RoboKit's Sharpa SimMPC example's (a 2 s `horizon`, 0.5 s nodes, 0.25 s per
    plan, the scene's steps, 0.01 s); `samples` and `iterations` per plan, noise `seed`; `penetration` scales Do as I
    Do's penalty on the hand's deepest penetration into the object past 5 mm (squared). Returns result.json."""
    import time

    import cv2
    import mujoco
    from robokit.helpers.sim_mpc import SimMPC, SimMPCConfig
    from robokit.helpers.sim_mpc.mjwarp import MjWarpSim
    from robokit.terms import StateTask

    from roboskills.tools.mujoco.engine import load, rollout
    from roboskills.tools.robokit.retarget import BASE

    capture, scene, out = cv2.VideoCapture(str(video)), Path(retarget) / "scene.xml", Path(out)
    fps = capture.get(cv2.CAP_PROP_FPS)
    size = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    (model, start), saved = load(scene), np.load(Path(retarget) / "trajectory.npz")
    ref, hold = saved["qpos"][:frames], saved["hold"][:frames] if "hold" in saved else np.zeros(len(saved["qpos"]))
    step = lambda seconds: int(np.rint(seconds / model.opt.timestep))  # noqa: E731
    config = SimMPCConfig(num_samples=samples, horizon_steps=step(horizon), node_steps=step(0.5), ctrl_steps=step(0.25),
                          max_iterations=iterations, min_improvement=-1.0, std_growth=2.0, noise_decay=0.75,
                          terminal_scale=1.0, seed=seed)  # fmt: skip
    steps = int(np.rint((len(ref) - 1) / fps / model.opt.timestep))  # to the last source frame
    clock = np.arange(steps + config.horizon_steps + config.ctrl_steps + 1) * model.opt.timestep * fps
    body = model.body("object")
    q, v = model.jnt_qposadr[body.jntadr[0]], model.jnt_dofadr[body.jntadr[0]]
    turn = slice(q + 3, q + 7)  # the object's quaternion, one sign throughout
    ref[:, turn] *= np.sign(np.sum(ref[:, turn] * ref[:1, turn], 1))[:, None]
    qpos_ref = np.stack([np.interp(clock, np.arange(len(ref)), c) for c in ref.T], 1)  # held past the last frame
    qpos_ref[:, turn] /= np.linalg.norm(qpos_ref[:, turn], axis=1, keepdims=True)
    ctrl_ref = qpos_ref[:, model.jnt_qposadr[model.actuator_trnid[:, 0]]]
    flex = [i for i in range(model.nu) if model.actuator(i).name.endswith(("_FE", "_PIP", "_DIP", "_IP"))]
    hold = np.interp(clock, np.arange(len(ref)), hold[: len(ref)].astype(float))[:, None]
    ctrl_ref[:, flex] += squeeze * hold  # a grip: position servos press only when told to go past the surface
    ctrl_ref = np.clip(ctrl_ref, *model.actuator_ctrlrange.T)
    names = [model.joint(j).name for j in model.dof_jntid]
    weight = np.array([0.1 if n in BASE[:3] else 0.03 if n in BASE[3:] else 0.003 for n in names])
    weight = np.where(np.isin(names, BASE), weight * (1 + (firm - 1) * hold), weight)  # the base `firm`er in hold
    weight[:, v : v + 6] = [focus] * 3 + [0.3 * focus] * 3
    bodies = [model.body(b).name if c else "" for b, c in zip(model.geom_bodyid, model.geom_contype)]  # colliding ones
    group = {f: np.array([b.startswith(f) for b in bodies]) for f in ("object", "right_thumb", ("right_index", "right_middle"))}  # fmt: skip
    grip = [(group[f], group["object"], hold[:, 0], touch) for f in list(group)[1:]]  # the thumb, index or middle on it
    ctrl_std = np.array([0.02 if model.actuator(i).name in BASE else 0.2 for i in range(model.nu)])
    start.qpos[:], start.ctrl[:] = qpos_ref[0], ctrl_ref[0]
    mujoco.mj_forward(model, start)
    initial = dict(qpos=start.qpos.copy(), qvel=start.qvel.copy())

    began, sim = time.monotonic(), MjWarpSim(model, start, nconmax=200, njmax=1000, cpu_execute=True)
    hand = np.array([b.startswith("right_") for b in bodies])
    cost = StateTask(qpos_ref, np.zeros((len(qpos_ref), model.nv)), weight, contacts=grip, margin=0.005,
                        penetration_groups=(hand, group["object"]), penetration_scale=penetration)  # fmt: skip
    (executed,), _ = SimMPC(sim, ctrl_ref, ctrl_std, [cost], config, ctrl_range=model.actuator_ctrlrange).solve(steps)
    seconds = time.monotonic() - began
    shutil.rmtree(out, ignore_errors=True)  # rollout writes a fresh folder
    report = rollout(scene, out, executed["ctrl"][:steps], fps=fps, frames=len(ref), initial=initial, camera="source",
                     size=size, video=True, preview=12, source=video)  # fmt: skip
    error = np.linalg.norm(np.load(out / "trajectory.npz")["qpos"][:, q : q + 3] - ref[:, q : q + 3], axis=1)
    result = dict(object_error_m=dict(mean=error.mean(), max=error.max(), last=error[-1]), mpc_seconds=seconds,
                  samples=samples, iterations=iterations, max_penetration_m=report["max_penetration_m"],
                  settings=dict(squeeze=squeeze, firm=firm, touch=touch, horizon=horizon, seed=seed,
                                penetration=penetration, focus=focus, timestep=model.opt.timestep))  # fmt: skip
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def check_rollout(retarget: Path, mpc: Path) -> dict:
    """track_with_mpc's check: the object in `mpc`'s rollout against `retarget`'s reference, per source frame: position error
    (mean, final), rotation error (mean), the lift it reaches where the reference lifts it 3 cm or more (sim over
    reference rise, the median over frames past half the reference's peak), a launch (frame-to-frame speed over
    `launch_mps` and 2.5x the reference's peak), and the share of `hold` frames with the hand on it (and with both the
    thumb and the index or middle finger on it); `failed` lists the LIMITS missed; `score` (lower is better) is
    position + 0.3 rotation + final error (m, rad) + 0.2 for a launch + 0.1 for a missed lift + 0.05 for each missing
    share of contact."""
    import mujoco
    from scipy.spatial.transform import Rotation

    model = mujoco.MjModel.from_binary_path(str(Path(mpc) / "scene.mjb"))
    data, body = mujoco.MjData(model), model.body("object")
    q = model.jnt_qposadr[body.jntadr[0]]
    saved, reference = np.load(Path(mpc) / "trajectory.npz"), np.load(Path(retarget) / "trajectory.npz")
    qpos, fps = saved["qpos"], 1 / np.median(np.diff(saved["source_time"]))
    sim, ref = qpos[:, q : q + 7], reference["qpos"][: len(qpos), q : q + 7]
    hold = reference["hold"][: len(qpos)] if "hold" in reference else np.zeros(len(qpos), bool)
    error = np.linalg.norm(sim[:, :3] - ref[:, :3], axis=1)
    turn = Rotation.from_quat(sim[:, 3:], scalar_first=True) * Rotation.from_quat(ref[:, 3:], scalar_first=True).inv()
    rise, peak = ref[:, 2] - ref[0, 2], np.max(ref[:, 2] - ref[0, 2])
    lift = float(np.median((sim[rise >= peak / 2, 2] - sim[0, 2]) / rise[rise >= peak / 2])) if peak >= 0.03 else None
    speed = lambda x: np.linalg.norm(np.diff(x[:, :3], axis=0), axis=1) * fps  # noqa: E731
    ref_speed = np.convolve(speed(ref), np.ones(5) / 5, mode="same").max() if len(ref) > 5 else 0.0
    bodies = model.geom_bodyid
    named = lambda prefix: [b for b in range(model.nbody) if model.body(b).name.startswith(prefix)]  # noqa: E731
    finger = lambda prefix: np.isin(bodies, named(prefix))  # noqa: E731
    on, thumb, other = bodies == body.id, finger("right_thumb"), finger(("right_index", "right_middle"))
    hand = finger("right_")
    touch = []
    for row in qpos[hold]:
        data.qpos[:] = row
        mujoco.mj_forward(model, data)
        pairs = [(c.geom1, c.geom2) for c in data.contact[: data.ncon] if c.dist <= 0.002]
        hit = lambda group: any((on[a] and group[b]) or (on[b] and group[a]) for a, b in pairs)  # noqa: E731
        touch.append((hit(hand), hit(thumb) and hit(other)))
    touch = np.array(touch, float).reshape(-1, 2)
    result = dict(
        pos_m=float(error.mean()), final_m=float(error[-1]), rot_rad=float(turn.magnitude().mean()),
        lift_ratio=lift, launch_mps=float(speed(sim).max()), reference_peak_mps=float(ref_speed),
        contact=float(touch[:, 0].mean()) if len(touch) else None,
        opposed=float(touch[:, 1].mean()) if len(touch) else None, hold_frames=int(hold.sum()),
    )  # fmt: skip
    launched = bool(result["launch_mps"] > max(LIMITS["launch_mps"], 2.5 * ref_speed))
    failed = [k for k in ("pos_m", "final_m", "rot_rad") if result[k] > LIMITS[k]]
    failed += ["lift_ratio"] * bool(lift is not None and lift < LIMITS["lift_ratio"]) + ["launch_mps"] * launched
    failed += ["contact"] * bool(result["contact"] is not None and result["contact"] < LIMITS["contact"])
    score = result["pos_m"] + 0.3 * result["rot_rad"] + result["final_m"] + 0.2 * launched
    score += 0.1 * ("lift_ratio" in failed) + 0.05 * (1 - (result["contact"] if result["contact"] is not None else 1))
    missed = ", ".join(f"{k} {result[k]:.3g} (limit {LIMITS[k]})" for k in failed)
    return dict(
        result,
        score=float(score),
        failed=failed,
        ok=not failed,
        reason=f"MPC's object missed {missed}" if failed else "",
    )


def feedback(retarget: Path, mpc: Path, out: Path, gain: float = 1.0):
    """out/: `retarget`'s scene and reference with the hand's base, while it holds the object, moved by `gain` times the
    object's error in `mpc`'s rollout (smoothed over 2 frames; iterative learning control, as the Codex baseline did):
    turned about the reference object by its rotation error (at most 0.5 rad), shifted by its position error (10 cm)."""
    import mujoco
    from scipy.ndimage import gaussian_filter1d
    from scipy.spatial.transform import Rotation

    from roboskills.tools.robokit.retarget import BASE

    shutil.rmtree(out, ignore_errors=True)
    shutil.copytree(retarget, out)
    model = mujoco.MjModel.from_binary_path(str(Path(retarget) / "scene.mjb"))
    q, adr = model.jnt_qposadr[model.body("object").jntadr[0]], [model.jnt_qposadr[model.joint(n).id] for n in BASE]
    saved = dict(np.load(Path(retarget) / "trajectory.npz"))
    ref, sim = saved["qpos"], np.load(Path(mpc) / "trajectory.npz")["qpos"]
    n = min(len(ref), len(sim))
    w = gain * gaussian_filter1d(saved["hold"][:n].astype(float), 2)[:, None]
    smooth = lambda x: gaussian_filter1d(x, 2, axis=0)  # noqa: E731
    target, now = (Rotation.from_quat(x[:n, q + 3 : q + 7], scalar_first=True) for x in (ref, sim))
    turn = Rotation.from_rotvec(np.clip(w * smooth((target * now.inv()).as_rotvec()), -0.5, 0.5))
    pivot, new = ref[:n, q : q + 3], ref.copy()
    shift = np.clip(w * smooth(pivot - sim[:n, q : q + 3]), -0.1, 0.1)
    new[:n, adr[:3]] = pivot + turn.apply(ref[:n, adr[:3]] - pivot) + shift
    euler = (turn * Rotation.from_euler("XYZ", ref[:n, adr[3:]])).as_euler("XYZ")
    new[:n, adr[3:]] = euler + 2 * np.pi * np.round((ref[:n, adr[3:]] - euler) / (2 * np.pi))
    np.savez(Path(out) / "trajectory.npz", **dict(saved, qpos=new))
