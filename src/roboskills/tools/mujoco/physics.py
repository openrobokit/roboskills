"""Real2sim layer 2, agent-free: fixed checks on a layer-1 delivery and bounded retries of the step a failed check
blames; the best attempt by the checks (never by the judge) is delivered.

Checks, each a measurement and a threshold (THRESHOLDS):
- recon (replay): `robot_iou`, the camera fit's robot silhouette IoU; `grasp_width`, the object's width across the
  jaws at the first closing against the gap the recorded gripper closure leaves.
- physics (both): `carried`, the object's distance from its kinematic track while held; `lifted`, the native lift
  over the kinematic one; `placed`, the final distance; `at_rest`, the final speed; `upright`, the final tilt from
  the kinematic pose; `tracking`, the arm's joint error; `stable`, the peak object speed and the deepest penetration.

Retries: a failed physics check reruns the native rollout over the PHYSICS ladder (contact parameters on the same scene
and controls), at most len(PHYSICS) - 1 attempts; physics checks that no rung passes (replay) reassemble the scene, as
layer 1 does, on every other SAM 3D sample layer 1 kept (shape/samples/), each then through the ladder (SAM 3D never
reruns here; without kept samples there is no re-pick). A hands attempt replaces layer 1's only when its checks
strictly improve (`improves`: every check layer 1 passes still passes, and more pass); a replay one when more pass and
carry and lift get no worse, within a tenth of their thresholds (`keeps`: an object that never moves can't win); among
those, the most checks, then the smallest carry, final and grasp-width errors.

    python -m roboskills.tools.mujoco.physics replay --workspace W --out O
    python -m roboskills.tools.mujoco.physics hands --delivery D --out O
"""

import json
import shutil
import time
from pathlib import Path

import numpy as np


THRESHOLDS = dict(robot_iou=0.6, grasp_width=0.03, carried=0.04, lifted=0.5, placed=0.06, at_rest=0.05, upright=30.0,
                  tracking=0.1, stable=3.0, penetration=0.02)  # fmt: skip


# contact parameters per attempt: object (and pad) sliding friction, contact dimension (6: torsional and rolling
# friction too), object mass factor, finger actuator force factor, grip window (hands: pinch distance, m)
PHYSICS = dict(
    layer1={},
    grip=dict(friction=2.5, condim=6, force=3.0),
    light=dict(friction=2.5, condim=6, force=3.0, mass=0.3),
    near=dict(friction=2.5, condim=6, force=3.0, mass=0.3, near=0.08),
)


STROKE = 0.085  # the 2F-85's opening (m)


def free_bodies(model) -> list[str]:
    """Bodies with a free joint: the scene's objects."""
    import mujoco

    return [model.body(model.jnt_bodyid[j]).name for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]  # fmt: skip


def arm_joints(model) -> np.ndarray:
    """qpos addresses of every Panda arm joint (joint1..7, any prefix)."""
    return np.array([model.jnt_qposadr[j] for j in range(model.njnt) if model.joint(j).name.split("_")[-1] in
                     {f"joint{i}" for i in range(1, 8)}])  # fmt: skip


def tilt(a: np.ndarray, b: np.ndarray) -> float:
    """Angle (deg) between the z axes of two wxyz quaternions' rotations."""
    import mujoco

    za, zb = np.zeros(9), np.zeros(9)
    mujoco.mju_quat2Mat(za, a), mujoco.mju_quat2Mat(zb, b)
    return float(np.degrees(np.arccos(np.clip(za.reshape(3, 3)[:, 2] @ zb.reshape(3, 3)[:, 2], -1, 1))))


def check_physics(model, recon: np.ndarray, native: np.ndarray, fps: float, penetration: float = 0.0) -> dict:
    """The physics checks of a native rollout `native` against its kinematic `recon` (qpos per source frame): each
    {value, pass}, the worst object counting. An object is held where the kinematic track lifts it over 2 cm."""
    rows = min(len(recon), len(native))
    recon, native, out = recon[:rows], native[:rows], {}
    worst = dict(carried=0.0, lifted=np.inf, placed=0.0, at_rest=0.0, upright=0.0, stable=0.0)
    for body in free_bodies(model):
        a = model.jnt_qposadr[model.body(body).jntadr[0]]
        kin, sim = recon[:, a : a + 3], native[:, a : a + 3]
        error = np.linalg.norm(kin - sim, axis=1)
        held = kin[:, 2] - kin[0, 2] > 0.02
        speed = np.linalg.norm(np.diff(sim, axis=0), axis=1) * fps
        if held.any():
            worst["carried"] = max(worst["carried"], float(np.median(error[held])))
            worst["lifted"] = min(
                worst["lifted"], float((sim[held, 2] - sim[0, 2]).max() / (kin[held, 2] - kin[0, 2]).max())
            )
        worst["placed"] = max(worst["placed"], float(error[-1]))
        worst["at_rest"] = max(worst["at_rest"], float(speed[-5:].mean()))
        worst["upright"] = max(worst["upright"], tilt(recon[-1, a + 3 : a + 7], native[-1, a + 3 : a + 7]))
        worst["stable"] = max(worst["stable"], float(speed.max()))
    arm = arm_joints(model)
    worst["tracking"] = float(np.sqrt(np.mean((recon[:, arm] - native[:, arm]) ** 2)))
    worst["penetration"] = float(penetration)
    for name, value in worst.items():
        if name == "lifted" and not np.isfinite(value):
            out[name] = dict(value=None, pass_=True)  # nothing lifted in the source track
        else:
            ok = value >= THRESHOLDS[name] if name == "lifted" else value <= THRESHOLDS[name]
            out[name] = dict(value=round(value, 4), pass_=bool(ok))
    return out


def adjust_physics(model, friction: float = 0.0, condim: int = 0, force: float = 1.0, mass: float = 1.0, **_):
    """`model` in place: every free body's colliding geoms (and the finger pads) at least `friction`, `condim`
    contacts, its mass and inertia times `mass`; the finger actuators' force range times `force`."""
    bodies = {model.body(b).id for b in free_bodies(model)}
    for g in range(model.ngeom):
        pad = "pad" in model.geom(g).name
        if (model.geom_bodyid[g] in bodies and model.geom_contype[g] | model.geom_conaffinity[g]) or pad:
            model.geom_friction[g, 0] = max(model.geom_friction[g, 0], friction)
            if condim:
                model.geom_condim[g] = condim
                model.geom_friction[g, 1:] = np.maximum(model.geom_friction[g, 1:], [0.02, 0.0005])
    for b in bodies:
        model.body_mass[b] *= mass
        model.body_inertia[b] *= mass
    for i in range(model.nu):
        if "fingers" in model.actuator(i).name:
            model.actuator_forcerange[i] *= force


def score(checks: dict) -> tuple[int, float]:
    """(checks passed, the carry, final and grasp width errors summed in m)."""
    value = lambda n: abs(checks.get(n, {}).get("value") or 0.0)
    return sum(c["pass_"] for c in checks.values()), value("carried") + value("placed") + value("grasp_width")


def improves(new: dict, old: dict) -> bool:
    """Whether checks `new` strictly improve on `old`: every check `old` passes still passes, and more pass."""
    kept = all(new.get(k, {}).get("pass_", False) for k, c in old.items() if c["pass_"])
    return kept and sum(c["pass_"] for c in new.values()) > sum(c["pass_"] for c in old.values())


def strict(checks: dict, first) -> object:
    """The key of `checks` (each a checks dict) that strictly improves on `first`'s with the best `score`, else
    `first`."""
    better = [k for k in checks if k != first and improves(checks[k], checks[first])]
    return max(better, key=lambda k: (score(checks[k])[0], -score(checks[k])[1])) if better else first


def keeps(new: dict, old: dict, slack: float = 0.1) -> bool:
    """Whether checks `new` keep `old`'s carry and lift: each of `carried` and `lifted` that `old` passes still passes,
    and each it fails is no further off than `slack` times its threshold (an object that never moves can't win)."""
    for name in ("carried", "lifted"):
        o, n = old.get(name, dict(pass_=True)), new.get(name, dict(pass_=False, value=None))
        if n["pass_"]:
            continue
        if o["pass_"] or o.get("value") is None or n.get("value") is None:
            return False
        worse = o["value"] - n["value"] if name == "lifted" else n["value"] - o["value"]
        if worse > slack * THRESHOLDS[name]:
            return False
    return True


def best(checks: dict, first) -> object:
    """The key of `checks` that keeps `first`'s carry and lift (`keeps`) and passes more checks, with the best
    `score`, else `first`."""
    passed = lambda k: score(checks[k])[0]
    better = [k for k in checks if k != first and keeps(checks[k], checks[first]) and passed(k) > passed(first)]
    return max(better, key=lambda k: (passed(k), -score(checks[k])[1])) if better else first


def choose(scores: dict, margin: float = 0.02):
    """The key of the best `scores` (from `score`): more checks passed, else an error at least `margin` (m) under
    the first key's (layer 1), else the first."""
    first = next(iter(scores))
    best = max(scores, key=lambda n: (scores[n][0], -scores[n][1]))
    better = scores[best][0] > scores[first][0] or (
        scores[best][0] == scores[first][0] and scores[best][1] < scores[first][1] - margin
    )
    return best if better else first


def replay_controls(w: Path, model) -> tuple[np.ndarray, int]:
    """Per-step native controls exactly as `pnp_replay.assemble` builds them, and the frame count: the servos LEAD s ahead
    of the recorded joints (as the kinematic replay has them: a clip from video's wrist turned), the gripper its
    recorded closing, commanded fully closed from the grasp to the release (assemble's `squeeze`, on unless
    scene/layer.json says otherwise; its `grasp`)."""
    from roboskills.tools.robokit.pnp_replay import LEAD

    r = np.load(w / "input/robot.npz")
    n = len(r["joint_position"])
    steps = np.rint(np.arange(n) / 30 / model.opt.timestep).astype(int)
    t = np.arange(steps[-1]) * model.opt.timestep * 30 + LEAD * 30
    joints = np.load(w / "scene/recon/trajectory.npz")["qpos"][:, :7]
    arm = np.stack([np.interp(t, np.arange(n), c) for c in joints.T], 1)
    info = json.loads((w / "scene/layer.json").read_text()) if (w / "scene/layer.json").is_file() else {}
    close = np.zeros(n)
    if info.get("grasp") and info.get("squeeze", True):
        close[info["grasp"][0] : info["grasp"][1] + 1] = 1
    grip = 255 * np.interp(t, np.arange(n), np.maximum(r["gripper_position"], close))
    return np.c_[np.clip(arm, *model.actuator_ctrlrange[:7].T), grip], n


def simulate_replay(w: Path, out: Path, params: dict) -> dict:
    """One native rollout of the assembled scene w/scene/scene.xml with `params` (PHYSICS) into out/; its physics
    checks. The XML, as `pnp_replay.assemble` rolls it out (not the .mjb, whose backdrop emission and offscreen size differ),
    so an unchanged scene reproduces layer 1's native rollout and video byte for byte."""
    import mujoco

    from roboskills.tools.mujoco.engine import load, rollout
    from roboskills.tools.robokit.pnp_replay import VIDEO, read_frames

    model, _ = load(w / "scene/scene.xml")
    adjust_physics(model, **params)
    out.mkdir(parents=True, exist_ok=True)
    mujoco.mj_saveModel(model, str(out / "scene.mjb"))
    ctrl, n = replay_controls(w, model)
    recon = np.load(w / "scene/recon/trajectory.npz")["qpos"]
    size = read_frames(w / VIDEO)[0].shape[:2]
    shutil.rmtree(out / "native", ignore_errors=True)
    report = rollout(out / "scene.mjb", out / "native", ctrl, fps=30, frames=n, initial=dict(qpos=recon[0]),
                     camera="source", size=size[::-1], preview=6, source=w / VIDEO)  # fmt: skip
    native = np.load(out / "native/trajectory.npz")["qpos"]
    return check_physics(model, recon, native, 30, report["max_penetration_m"])


def check_replay_recon(w: Path) -> dict:
    """`robot_iou` from the camera fit (when there is one); `grasp_width`: the object's extent across the jaws at the pinch height at the
    first closing (kinematic) against STROKE times the open fraction the recorded closure leaves."""
    import mujoco

    from roboskills.tools.mujoco.engine import load
    from roboskills.tools.robokit.pnp_replay import grasp_frames

    out, camera = {}, json.loads((w / "camera/camera.json").read_text())
    if "iou" in camera:  # a camera fitted to the robot's silhouette; one from MoGe alone has no IoU to check
        out["robot_iou"] = dict(value=round(camera["iou"], 3), pass_=camera["iou"] >= THRESHOLDS["robot_iou"])
    grasp = grasp_frames(w)
    if grasp is None:
        return out
    model, data = load(w / "scene/recon/scene.mjb")
    data.qpos[:] = np.load(w / "scene/recon/trajectory.npz")["qpos"][grasp[0]]
    mujoco.mj_kinematics(model, data)
    across = data.body("left_pad").xpos - data.body("right_pad").xpos
    across /= np.linalg.norm(across)
    pinch = data.site("pinch").xpos
    points = object_points(model, data)
    near = points[np.abs((points - pinch) @ data.site("pinch").xmat.reshape(3, 3)[:, 2]) < 0.015]  # the pads' height
    width = float(np.ptp(near @ across)) if len(near) else 0.0
    g = np.load(w / "input/robot.npz")["gripper_position"][grasp[0] : grasp[1] + 1].max()
    gap = STROKE * (1 - g)
    out["grasp_width"] = dict(value=round(width - gap, 4), width=round(width, 4), gap=round(gap, 4),
                              pass_=bool(abs(width - gap) <= THRESHOLDS["grasp_width"]))  # fmt: skip
    return out


def object_points(model, data) -> np.ndarray:
    """World vertices of every free body's colliding mesh geoms."""
    import mujoco

    bodies, found = {model.body(b).id for b in free_bodies(model)}, []
    for g in range(model.ngeom):
        if (
            model.geom_bodyid[g] in bodies
            and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
            and model.geom_contype[g]
        ):
            m = model.geom_dataid[g]
            v = model.mesh_vert[model.mesh_vertadr[m] : model.mesh_vertadr[m] + model.mesh_vertnum[m]]
            found.append(v @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g])
    return np.concatenate(found) if found else np.zeros((0, 3))


def samples_replay(w: Path, out: Path) -> dict[str, Path]:
    """Layer-1 workspace `w` reassembled, as layer 1 assembles it, on each other SAM 3D sample layer 1 kept
    (shape/samples/<seed>/, written with `keep`; `pnp_replay.assemble(sample=...)`): {seed: out/<seed>/} (input, camera,
    depth, masks and shape linked from `w`). Empty without kept samples. SAM 3D never runs here."""
    from roboskills.tools.robokit import pnp_replay

    kept = sorted(p.parent.name for p in (w / "shape/samples").glob("*/mesh.ply"))
    chosen, found = str(json.loads((w / "shape/pose.json").read_text()).get("seed")), {}
    for seed in (s for s in kept if s != chosen):
        found[seed] = trial = out / seed
        trial.mkdir(parents=True, exist_ok=True)
        for name in ("input", "camera", "depth", "masks", "shape"):
            (trial / name).exists() or (trial / name).symlink_to(w / name, target_is_directory=True)
        pnp_replay.assemble(trial, sample=int(seed))
    return found


def check_replay(w: Path, out: Path, attempts: int) -> dict:
    """Recon checks, then the PHYSICS ladder (up to `attempts`, stopping once all pass) on workspace `w`."""
    from roboskills.tools.mujoco.engine import load

    recon, tried = check_replay_recon(w), {}
    for name, params in list(PHYSICS.items())[:attempts]:
        if "near" in params:  # hands' grip window; replay closes the gripper as recorded
            continue
        if name == "layer1":
            model, _ = load(w / "scene/scene.mjb")
            native = np.load(w / "scene/native/trajectory.npz")["qpos"]
            depth = json.loads((w / "scene/native/diagnostics.json").read_text())["max_penetration_m"]
            checks = check_physics(model, np.load(w / "scene/recon/trajectory.npz")["qpos"], native, 30, depth)
            video = w / "scene/mujoco_ext1.mp4"
        else:
            checks, video = simulate_replay(w, out / name, params), out / name / "native/mujoco_ext1.mp4"
        scene, native = (w / "scene", w / "scene/native") if name == "layer1" else (out / name, out / name / "native")
        tried[name] = dict(params=params, checks=checks, score=score({**recon, **checks}), video=str(video),
                           scene=str(scene / "scene.mjb"), trajectory=str(native / "trajectory.npz"))  # fmt: skip
        if all(c["pass_"] for c in checks.values()):
            break
    return dict(workspace=str(w), recon=recon, recon_score=score(recon), physics=tried)


def route_replay(attempt: dict, masks: Path) -> dict:
    """route.py's verdict ({pass, score, reasons}) on an attempt's native rollout (check_replay's) against the
    source's per-frame object masks."""
    from roboskills.tools.mujoco.engine import load
    from roboskills.tools.mujoco.route import score_rollout

    found = score_rollout(load(attempt["scene"])[0], np.load(attempt["trajectory"])["qpos"], None, Path(masks))
    return {k: found[k] for k in ("pass", "score", "reasons")}


def run_replay(workspace: Path, out: Path, attempts: int = len(PHYSICS), masks: Path | None = None) -> dict:
    """Layer 2 on a layer-1 replay workspace: checks and retries (the physics ladder; when no rung passes the physics
    checks, the scene rebuilt on every other kept SAM 3D sample, the ladder on each), out/recon_ext1.mp4 (layer 1's, or
    a sample's when its recon checks strictly improve) and out/mujoco_ext1.mp4 (layer 1's, or the attempt that keeps
    layer 1's carry and lift and passes more of the recon and physics checks: `best`, when it does not lower route's
    score, route.py, against `masks`: default the workspace's per-frame SAM 3 masks), out/layer2.json (every
    attempt's checks and route verdict, the choice)."""
    w, out, began = Path(workspace).resolve(), Path(out).resolve(), time.monotonic()
    out.mkdir(parents=True, exist_ok=True)
    found = dict(layer1=check_replay(w, out / "layer1", attempts))
    passed = any(all(c["pass_"] for c in p["checks"].values()) for p in found["layer1"]["physics"].values())
    for seed, trial in samples_replay(w, out / "samples").items() if not passed else ():
        found[f"sample{seed}"] = check_replay(trial, out / f"sample{seed}", attempts)
    shape, first = strict({n: found[n]["recon"] for n in found}, "layer1"), ("layer1", "layer1")
    physics = checked = best({(n, p): {**found[n]["recon"], **found[n]["physics"][p]["checks"]} for n in found
                              for p in found[n]["physics"]}, first)  # fmt: skip
    masks = Path(masks) if masks else next((m for m in (w / "masks", w / "cache/masks") if len(list(m.glob("*.png"))) > 1), w / "masks")  # fmt: skip
    for attempt in (a for f in found.values() for a in f["physics"].values()):
        attempt["route"] = route_replay(attempt, masks)
    routed = lambda k: -1.0 if (s := found[k[0]]["physics"][k[1]]["route"]["score"]) is None else s  # none: no evidence
    if routed(physics) < routed(first):  # route, the final acceptance: a kept change must not lower its score
        physics = first
    shutil.copy(Path(found[shape]["workspace"]) / "scene/recon_ext1.mp4", out / "recon_ext1.mp4")
    shutil.copy(found[physics[0]]["physics"][physics[1]]["video"], out / "mujoco_ext1.mp4")
    report = dict(attempts=found, recon_best=shape, physics_checked=list(checked), physics_best=list(physics),
                  masks=str(masks), seconds=round(time.monotonic() - began, 1))  # fmt: skip
    (out / "layer2.json").write_text(json.dumps(report, indent=2) + "\n")
    (out / "timing.json").write_text(json.dumps(dict(seconds=report["seconds"])) + "\n")
    return report


def grip_window(model, recon: np.ndarray, ctrl: np.ndarray, near: float) -> np.ndarray:
    """`ctrl` with each arm's fingers fully closed (255) where its pinch is within `near` of an object's surface in the
    kinematic `recon` and the recorded closure is over 5% (blips under 3 frames removed)."""
    import mujoco
    from scipy.ndimage import binary_closing, binary_opening

    data, ctrl = mujoco.MjData(model), ctrl.copy()
    sites = [s for s in range(model.nsite) if model.site(s).name.endswith("pinch")]
    distance = np.zeros((len(sites), len(recon)))
    for f, q in enumerate(recon):
        data.qpos[:] = q
        mujoco.mj_kinematics(model, data)
        points = object_points(model, data)
        for i, s in enumerate(sites):
            distance[i, f] = np.linalg.norm(points - data.site_xpos[s], axis=1).min() if len(points) else np.inf
    for i, s in enumerate(sites):
        side = model.site(s).name.removesuffix("pinch")
        a = model.actuator(f"{side}fingers_actuator").id
        held = binary_opening(binary_closing((distance[i] < near) & (ctrl[:, a] > 0.05 * 255), np.ones(3)), np.ones(3))
        ctrl[held, a] = 255
    return ctrl


def simulate_hands(d: Path, out: Path, params: dict) -> dict:
    """One native rollout of layer 1's hands scene d/scene.mjb with `params` (PHYSICS) into out/ (trajectory.npz,
    mujoco_ext1.mp4); its physics checks."""
    import mujoco

    from roboskills.tools.mujoco.scene import render_video, rollout

    model = mujoco.MjModel.from_binary_path(str(d / "scene.mjb"))
    adjust_physics(model, **params)
    report, recon = json.loads((d / "report.json").read_text()), np.load(d / "recon.npz")["qpos"]
    ctrl = np.load(d / "trajectory.npz")["ctrl"]
    if params.get("near"):
        ctrl = grip_window(model, recon, ctrl, params["near"])
    out.mkdir(parents=True, exist_ok=True)
    physics = rollout(model, recon[0], ctrl, report["fps"], out / "trajectory.npz", report["objects"])
    native = np.load(out / "trajectory.npz")["qpos"]
    cameras = np.array(json.loads((d / "world.json").read_text())["world_from_reference"]) @ np.load(d / "camera.npz")["reference_from_camera"]  # fmt: skip
    render_video(model, native, cameras, out / "mujoco_ext1.mp4", (model.vis.global_.offwidth, model.vis.global_.offheight), report["fps"])  # fmt: skip
    return check_physics(model, recon, native, report["fps"], physics["max_penetration_m"])


def run_hands(delivery: Path, out: Path, attempts: int = len(PHYSICS)) -> dict:
    """Layer 2 on layer 1's hands delivery folder: physics checks, retries, out/recon_ext1.mp4 (layer 1's),
    out/mujoco_ext1.mp4 (layer 1's, or the attempt whose checks strictly improve on it: `strict`) and out/layer2.json."""
    import mujoco

    d, out, began = Path(delivery).resolve(), Path(out).resolve(), time.monotonic()
    out.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_binary_path(str(d / "scene.mjb"))
    fps, tried = json.loads((d / "report.json").read_text())["fps"], {}
    for name, params in list(PHYSICS.items())[:attempts]:
        if name == "layer1":
            depth = json.loads((d / "report.json").read_text())["physics"]["max_penetration_m"]
            checks = check_physics(model, np.load(d / "recon.npz")["qpos"], np.load(d / "trajectory.npz")["qpos"], fps, depth)  # fmt: skip
            video = d / "mujoco_ext1.mp4"
        else:
            checks, video = simulate_hands(d, out / name, params), out / name / "mujoco_ext1.mp4"
        tried[name] = dict(params=params, checks=checks, score=score(checks), video=str(video))
        if all(c["pass_"] for c in checks.values()):
            break
    best = strict({n: tried[n]["checks"] for n in tried}, "layer1")
    shutil.copy(tried[best]["video"], out / "mujoco_ext1.mp4")
    shutil.copy(d / "recon_ext1.mp4", out / "recon_ext1.mp4")
    report = dict(physics=tried, best=best, seconds=round(time.monotonic() - began, 1))
    (out / "layer2.json").write_text(json.dumps(report, indent=2) + "\n")
    (out / "timing.json").write_text(json.dumps(dict(seconds=report["seconds"])) + "\n")
    return report


if __name__ == "__main__":
    import tyro

    tyro.extras.subcommand_cli_from_dict(dict(replay=run_replay, hands=run_hands))
