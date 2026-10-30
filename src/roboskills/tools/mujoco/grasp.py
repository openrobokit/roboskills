"""A harness for the pnp step's robot motion: a clip as pick-and-place left it (pnp_replay.py, pnp_hands.py) with its
scene, object tracks and first state kept, a candidate robot motion driven through the actuators exactly as layer 1
drives its own, the physics checks (physics.py `check_physics`) and mujoco_ext1.mp4 rendered as layer 1 renders it.

A candidate is an .npz: `arm` [frames, 7 per arm] joint targets (hands: left then right) and `grip` [frames, arms]
the gripper command (0 open .. 1 fully closed; it stops on contact at the finger force limit). Replay's servos lead the
arm and gripper by LEAD s, hands' arms by AHEAD frames, as layer 1. The kinematic reference for the checks is layer
1's (objects as tracked) with the candidate's arm in place of layer 1's; `arm_change` is its RMS from layer 1's (rad).

    python -m roboskills.tools.mujoco.grasp prepare --src SRC --clip CLIP --source SOURCE   # replay workspace or hands delivery
    python -m roboskills.tools.mujoco.grasp rollout --clip CLIP --candidate C.npz --out OUT [--no-video]
"""

import json
import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from roboskills.tools.mujoco import physics as checks


os.environ.setdefault("MUJOCO_GL", "egl")
LEAD, AHEAD = 0.1, 2  # replay's servo lead (s); hands' arm lead (frames)


def load(clip: Path):
    """(model, recon qpos, clip.json, arm prefixes) of a prepared clip folder."""
    import mujoco

    clip = Path(clip).resolve()
    facts = json.loads((clip / "clip.json").read_text())
    scene = clip / facts.get("scene", "scene.xml" if facts["kind"] == "replay" else "scene.mjb")
    model = (mujoco.MjModel.from_xml_path if scene.suffix == ".xml" else mujoco.MjModel.from_binary_path)(str(scene))
    return model, np.load(clip / "recon.npz")["qpos"], facts, facts["arms"]


def joints(model, prefix: str) -> list[int]:
    return [int(model.jnt_qposadr[model.joint(f"{prefix}joint{j}").id]) for j in range(1, 8)]


def controls(model, facts: dict, arm: np.ndarray, grip: np.ndarray) -> np.ndarray:
    """Layer 1's actuator commands for a candidate: replay per physics step, hands per source frame."""
    n, fps, arms = len(arm), facts["fps"], facts["arms"]
    grip = np.clip(np.asarray(grip, float).reshape(n, len(arms)), 0, 1)
    if facts["kind"] == "replay":  # pnp_replay.py `assemble`: interpolated LEAD s ahead
        steps = np.rint(np.arange(n) / fps / model.opt.timestep).astype(int)
        t = np.arange(steps[-1]) * model.opt.timestep * fps + LEAD * fps
        arm, grip = (np.stack([np.interp(t, np.arange(n), c) for c in x.T], 1) for x in (arm, grip))
    else:  # pnp_hands.py `robot`: the arm AHEAD frames ahead, the fingers as commanded
        arm = arm[np.minimum(np.arange(n) + AHEAD, n - 1)]
    ctrl = np.zeros((len(arm), model.nu))
    for i, p in enumerate(arms):
        ctrl[:, [model.actuator(f"{p}actuator{j}").id for j in range(1, 8)]] = arm[:, 7 * i : 7 * i + 7]
        ctrl[:, model.actuator(f"{p}fingers_actuator").id] = 255 * grip[:, i]
    return np.clip(ctrl, *model.actuator_ctrlrange.T)


def rollout(clip: Path, candidate: Path, out: Path, video: bool = True) -> dict:
    """The candidate's native rollout into `out` (trajectory.npz, mujoco_ext1.mp4, candidate.npz, checks.json): layer
    2's physics checks, their score (passed, carry + final error in m), arm_change and each object's track error."""
    clip, out = Path(clip).resolve(), Path(out).resolve()
    model, recon, facts, arms = load(clip)
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    shutil.copy(candidate, out / "candidate.npz")  # what was rolled out, kept
    c = np.load(out / "candidate.npz")
    arm, grip = np.asarray(c["arm"], float), np.asarray(c["grip"], float)
    adr = [a for p in arms for a in joints(model, p)]
    assert arm.shape == (len(recon), len(adr)), f"arm must be [{len(recon)}, {len(adr)}], got {arm.shape}"
    ctrl, fps = controls(model, facts, arm, grip), facts["fps"]
    if facts["kind"] == "replay":
        from roboskills.tools.mujoco.engine import rollout as run

        report = run(clip / facts.get("scene", "scene.xml"), out / "native", ctrl, fps=fps, frames=len(recon), initial=dict(qpos=recon[0]),
                     camera="source", size=tuple(facts["size"]), video=video, preview=6 if video else 0,
                     source=clip / "video.mp4")  # fmt: skip
        native = np.load(out / "native/trajectory.npz")["qpos"]
        np.savez(out / "trajectory.npz", qpos=native, ctrl=ctrl)
        video and shutil.copy(out / "native/mujoco_ext1.mp4", out / "mujoco_ext1.mp4")
    else:
        from roboskills.tools.mujoco.scene import render_video
        from roboskills.tools.mujoco.scene import rollout as run

        report = run(model, recon[0], ctrl, fps, out / "trajectory.npz", facts["objects"])
        native = np.load(out / "trajectory.npz")["qpos"]
        cameras = np.array(json.loads((clip / "world.json").read_text())["world_from_reference"]) @ np.load(clip / "camera.npz")["reference_from_camera"]  # fmt: skip
        size = (model.vis.global_.offwidth, model.vis.global_.offheight)
        video and render_video(model, native, cameras, out / "mujoco_ext1.mp4", size, fps)
    reference = recon.copy()
    reference[:, adr] = arm
    found = checks.check_physics(model, reference, native, fps, report["max_penetration_m"])
    every = np.linspace(0, len(recon) - 1, 12).astype(int)
    track = {}
    for body in checks.free_bodies(model):
        a = model.jnt_qposadr[model.body(body).jntadr[0]]
        if np.ptp(recon[:, a : a + 3], axis=0).max() > 0.01:  # the moving ones
            track[body] = dict(frames=every.tolist(), kin_z=recon[every, a + 2].round(3).tolist(),
                               sim_z=native[every, a + 2].round(3).tolist(),
                               error=np.linalg.norm(native[every, a : a + 3] - recon[every, a : a + 3], axis=1).round(3).tolist())  # fmt: skip
    layer1 = np.load(clip / "layer1.npz")["arm"]
    result = dict(checks={k: [v["value"], v["pass_"]] for k, v in found.items()}, score=checks.score(found),
                  arm_change=round(float(np.sqrt(np.mean((arm - layer1) ** 2))), 4), track=track)  # fmt: skip
    (out / "checks.json").write_text(json.dumps(result) + "\n")
    with open(clip / "tries.jsonl", "a") as log:
        log.write(json.dumps(dict(out=str(out), score=result["score"])) + "\n")
    return result


def ik(clip: Path, prefix: str, poses: np.ndarray, seed: np.ndarray | None = None, steps: int = 60) -> np.ndarray:
    """Arm joints [frames, 7] putting arm `prefix`'s pinch site (between the pads, its z the approach) at `poses`
    [frames, 4, 4] (world): damped least squares on the clip's own model, each frame from `seed` [frames, 7] (default
    layer 1's arm at that frame), pulled lightly toward it in the null space. CPU only."""
    import mujoco

    model, recon, _, arms = load(clip)
    data, site, adr = mujoco.MjData(model), model.site(f"{prefix}pinch").id, joints(model, prefix)
    dof = [model.jnt_dofadr[model.joint(f"{prefix}joint{j}").id] for j in range(1, 8)]
    low, high = model.jnt_range[[model.joint(f"{prefix}joint{j}").id for j in range(1, 8)]].T
    jp, jr, out = np.zeros((3, model.nv)), np.zeros((3, model.nv)), []
    seed = recon[:, adr] if seed is None else np.asarray(seed, float)
    for f, target in enumerate(np.asarray(poses, float)):
        data.qpos[:] = recon[min(f, len(recon) - 1)]
        q = seed[min(f, len(seed) - 1)].copy()
        for _ in range(steps):
            data.qpos[adr] = q
            mujoco.mj_kinematics(model, data)
            mujoco.mj_comPos(model, data)
            mujoco.mj_jacSite(model, data, jp, jr, site)
            r = data.site_xmat[site].reshape(3, 3)
            error = np.r_[target[:3, 3] - data.site_xpos[site], 0.5 * np.cross(r, target[:3, :3], axis=0).sum(1)]
            j = np.r_[jp[:, dof], jr[:, dof]]
            step = j.T @ np.linalg.solve(j @ j.T + 1e-3 * np.eye(6), error)
            step += 0.05 * (np.eye(7) - np.linalg.pinv(j) @ j) @ (seed[min(f, len(seed) - 1)] - q)
            q = np.clip(q + step * min(1.0, 0.2 / (np.abs(step).max() + 1e-9)), low, high)  # 0.2 rad at most
        out.append(q.copy())
    return np.array(out)


def replay_facts(model, recon: np.ndarray, grasp: tuple[int, int], layer: dict) -> dict:
    """What an agent would otherwise read out of a replay scene (world frame, m): the object's extent, place and contact
    parameters at frame 0 and at the grasp, the table, and the gripper (its pinch at layer 1's grasp)."""
    import mujoco

    data, body = mujoco.MjData(model), model.body("object").id
    geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] == body and model.geom_contype[g]]
    at = {}
    for name, frame in (("frame0", 0), ("grasp", grasp[0])):
        data.qpos[:] = recon[frame]
        mujoco.mj_kinematics(model, data)
        points = checks.object_points(model, data)
        at[name] = dict(frame=int(frame), low=points.min(0).round(4).tolist(), high=points.max(0).round(4).tolist())
    r, pad = data.site("pinch").xmat.reshape(3, 3), model.geom("left_pad1").id
    size = (np.array(at["frame0"]["high"]) - at["frame0"]["low"]).round(4).tolist()
    thing = dict(size=size, **at, mass_kg=round(float(model.body_subtreemass[body]), 4),
                 friction=model.geom_friction[geoms[0]].round(4).tolist(), collision=layer.get("collision"))  # fmt: skip
    gripper = dict(stroke=0.085, finger_force_limit_N=float(model.actuator_forcerange[model.actuator("fingers_actuator").id, 1]),
                   pad_box_half_size=model.geom_size[pad].round(4).tolist(), pinch_at_grasp=data.site("pinch").xpos.round(4).tolist(),
                   approach_axis_at_grasp=r[:, 2].round(3).tolist(), closing_axis_at_grasp=r[:, 1].round(3).tolist())  # fmt: skip
    return dict(table_z=round(layer["table"], 4), object=thing, gripper=gripper)


def prepare(src: Path, clip: Path, source: Path, physics: str = "layer1", native: Path | None = None) -> dict:
    """A clip folder from a layer-1 replay workspace (scene/ assembled) or hands delivery: the scene, recon.npz,
    layer1.npz (layer 1's own motion as a candidate), recon_ext1.mp4, the source video, clip.json; then layer1/, the
    rollout of layer1.npz (it must reproduce layer 1's checks: layer1_delivered.json). Replay: scene.mjb, the scene with
    layer 2's contact parameters `physics` (checks.PHYSICS), as layer 2 rolls it out; `native` the delivered rollout's
    folder (default src/scene/native)."""
    import mujoco

    src, clip, source = Path(src).resolve(), Path(clip).resolve(), Path(source).resolve()
    clip.mkdir(parents=True, exist_ok=True)
    (clip / "video.mp4").unlink(missing_ok=True)
    shutil.copyfile(source / "video.mp4", clip / "video.mp4")
    facts = dict(task=json.loads((source / "metadata.json").read_text())["instruction"])
    if (src / "scene/scene.xml").exists():  # replay
        from roboskills.tools.robokit.pnp_replay import grasp_frames

        root = ET.parse(src / "scene/scene.xml").getroot()
        for node in root.iter():
            if (f := node.get("file")) and Path(f).is_absolute() and (src / "scene" / Path(f).name).is_file():
                shutil.copy(src / "scene" / Path(f).name, clip)  # its own assets (written on any machine)
        fix_paths(root, [clip])
        ET.ElementTree(root).write(clip / "scene.xml", encoding="unicode")
        recon, grasp = np.load(src / "scene/recon/trajectory.npz")["qpos"], grasp_frames(src)
        closed = np.zeros(len(recon))
        closed[grasp[0] : grasp[1] + 1] = 1  # layer 1 squeezes: fully closed from the grasp to the release
        grip = np.maximum(np.load(src / "input/robot.npz")["gripper_position"], closed)[:, None]
        native = Path(native).resolve() if native else src / "scene/native"
        delivered = native / "trajectory.npz", json.loads((native / "diagnostics.json").read_text())["max_penetration_m"]  # fmt: skip
        model = mujoco.MjModel.from_xml_path(str(clip / "scene.xml"))
        checks.adjust_physics(model, **checks.PHYSICS[physics])  # as checks.simulate_replay: the XML's model adjusted,
        mujoco.mj_saveModel(model, str(clip / "scene.mjb"))  # saved (its meshes and textures in, so it loads anywhere)
        facts.update(scene="scene.mjb", physics=physics)
        facts.update(kind="replay", fps=30.0, arms=[""], objects=["object"], layer1_grasps=[list(grasp)],
                     size=[int(x) for x in model.cam_resolution[model.camera("source").id]])  # fmt: skip
        facts.update(replay_facts(model, recon, grasp, json.loads((src / "scene/layer.json").read_text())),
                     recorded="estimated" not in json.loads((src / "input/camera.json").read_text()))  # fmt: skip
        shutil.copy(src / "scene/recon_ext1.mp4", clip)
    else:  # hands
        for name in ("scene.mjb", "recon.npz", "report.json", "world.json", "camera.npz", "recon_ext1.mp4"):
            shutil.copy(src / name, clip)
        report, recon = json.loads((src / "report.json").read_text()), np.load(src / "recon.npz")["qpos"]
        model = mujoco.MjModel.from_binary_path(str(clip / "scene.mjb"))
        ctrl = np.load(src / "trajectory.npz")["ctrl"]
        grip = np.stack([ctrl[:, model.actuator(f"{s}_fingers_actuator").id] / 255 for s in ("left", "right")], 1)
        delivered = src / "trajectory.npz", report["physics"]["max_penetration_m"]
        facts.update(kind="hands", fps=report["fps"], arms=["left_", "right_"], objects=report["objects"],
                     names=report["names"], layer1_grasps={s: report[s].get("grasps", report[s].get("steps")) for s in ("left", "right")},
                     welds=[model.equality(i).name for i in range(model.neq) if model.equality(i).name.startswith("hold:")])  # fmt: skip
    np.savez(clip / "recon.npz", qpos=recon)
    facts.update(frames=len(recon), arm_qpos=[a for p in facts["arms"] for a in joints(model, p)],
                 object_qpos={b: int(model.jnt_qposadr[model.body(b).jntadr[0]]) for b in checks.free_bodies(model)})  # fmt: skip
    (clip / "clip.json").write_text(json.dumps(facts, indent=1) + "\n")
    np.savez(clip / "layer1.npz", arm=recon[:, facts["arm_qpos"]], grip=grip)
    native = np.load(delivered[0])["qpos"]
    found = checks.check_physics(model, recon, native, facts["fps"], delivered[1])
    (clip / "layer1_delivered.json").write_text(json.dumps(dict(checks={k: [v["value"], v["pass_"]] for k, v in found.items()}, score=checks.score(found))) + "\n")  # fmt: skip
    return rollout(clip, clip / "layer1.npz", clip / "layer1")


def fix_paths(root: ET.Element, roots: list[Path]):
    """Point the MJCF tree's absolute files and directories (and relative ones when there is no meshdir) at this
    machine: the MuJoCo tool's franka/ for the robot's meshes, else the first of `roots` holding the file's longest
    tail."""
    from roboskills.tools.mujoco.mujoco import FRANKA

    def local(value: str) -> str:
        path = Path(value)
        if path.is_absolute() and path.exists():
            return value
        if "/tools/mujoco/franka" in value:
            return str(FRANKA.parent / value.split("/tools/mujoco/franka", 1)[1].lstrip("/"))
        for r in roots:
            for k in range(min(len(path.parts), 4), 0, -1):
                if (r / Path(*path.parts[-k:])).exists():
                    return str(r / Path(*path.parts[-k:]))
        return value

    compiler = root.find("compiler")
    meshdir = compiler is not None and compiler.get("meshdir")
    for key in ("meshdir", "texturedir"):
        if compiler is not None and compiler.get(key):
            compiler.set(key, local(compiler.get(key)))
    for node in root.iter():
        if node.get("file") and (Path(node.get("file")).is_absolute() or not meshdir):
            node.set("file", local(node.get("file")))


if __name__ == "__main__":
    import tyro

    print(json.dumps(tyro.extras.subcommand_cli_from_dict(dict(prepare=prepare, rollout=rollout))))
