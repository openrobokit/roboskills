"""Pick and place, the workflow's robot step: a scene of the source's objects and Panda arms with 2F-85s
(tools/mujoco/scene.py), the arms through keyframes and RoboKit IK timed to the source, rolled out natively. The mode
is what the clip shows: `replay`, a robot (input/robot.npz, recorded or from video) replayed through position servos
(tools/robokit/pnp_replay.py); `hands`, human hands, two arms taking each object they pick and place
(tools/robokit/pnp_hands.py). Its check is tools/mujoco/route.py's against the source's masks (`check_route`);
`retry` runs tools/mujoco/physics.py's ladder.

    python -m roboskills.workflows.real2sim --workspace W --tools pnp [--retry]
"""

import json
import shutil
from pathlib import Path
from typing import Literal


def mode_of(w: Path) -> str:
    """replay with a robot in input/ (recorded joints, or a robot from video), else hands."""
    return "replay" if (Path(w) / "input/robot.npz").exists() else "hands"


def pick_and_place(workspace: Path, out: Path | None = None, mode: Literal["auto", "replay", "hands"] = "auto",
                   budget: float = 90.0, retry: bool = False) -> dict:  # fmt: skip
    """out/ (default workspace/pnp/; replay perceives first when without camera/camera.json): mujoco_ext1.mp4 and
    recon_ext1.mp4, the rollout's scene and trajectory.npz (hands: robot's out/; replay: assemble's workspace/scene/,
    its videos and layer.json copied), report.json. Hands retries in physics for up to `budget` seconds. With `retry`,
    physics.py's ladder on it into out/layer2/, whose videos and choice replace out's."""
    w = Path(workspace).resolve()
    out, mode = Path(out or w / "pnp").resolve(), mode_of(w) if mode == "auto" else mode
    out.mkdir(parents=True, exist_ok=True)
    if mode == "replay":
        from roboskills.tools.robokit.pnp_replay import assemble, perceive

        (w / "camera/camera.json").exists() or perceive(w)  # the camera, object and, without telemetry, the robot
        (w / "scene").is_symlink() and (w / "scene").unlink()  # a retry set's link to the first run's: its own instead
        assemble(w)
        for name in ("mujoco_ext1.mp4", "recon_ext1.mp4", "layer.json"):
            out == w / "scene" or shutil.copy(w / "scene" / name, out / name)
        report = json.loads((w / "scene/layer.json").read_text())
    else:
        from roboskills.tools.robokit.pnp_hands import robot

        for name in ("camera.npz", "reference.npz"):  # the reader's camera and reference frame
            (w / "camera" / name).exists() and not (out / name).exists() and shutil.copy(w / "camera" / name, out)
        robot(w, out, budget=budget)
        report = json.loads((out / "report.json").read_text())
    if retry:
        from roboskills.tools.mujoco import physics

        found = physics.run_replay(w, out / "layer2") if mode == "replay" else physics.run_hands(out, out / "layer2")
        for name in ("mujoco_ext1.mp4", "recon_ext1.mp4"):
            shutil.copy(out / "layer2" / name, out / name)
        report["layer2"] = {k: found[k] for k in ("physics_best", "recon_best", "best") if k in found}
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    return report


def check_route(w: Path, out: Path, masks: Path | None = None) -> dict:
    """The pnp step's check: route.py's verdict on out/'s delivered rollout against the source's per-frame masks
    (`ok`, a `reason`, its `score`)."""
    from roboskills.tools.mujoco.route import route

    found = route(out, w, masks=masks)
    return dict(ok=bool(found["pass"]), reason="; ".join(found["reasons"]), score=found["score"])
