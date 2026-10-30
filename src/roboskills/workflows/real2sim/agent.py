"""The `agent` tool: Codex, only where track's checks still fail, at a fraction of a prompt-only agent's tokens. It gets a
compact prompt: the clip's check numbers, its overlay image, `r2s`'s help (RoboKit's edit.py) and a short hint list, and is
stopped at `seconds` or after `commands` shell commands. A tool-free call then judges source | render pairs with five
fixed questions; any "no" sends it back with the answers. Vision models never rerun; Codex never sees ground truth.
"""

import filecmp
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from roboskills.core import llm
from roboskills.core.process import terminate


QUESTIONS = dict(place="Is the object in the right place?", size="Is it the right size?",
                 orientation="Is it oriented right?", motion="Does it move when and as the hand moves it?",
                 contact="Is it free of floating or penetrating?")  # fmt: skip
SYSTEM = "You are a careful engineer checking and fixing 3D reconstructions of hand-held objects. Be brief.\n"
FIX = """Fix the 3D reconstruction of a {object} that a right hand handles in a {n}-frame video (reference frame {ref}: the object before the hand moves it). Its code checks still fail: {failed}.
You are in its workspace. Change it only through the `r2s` command (help below). Do not read code or other files; never rerun vision models.

Now (`r2s look`):
{report}
Attached: that agent/look.png. Per frame, the source cropped round the object: SAM 3's object mask outlined red, the reconstruction's silhouette green, the hand tinted blue.
{answers}
Techniques that work:
1. Size first: compare `size` with the real {object}'s typical dimensions. A box or can whose size or shape is off: `r2s shape box|cylinder` with its real dimensions in cm. Other objects vary more: stretch them (`r2s shape mesh`) only when the mesh is clearly malformed (flat, swollen).
2. Then `r2s refit`: each frame's pose fitted to the mask outline, hand pixels excluded, then smoothed (`--weight 0.1` follows the mask more closely).
3. Upside down or turned: `r2s rotate`, then `r2s refit`.
4. Depth (distance from the camera) is the least certain number; keep it unless the image clearly contradicts it (`r2s scale S`: the same image, S times deeper).
5. residual is how far the mesh's surface lies from the measured depth: a large rise after an edit (well above 0.05) means the size or pose no longer fits the depth, so `r2s undo` it. The hand hides part of the mask, so iou need not pass.

Limits: at most {commands} shell commands and {minutes} minutes. Every edit prints the numbers and redraws agent/look.png; view it only when the numbers leave doubt. End with one line: what you changed.

{usage}"""
JUDGE = """Judge a 3D reconstruction of a {object} that a right hand handles in a video. The attached image has one row per frame (frame number top left): left, the source video cropped round the object; right, the reconstructed object drawn from the same camera in its colours over the darkened source, same crop. The reconstructed object measures {size} cm; judge its size by these against a real {object}'s, not by its thickness in the drawing. Where the hand hides the object, judge what remains visible.
Answer each question yes or no, with a reason of at most 12 words:
{questions}
Reply with JSON only: {{"place": ["yes" or "no", "reason"], "size": [...], "orientation": [...], "motion": [...], "contact": [...]}}"""


def finish(w: Path, support: str = "pedestal") -> dict:
    """Install the state: the first align/, track/, collision/ and scene/ kept in agent/before/, the state's mesh and
    poses copied in (the reference pose into align/pose.json) and track's status rechecked; then real2sim's runner, with
    the tools the workspace ran, reruns coacd on a changed mesh and mujoco. Its record (`checks`: track's new status)."""
    from roboskills.tools.robokit.edit import STATE, State
    from roboskills.workflows import real2sim

    s, keep = State(w), w / "agent/before"
    tools = [name for name, (_, done) in real2sim.TOOLS.items() if name != "agent" and (w / done).exists()]
    if not keep.exists():
        keep.mkdir(parents=True)
        for name in ("align", "track", "collision", "scene"):
            shutil.copytree(w / name, keep / name)
    changed = not filecmp.cmp(w / STATE / "mesh.ply", keep / "align/mesh.ply", shallow=False)
    shutil.copy(w / STATE / "mesh.ply", w / "align/mesh.ply")
    pose = json.loads((keep / "align/pose.json").read_text())
    (w / "align/pose.json").write_text(
        json.dumps(dict(pose, camera_from_object=s.poses[s.ref].tolist()), indent=2) + "\n"
    )
    np.savez(w / "track/poses.npz", **dict(np.load(keep / "track/poses.npz")) | dict(camera_from_object=s.poses))
    p = real2sim.inputs(tools, support)
    record = dict(mesh_changed=changed, checks=real2sim.call(w, "track", p, fresh=False))
    for name in ["coacd"] * changed + ["mujoco"]:  # rerun by the runner, as unfinished
        shutil.rmtree(w / real2sim.folder(name), ignore_errors=True)
    real2sim.run(w, tools, support)
    return record


def r2s(o, *args):
    """`r2s ARGS` in the workspace (the runner's own calls: no tokens); its stdout."""
    env = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
    out = subprocess.run([o / "bin/r2s", *args], cwd=o.parent, env=env, capture_output=True, text=True, check=False)
    if out.returncode:
        raise RuntimeError(f"r2s {' '.join(args)} failed: {out.stderr[-2000:]}")
    return out.stdout.strip()


def codex(o, step, text, images, *, shell, seconds, commands, model, effort):
    """One `codex exec` logged in o/`step`/, tried once more when the model is at capacity: `shell` gives it the shell
    and view_image in the workspace, else no tools; stopped at `seconds` or past `commands` commands. Its receipt."""
    for log in (o / step, o / f"{step}-retry"):
        record = codex_once(o, log, text, images, shell, seconds, commands, model, effort)
        said = (log / "stdout.jsonl").read_text() + (log / "stderr.txt").read_text()
        if record["status"] != "codex failed" or "capacity" not in said.lower():
            break
    return record


def codex_once(o, log, text, images, shell, seconds, commands, model, effort):
    log.mkdir(parents=True, exist_ok=True)
    (log / "prompt.md").write_text(text)
    (log / "system.txt").write_text(SYSTEM)
    env = dict(os.environ, PATH=f"{o / 'bin'}{os.pathsep}{os.environ['PATH']}", OMP_NUM_THREADS="4",
               MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")  # fmt: skip
    env.pop("VIRTUAL_ENV", None)
    reply, command = log / "reply.txt", llm.codex_command(log, model, effort, images=images)
    if shell:  # the short instructions stay: Codex's own cost 2.5 times the tokens
        command[command.index("--sandbox") + 1] = "danger-full-access"
        for i, value in enumerate(command):
            if value in ("shell_tool", "view_image") and command[i - 1] == "--disable":
                command[i - 1] = "--enable"
    start, state = time.monotonic(), "finished"
    with (log / "stdout.jsonl").open("w") as out, (log / "stderr.txt").open("w") as err:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=out, stderr=err, text=True, env=env,
                                   cwd=o.parent if shell else log, start_new_session=True)  # fmt: skip
        process.stdin.write(text)
        process.stdin.close()
        while process.poll() is None:
            time.sleep(2)
            lines = (log / "stdout.jsonl").read_text().splitlines()
            started = sum('"item.started"' in x and '"command_execution"' in x for x in lines)
            if time.monotonic() - start > seconds:
                state = f"stopped at the {seconds} s cap"
            elif shell and started > commands:
                state = f"stopped past {commands} commands"
            else:
                continue
            terminate(process, 10)
            break
    rows = [json.loads(x) for x in (log / "stdout.jsonl").read_text().splitlines() if x.startswith("{")]
    final = next((r for r in reversed(rows) if r.get("type") == "turn.completed"), {})
    if state == "finished" and (process.returncode or not final):
        state = "codex failed"
    done = [r for r in rows if r.get("type") == "item.completed" and r.get("item", {}).get("type") == "command_execution"]  # fmt: skip
    record = dict(step=log.name, status=state, seconds=round(time.monotonic() - start, 1), commands=len(done),
                  usage=final.get("usage", {}), reply=reply.read_text() if reply.exists() else "")  # fmt: skip
    (log / "receipt.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def usage(o):
    """`r2s --help` and each command's, without the -h lines."""
    text = [r2s(o, "--help")] + [r2s(o, name, "--help") for name in ("look", "shape", "scale", "rotate", "move", "refit")]  # fmt: skip
    return "\n\n".join("\n".join(x for x in t.splitlines() if "-h, --help" not in x) for t in text)


def fix(w, o, rnd, failed, answers, support, seconds, commands, **settings):
    """One round of Codex's edits through r2s, then `finish` installs them. Its receipt, with `finish`'s record."""
    report, note = r2s(o, "look"), ""
    if answers:
        no = {k: v for k, v in answers.items() if str(v[0]).lower() != "yes"}
        note = "\nA visual check of the current reconstruction answered no to: " + "; ".join(f"{QUESTIONS[k]} ({v[1]})" for k, v in no.items()) + "\n"  # fmt: skip
    reference = json.loads(p.read_text())["frame"] if (p := w / "input/reference.json").exists() else 0
    text = FIX.format(object=(w / "input/object.txt").read_text().strip(), n=len(list((w / "masks").glob("*.png"))),
                      ref=reference, failed=", ".join(failed) or "none (a visual check failed)", report=report,
                      answers=note, commands=commands, minutes=seconds // 60, usage=usage(o))  # fmt: skip
    image = o / f"look_r{rnd}.png"
    shutil.copy(o / "look.png", image)
    record = codex(o, f"fix-r{rnd}", text, [image], shell=True, seconds=seconds, commands=commands, **settings)
    if (o / "edits.jsonl").exists():
        record["finish"] = finish(w, support)
    return record


def judge(w, o, rnd, seconds=180, **settings):
    """One tool-free call on source | render pairs: the five QUESTIONS. Its receipt, with `answers` and each `no`."""
    name = f"judge_r{rnd}.png"
    size = re.search(r"size: ([0-9. x]+) cm", r2s(o, "look", "--render", "--out", f"agent/{name}")).group(1)
    text = JUDGE.format(object=(w / "input/object.txt").read_text().strip(), size=size,
                        questions="\n".join(f"{k}: {q}" for k, q in QUESTIONS.items()))  # fmt: skip
    record = codex(o, f"judge-r{rnd}", text, [o / name], shell=False, seconds=seconds, commands=0, **settings)
    match = re.search(r"\{.*\}", record["reply"], re.S)
    try:
        answers = {k: v for k, v in json.loads(match.group(0)).items() if k in QUESTIONS} if match else {}
    except ValueError:
        answers = {}
    return record | dict(answers=answers, no=[k for k, v in answers.items() if str(v[0]).lower() != "yes"])


def kept(nos, residuals, slack=0.01):
    """The round whose state stays: a later one replaces the kept one when judged with no more "no"s and its residual
    check no more than `slack` above the kept one's (the visual check cannot see depth; the residual check does)."""
    best = 0
    for i in range(1, len(nos)):
        if len(nos[i]) <= len(nos[best]) and residuals[i] <= residuals[best] + slack:
            best = i
    return best + 1


def agent(w: Path, o: Path, p: dict, rounds: int = 2, seconds: int = 480, commands: int = 8,
          model: str = "gpt-6.1-sol", effort: str = "low") -> dict:  # fmt: skip
    """`o`/agent.json: nothing where track's checks pass; else up to `rounds` rounds of `fix`, each `judge`d, a "no"
    sending it back with the answers, and the `kept` state installed. Its `tokens` are what every call reported."""
    if p["shape"] != "align":
        raise ValueError("agent edits align/'s object: leave fit out of the tools")
    shutil.rmtree(o, ignore_errors=True)
    (o / "bin").mkdir(parents=True)
    track = json.loads((w / "track/status.json").read_text())
    record = dict(failed=track.get("failed", []), rounds=0, tokens={})
    if not record["failed"]:
        (o / "agent.json").write_text(json.dumps(record, indent=1) + "\n")
        return record
    (o / "bin/r2s").write_text(f'#!/bin/sh\nexec {sys.executable} -m roboskills.tools.robokit.edit -w {w} "$@"\n')
    (o / "bin/r2s").chmod(0o755)
    settings, calls, nos, residuals, answers = dict(model=model, effort=effort), [], [], [], None
    for rnd in range(1, rounds + 1):
        fixed = fix(w, o, rnd, record["failed"], answers, p["support"], seconds, commands, **settings)
        residuals.append(fixed["finish"]["checks"]["residual"] if "finish" in fixed else residuals[-1] if residuals else track["residual"])  # fmt: skip
        shutil.copytree(o / "state", o / f"state_r{rnd}")
        verdict = judge(w, o, rnd, **settings)
        calls += [fixed, verdict]
        nos.append(verdict["no"])
        answers = verdict["answers"]
        if not verdict["no"] or not answers:
            break
    best = kept(nos, residuals)
    if best != len(nos):  # back to the better judged state
        shutil.rmtree(o / "state")
        shutil.copytree(o / f"state_r{best}", o / "state")
        finish(w, p["support"])
    keys = {k for c in calls for k in c["usage"]}
    record |= dict(rounds=len(nos), kept_round=best, no=nos, residual=residuals, seconds=round(sum(c["seconds"] for c in calls), 1),
                   commands=sum(c["commands"] for c in calls), tokens={k: sum(c["usage"].get(k, 0) for c in calls) for k in keys})  # fmt: skip
    (o / "agent.json").write_text(json.dumps(record, indent=1) + "\n")
    return record
