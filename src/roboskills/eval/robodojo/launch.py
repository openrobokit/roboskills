"""`make_vector` for `python -m roboskills.workflows.agent_as_policy.run`: RoboDojo's vector sim (`sim/vector.py`: one
Isaac scene, one env per layout) in a container on one GPU of this host, and one clocked RoboDojo robot per env
(`robot.Clocked`).

    python -m roboskills.workflows.agent_as_policy.run --adapter roboskills.eval.robodojo.launch \
        --out runs/harness_parallel/<dir> -- task=general_pickup layouts=30,31 gpu=0 mode=dev

kw (strings): task, layouts (comma list), gpu, mode (dev|eval), suite (legacy: layouts.py's tasks; full5: the 42 canonical
tasks, eval layouts 0-4), seed (2), obs (full: simulator RGB-D with camera calibration; official: RoboDojo's eval
observation, camera RGB only, its depth and calibration estimated by `rgbd.py`), runtime ($RD_RUNTIME: native inside
the sim's own image, as a Kubernetes Pod: Isaac Sim's python runs here; else docker, pulling the image once, or apptainer
where docker cannot), image ($RD_IMAGE or `IMAGE`; apptainer: $RD_SANDBOX, else that image's sandbox, built once),
docker_args ($RD_DOCKER_ARGS), vendor ($ROBODOJO_VENDOR: the RoboDojo clone); the launcher adds out
(its --out). One image serves every user: the sim runs as the caller, and Kit and OptiX caches persist per user (docker
volumes $RD_CACHE_PREFIX-{kit,optix}cache, default sk-rd-<uid>; apptainer: <tmp>/rd-<uid>/cache on the host's own disk).
Episodes save no video.mp4 unless MANIP_VIDEO=1. The sim logs every tick's commands to `sim_events.jsonl`, so a finished
batch replays later, its videos re-rendered:

    python -m roboskills.eval.robodojo.launch replay runs/harness_parallel/<dir> gpu=0   # -> <dir>/replay/
"""

import json
import os
import secrets
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from roboskills.eval.robodojo import vendor as vendor_dir
from roboskills.eval.robodojo.layouts import FIXES, check_layout
from roboskills.eval.robodojo.robot import DT, make
from roboskills.eval.runtime import Server, pick
from roboskills.eval.wire import free_port


HERE = Path(__file__).resolve().parent
PACKAGES = HERE.parents[2]  # the directory holding `roboskills` (src/ in a checkout), read-only in the container
OPTIX = "/var/tmp/OptixCache_robodojo"  # OptiX's own default names the user, who differs per caller
ENV = dict(ACCEPT_EULA="Y", PRIVACY_CONSENT="Y", OMNI_KIT_ACCEPT_EULA="YES", OMP_NUM_THREADS="4", PYTHONUNBUFFERED="1",
           OPTIX_CACHE_PATH=OPTIX)  # fmt: skip
IMAGE = "ghcr.io/openrobokit/roboskills-robodojo:0.2.0"  # docker/robodojo
PYTHON = "/isaac-sim/python.sh"  # Isaac Sim's own python, in the image
EXTS = "Kit/Isaac-Sim/5.1/exts/3"  # under Kit's data: its registry extensions, as run.py gives every container
FPS = round(1 / (DT * 8))  # replay videos: a head frame per rendered observation, about every 8 steps as robot.py's


_open = {}  # the simulator's server, for close()


def make_vector(
    task,
    layouts,
    gpu,
    out,
    mode="dev",
    suite="legacy",
    seed="2",
    obs="full",
    image=None,
    docker_args=None,
    vendor=None,
    boot_s="1200",
    runtime=None,
):
    layouts, seed = [int(x) for x in layouts.split(",")], int(seed)
    if obs not in ("full", "official"):
        raise ValueError("obs is full or official")
    for layout in layouts:
        check_layout(task, layout, seed, mode, suite)
    out = Path(out).resolve()
    vendor = vendor_dir(vendor)
    server = _server(gpu, out, vendor, image, docker_args, runtime)
    ports = [free_port() for _ in layouts]
    log = out / "vector_sim.log"
    server.start([PYTHON, HERE / "sim/vector.py", "--ports", ",".join(map(str, ports)), "--layouts",
                  ",".join(map(str, layouts)), "--task", task, "--mode", mode, "--suite", suite, "--vendor", vendor,
                  "--seed", seed, "--events", out / "sim_events.jsonl"] + (["--official"] if obs == "official" else []),
                 log)  # fmt: skip
    server.ready(log, "ready vector_sim num_envs=", boot_s)

    video = os.environ.get("MANIP_VIDEO", "0") == "1"

    def robot(i):  # the robot's own parsing and reset, on env i's socket
        fix = {"fix": FIXES[task]} if task in FIXES else {}
        return make(task=task, layout=layouts[i], seed=seed, sim=ports[i], mode=mode, obs=obs, suite=suite,
                    clocked=True, video=video, vendor=vendor, out=str(out / f"env{i}"), ws=str(Path.cwd()), **fix)  # fmt: skip

    try:
        with ThreadPoolExecutor(len(layouts)) as pool:  # the envs connect together
            return list(pool.map(robot, range(len(layouts))))
    except BaseException:
        close()
        raise


def replay(run, gpu, image=None, docker_args=None, vendor=None, runtime=None):
    """Replay the finished batch in `run` (a make_vector `out`) from its `sim_events.jsonl` (`sim/replay.py`) on `gpu`:
    `run/replay/env{i}.mp4` per env and `replay.json`, each episode's logged and replayed outcome and final gaps."""
    run = Path(run).resolve()
    out = run / "replay"
    out.mkdir(exist_ok=True)
    vendor = vendor_dir(vendor)
    server = _server(gpu, run, vendor, image, docker_args, runtime)
    try:
        code = server.start([PYTHON, HERE / "sim/replay.py", "--events", run / "sim_events.jsonl", "--out", out,
                             "--vendor", vendor], out / "replay_sim.log").wait()  # fmt: skip
    finally:
        close()
    if code or not (out / "replay.json").exists():
        raise RuntimeError(f"replay failed (exit {code}); see {out / 'replay_sim.log'}")
    import imageio

    for frames in sorted(out.glob("env*_frames.npz")):
        imageio.mimwrite(out / f"{frames.name.removesuffix('_frames.npz')}.mp4", np.load(frames)["rgb"], fps=FPS)
        frames.unlink()
    return json.loads((out / "replay.json").read_text())


def _server(gpu, out, vendor, image, docker_args, runtime):
    """The sim's runtime on `gpu` with `out` writable (`roboskills.eval.runtime`). Kit's and OptiX's caches persist per
    user: docker volumes, or per GPU on the host's own disk."""
    sandbox, image = image or os.environ.get("RD_SANDBOX"), image or os.environ.get("RD_IMAGE", IMAGE)
    runtime = pick(runtime, image, "RD_RUNTIME", PYTHON)
    mounts = [(PACKAGES, PACKAGES, "ro"), (out, out, "rw"), (vendor.parent, vendor.parent, "ro")]
    name = f"rd-harness-{os.getuid()}-{secrets.token_hex(4)}"
    if runtime == "native":  # already in the image: no container, `gpu` of this host's (or Pod's) own
        server = Server(runtime, None, gpu, env=ENV)
    elif runtime == "docker":
        args = os.environ.get("RD_DOCKER_ARGS", "") if docker_args is None else docker_args
        server = isaac_server(runtime, image, gpu, name=name, env=ENV, mounts=mounts,
                              cache=os.environ.get("RD_CACHE_PREFIX", f"sk-rd-{os.getuid()}"), docker_args=args)  # fmt: skip
    else:
        server = isaac_server(runtime, sandbox or image, gpu, name=name, env=ENV, mounts=mounts)
    _open["server"] = server
    return server


def isaac_server(runtime, image, gpu, *, name, env, mounts, cache=None, docker_args="", workdir=None):
    """A docker or apptainer `Server` for Isaac Sim 5.1 in RoboDojo's image or one built on it (`image`: the docker
    image; apptainer, also a sandbox directory). Kit's and OptiX's caches persist per user: the docker volumes
    <cache>-kitcache and <cache>-optixcache; apptainer, <tmp>/rd-<uid>/cache/gpu<gpu> on the host's own disk, with Kit's
    data and logs private. Kit gets the registry extensions in ~/.roboskills/runs/docker/kit-exts, when there are some."""
    mounts = list(mounts)
    if runtime == "docker":
        mounts += [(f"{cache}-kitcache", "/isaac-sim/kit/cache", "rw"), (f"{cache}-optixcache", OPTIX, "rw")]
        server = Server(runtime, image, gpu, name=name, env=dict(env, HOME="/tmp"), mounts=mounts, workdir=workdir,
                        docker_args=docker_args)  # fmt: skip
    else:
        local = Path(tempfile.gettempdir()) / f"rd-{os.getuid()}" / f"cache/gpu{gpu}"
        for d in ("kit", "var-tmp"):
            (local / d).mkdir(parents=True, exist_ok=True)
        mounts += [(local / "kit", "/isaac-sim/kit/cache", "rw"), (local / "var-tmp", "/var/tmp", "rw")]  # OPTIX inside
        server = Server(runtime, image, gpu, name=name, env=env, mounts=mounts, home=True, workdir=workdir,
                        scratch=("/isaac-sim/kit/data", "/isaac-sim/kit/logs"))  # fmt: skip
    exts = Path.home() / ".roboskills/runs/docker/kit-exts"
    try:
        if exts.is_dir() and runtime == "docker":
            subprocess.run(server.prefix + ["mkdir", "-p", f"/isaac-sim/kit/data/{EXTS}"], check=True)
            subprocess.run(["docker", "cp", f"{exts}/.", f"{server.name}:/isaac-sim/kit/data/{EXTS}"], check=True,
                           capture_output=True)  # fmt: skip
        elif exts.is_dir():
            shutil.copytree(exts, server.scratch_dir("/isaac-sim/kit/data") / EXTS)
    except BaseException:
        server.close()
        raise
    return server


def close():
    """Stop the simulator and remove its container or scratch (only ours)."""
    server = _open.pop("server", None)
    if server is not None:
        server.close()


if __name__ == "__main__":
    if sys.argv[1:2] != ["replay"] or len(sys.argv) < 3:
        sys.exit(
            "usage: python -m roboskills.eval.robodojo.launch replay RUN gpu=G [runtime= image= docker_args= vendor=]"
        )
    episodes = replay(sys.argv[2], **dict(kv.split("=", 1) for kv in sys.argv[3:]))["episodes"]
    print(json.dumps(episodes, indent=1))
