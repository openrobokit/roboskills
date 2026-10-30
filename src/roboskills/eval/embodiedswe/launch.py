"""`make` for `python -m roboskills.workflows.agent_as_policy.run`: one EmbodiedSWE-Bench episode, its simulator
(`sim/server.py`) in the image on one GPU of this host, and the robot behind the shared Robot contract
(`robot.EmbodiedSWE`).

    python -m roboskills.workflows.agent_as_policy.run --adapter roboskills.eval.embodiedswe.launch \
        --out runs/embodiedswe/<dir> \
        -- task=puzzle.classify_objects.g1.joint seed=1 gpu=0 out=runs/embodiedswe/<dir>/server

kw (strings): task (a cohort preset, `TASKS`), seed (1, 2), out (the episode's files: the robot model, the simulator's
log), gpu, vendor ($ESWE_VENDOR: the EmbodiedSWE checkout with its assets), runtime ($ESWE_RUNTIME: native inside the
image itself, as a Kubernetes Pod; else docker, pulling the image once, or apptainer where docker cannot), image
($ESWE_IMAGE, else `IMAGE`; apptainer: $ESWE_SANDBOX, else that image's sandbox, built once), docker_args ($ESWE_DOCKER_ARGS), frame_every (control steps per video frame), boot_s. The simulator
runs as the caller, in the checkout (mounted read-only at /bench, its .cache writable: the asset fetcher's lock). Kit's
shader and OptiX caches are RoboDojo's (the same Isaac Sim 5.1, so warm): docker volumes <prefix>-kitcache and
<prefix>-optixcache ($ESWE_CACHE_PREFIX, else sk-rd-<uid>); apptainer, <tmp>/rd-<uid>/cache/gpu<gpu>.
"""

import os
import secrets
import time
from pathlib import Path

from roboskills.eval.embodiedswe import SEEDS, TASKS
from roboskills.eval.embodiedswe import vendor as vendor_dir
from roboskills.eval.embodiedswe.robot import EmbodiedSWE
from roboskills.eval.robodojo.launch import isaac_server
from roboskills.eval.runtime import Server, pick
from roboskills.eval.wire import free_port


HERE = Path(__file__).resolve().parent
PACKAGES = HERE.parents[2]  # the directory holding `roboskills` (src/ in a checkout), read-only in the container
IMAGE = "ghcr.io/openrobokit/roboskills-embodiedswe:0.2.0"  # docker/embodiedswe
PYTHON = "/isaac-sim/python.sh"  # Isaac Sim's own python, in the image
OPTIX = "/var/tmp/OptixCache_robodojo"
BENCH = "/bench"  # the checkout, in a container
ENV = dict(ACCEPT_EULA="Y", PRIVACY_CONSENT="Y", OMNI_KIT_ACCEPT_EULA="YES", OMP_NUM_THREADS="4", PYTHONUNBUFFERED="1",
           OPTIX_CACHE_PATH=OPTIX, HF_HUB_OFFLINE="1")  # fmt: skip

_open = {}  # the simulator's server, for close()


def make(task, seed="1", out=None, gpu=None, vendor=None, runtime=None, image=None, docker_args=None,
         frame_every="20", boot_s="1500"):  # fmt: skip
    if task not in TASKS:
        raise ValueError(f"unknown task {task!r}; the cohort is {tuple(TASKS)}")
    if str(seed) not in SEEDS:
        raise ValueError(f"seed is one of {SEEDS}")
    out = Path(out or f"runs/embodiedswe/{task}-s{seed}").resolve()
    out.mkdir(parents=True, exist_ok=True)
    gpu = str(gpu if gpu is not None else os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
    port = free_port()
    server = _server(gpu, out, vendor_dir(vendor), runtime, image, docker_args)
    log = out / "embodiedswe_sim.log"
    t0 = time.monotonic()
    server.start([PYTHON, HERE / "sim/server.py", task, seed, "--port", port, "--out", out, "--frame-every",
                  frame_every], log)  # fmt: skip
    server.ready(log, "\nready ", boot_s)
    print(f"embodiedswe: simulator ready in {time.monotonic() - t0:.0f} s ({server.runtime}, gpu {gpu})", flush=True)
    try:
        return EmbodiedSWE(port, task, seed, out, int(frame_every))
    except BaseException:
        close()
        raise


def _server(gpu, out, vendor, runtime, image, docker_args):
    """The simulator's runtime on `gpu` with `out` writable, in the checkout (`roboskills.eval.runtime`)."""
    sandbox, image = image or os.environ.get("ESWE_SANDBOX"), image or os.environ.get("ESWE_IMAGE", IMAGE)
    runtime = pick(runtime, image, "ESWE_RUNTIME", PYTHON)
    if runtime == "native":  # already in the image: no container, `gpu` of this host's (or Pod's) own
        _open["server"] = Server(runtime, None, gpu, env=dict(ENV, PYTHONPATH=str(vendor)), workdir=str(vendor))
        return _open["server"]
    (vendor / ".cache").mkdir(exist_ok=True)
    mounts = [(PACKAGES, PACKAGES, "ro"), (out, out, "rw"), (vendor, BENCH, "ro"),
              (vendor / ".cache", f"{BENCH}/.cache", "rw")]  # fmt: skip
    name, env = f"eswe-{os.getuid()}-{secrets.token_hex(4)}", dict(ENV, PYTHONPATH=BENCH)
    if runtime == "docker":
        args = os.environ.get("ESWE_DOCKER_ARGS", "") if docker_args is None else docker_args
        _open["server"] = isaac_server(runtime, image, gpu, name=name, env=env, mounts=mounts,
                                       cache=os.environ.get("ESWE_CACHE_PREFIX", f"sk-rd-{os.getuid()}"),
                                       docker_args=args, workdir=BENCH)  # fmt: skip
    else:
        _open["server"] = isaac_server(runtime, sandbox or image, gpu, name=name, env=env, mounts=mounts, workdir=BENCH)
    return _open["server"]


def close():
    """Stop the simulator and remove its container or scratch (only ours)."""
    server = _open.pop("server", None)
    if server is not None:
        server.close()
