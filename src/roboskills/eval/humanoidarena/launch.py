"""`make` for `python -m roboskills.workflows.agent_as_policy.run`: one HumanoidArena official test episode, its simulator
(`sim/server.py`) in the image on one GPU of this host, and the G1 behind the shared Robot contract (`robot.HumanoidArena`).

    python -m roboskills.workflows.agent_as_policy.run --adapter roboskills.eval.humanoidarena.launch \
        --out runs/humanoidarena/<dir> \
        -- task=sit_sofa seed=1 gpu=0 out=runs/humanoidarena/<dir>/server

kw (strings): task (`TASKS`), seed (1, 2: the official test groups 0 and 1, repeat 0), out (the episode's files: the arm
model, the simulator's log), gpu, vendor ($HA_VENDOR: the HumanoidArena checkout with its assets, the SONIC models and
the G1 description), runtime ($HA_RUNTIME: native inside the image itself, as a Kubernetes Pod: Isaac Sim's python runs
here; else docker, pulling the image once, or apptainer where docker cannot), image ($HA_IMAGE or `IMAGE`; apptainer:
$HA_SANDBOX, else that image's sandbox, built once), docker_args ($HA_DOCKER_ARGS),
frame_every (control steps per video frame), boot_s. The simulator runs in `roboskills.eval.runtime`'s Server: under
docker as root in a container that sees only `gpu`, under Apptainer as the caller. Isaac Sim's caches (shaders, Kit
data) live in one of 16 slots per user and host, held with a lock (<tmp>/sk-ha-<uid>/cache-<k>.lock) while a simulator
runs: concurrent simulators never share one, later ones reuse its warm caches (docker: the volume ha-<uid>-cache-<k> at
/cache, since some dockerds see only /home of the host's paths; apptainer: <tmp>/sk-ha-<uid>/cache-<k>, also the
home).
"""

import fcntl
import os
import tempfile
import time
from pathlib import Path

from roboskills.eval.humanoidarena import GROUPS, TASKS, episode_seed
from roboskills.eval.humanoidarena import vendor as vendor_dir
from roboskills.eval.runtime import Server, pick
from roboskills.eval.wire import free_port


HERE = Path(__file__).resolve().parent
PACKAGES = HERE.parents[2]  # the directory holding `roboskills` (src/ in a checkout), read-only in the container
IMAGE = "ghcr.io/openrobokit/roboskills-humanoidarena:0.2.0"  # docker/humanoidarena
PYTHON = "/isaac-sim/python.sh"  # Isaac Sim's own python, in the image
ENV = dict(OMNI_KIT_ACCEPT_EULA="YES", ACCEPT_EULA="Y", PRIVACY_CONSENT="Y", OMNI_KIT_ALLOW_ROOT="1", HOME="/cache/home",
           PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="8", NVIDIA_DRIVER_CAPABILITIES="all")  # fmt: skip
ULIMITS = "--ulimit memlock=-1 --ulimit stack=67108864"  # (docker)
SLOTS = 16
VULKAN = (
    "f=$(ls /etc/vulkan/icd.d/nvidia_icd*.json /usr/share/vulkan/icd.d/nvidia_icd*.json 2>/dev/null | head -1); "
    '[ -n "$f" ] && export VK_DRIVER_FILES=$f VK_ICD_FILENAMES=$f; mkdir -p "$HOME"; exec "$@"'
)

_open = {}  # the simulator's server and its cache slot, for close()


def start(
    task,
    seed="1",
    out=None,
    gpu=None,
    vendor=None,
    image=None,
    docker_args=None,
    frame_every="25",
    boot_s="1800",
    runtime=None,
):
    """Boot the simulator; returns its port and `out` (and leaves it running until close())."""
    if task not in TASKS:
        raise ValueError(f"unknown task {task!r}; the cohort is {tuple(TASKS)}")
    if str(seed) not in GROUPS:
        raise ValueError(f"seed is one of {tuple(GROUPS)} (official test groups 0 and 1, repeat 0)")
    out = Path(out or f"runs/humanoidarena/{task}-s{seed}").resolve()
    out.mkdir(parents=True, exist_ok=True)
    vendor = vendor_dir(vendor)
    gpu = str(gpu if gpu is not None else os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
    port, log, t0 = free_port(), out / "humanoidarena_sim.log", time.monotonic()
    try:
        server = _server(gpu, vendor, image, docker_args, runtime)
        server.start(["/bin/bash", "-c", VULKAN, "_", PYTHON, HERE / "sim/server.py", task, episode_seed(task, seed),
                      "--port", port, "--arena", vendor / "HumanoidArena/isaaclab_twist2_g1", "--sonic", vendor / "sonic",
                      "--frame-every", frame_every], log)  # fmt: skip
        server.ready(log, "\nready ", boot_s)
    except BaseException:
        close()
        raise
    print(f"humanoidarena: simulator ready in {time.monotonic() - t0:.0f} s ({server.runtime}, gpu {gpu})", flush=True)
    return port, out


def _server(gpu, vendor, image, docker_args, runtime):
    """The simulator's runtime on `gpu` (`roboskills.eval.runtime`), holding a cache slot."""
    sandbox, image = image or os.environ.get("HA_SANDBOX"), image or os.environ.get("HA_IMAGE", IMAGE)
    runtime = pick(runtime, image, "HA_RUNTIME")  # (other images have PYTHON)
    k, cache = _slot()
    name = f"ha-{os.getuid()}-{os.getpid()}-{k}"
    mounts = [(PACKAGES, PACKAGES, "ro"), (vendor, vendor, "ro")]
    if runtime == "native":  # already in the image: no container; the slot's directory is the home
        server = Server(runtime, None, gpu, name=name, env=dict(ENV, HOME=str(cache / "home")))
    elif runtime == "docker":  # as root (the image's user), its caches and home in the slot's volume
        args = os.environ.get("HA_DOCKER_ARGS", "") if docker_args is None else docker_args
        server = Server(runtime, image, gpu, name=name, env=ENV, root=True,
                        mounts=mounts + [(f"ha-{os.getuid()}-cache-{k}", "/cache", "rw")],
                        docker_args=f"{ULIMITS} {args}")  # fmt: skip
    else:  # Kit writes its caches, data and logs inside /isaac-sim: on the host's disk, not the overlay's memory
        for d in ("home", "kit-cache", "var-tmp"):
            (cache / d).mkdir(parents=True, exist_ok=True)
        mounts += [(cache, "/cache", "rw"), (cache / "kit-cache", "/isaac-sim/kit/cache", "rw"),
                   (cache / "var-tmp", "/var/tmp", "rw")]  # fmt: skip
        env = {k: v for k, v in ENV.items() if k != "HOME"}  # (apptainer sets HOME itself: the slot's home)
        server = Server(runtime, sandbox or image, gpu, name=name, env=env, mounts=mounts, home=cache / "home",
                        scratch=("/isaac-sim/kit/data", "/isaac-sim/kit/logs", "/dev/shm"))  # fmt: skip
        #   (upstream's image writer maps shared memory; --contain's /dev/shm is a few MiB of the session's memory)
    _open["server"] = server
    return server


def make(
    task,
    seed="1",
    out=None,
    gpu=None,
    vendor=None,
    image=None,
    docker_args=None,
    frame_every="25",
    boot_s="1800",
    runtime=None,
):
    from roboskills.eval.humanoidarena.robot import HumanoidArena

    port, out = start(task, seed, out, gpu, vendor, image, docker_args, frame_every, boot_s, runtime)
    try:
        return HumanoidArena(port, task, seed, out, frame_every)
    except BaseException:
        close()
        raise


def _slot():
    """The first cache slot no running simulator of this user holds on this host, locked until close(): (k, its
    directory on this host's disk)."""
    root = Path(tempfile.gettempdir()) / f"sk-ha-{os.getuid()}"
    root.mkdir(parents=True, exist_ok=True)
    for k in range(SLOTS):
        lock = open(root / f"cache-{k}.lock", "a")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock.close()
            continue
        _open["slot"] = lock
        return k, root / f"cache-{k}"
    raise RuntimeError(f"all {SLOTS} HumanoidArena cache slots in {root} are in use")


def close():
    """Stop the simulator and remove its container or scratch (only ours); release the cache slot."""
    server = _open.pop("server", None)
    if server is not None:
        server.close()
    if _open.get("slot"):
        _open.pop("slot").close()
