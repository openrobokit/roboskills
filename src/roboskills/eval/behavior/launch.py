"""`make` for `python -m roboskills.workflows.agent_as_policy.run`: one BEHAVIOR-1K episode, its simulator (`sim/server.py`) in
the official image on one GPU of this host, and the R1Pro behind the shared Robot contract (`robot.Behavior`).

    python -m roboskills.workflows.agent_as_policy.run --adapter roboskills.eval.behavior.launch \
        --out runs/behavior/<dir> \
        -- task=dispose_of_glass seed=1 gpu=0 out=runs/behavior/<dir>/server

kw (strings): task (a cohort task, `TASKS`), seed (1-3: the challenge's public test instances 301-303), out (the
episode's files: the robot model, the simulator's log), gpu, data ($BEHAVIOR_DATA: the dataset), runtime ($BH_RUNTIME:
native inside the image itself, as a Kubernetes Pod: OmniGibson's python runs here; else docker, or apptainer where docker
or its image is missing), image (docker: $BH_IMAGE or the official image by digest; apptainer: $BH_SANDBOX or
~/.roboskills/images/behavior-3.9.2, a sandbox of it), docker_args ($BH_DOCKER_ARGS), frame_every (control steps per
video frame), boot_s. The simulator runs as the caller. OmniGibson's app data (Kit's shader and texture caches) lives in
one of 16 slots per user and host, held with a lock (<tmp>/sk-bh-<uid>/appdata-<k>.lock) while a simulator runs:
concurrent simulators never share one, later ones reuse its warm caches. Docker keeps a slot in the volume
sk-bh-<uid>-appdata-<k> (some dockerds see only /home of the host's paths), native and apptainer runs in
<tmp>/sk-bh-<uid>/appdata-<k>.
"""

import fcntl
import os
import tempfile
import time
from pathlib import Path

from roboskills.eval.behavior import INSTANCES, TASKS
from roboskills.eval.behavior import data as data_dir
from roboskills.eval.behavior.robot import Behavior
from roboskills.eval.runtime import Server, pick
from roboskills.eval.wire import free_port


HERE = Path(__file__).resolve().parent
PACKAGES = HERE.parents[2]  # the directory holding `roboskills` (src/ in a checkout), read-only in the container
IMAGE = "docker.io/stanfordvl/behavior:3.9.2@sha256:2702900192741904e69a573aa7869ea51c592586f9791050796efc37faa893fb"
PYTHON = "/opt/conda/envs/behavior/bin/python"  # OmniGibson's python, in the image
APPS = "/opt/conda/envs/behavior/lib/python3.11/site-packages/isaacsim/apps"  # OmniGibson copies its .kit file here
DATA, APPDATA = "/data", "/cache/appdata"  # where the image expects the dataset and OmniGibson's app data
ENV = dict(OMNI_KIT_ACCEPT_EULA="YES", ACCEPT_EULA="Y", PRIVACY_CONSENT="Y", OMNIGIBSON_HEADLESS="1", PYTHONHASHSEED="0",
           PYTHONUNBUFFERED="1", OMP_NUM_THREADS="8", MKL_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8")  # fmt: skip
SLOTS = 16

_open = {}  # the simulator's server and its app-data slot, for close()


def make(task, seed="1", out=None, gpu=None, data=None, runtime=None, image=None, docker_args=None, frame_every="10",
         boot_s="1800"):  # fmt: skip
    if task not in TASKS:
        raise ValueError(f"unknown task {task!r}; the cohort is {tuple(TASKS)}")
    if str(seed) not in INSTANCES:
        raise ValueError(f"seed is one of {tuple(INSTANCES)} (the challenge's public test instances 301-303)")
    out = Path(out or f"runs/behavior/{task}-s{seed}").resolve()
    out.mkdir(parents=True, exist_ok=True)
    data = data_dir(data)
    gpu = str(gpu if gpu is not None else os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
    port, log, t0 = free_port(), out / "behavior_sim.log", time.monotonic()
    try:  # a simulator that does not boot releases its slot too
        server, runtime = _server(gpu, data, image, docker_args, runtime)
        server.start([PYTHON, HERE / "sim/server.py", task, INSTANCES[str(seed)], "--port", port, "--frame-every",
                      frame_every], log)  # fmt: skip
        server.ready(log, "\nready ", boot_s)
    except BaseException:
        close()
        raise
    print(f"behavior: simulator ready in {time.monotonic() - t0:.0f} s ({runtime}, gpu {gpu})", flush=True)
    try:
        return Behavior(port, task, seed, out, data, None if runtime == "native" else DATA, frame_every)
    except BaseException:
        close()
        raise


def _slot():
    """The first app-data slot no running simulator of this user holds on this host, locked until close(): (k, its
    directory)."""
    root = Path(tempfile.gettempdir()) / f"sk-bh-{os.getuid()}"
    root.mkdir(parents=True, exist_ok=True)
    for k in range(SLOTS):
        lock = open(root / f"appdata-{k}.lock", "a")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock.close()
            continue
        _open["slot"] = lock
        path = root / f"appdata-{k}"
        path.mkdir(exist_ok=True)
        return k, path
    raise RuntimeError(f"all {SLOTS} BEHAVIOR app-data slots in {root} are in use")


def _server(gpu, data, image, docker_args, runtime):
    """The simulator's runtime on `gpu` (`roboskills.eval.runtime`), holding an app-data slot; docker without the image
    runs apptainer."""
    image = image or os.environ.get("BH_IMAGE", IMAGE)
    runtime = pick(runtime, image, "BH_RUNTIME", PYTHON)
    k, appdata = _slot()
    user = dict(
        HOME=tempfile.gettempdir() if runtime == "native" else "/tmp",
        USER=os.environ.get("USER", "sim"),
        LOGNAME=os.environ.get("USER", "sim"),
    )  # fmt: skip  (torch looks the user up; the container has none)
    if runtime == "native":  # already in the image: no container, `gpu` of this host's (or Pod's) own
        env = dict(ENV, **user, OMNIGIBSON_GPU_ID="0", OMNIGIBSON_DATA_PATH=str(data),
                   OMNIGIBSON_APPDATA_PATH=str(appdata))  # fmt: skip
        _open["server"] = Server(runtime, None, gpu, env=env)
        return _open["server"], runtime
    env = dict(ENV, **user, OMNIGIBSON_DATA_PATH=DATA, OMNIGIBSON_APPDATA_PATH=APPDATA)
    name = f"bh-{os.getuid()}-{os.getpid()}-{k}"
    mounts = [(PACKAGES, PACKAGES, "ro"), (data, DATA, "ro")]
    if runtime == "docker":
        args = os.environ.get("BH_DOCKER_ARGS", "") if docker_args is None else docker_args
        mounts.append((f"sk-bh-{os.getuid()}-appdata-{k}", APPDATA, "rw"))
        server = _open["server"] = Server(runtime, image, gpu, name=name, env=env, mounts=mounts, docker_args=args)
        server.root(
            "/bin/bash", "-c", f"chmod a+w {APPS} && chown {os.getuid()}:{os.getgid()} {APPDATA}"
        )  # in this container only
        return server, runtime
    sandbox = (
        image
        if image != IMAGE
        else os.environ.get("BH_SANDBOX", str(Path.home() / ".roboskills/images/behavior-3.9.2"))
    )
    _open["server"] = Server(runtime, sandbox, gpu, name=name, env=env, mounts=mounts + [(appdata, APPDATA, "rw")])
    return _open["server"], runtime


def close():
    """Stop the simulator and remove its container or scratch (only ours); release the app-data slot."""
    server = _open.pop("server", None)
    if server is not None:
        server.close()
    if _open.get("slot"):
        _open.pop("slot").close()
