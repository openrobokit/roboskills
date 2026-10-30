"""One episode of an in-process adapter served over `roboskills.eval.wire`, and the host's `Robot` for it.

A benchmark whose simulator runs in Python of its own (MuJoCo, SAPIEN) keeps its `Robot` class and gives its
`launch.py` two functions: `episode(**kw)`, which builds that Robot for one episode inside the server, and `make`,
which starts the server and returns its `Remote`:

    def make(task, seed="1", out="runs/x", runtime=None, image=None, gpu=None):
        return remote(__name__, dict(task=task, seed=seed, out=out), out=out, image=image or IMAGE, runtime=runtime, gpu=gpu)

    python -m roboskills.eval.serve roboskills.eval.libero.launch --port P --out DIR [--export] -- task=0 seed=1

`runtime` (native, docker or apptainer; default $ROBOSKILLS_RUNTIME, else native) is where the server runs
(`roboskills.eval.runtime`): natively it is a process of this host's Python and environment; in a container it is the
benchmark's image (pulled from GHCR; Apptainer: its sandbox, built from there once), with this host's `roboskills`
package mounted read-only at /src/roboskills and DIR read-write at its own path. There (`--export`) the server copies each arm's model (one file per model: arms that share one share it),
together with the meshes it names, into DIR, so the host loads them. It
serves `facts` once (arms, `dt`, a mobile `base`, a legged `body`, `table_z`, whether it masks the robot), then `obs`,
`follow` (the host answers each `stop` check, which gets the proprioception and cannot call the robot), `finish` (into DIR/finish,
which the host copies to the directory asked for) and the adapter's own `REMOTE` methods (e.g. a privileged `scene`).
Every reply carries `ended` and the arms' current `T_world_base`. The host holds only `Remote`: the
simulator, its state and its files stay with the server.
"""

import argparse
import dataclasses
import functools
import hashlib
import importlib
import os
import shutil
import socket
import sys
import tempfile
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path

from roboskills.eval.runtime import RUNTIMES, Server
from roboskills.eval.wire import free_port, recv, send


PACKAGE = Path(__file__).resolve().parents[1]  # this `roboskills` (src/roboskills in a checkout), /src/roboskills there
_open = []  # this process's Remotes, for close()


def remote(module, kw, *, out, image, runtime=None, gpu=None, boot_s=900, python=None):
    """Start an episode server; native `python` optionally selects the simulator's own interpreter.

    Preserve its venv symlink: resolving it would select the base Python without its installed SDK.
    """
    runtime = runtime or os.environ.get("ROBOSKILLS_RUNTIME", "native")
    if runtime not in RUNTIMES:
        raise ValueError(f"runtime is one of {RUNTIMES}, not {runtime!r}")
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    gpu = str(gpu if gpu is not None else os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
    port = free_port()
    args = [module, "--port", port, "--out", out] + (["--export"] if runtime != "native" else [])
    args += ["--"] + [f"{k}={v}" for k, v in kw.items() if v is not None]
    if runtime == "native":
        server, python = Server(runtime, None, gpu), str(Path(python).absolute()) if python else sys.executable
    else:
        # caches (Warp's, numba's) go to a writable home and /tmp: Apptainer's layer over the image is a small tmpfs
        env = dict(HOME="/tmp", NUMBA_CACHE_DIR="/tmp/numba", MUJOCO_EGL_DEVICE_ID="0", PYTHONUNBUFFERED="1")
        mounts = [(PACKAGE, "/src/roboskills", "ro"), (out, out, "rw")]
        server = Server(runtime, image, gpu, env=env, mounts=mounts, home=True)
        python = "python"
    log = out / "sim.log"
    server.start([python, "-m", "roboskills.eval.serve", *args], log)
    server.ready(log, "\nready ", boot_s)
    try:
        robot = Remote(server, port, out)
    except BaseException:
        server.close()
        raise
    _open.append(robot)
    return robot


def episode_dir(out, default):
    """`out`, else a new directory named after `default` beside it (concurrent episodes never share one); resolved."""
    if out is None:
        Path(default).parent.mkdir(parents=True, exist_ok=True)
        out = tempfile.mkdtemp(prefix=f"{Path(default).name}-", dir=Path(default).parent)
    return Path(out).resolve()


def close():
    """Close every Remote this process started (an adapter's `close`, which the launcher calls at exit)."""
    while _open:
        _open.pop().close()


class Remote:
    """The host's `roboskills.eval.interface.Robot` for one served episode."""

    def __init__(self, server, port, out):
        self.server, self.out, self.ended, self._stop, self._masks = server, Path(out), False, None, {}
        self.lost, self._stopping = None, False
        self.sock = socket.create_connection(("127.0.0.1", port), timeout=3600)
        facts = self._call(op="facts")
        self.arms, self.dt = facts["arms"], facts["dt"]
        for name in ("base", "body", "table_z"):
            if facts[name] is not None:
                setattr(self, name, facts[name])
        if facts["masks"]:
            self.robot_mask = self._robot_mask
        for name in facts["remote"]:
            setattr(self, name, functools.partial(self._method, name))

    def _call(self, **msg):
        if self._stopping:  # the server is waiting for the stop answer, not for a request
            raise RuntimeError("a motion's stop check gets its proprioception; it cannot call the robot")
        if self.lost:  # the stream may hold an earlier reply: no later one can be trusted
            raise RuntimeError(f"the simulator is gone ({self.lost}); see {self.out / 'sim.log'}")
        failed = None
        try:
            send(self.sock, msg)
            while "stop?" in (rep := recv(self.sock)):  # a check of the motion's `stop`, on its proprioception
                self._stopping = True
                try:
                    stop = bool(self._stop(rep["stop?"]))
                except BaseException as e:  # noqa: BLE001  (the motion ends, then it raises here)
                    failed, stop = e, True
                finally:
                    self._stopping = False
                send(self.sock, {"stop": stop})
        except (OSError, EOFError) as e:
            self.lost = repr(e)
            raise RuntimeError(f"the simulator is gone ({e!r}); see {self.out / 'sim.log'}") from None
        except BaseException:  # interrupted mid-exchange
            self.lost = "interrupted mid-request"
            raise
        if failed is not None:
            raise failed
        if "error" in rep:
            raise RuntimeError(f"simulator: {rep['error']}")
        self.ended = rep["ended"]
        for name, T in rep["poses"].items():  # in place: the skills may hold the arrays
            if hasattr(self, "arms"):
                self.arms[name].T_world_base[...] = T
        return rep

    def obs(self, cams=True):
        rep = self._call(op="obs", cams=cams)
        if cams:
            self._masks = rep.get("masks", {})
        return rep["obs"]

    def reset(self, seed):
        """Reset an adapter-owned episode while preserving static host models and the server process."""
        facts = self._call(op="reset", seed=int(seed))["facts"]
        for name, arm in facts["arms"].items():
            self.arms[name].home[...] = arm.home
            self.arms[name].T_world_base[...] = arm.T_world_base
        self._masks, self._stop = {}, None

    def _robot_mask(self, cam, obs):
        return self._masks.get(cam) if obs.cams else None

    def follow(self, path, stop=None):
        self._stop = stop
        return self._call(op="follow", path=path, stop=stop is not None)["follow"]

    def finish(self, out):
        result = self._call(op="finish")["result"]
        if Path(out).resolve() != self.out / "finish":
            shutil.copytree(self.out / "finish", out, dirs_exist_ok=True)
        return result

    def _method(self, name, *args):
        return self._call(op="call", name=name, args=list(args))["result"]

    def close(self):
        try:
            self._call(op="close")
        except RuntimeError:
            pass
        self.sock.close()
        self.server.close()


# ---- the server


FILES = ("file", "fileright", "fileleft", "fileup", "filedown", "filefront", "fileback")  # an MJCF element's files


def portable(src, dest, copy=True):
    """Write the MJCF or URDF `src` (a path, or a yourdfpy.URDF with its edits) to `dest`, its includes inlined and
    every file it names (meshes, textures, models) named by absolute path, and with `copy` copied into `assets/` beside
    it first (for a host that sees only `dest`'s directory); returns `dest` as a string."""
    dest = Path(dest).resolve()
    assets = dest.parent / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    resolve = getattr(src, "_filename_handler", None)  # a yourdfpy.URDF resolves its meshes this way
    if resolve is not None:
        src.write_xml_file(dest)
        src = dest
    tree = ET.parse(src)
    root, base = tree.getroot(), Path(src).resolve().parent

    def place(path, model=False):
        path = Path(path).resolve()
        target = assets / f"{hashlib.sha1(str(path).encode()).hexdigest()[:10]}-{path.name}"
        if model:
            return portable(path, target, copy)
        if not copy:
            return str(path)
        if not target.exists():
            shutil.copyfile(path, target)
        return str(target)

    if root.tag == "mujoco":
        _inline(root, base)
        dirs = {}  # the compiler's directories hold for every file, included ones' too
        for compiler in root.iter("compiler"):
            dirs.update(compiler.attrib)
        for el in root.iter():
            kind = "meshdir" if el.tag in ("mesh", "skin") else "texturedir" if el.tag in ("texture", "hfield") else ""
            for key in FILES:
                name = el.get(key)
                if name is None:
                    continue
                path = Path(name) if os.path.isabs(name) else base / dirs.get(kind, dirs.get("assetdir", "")) / name
                if kind and key == "file" and el.get("name") is None:  # named after its file, which the copy renames
                    el.set("name", Path(name).stem)
                el.set(key, place(path, model=el.tag == "model"))
        for compiler in root.iter("compiler"):
            for key in ("meshdir", "texturedir", "assetdir"):
                compiler.attrib.pop(key, None)
    else:
        for el in root.iter("mesh"):
            name = el.get("filename")
            path = os.path.abspath(resolve(name)) if resolve else name.removeprefix("file://")
            el.set("filename", place(path if os.path.isabs(path) else base / path))
    tree.write(dest)
    return str(dest)


def _inline(parent, base):
    """Replace each <include> under `parent` by its file's elements, as MuJoCo does (its path relative to `base`, the
    main file's directory)."""
    i = 0
    while i < len(parent):
        if parent[i].tag == "include":
            name = parent[i].get("file")
            parent[i : i + 1] = list(ET.parse(name if os.path.isabs(name) else base / name).getroot())
        else:
            _inline(parent[i], base)
            i += 1


def facts(robot, out, export):
    """The Robot's facts; a model that is a file (always, when exported) or a loaded URDF goes as a portable file, one
    per model (the host keys models by it)."""
    arms, models = {}, {}
    for name, arm in robot.arms.items():
        if export or not isinstance(arm.model, (str, Path)):
            key = str(Path(arm.model).resolve()) if isinstance(arm.model, (str, Path)) else id(arm.model)
            if key not in models:
                models[key] = portable(arm.model, out / "model" / name / "robot.xml", copy=export)
            arm = dataclasses.replace(arm, model=models[key])
        arms[name] = arm
    return dict(arms=arms, dt=robot.dt, base=getattr(robot, "base", None), body=getattr(robot, "body", None),
                table_z=getattr(robot, "table_z", None), masks=hasattr(robot, "robot_mask"),
                remote=list(getattr(robot, "REMOTE", ())))  # fmt: skip


def handle(robot, req, conn, out, export):
    op = req["op"]
    if op == "reset":
        robot.reset(req["seed"])
        return {"facts": facts(robot, out, export)}
    if op == "facts":
        return facts(robot, out, export)
    if op == "obs":
        obs = robot.obs(cams=req["cams"])
        rep = {"obs": obs}
        if req["cams"] and hasattr(robot, "robot_mask"):
            rep["masks"] = {cam: robot.robot_mask(cam, obs) for cam in obs.cams}
        return rep
    if op == "follow":

        def stop(obs):
            send(conn, {"stop?": obs})
            return recv(conn)["stop"]

        return {"follow": robot.follow(req["path"], stop if req["stop"] else None)}
    if op == "finish":
        (out / "finish").mkdir(parents=True, exist_ok=True)
        return {"result": robot.finish(out / "finish")}
    if op == "call":
        if req["name"] not in getattr(robot, "REMOTE", ()):
            raise PermissionError(f"{req['name']!r} is not one of this adapter's REMOTE methods")
        return {"result": getattr(robot, req["name"])(*req["args"])}
    if op == "close":
        return {}
    raise ValueError(f"unknown op {op!r}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("module", help="the adapter's launch module, whose episode(**kw) builds the Robot")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--out", type=Path, required=True, help="the episode's directory, shared with the host")
    p.add_argument("--export", action="store_true", help="copy robot models into --out (the host cannot see ours)")
    argv = sys.argv[1:]
    cut = argv.index("--") if "--" in argv else len(argv)
    a = p.parse_args(argv[:cut])
    kw = dict(x.split("=", 1) for x in argv[cut + 1 :])
    lst = socket.create_server(("127.0.0.1", a.port))  # bound first: a taken port fails fast, before the episode
    robot = importlib.import_module(a.module).episode(**kw)
    print(f"\nready {a.module} port {a.port}", flush=True)
    conn, _ = lst.accept()
    lst.close()
    while True:
        try:
            req = recv(conn)
        except (EOFError, OSError):
            break
        try:
            rep = handle(robot, req, conn, a.out, a.export)
        except Exception as e:  # noqa: BLE001  (the reply carries it)
            traceback.print_exc()
            rep = {"error": f"{type(e).__name__}: {e}"}
        poses = {name: arm.T_world_base for name, arm in robot.arms.items()}
        rep.update(ended=bool(getattr(robot, "ended", False)), poses=poses)
        try:
            send(conn, rep)
        except (
            TypeError,
            ValueError,
        ) as e:  # a reply the wire cannot carry (sends nothing): this call fails, not the episode
            traceback.print_exc()
            send(conn, {"error": f"{type(e).__name__}: {e}", "ended": rep["ended"], "poses": poses})
        if req.get("op") == "close":
            break
    if hasattr(robot, "close"):
        robot.close()


if __name__ == "__main__":
    main()
