"""Where a simulator server runs, for every benchmark: natively (a process of this host, or of the image when this
process is already inside it, as a Kubernetes Pod), in a Docker container of its image, or in an Apptainer sandbox of
the same image; on one GPU, as the caller (or root in a Docker container whose image needs it).

    server = Server(pick(runtime, image, "LB_RUNTIME"), image, gpu, env={...}, mounts=[(src, dst, "ro"), ...])
    server.start([python, "-m", ...], log)          # the simulator's process; its output goes to `log`
    server.ready(log, "\\nready ", boot_s=600)      # wait for its marker, or close and raise
    ...
    server.close()                                  # stop it, remove its container or scratch

Images are pulled, never built here: docker/<name>/Dockerfile, pushed to GHCR as ghcr.io/openrobokit/roboskills-<name>
(docker/README.md). A container sees the host only through `mounts` (a Docker source that is not a path is a named
volume) and shares the host's network, so a server listens on 127.0.0.1. Docker needs NVIDIA's container toolkit.
Apptainer runs a sandbox directory of the image (`sandbox`: built from the registry once per host) read-only behind a
private writable layer, only `gpu` visible through NVIDIA's container CLI, with `/tmp` and each `scratch` path private
to this server.

A model or tool's image (SAM 3, MoGe, Kimodo, ...) runs one command in a fresh container (`contained`), from Python or,
the working directory mounted, from a shell:

    python -m roboskills.eval.runtime [--mount SRC[:DST[:ro]]]... [--env NAME]... IMAGE COMMAND...
"""

import argparse
import atexit
import contextlib
import csv
import getpass
import glob
import io
import os
import secrets
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path


RUNTIMES = ("native", "docker", "apptainer")


def pick(runtime, image, var, inside=None):
    """`runtime`, else $`var`, else native when `inside` (a path only the image has) exists here, else docker when
    docker has `image` or pulls it, else apptainer."""
    runtime = runtime or os.environ.get(var) or ("native" if inside and Path(inside).exists() else None)
    if not runtime:
        runtime = "docker" if docker_has(image) else "apptainer"
    if runtime not in RUNTIMES:
        raise ValueError(f"runtime is one of {RUNTIMES}, not {runtime!r}")
    return runtime


def docker_has(image):
    """Whether docker runs here with `image`, pulling it from its registry when missing (False: no docker, no access
    to the registry, or no room for it)."""
    if not shutil.which("docker"):
        return False

    def docker(*args):
        return subprocess.run(["docker", *args], capture_output=True, check=False).returncode == 0

    if docker("image", "inspect", image):
        return True
    if not docker("info"):  # a daemon this user cannot reach
        return False
    print(f"pulling {image} (once per host)", file=sys.stderr, flush=True)
    return docker("pull", image)


def contained(image, argv, mounts=(), *, name=None, workdir=None, env=()):
    """The command running `argv` in a fresh container of `image` (one model or tool, docker/<name>) where docker runs,
    the image pulled once; None where it cannot. Its stdin and stdout are the caller's; it runs as the caller (uid, gid,
    $USER, a fresh home: the build leaves root's caches in the image's) on the host's network, sees the GPUs of $CUDA_VISIBLE_DEVICES (else the first) and the host only through
    `mounts`: paths (read-write, at the same place; outside /tmp, which can be a session's own and invisible to the
    docker daemon, as on some hosts) or (source, destination, "ro"|"rw"). `name`: the container's, for
    `docker rm -f` after a killed client; `env`: names of this process's variables to pass in (those set). Under
    $ROBOSKILLS_RUN the container carries its label, for `remove_containers`.
    On Linux, preserve the caller's CPU affinity through Docker's daemon boundary."""
    if not docker_has(image):
        return None
    gpus = os.environ.get("CUDA_VISIBLE_DEVICES") or "0"
    cmd = ["docker", "run", "--rm", "-i", "--init", "--network", "host", "--shm-size", "8g", "--gpus", f'"device={gpus}"',
           "-u", f"{os.getuid()}:{os.getgid()}", "-e", f"USER={getpass.getuser()}", "-e", "HOME=/tmp/home",
           "-e", "NVIDIA_DRIVER_CAPABILITIES=all"]  # fmt: skip
    if hasattr(os, "sched_getaffinity"):
        cmd += ["--cpuset-cpus", ",".join(map(str, sorted(os.sched_getaffinity(0))))]
    cmd += _mounts((m, m, "rw") if isinstance(m, (str, Path)) else m for m in mounts)
    cmd += [x for k in env if k in os.environ for x in ("-e", k)]
    cmd += (["--name", name] if name else []) + (["-w", str(workdir)] if workdir else [])
    cmd += ["--label", f"roboskills.run={os.environ['ROBOSKILLS_RUN']}"] if os.environ.get("ROBOSKILLS_RUN") else []
    return cmd + ["--entrypoint", str(argv[0]), image, *map(str, argv[1:])]


def remove_containers(run):
    """Remove the containers `contained` started under $ROBOSKILLS_RUN=`run`: a killed client leaves its own running."""
    if shutil.which("docker"):
        ps = ["docker", "ps", "-aq", "--filter", f"label=roboskills.run={run}"]
        if ids := subprocess.run(ps, capture_output=True, text=True, check=False).stdout.split():
            subprocess.run(["docker", "rm", "-f", *ids], capture_output=True, check=False)


def _mounts(mounts):
    """docker's --mount arguments for (source, destination, "ro"|"rw"): a source that is not a path is a named volume."""
    args = []
    for source, dest, mode in mounts:  # --mount: some dockerds mangle `-v path:path:ro`
        kind = "bind" if str(source).startswith("/") else "volume"
        spec = io.StringIO()  # CSV: a path with a comma goes quoted
        csv.writer(spec, lineterminator="").writerow(
            [f"type={kind}", f"source={source}", f"target={dest}"] + (["readonly"] if mode == "ro" else [])
        )
        args += ["--mount", spec.getvalue()]
    return args


def sandbox(image):
    """The Apptainer sandbox of `image`: a directory as it is, else ~/.roboskills/images/<name>-<tag>, built from the
    registry (docker://`image`) when missing. A private registry takes $APPTAINER_DOCKER_USERNAME and _PASSWORD."""
    if str(image).startswith("/"):
        return Path(image)
    path = Path.home() / ".roboskills/images" / str(image).rsplit("/", 1)[-1].split("@")[0].replace(":", "-")
    if not path.is_dir():
        print(f"building {path} from docker://{image} (once per host)", file=sys.stderr, flush=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["apptainer", "build", "--userns", "--sandbox", "--fix-perms", str(path), f"docker://{image}"],
                       check=True)  # fmt: skip
    return path


class Server:
    def __init__(self, runtime, image, gpu, *, name=None, env=None, mounts=(), scratch=(), home=False, workdir=None,
                 root=False, docker_args=""):  # fmt: skip
        """`image`: the docker image (Apptainer: its sandbox, or a sandbox directory); `env`: the simulator's environment
        (natively, on top of this process's); `mounts`: (source, dest, "ro"|"rw");
        `scratch`: container paths given a private writable directory (Apptainer); `home`: a home of its own too
        (Apptainer: True, a private one; a path, that directory); `workdir`: where it runs; `root`: as root (Docker)."""
        self.runtime, self.gpu, self.proc, self.container, self.workdir = runtime, str(gpu), None, None, workdir
        atexit.register(self.close)  # an exiting host leaves no container or scratch behind (a killed one may)
        self.name = name or f"sk-{os.getuid()}-{secrets.token_hex(4)}"
        self.scratch = Path(tempfile.gettempdir()) / f"sk-{os.getuid()}" / self.name
        env = dict(env or {})
        if runtime == "native":
            self.prefix, self.env = [], dict(os.environ, **env, CUDA_VISIBLE_DEVICES=self.gpu)
        elif runtime == "docker":
            self.prefix, self.env = self._docker(image, env, mounts, docker_args, root), None
        else:
            self.prefix, self.env = self._apptainer(image, env, mounts, scratch, home)

    def _docker(self, image, env, mounts, docker_args, root):
        cmd = ["docker", "run", "-d", "--rm", "--init", "--name", self.name, "--gpus", f"device={self.gpu}",
               "--network", "host", "--ipc", "private", "--shm-size", "8g", "--entrypoint", "/bin/bash"]  # fmt: skip
        for k, v in dict(env, NVIDIA_DRIVER_CAPABILITIES="all").items():
            cmd += ["-e", f"{k}={v}"]
        # Vulkan needs NVIDIA's compiler library; old toolkits omit it
        gpucomp = [(lib, lib, "ro") for lib in glob.glob("/usr/lib/x86_64-linux-gnu/libnvidia-gpucomp.so.*")]
        cmd += _mounts(list(mounts) + gpucomp)
        started = subprocess.run(cmd + shlex.split(docker_args) + [image, "-c", "sleep infinity"], capture_output=True,
                                 text=True, check=False)  # fmt: skip
        if started.returncode:
            raise RuntimeError(f"docker could not start {image}: {started.stderr.strip()}")
        self.container = self.name
        user = [] if root else ["-u", f"{os.getuid()}:{os.getgid()}"]
        return ["docker", "exec", *user, *(["-w", self.workdir] if self.workdir else []), self.name]

    def _apptainer(self, image, env, mounts, scratch, home):
        """NVIDIA's container CLI has no NGX capability, so the host's NGX library is bound in (Isaac Sim's DLAA needs it)."""
        uuid = subprocess.check_output(
            ["nvidia-smi", "-i", self.gpu, "--query-gpu=uuid", "--format=csv,noheader"], text=True
        ).strip()
        binds = list(mounts) + [(lib, lib, "ro") for lib in glob.glob("/usr/lib/x86_64-linux-gnu/libnvidia-ngx.so.*")]
        for dest in ("/tmp", *scratch):  # --contain's own /tmp is the session directory, 16 MiB on some hosts
            source = self.scratch_dir(dest)
            source.mkdir(parents=True, exist_ok=True)
            binds.append((source, dest, "rw"))
        cmd = ["apptainer", "exec", "--userns", "--contain", "--cleanenv", "--nvccli", "--writable-tmpfs"]
        if home:
            home = self.scratch / "home" if home is True else Path(home)
            home.mkdir(parents=True, exist_ok=True)
            cmd += ["--home", f"{home}:{Path.home()}"]
        if self.workdir:
            cmd += ["--pwd", str(self.workdir)]
        for k, v in env.items():
            cmd += ["--env", f"{k}={v}"]
        for source, dest, mode in binds:
            cmd += ["--bind", f"{source}:{dest}:{mode}"]
        host = dict(os.environ, NVIDIA_VISIBLE_DEVICES=uuid,
                    NVIDIA_DRIVER_CAPABILITIES="compute,utility,graphics,video,display")  # fmt: skip
        return cmd + [str(sandbox(image))], host

    def scratch_dir(self, dest):
        """The host directory behind the container path `dest` of `scratch` (Apptainer)."""
        return self.scratch / dest.strip("/").replace("/", "-")

    def root(self, *cmd):
        """Run `cmd` as root in the Docker container (to adapt the image to the caller, in this container only)."""
        subprocess.run(["docker", "exec", "-u", "0", self.name, *cmd], check=True, capture_output=True)

    def start(self, cmd, log):
        """Start the simulator's process `cmd` in its runtime, its output in `log`; returns the process."""
        cwd = self.workdir if self.runtime == "native" else None
        self.proc = subprocess.Popen(self.prefix + [str(c) for c in cmd], stdout=open(log, "w"), stderr=subprocess.STDOUT,
                                     env=self.env, cwd=cwd, start_new_session=True)  # fmt: skip
        return self.proc

    def ready(self, log, marker, boot_s):
        """Wait until `marker` shows in `log` (at a line start if it begins with a newline); close and raise if the
        process ends or `boot_s` passes first."""
        end = time.monotonic() + float(boot_s)
        while marker not in "\n" + Path(log).read_text(errors="replace"):
            if self.proc.poll() is not None or time.monotonic() > end:
                self.close()
                raise RuntimeError(f"the simulator did not start ({self.runtime}); see {log}")
            time.sleep(1)

    def close(self):
        """Stop the simulator and remove its container or scratch (only ours)."""
        proc, self.proc = self.proc, None
        if self.container:
            if proc is not None and proc.poll() is None:
                proc.terminate()
            subprocess.run(["docker", "rm", "-f", self.container], capture_output=True, check=False)
            self.container = None
        elif proc is not None:  # native or apptainer (whose own process can exit before the simulator's): the group
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGTERM)
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=30)
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
        if self.scratch.exists():
            for root, dirs, _ in os.walk(self.scratch):  # Kit makes screenshots/ mode 000, which rmtree cannot empty
                for d in dirs:
                    with contextlib.suppress(OSError):
                        os.chmod(os.path.join(root, d), 0o700)
            shutil.rmtree(self.scratch, ignore_errors=True)


def main():
    """One command in a fresh container of a model or tool's image (`contained`), the working directory mounted and kept:
    what an agent's setup files call it through."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--mount", action="append", default=[], help="SRC[:DST[:ro]] (DST defaults to SRC)")
    parser.add_argument("--env", action="append", default=[], help="a variable of this shell to pass in, when set")
    parser.add_argument("image")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    mounts = [(Path.cwd(), Path.cwd(), "rw")]
    for m in args.mount:
        source, dest, mode = (m.split(":") + [None, None])[:3]
        mounts.append((Path(source).expanduser().resolve(), dest or Path(source).expanduser().resolve(), mode or "rw"))
    if not args.command or not (cmd := contained(args.image, args.command, mounts, workdir=Path.cwd(), env=args.env)):
        sys.exit(f"docker cannot run {args.image} here" if args.command else "no command")
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    main()
