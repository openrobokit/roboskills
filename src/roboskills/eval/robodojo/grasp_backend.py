"""GraspKit's learned ARX X5 grasp policy as the backend of `action.grasp_poses`
(`GRASP_BACKEND=roboskills.eval.robodojo.grasp_backend`), with no heuristic fallback.

The policy runs in its own process and venv (GraspKit's `graspkit.policies.arx_x5_server`), started here on first use
and reached over a unix socket. Its grasps (world TCP, pre-grasp, jaw commands) are kept when the hand travels within the approach cone and the
open hand's straight way in from the pre-grasp clears the table, the obstacles, the avoid points and the object (the
robot's own hand model, `RobotModel.hand_points`), in the policy's order, near-duplicates dropped. Reachability is
the planner's own check on the returned list.

Env: GRASP_POLICY_PYTHON (the python of a venv with GraspKit 0.1.0 or later), GRASP_POLICY_CKPT and
GRASP_POLICY_RANKER (local generator and ranker checkpoints; default: the published ones, fetched from GraspKit's HF dataset by
`graspkit.assets.policies.skill_policy_paths`), GRASP_POLICY_RANK (GraspKit's ordering; default ranker_reach),
GRASP_POLICY_N (samples per call, default 128), GRASP_POLICY_ARGS (more server flags, e.g. `--threads 2`),
GRASP_POLICY_SOCKET (an already running server shared by several episodes; none is started), GRASP_FILTER_COLLISION=0
(skip the swept-hand checks). Every call appends a line to `<run>/grasp_backend.jsonl`.
"""

import atexit
import json
import os
import shlex
import socket
import subprocess
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from roboskills.tools.robokit.action import Pose


_SERVER = {}


# ---- the policy process


def _server_cmd(sock):
    env = os.environ
    cmd = [env["GRASP_POLICY_PYTHON"], "-m", "graspkit.policies.arx_x5_server", "--socket", sock, "--exit-with-parent"]
    for flag, key in (
        ("--ckpt", "GRASP_POLICY_CKPT"),
        ("--rank", "GRASP_POLICY_RANK"),
        ("--ranker", "GRASP_POLICY_RANKER"),
    ):
        if env.get(key):
            cmd += [flag, env[key]]
    return cmd + shlex.split(env.get("GRASP_POLICY_ARGS", ""))


def server(log_dir=None, timeout=300.0):
    """The policy server's socket path: GRASP_POLICY_SOCKET, else the one started in this process, else one started now."""
    if os.environ.get("GRASP_POLICY_SOCKET"):
        return os.environ["GRASP_POLICY_SOCKET"]
    p = _SERVER.get("proc")
    if p is not None and p.poll() is None:
        return _SERVER["sock"]
    tmp = Path(f"/tmp/sk-{os.getuid()}")
    tmp.mkdir(exist_ok=True)
    sock = str(tmp / f"grasp_policy_{os.getpid()}.sock")
    log = Path(log_dir or tmp) / "grasp_server.log"
    cmd = _server_cmd(sock)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME"))}
    with open(log, "a") as f:  # (the child keeps its own handle)
        f.write(f"$ {shlex.join(cmd)}\n")
        f.flush()
        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=str(Path(cmd[0]).parents[2]))
    _SERVER.update(proc=p, sock=sock)
    atexit.register(lambda: p.poll() is None and p.terminate())
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if p.poll() is not None:
            raise RuntimeError(f"the grasp policy server exited ({p.returncode}); see {log}")
        if Path(sock).exists():
            try:
                _call(sock, dict(op="info"))
                _SERVER["start_s"] = round(time.monotonic() - t0, 1)
                return sock
            except OSError:
                pass
        time.sleep(0.5)
    p.terminate()
    raise RuntimeError(f"the grasp policy server did not come up in {timeout:.0f} s; see {log}")


def _call(sock, req, timeout=120.0):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        s.connect(sock)
        with s.makefile("rw") as f:
            f.write(json.dumps(req) + "\n")
            f.flush()
            out = json.loads(f.readline())
    if not out.get("ok"):
        raise RuntimeError(f"grasp policy server: {out.get('error')}")
    return out


def propose(P, tz, scene, n, seed, log_dir=None):
    """The policy's grasps on object points P: world TCP poses (n, 3, 4, 4) of pre-grasp / grasp / hold, opening
    fractions (n, 3), scores (n,) or None, and the server's seconds."""
    req = dict(
        op="propose",
        object=np.asarray(P, float).tolist(),
        n=int(n),
        seed=int(seed),
        table_z=float(tz),
        scene=None if scene is None else np.asarray(scene, float).tolist(),
    )
    out = _call(server(log_dir), req)
    score = None if out["score"] is None else np.asarray(out["score"])
    return np.asarray(out["T"], float), np.asarray(out["u"], float), score, out["seconds"]


# ---- filters and ranking (the robot's own hand model: `RobotModel.hand_points`, TCP frame)


def _crop(X, c, r=0.3, voxel=0.005):
    if X is None or not len(X):
        return None
    X = np.asarray(X, float)
    X = X[np.linalg.norm(X - c, axis=1) < r]
    return X[np.unique(np.floor(X / voxel).astype(np.int64), axis=0, return_index=True)[1]] if len(X) else None


def swept(hand_pts, T, T_pre, steps=5):
    """World points (n, steps * M, 3) of the hand (TCP-frame points) on its straight way in, from the pre-grasp
    T_pre to the grasp T (n, 4, 4; the grasp's orientation throughout, as the policy's pre-grasp keeps it)."""
    W = np.einsum("nij,mj->nmi", T[:, :3, :3], hand_pts) + T[:, None, :3, 3]
    back = T_pre[:, None, :3, 3] - T[:, None, :3, 3]
    return np.concatenate([W + back * (1.0 - s) for s in np.linspace(0.0, 1.0, steps)], 1)


def _gap(W, X, cap=0.05):
    if X is None or not len(X):
        return np.full(len(W), cap)
    d, _ = cKDTree(np.asarray(X, float)).query(W.reshape(-1, 3), distance_upper_bound=cap, workers=8)
    return np.minimum(d, cap).reshape(W.shape[:2]).min(1)


def select(
    T,
    T_pre,
    score,
    P,
    tz,
    hand_pts,
    obstacles=None,
    avoid=None,
    approach=None,
    cone=30.0,
    keep=48,
    obj_tol=0.002,
    obst_tol=0.006,
    avoid_tol=0.01,
    table_tol=-0.002,
):
    """Indices of the grasps T (world TCP, n x 4 x 4; pre-grasps T_pre) that pass, best first (the policy's `score`,
    higher first; without one, the TCP nearest the object's middle), one per 1 cm / 20 deg, and the rejections."""
    ok, why = np.isfinite(T).all((1, 2)), {}
    x = T[:, :3, 0]
    if approach is not None:
        bad = x @ (np.asarray(approach, float) / np.linalg.norm(approach)) < np.cos(np.radians(cone))
    else:
        bad = x[:, 2] > np.cos(np.radians(80.0))  # not from under the table (100 deg around down)
    why["cone"] = int(bad.sum())
    ok &= ~bad
    P = np.asarray(P, float)
    idx = np.flatnonzero(ok)
    if len(idx) and os.environ.get("GRASP_FILTER_COLLISION", "1") != "0":
        W = swept(hand_pts, T[idx], T_pre[idx])
        for k, hit in (("table", W[..., 2].min(1) < tz + table_tol), ("obstacle", _gap(W, obstacles) < obst_tol),
                       ("avoid", _gap(W, avoid) < avoid_tol), ("object", _gap(W, P) < obj_tol)):  # fmt: skip
            why[k] = int((ok[idx] & hit).sum())
            ok[idx[hit]] = False
    key = -np.asarray(score, float) if score is not None else np.linalg.norm(T[:, :3, 3] - P.mean(0), axis=1)
    kept = []
    for i in np.argsort(key, kind="stable"):
        if ok[i] and all(np.linalg.norm(T[i, :3, 3] - T[j, :3, 3]) > 0.01 or T[i, :3, 0] @ T[j, :3, 0] < 0.94
                         or abs(T[i, :3, 1] @ T[j, :3, 1]) < 0.94 for j in kept):  # fmt: skip
            kept.append(i)
            if len(kept) >= keep:
                break
    why["passed"] = int(ok.sum())
    return np.array(kept, int), why


def _log(S, **kw):
    line = json.dumps(kw, default=lambda v: np.asarray(v).tolist())
    if getattr(S, "out", None) is not None:
        with open(Path(S.out) / "grasp_backend.jsonl", "a") as f:
            f.write(line + "\n")
    print("grasp_backend", line, flush=True)


def candidates(S, a, P, tz, obst, tgt, avoid_pts, ax, cn, seed, optimizer=False):
    """`action.grasp_poses`'s candidates under this backend: the policy's grasps for arm `a`, best first. `ax`, `cn`:
    the approach axis and cone (None: any way not from under the table); the policy has no approach input, so with
    one it samples 4x and keeps those in the cone. `optimizer` (a second pass, when no first-pass grasp was
    reachable): 4x the samples, another seed. `tgt` (a part to hold): grasps whose TCP is not within 2 cm of it are
    dropped."""
    h = S.robot.arms[a].hand
    if h.kind != "jaw":
        raise NotImplementedError("the openloop grasp backend covers parallel jaws only")
    t0 = time.monotonic()
    n = int(os.environ.get("GRASP_POLICY_N", "128")) * (4 if optimizer else 1) * (4 if ax is not None else 1)
    near = _crop(obst, P.mean(0))
    Tall, u, score, net_s = propose(P, tz, near, n, seed + (1000 if optimizer else 0), getattr(S, "out", None))
    T, T_pre = Tall[:, 1], Tall[:, 0]
    u_in = np.clip(np.ceil(u[:, 0].max() * 10) / 10, 0.1, 1.0)  # the hand on its way in, a little wider if anything
    keep, why = select(T, T_pre, score, P, tz, S.kin.hand_points(a, np.r_[u_in]), obstacles=near, avoid=avoid_pts,
                       approach=ax, cone=cn)  # fmt: skip
    if tgt is not None and len(tgt):  # on the part: its points between the pads
        tt = cKDTree(np.asarray(tgt, float))
        close = [i for i in keep if tt.query(T[i, :3, 3])[0] < 0.02]
        why["off_part"] = len(keep) - len(close)
        keep = close
    cands = [Pose("grasp", a, tcp=T[i], ee_pre=T_pre[i], hand_pre=u[i, 0:1], hand=u[i, 2:3], rank=k)
             for k, i in enumerate(keep)]  # fmt: skip
    _log(
        S,
        arm=a,
        second_pass=bool(optimizer),
        n_points=len(P),
        proposals=n,
        kept=len(cands),
        rejected=why,
        approach=None if ax is None else np.round(ax, 3),
        cone=cn,
        obstacles=0 if near is None else len(near),
        policy_s=net_s,
        seconds=round(time.monotonic() - t0, 3),
        server_start_s=_SERVER.pop("start_s", None),
    )
    return cands
