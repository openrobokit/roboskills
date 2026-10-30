"""RGB + joints -> what the skills need from a camera (depth, intrinsics, pose), for RoboDojo's official observation.

Intrinsics and poses: a self-calibration (`cams.json`, robot facts measured once offline from dev-episode RGB and
joints with the robot as the target: examples/robodojo_claude/rgbpose/calib.py): the head camera fixed in the world,
each wrist camera fixed on its arm's link6 and posed by forward kinematics. Square pixels, principal point at the
centre. Depth: a monocular model (MoGe-2, `depth_worker.py`) gives each image's depth up to an affine map
(d = a * d_model + b), fitted per image to the table: the horizontal plane the arms are mounted on (their base
height). The robot's own pixels get its model's depth (rendered at the measured joints).
"""

import json
import subprocess
import tempfile
import threading
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
CAMS_JSON = HERE / "cams.json"
SIDES = ("left", "right")


class RobotModel:
    """One arm's URDF stood at each side's base (`T_base`: {side: (4, 4)}): surface points of every link (area-weighted
    samples of its visual meshes) and forward kinematics."""

    def __init__(self, urdf, T_base, n=120000, seed=0):
        import trimesh
        import yourdfpy

        self.u = yourdfpy.URDF.load(urdf, load_meshes=False, build_collision_scene_graph=False)
        meshes = {}
        for name, link in self.u.link_map.items():
            for v in link.visuals:
                if v.geometry is not None and v.geometry.mesh is not None:
                    m = trimesh.load(Path(urdf).parent / v.geometry.mesh.filename, force="mesh")  # relative to the URDF
                    if v.origin is not None:
                        m.apply_transform(v.origin)
                    meshes.setdefault(name, []).append(m)
        per = n / sum(m.area for ms in meshes.values() for m in ms)
        rng = np.random.default_rng(seed)
        self.pts = {k: np.concatenate([trimesh.sample.sample_surface(m, max(20, int(m.area * per)), seed=int(rng.integers(1 << 30)))[0]
                                       for m in ms]) for k, ms in meshes.items()}  # fmt: skip
        self.T_base, self.cfg = {s: np.asarray(T, float) for s, T in T_base.items()}, {}

    def set(self, state):
        """Pose the arms at a state dict ({side}_arm_joint_state (6,), {side}_gripper_joint_state (finger travel, m))."""
        for s in SIDES:
            q, g = (
                np.asarray(state[f"{s}_arm_joint_state"], float),
                np.asarray(state[f"{s}_gripper_joint_state"], float),
            )
            self.cfg[s] = {f"joint{i + 1}": float(q[i]) for i in range(6)} | {
                "joint7": float(g[0]),
                "joint8": float(g[0]),
            }

    def T(self, side, link):
        """World pose of `link` of the arm on `side`, as set."""
        self.u.update_cfg(self.cfg[side])
        return self.T_base[side] @ self.u.get_transform(link, self.u.base_link)

    def points(self, skip=("base_link",)):
        out = []
        for s in SIDES:
            self.u.update_cfg(self.cfg[s])
            for k, P in self.pts.items():
                if k not in skip:
                    T = self.T_base[s] @ self.u.get_transform(k, self.u.base_link)
                    out.append(P @ T[:3, :3].T + T[:3, 3])
        return np.concatenate(out)


class MonoDepth:
    """MoGe-2 in a worker process, loaded once: `uv run` of `depth_worker.py` (its own dependencies; the weights
    download once into the Hugging Face cache) on $CUDA_VISIBLE_DEVICES's first GPU. `shared()`: one per process,
    which the envs of a vector sim take turns on, rather than a model per env on the same GPU."""

    _shared, _making = None, threading.Lock()

    @classmethod
    def shared(cls):
        with cls._making:
            if cls._shared is None:
                cls._shared = cls()
            return cls._shared

    def __init__(self):
        self.lock, self.dir = threading.Lock(), tempfile.TemporaryDirectory(prefix="depth-")  # requests and replies
        self.p = subprocess.Popen(["uv", "run", "--quiet", str(HERE / "depth_worker.py")], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, text=True, bufsize=1)  # fmt: skip
        self.runtime = self._reply()

    def _reply(self):
        for line in self.p.stdout:
            if line.startswith("DEPTH "):
                r = json.loads(line[6:])
                if not r.get("ok"):
                    raise RuntimeError(f"depth worker: {r.get('error')}")
                return r
        raise RuntimeError("the depth worker exited (see the server log)")

    def __call__(self, rgbs, fovs):
        with self.lock:
            d = self.dir.name
            np.savez(Path(d) / "in.npz", **{f"rgb_{i}": np.asarray(x, np.uint8) for i, x in enumerate(rgbs)})
            self.p.stdin.write(
                json.dumps(dict(images=str(Path(d) / "in.npz"), fov_x=list(fovs), out=str(Path(d) / "out.npz"))) + "\n"
            )
            self.p.stdin.flush()
            self._reply()
            got = np.load(Path(d) / "out.npz")
            return [got[f"depth_{i}"] for i in range(len(rgbs))]


def intrinsics(f, hw):
    return np.array([[f, 0, (hw[1] - 1) / 2], [0, f, (hw[0] - 1) / 2], [0, 0, 1.0]])


def zbuffer(P, K, T, hw, near=0.02):
    """Nearest depth per pixel of world points P seen by camera (K, T); NaN where none land."""
    Pc = (P - T[:3, 3]) @ T[:3, :3]
    Pc = Pc[Pc[:, 2] > near]
    u = np.round(K[0, 0] * Pc[:, 0] / Pc[:, 2] + K[0, 2]).astype(int)
    v = np.round(K[1, 1] * Pc[:, 1] / Pc[:, 2] + K[1, 2]).astype(int)
    ok = (u >= 0) & (u < hw[1]) & (v >= 0) & (v < hw[0])
    z = np.full(hw[0] * hw[1], np.inf)
    np.minimum.at(z, v[ok] * hw[1] + u[ok], Pc[ok, 2])
    z = z.reshape(hw)
    return np.where(np.isfinite(z), z, np.nan)


def ray_z(K, T, hw):
    """World z component of each pixel's ray per unit of camera depth."""
    v, u = np.mgrid[: hw[0], : hw[1]]
    d = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones(hw)], -1)
    return d @ T[2, :3]


def fit_affine(dm, rz, tz, h, iters=4, max_depth=3.0, min_table=20000):
    """Fit depth = a * dm + b so that the table, the horizontal plane z = h, lies where it should: a table pixel's depth
    is (h - tz) / rz. Table pixels are re-selected each round as those the current fit puts near the plane (objects
    and the robot drop out). Returns (a, b, n_table) or None if too little table is in view (under `min_table`
    pixels, where another plane, a cabinet's top or front, can pass for it)."""
    ok = np.isfinite(dm) & (dm > 0) & (rz < -0.05)  # rays that can reach a table below the camera
    dp = np.full(dm.shape, np.nan)
    dp[ok] = (h - tz) / rz[ok]
    ok &= dp < max_depth
    if ok.sum() < 500:
        return None
    a, b, tol, n = float(np.median(dp[ok] / dm[ok])), 0.0, 0.03, 0
    for _ in range(iters):
        z = tz + (a * dm + b) * rz
        sel = np.flatnonzero((ok & (np.abs(z - h) < tol)).ravel())
        if len(sel) < 200:
            return None
        n, sel = len(sel), sel[:: max(1, len(sel) // 20000)]
        x, y = dm.ravel()[sel], dp.ravel()[sel]
        if np.percentile(y, 95) > 1.5 * np.percentile(y, 5):  # the table spans a range of depths: fit a and b
            a, b = np.linalg.lstsq(np.c_[x, np.ones(len(sel))], y, rcond=None)[0]
        else:  # seen head-on it does not (b would absorb everything): a scale alone
            a, b = float(np.median(y / x)), 0.0
        tol = max(0.008, tol / 2)
    return (float(a), float(b), n) if n >= min_table else None


def ramps(d, rel=0.04, k=5):
    """Pixels on a depth ramp: the depth varies by more than `rel` of itself within k x k (object outlines, where a
    monocular model interpolates between the object and what is behind it)."""
    import cv2

    z = np.nan_to_num(d, nan=0.0).astype(np.float32)
    ker = np.ones((k, k), np.uint8)
    return (cv2.dilate(z, ker) - cv2.erode(z, ker)) > rel * np.maximum(z, 1e-3)


def robot_depth(P, K, T, hw):
    """The robot's own depth, rendered from its model points: holes between points filled, NaN elsewhere."""
    import cv2

    z = zbuffer(P, K, T, hw)
    zf = np.where(np.isfinite(z), z, 1e3).astype(np.float32)
    zf = cv2.erode(zf, np.ones((5, 5), np.uint8))  # nearest of the neighbourhood fills the holes
    m = cv2.morphologyEx(np.isfinite(z).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)).astype(bool)
    return np.where(m & (zf < 1e3), zf, np.nan)


class RGBD:
    """Per observation: {cam: (depth, K, T_world_cam)} from the cameras' RGB and the robot state. `urdf`: one arm's,
    stood at `T_base` ({side: (4, 4)}); `table_z`: the table top's height, the plane the monocular depth is scaled on."""

    def __init__(self, urdf, T_base, table_z, cams=CAMS_JSON, mono=None):
        self.cal = json.loads(Path(cams).read_text())
        self.model = RobotModel(urdf, T_base)
        self.mono = mono or MonoDepth.shared()
        self.h = float(table_z)
        self.log = []

    def camera(self, name, hw):
        c = self.cal[name]
        T = np.asarray(c["T"], float)
        if c["parent"] != "world":  # "<side>_<link>" in cams.json
            T = self.model.T(*c["parent"].split("_", 1)) @ T
        return intrinsics(c["f"], hw), T

    def __call__(self, rgbs, state):
        self.model.set(state)
        names = [n for n in rgbs if n in self.cal]
        geo = {n: self.camera(n, rgbs[n].shape[:2]) for n in names}
        fovs = [float(np.degrees(2 * np.arctan(rgbs[n].shape[1] / 2 / geo[n][0][0, 0]))) for n in names]
        dms = dict(zip(names, self.mono([rgbs[n] for n in names], fovs)))
        P = self.model.points()
        out, info = {}, {}
        for n in names:
            (K, T), hw, dm = geo[n], rgbs[n].shape[:2], dms[n]
            fit = fit_affine(dm, ray_z(K, T, hw), T[2, 3], self.h)
            if fit is None:
                info[n] = "no table in view"
                continue
            a, b, nt = fit
            d = (a * dm + b).astype(np.float32)
            d[ramps(d)] = np.nan  # the model blends depth across an object's outline: no points there
            zr = robot_depth(P, K, T, hw)
            d = np.where(np.isfinite(zr), zr, d).astype(np.float32)  # the robot where its model says
            d[~np.isfinite(d) | (d <= 0)] = np.nan
            out[n] = (d, K, T)
            info[n] = dict(a=round(a, 4), b=round(b, 4), table_px=nt)
        self.log.append(info)
        return out
