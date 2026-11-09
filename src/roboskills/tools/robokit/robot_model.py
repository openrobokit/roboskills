"""The arms' RoboKit models: joint vectors, forward kinematics, TCP poses and the robot's surface. Facts about the
robot only: nothing about the scene, nothing that moves it.

Frames: world, metres. Each arm's `model` is a RoboKit-loadable URDF/MJCF,
placed by `T_world_base`, read again at every use (a mobile base moves it); its TCP is `hand.T_ee_tcp` in its `ee_link` frame. `fq` is one model's full actuated-joint
vector (arms and hand joints). `info[arm]` says where the arm sits in its model: `M` (the loaded description, shared by
the arms it holds at the same base: `rk`, the RoboKit Robot with collision spheres; `names`, its actuated joints, the order of `fq`; `root`,
the `Arm` whose `T_world_base` places it; `desc`, what RoboKit loaded), `idx` (the arm's joints in `fq`, in command
order), `ee` (its `ee_link` among the model's links), `hidx` (its hand's joints in `fq`) and `hand_links` (`ee_link`
and every link hanging off it)."""

import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from roboskills.tools.robokit.geom import voxel_ids


def fit_spheres(mesh, slack=0.02, samples=3000, cap=None):
    """Spheres (centres (K, 3), radii (K,)) covering `mesh` (a trimesh, its link's frame) that reach about `slack` m
    past it at most: candidate centres inside the mesh (its filled voxels, `slack` / 2 apart) and on its surface, each
    with its distance to the surface plus `slack` as its radius, taken greedily until every surface sample is inside
    one. A thick part gets a few large spheres and a thin plate many small ones; a mesh that is not closed gets only
    surface-centred spheres of radius `slack`. With `cap`, at most `cap` spheres: the samples still uncovered then
    grow their nearest sphere until it reaches them."""
    import trimesh
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(0)
    S = trimesh.sample.sample_surface_even(mesh, samples, seed=0)[0]
    V = np.asarray(mesh.vertices)
    S = np.vstack([S, V[rng.choice(len(V), min(len(V), samples // 3), replace=False)]])
    inner = np.zeros((0, 3))
    if mesh.is_watertight:
        inner = mesh.voxelized(max(slack / 2, 0.002)).fill().points
        inner = inner[mesh.contains(inner)]  # a fill also closes concave pockets
    tree = cKDTree(S)
    cand = np.vstack([inner, S])
    radii = tree.query(cand)[0] + slack
    covers = tree.query_ball_point(cand, radii)
    left, C, R = np.ones(len(S), bool), [], []
    while left.any() and len(C) < (cap or len(cand)):
        k = int(np.argmax([left[c].sum() for c in covers]))
        C.append(cand[k])
        R.append(radii[k])
        left[covers[k]] = False
    C, R = np.array(C), np.array(R)
    if left.any():
        d, i = cKDTree(C).query(S[left])
        np.maximum.at(R, i, d)
    return C, R


def write_collision_spheres(rk, path, slack=0.02, visual=(), budget=16, spacing=0.035):
    """Collision spheres for a model without a sphere file, written to `path` (the YAML RoboKit loads as
    `collision_spheres_path`): `fit_spheres` on each link's collision mesh, or for the links in `visual` that have
    none, on its visual mesh (a hand's palm whose geom is set not to collide is still there). A link that needs more
    than `budget` spheres (a mesh that is not closed, a large thin part) is fitted again with 1.5 times the slack,
    up to 5 cm, where it is capped at `budget`: planning time grows with the sphere count. A link no joint moves
    (a pedestal, a controller box, a mobile base) keeps one circumscribing sphere per slice of its bounding box,
    `spacing` long (at most `budget` of them): fitting a robosuite pedestal took 2.5 minutes for nothing that moves.
    The bounding boxes the moving links had before reached up to 6 cm past a YAM fingertip: the planner then kept the
    TCP 2.5-3 cm over the table and no grasp reached a 2 cm object."""
    import yaml

    pj, moving = rk.spec.link_parent_joint_indices, set(rk.spec.nonfixed_joint_names)
    moves = np.array([j >= 0 and rk.spec.joint_names[j] in moving for j in pj])
    moves |= (np.asarray(rk.spec.link_ancestor_links_mask, bool) & moves[None]).any(1)  # or an ancestor link moves
    spheres = {}
    for k, name in enumerate(rk.link_names):
        scene = rk.spec.link_collision_geometries.get(name)
        if (scene is None or not scene.geometry) and name in visual:
            scene = rk.spec.link_visual_geometries.get(name)
        if scene is None or not scene.geometry:
            continue
        mesh, s = scene.to_mesh(), slack
        if not moves[k]:
            low, high = mesh.bounds
            axis = int(np.argmax(high - low))
            count = int(np.clip(np.ceil((high[axis] - low[axis]) / spacing), 1, budget))
            width = (high[axis] - low[axis]) / count
            C = np.repeat((low + high)[None] / 2, count, 0)
            C[:, axis] = low[axis] + (np.arange(count) + 0.5) * width
            R = np.full(count, max(float(np.sqrt(np.sum(np.delete(high - low, axis) ** 2) + width**2) / 2), 0.001))
        else:
            C, R = fit_spheres(mesh, s, cap=budget if s >= 0.05 else None)
            while len(R) > budget:
                s = min(s * 1.5, 0.05)
                C, R = fit_spheres(mesh, s, cap=budget if s >= 0.05 else None)
        spheres[name] = [{"center": c.tolist(), "radius": float(r)} for c, r in zip(C, R)]
    Path(path).write_text(yaml.safe_dump({"collision_spheres": spheres}))
    return Path(path)


class RobotModel:
    def __init__(self, robot):
        import warp as wp
        from robokit.robo import Robot as RK

        wp.config.quiet = True
        self.device = "cuda:0" if wp.get_cuda_device_count() else "cpu"
        self.robot, self.info, models, self._hand_pts, self._surface = robot, {}, {}, {}, {}
        for a, arm in robot.arms.items():
            desc = arm.model
            key = (
                desc if isinstance(desc, (str, Path)) else id(desc),
                np.asarray(arm.T_world_base, float).round(6).tobytes(),
            )
            if key not in models:
                with tempfile.TemporaryDirectory(prefix="sk_robot_") as d:  # for spheres generated from the meshes
                    sph = arm.collision_spheres or self._spheres(
                        key[0], RK.load(desc, load_meshes=True), arm.ee_link, Path(d)
                    )
                    m = RK.load(desc, load_meshes=True, load_collision_spheres=True, collision_spheres_path=str(sph))
                models[key] = SimpleNamespace(rk=m, names=list(m.spec.actuated_joint_names), root=arm, desc=desc)
            M, ee = models[key], models[key].rk.link_names.index(arm.ee_link)
            below = np.asarray(M.rk.spec.link_ancestor_links_mask, bool)[:, ee]  # links hanging off ee_link: the hand
            self.info[a] = SimpleNamespace(M=M, idx=[M.names.index(j) for j in arm.joints], ee=ee,
                                           hidx=[M.names.index(j) for j in arm.hand.joints],
                                           hand_links=np.flatnonzero(below | (np.arange(len(below)) == ee)))  # fmt: skip

    _sphere_yaml = {}  # generated sphere files by description, for every RobotModel of the process (a fit takes seconds)

    @staticmethod
    def _spheres(key, rk, ee_link, d):
        """A sphere file in directory `d` for description `key` (loaded as `rk`) that has none: `write_collision_spheres`,
        its hand (`ee_link` and the links hanging off it) fitted on visual meshes where it has no collision mesh. Kept
        for a description given as a path only (an in-memory one's id may be reused)."""
        key = (key, ee_link) if isinstance(key, (str, Path)) else None
        if key is None or key not in RobotModel._sphere_yaml:
            ee = rk.link_names.index(ee_link)
            below = np.asarray(rk.spec.link_ancestor_links_mask, bool)[:, ee]
            hand = [n for k, n in enumerate(rk.link_names) if below[k] or k == ee]
            text = write_collision_spheres(rk, d / "spheres.yml", visual=hand).read_text()
            if key is None:
                return d / "spheres.yml"
            RobotModel._sphere_yaml[key] = text
        (d / "spheres.yml").write_text(RobotModel._sphere_yaml[key])
        return d / "spheres.yml"

    def fq(self, a, obs=None):
        """Arm `a`'s model's full joint vector from an observation (now)."""
        obs, i = obs or self.robot.obs(cams=False), self.info[a]
        f = np.zeros(len(i.M.names), np.float32)
        for b, arm in self.robot.arms.items():
            if (j := self.info[b]).M is i.M:
                f[j.idx] = obs.q[b]
                if j.hidx:
                    ub = np.asarray(obs.hand[b], float)
                    f[j.hidx] = ub if arm.hand.model_q is None else arm.hand.model_q(ub)
        return f

    def set_arm(self, a, fq, q):
        (fq := np.array(fq, np.float32))[..., self.info[a].idx] = q
        return fq

    def links(self, a, fq):
        """World poses of all links of arm `a`'s model at `fq`: (L, 4, 4), or (B, L, 4, 4) for a batch."""
        M, fq = self.info[a].M, np.asarray(fq, np.float32)
        T = M.root.T_world_base @ M.rk.forward_kinematics_via_matrix_numpy(fq.reshape(-1, fq.shape[-1]), device="cpu")
        return T[0] if fq.ndim == 1 else T

    def tcp(self, a, fq=None):
        """World TCP pose of arm `a` (now, or at `fq`; (B, 4, 4) for a batch)."""
        L = self.links(a, self.fq(a) if fq is None else fq)
        return L[..., self.info[a].ee, :, :] @ self.robot.arms[a].hand.T_ee_tcp

    def points(self, a, fq, links=None, spacing=0.004):
        """World surface points of arm `a`'s model at `fq` (all links, or link indices `links`): (N, 3), about `spacing`
        apart: each link's visual mesh (else its collision mesh) sampled once, one point kept per cell, in its frame."""
        from robokit.xform.numpy import transform_points

        M, L, out = self.info[a].M, self.links(a, fq), [np.zeros((0, 3))]
        for k in range(len(M.rk.link_names)) if links is None else links:
            if (key := (id(M), k)) not in self._surface:
                import trimesh

                name, P = M.rk.link_names[k], np.zeros((0, 3))
                scenes = [g[name] for g in (M.rk.spec.link_visual_geometries, M.rk.spec.link_collision_geometries) if g[name].geometry]  # fmt: skip
                if scenes:
                    mesh = scenes[0].to_mesh()
                    P = trimesh.sample.sample_surface(mesh, int(np.clip(4 * mesh.area / spacing**2, 20, 4e5)), seed=0)[
                        0
                    ]
                self._surface[key] = P[voxel_ids(P, spacing)] if len(P) else P
            out.append(transform_points(self._surface[key], L[k]))
        return np.concatenate(out)

    def robot_points(self, obs):
        """Surface points of the whole robot now (each model once): (N, 3)."""
        return np.concatenate([self.points(a, self.fq(a, obs)) for a in {id(self.info[a].M): a for a in self.robot.arms}.values()])  # fmt: skip

    def hand_points(self, a, u):
        """Surface points of arm `a`'s hand at command `u`, in its TCP frame, ~6 mm apart (M, 3); what a grasp backend
        sweeps along its approach (`S.kin.hand_points`)."""
        from robokit.xform.numpy import transform_points

        key = (a, tuple(np.round(np.atleast_1d(u), 3)))
        if key not in self._hand_pts:
            arm, i = self.robot.arms[a], self.info[a]
            f = np.zeros(len(i.M.names), np.float32)
            f[i.idx] = arm.home
            if i.hidx:
                f[i.hidx] = u if arm.hand.model_q is None else arm.hand.model_q(np.asarray(u, float))
            P = self.points(a, f, i.hand_links)
            self._hand_pts[key] = transform_points(P[voxel_ids(P, 0.006)], np.linalg.inv(self.tcp(a, f)))
        return self._hand_pts[key]

    def reach_heights(self, n=1500, below=0.6):
        """(lo, hi) TCP heights the arms reach: the 3rd..99th percentile over random postures within the limits, with lo
        at most `below` m under the lowest shoulder (an arm's first joint), as random postures seldom point down to a
        table under an arm on a pedestal. A mobile robot's lo is 0.15 m over the floor it drives on: it works on low
        tables too, and a table under its hands' reach still holds what they reach for."""
        rng, zs, shoulders = np.random.default_rng(0), [], []
        for a, arm in self.robot.arms.items():
            fq, spec = self.fq(a), self.info[a].M.rk.spec
            Q = self.set_arm(a, np.repeat(fq[None], n, 0), rng.uniform(arm.lo, arm.hi, (n, len(arm.lo))))
            zs.append(self.tcp(a, Q)[:, 2, 3])
            link = list(spec.link_parent_joint_indices).index(spec.joint_names.index(arm.joints[0]))
            shoulders.append(self.links(a, fq)[link, 2, 3])
        lo, hi = np.percentile(np.concatenate(zs), [3, 99])
        if (base := getattr(self.robot, "base", None)) is not None:
            return base.floor_z + 0.15, float(hi)
        return min(float(lo), float(min(shoulders)) - below), float(hi)
