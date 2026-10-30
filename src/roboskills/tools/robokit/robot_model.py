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


def write_collision_spheres(rk, path, spacing=0.035):
    """Conservative collision spheres for a model without a sphere file, written to `path` (the YAML RoboKit loads as
    `collision_spheres_path`): each link's collision-mesh bounding box, cut along its longest axis into cells at most
    `spacing` long, one circumscribing sphere each."""
    import yaml

    spheres = {}
    for name, scene in rk.spec.link_collision_geometries.items():
        if not scene.geometry:
            continue
        low, high = scene.to_mesh().bounds
        axis = int(np.argmax(high - low))
        count = max(1, int(np.ceil((high[axis] - low[axis]) / spacing)))
        width = (high[axis] - low[axis]) / count
        radius = max(float(np.sqrt(np.sum(np.delete(high - low, axis) ** 2) + width**2) / 2), 0.001)
        centers = np.repeat((low + high)[None] / 2, count, 0)
        centers[:, axis] = low[axis] + (np.arange(count) + 0.5) * width
        spheres[name] = [{"center": c.tolist(), "radius": radius} for c in centers]
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
                    sph = arm.collision_spheres or write_collision_spheres(RK.load(desc, load_meshes=True), Path(d) / "spheres.yml")  # fmt: skip
                    m = RK.load(desc, load_meshes=True, load_collision_spheres=True, collision_spheres_path=str(sph))
                models[key] = SimpleNamespace(rk=m, names=list(m.spec.actuated_joint_names), root=arm, desc=desc)
            M, ee = models[key], models[key].rk.link_names.index(arm.ee_link)
            below = np.asarray(M.rk.spec.link_ancestor_links_mask, bool)[:, ee]  # links hanging off ee_link: the hand
            self.info[a] = SimpleNamespace(M=M, idx=[M.names.index(j) for j in arm.joints], ee=ee,
                                           hidx=[M.names.index(j) for j in arm.hand.joints],
                                           hand_links=np.flatnonzero(below | (np.arange(len(below)) == ee)))  # fmt: skip

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
