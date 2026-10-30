"""Read-only MuJoCo collision geometry and joint measurements; no task annotations.

Compiled mesh vertices already include MuJoCo's asset scaling and recentering.
Their owning geom's measured frame places them in the world. Applying the asset
transform again would move them twice. See MuJoCo's overview of body/geom frames:
https://mujoco.readthedocs.io/en/3.3.2/overview.html#bodies-geoms-sites
"""

import hashlib

import numpy as np


def name(model, kind, index):
    lookup = getattr(model, f"{kind}_id2name", None)
    return (lookup(index) if lookup is not None else getattr(model, kind)(index).name) or f"{kind}{index}"


def physical_bodies(model, robot_roots):
    """Physical collision bodies, excluding the complete robot subtree."""
    excluded = set(int(i) for i in robot_roots)
    for body in range(1, model.nbody):
        if int(model.body_parentid[body]) in excluded:
            excluded.add(body)
    bodies = {
        int(model.geom_bodyid[g]) for g in range(model.ngeom) if model.geom_contype[g] or model.geom_conaffinity[g]
    }
    return sorted(bodies - excluded)


def layout_fingerprint(model, data, robot_roots):
    """Host-only six-decimal collision-body transforms; return a hash, never poses."""
    bodies = physical_bodies(model, robot_roots)
    values = np.c_[np.asarray(data.xpos)[bodies], np.asarray(data.xmat)[bodies]].round(6)
    # Normalize signed zeros: their IEEE bytes differ despite equal positions.
    values[values == 0] = 0
    return hashlib.sha256(values.astype("<f8").tobytes()).hexdigest()


def surface(mesh):
    """Deterministic surface samples, including small faces of simple primitives."""
    faces = np.asarray(mesh.triangles, float)
    if len(faces) > 512:
        faces = faces[np.linspace(0, len(faces) - 1, 512).astype(int)]
    divisions = min(12, max(2, int(np.ceil(np.sqrt(1024 / len(faces))))))
    weights = np.array(
        [
            (a / divisions, b / divisions, 1 - (a + b) / divisions)
            for a in range(divisions + 1)
            for b in range(divisions + 1 - a)
        ]
    )
    points = np.einsum("ij,kjl->kil", weights, faces).reshape(-1, 3)
    if len(points) > 2048:
        points = points[np.linspace(0, len(points) - 1, 2048).astype(int)]
    return np.unique(points, axis=0)


def geometry(model, index):
    """Sample the collision primitive in its compiled geom frame."""
    import mujoco
    import trimesh

    kind, size = int(model.geom_type[index]), np.asarray(model.geom_size[index], float)
    types = mujoco.mjtGeom
    if kind == types.mjGEOM_MESH:
        mid = int(model.geom_dataid[index])
        start, count = int(model.mesh_vertadr[mid]), int(model.mesh_vertnum[mid])
        vertices = np.asarray(model.mesh_vert[start : start + count], float)
        # MuJoCo collision meshes are convex, even when the visual mesh is not.
        mesh = trimesh.convex.convex_hull(vertices)
    elif kind == types.mjGEOM_BOX:
        mesh = trimesh.creation.box(extents=2 * size)
    elif kind == types.mjGEOM_SPHERE:
        mesh = trimesh.creation.icosphere(subdivisions=2, radius=size[0])
    elif kind == types.mjGEOM_ELLIPSOID:
        mesh = trimesh.creation.icosphere(subdivisions=2)
        mesh.vertices *= size
    elif kind == types.mjGEOM_CYLINDER:
        mesh = trimesh.creation.cylinder(radius=size[0], height=2 * size[1], sections=32)
    elif kind == types.mjGEOM_CAPSULE:
        mesh = trimesh.creation.capsule(radius=size[0], height=2 * size[1], count=[16, 16])
    else:
        # Infinite planes and heightfields have no finite surface representation
        # here. The observed collision planner still owns support-plane handling.
        return None
    return surface(mesh)


class CollisionScene:
    """Cache static collision surfaces; read only current physical measurements."""

    def __init__(self, model, robot_roots):
        self.model = model
        bodies = set(physical_bodies(model, robot_roots))
        self.geoms = {}
        for geom in range(model.ngeom):
            if int(model.geom_bodyid[geom]) not in bodies:
                continue
            if not (model.geom_contype[geom] or model.geom_conaffinity[geom]):
                continue
            points = geometry(model, geom)
            if points is not None and len(points) >= 4:
                self.geoms.setdefault(int(model.geom_bodyid[geom]), []).append((geom, points))

    def snapshot(self, data):
        model, objects = self.model, []
        for body, geoms in self.geoms.items():
            clouds = [points @ np.asarray(data.geom_xmat[g]).reshape(3, 3).T + data.geom_xpos[g] for g, points in geoms]
            points = np.vstack(clouds)
            pose = np.eye(4)
            pose[:3, :3] = np.asarray(data.xmat[body]).reshape(3, 3)
            pose[:3, 3] = data.xpos[body]
            joints = []
            for joint in range(model.njnt):
                if int(model.jnt_bodyid[joint]) != body:
                    continue
                kind = int(model.jnt_type[joint])
                count = {0: 7, 1: 4, 2: 1, 3: 1}[kind]
                start = int(model.jnt_qposadr[joint])
                joints.append(
                    dict(
                        name=name(model, "joint", joint),
                        kind={0: "free", 1: "ball", 2: "slide", 3: "hinge"}[kind],
                        qpos=np.asarray(data.qpos[start : start + count]).tolist(),
                        anchor=np.asarray(data.xanchor[joint]).tolist(),
                        axis=np.asarray(data.xaxis[joint]).tolist() if kind in (2, 3) else None,
                        range=np.asarray(model.jnt_range[joint]).tolist() if model.jnt_limited[joint] else None,
                    )
                )
            label = name(model, "body", body)
            objects.append(
                dict(
                    id=label,
                    label=label,
                    parent=name(model, "body", int(model.body_parentid[body])),
                    pose=pose.tolist(),
                    points=points.tolist(),
                    lower=points.min(axis=0).tolist(),
                    upper=points.max(axis=0).tolist(),
                    joints=joints,
                )
            )
        return dict(frame="world", units="metres/radians", objects=objects)
