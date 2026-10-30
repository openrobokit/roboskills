"""Read-only physical geometry from RoboDojo; no task labels, rewards or success rules. Raw simulator poses and
local surface points only: the host (`robot.RoboDojo._scene`) makes world poses and points, so this runs in Isaac Sim's
Python without RoboKit."""

import numpy as np


def array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, float)


def mesh_points(prim, stage, limit=1800):
    """Sample authored mesh triangles in the object's local frame, deterministically."""
    from pxr import Usd, UsdGeom

    transforms = UsdGeom.XformCache()
    root = np.asarray(transforms.GetLocalToWorldTransform(prim), float).T
    # Physics poses omit authored scale. Preserve that scale in cached local
    # geometry, then apply only the current physical rigid pose on each read.
    rigid = root.copy()
    u, _, vt = np.linalg.svd(root[:3, :3])
    rigid[:3, :3] = u @ vt
    inverse = np.linalg.inv(rigid)
    triangles = []
    for child in Usd.PrimRange(prim):
        if not child.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(child)
        points = np.asarray(mesh.GetPointsAttr().Get(), float)
        if not len(points):
            continue
        transform = inverse @ np.asarray(transforms.GetLocalToWorldTransform(child), float).T
        points = points @ transform[:3, :3].T + transform[:3, 3]
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), int)
        cursor = 0
        for count in mesh.GetFaceVertexCountsAttr().Get():
            face = indices[cursor : cursor + count]
            for index in range(1, count - 1):
                triangles.append(points[face[[0, index, index + 1]]])
            cursor += count
    if not triangles:
        raise ValueError("Object has no sampleable mesh triangles")
    triangles = np.asarray(triangles)
    areas = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1)
    if not np.isfinite(areas).all() or areas.sum() <= 0:
        raise ValueError("Object mesh is degenerate")
    rng = np.random.default_rng(0)
    chosen = triangles[rng.choice(len(triangles), size=limit, p=areas / areas.sum())]
    uv = rng.random((limit, 2))
    reflected = uv.sum(1) > 1
    uv[reflected] = 1 - uv[reflected]
    return chosen[:, 0] + uv[:, :1] * (chosen[:, 1] - chosen[:, 0]) + uv[:, 1:] * (chosen[:, 2] - chosen[:, 0])


def snapshot(env):
    from isaacsim.core.utils.stage import get_current_stage

    stage = get_current_stage()
    cache = getattr(env, "public_geometry_cache", None)
    if cache is None:
        env.public_geometry_cache = cache = {}
    objects, errors = [], []
    layout = env.scene_manager.layout_manager
    for kind, collection in layout.object_records_by_type.items():
        if kind in {"Light", "Background", "Room"}:
            continue
        for record in collection.layout_records_by_env[0]:
            name = record["inst_name"]
            obj = layout.get_scene_object(0, name)
            if obj is None:
                continue
            try:
                path = str(obj.prim_path)
                prim = stage.GetPrimAtPath(path)
                key = (id(env.sim), path)
                if key not in cache:
                    cache[key] = mesh_points(prim, stage)
                position, quaternion = obj.get_world_pose()
                objects.append(
                    dict(
                        id=name,
                        label=record.get("category", getattr(obj, "category_name", name)),
                        position=array(position).tolist(),
                        quaternion_wxyz=array(quaternion).tolist(),
                        local_points=cache[key].tolist(),
                        geometry="native_mesh_surface",
                    )
                )
            except Exception as error:
                errors.append(dict(object=name, error=type(error).__name__, message=str(error)))
    return dict(frame="world", steps=int(env.take_action_cnt[0]), objects=objects, errors=errors)
