"""Read only physical SAPIEN collision geometry, without task-specific actor roles."""

import numpy as np


def shape_mesh(shape):
    import trimesh

    name = type(shape).__name__
    if hasattr(shape, "get_vertices"):
        return trimesh.Trimesh(
            vertices=np.asarray(shape.get_vertices()) * np.asarray(shape.get_scale()),
            faces=np.asarray(shape.get_triangles()).reshape(-1, 3),
            process=False,
        )
    if name.endswith("Box"):
        return trimesh.creation.box(extents=2 * np.asarray(shape.get_half_size()))
    if name.endswith("Sphere"):
        return trimesh.creation.icosphere(subdivisions=2, radius=shape.get_radius())
    if name.endswith("Capsule"):
        mesh = trimesh.creation.capsule(height=2 * shape.get_half_length(), radius=shape.get_radius())
        # PhysX capsules extend along x; trimesh creates them along z.
        mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
        return mesh
    if name.endswith("Cylinder"):
        mesh = trimesh.creation.cylinder(radius=shape.get_radius(), height=2 * shape.get_half_length())
        mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
        return mesh
    raise ValueError(f"Unsupported physical shape {name}")


def snapshot(robot):
    import trimesh

    objects, planes, errors = [], [], []
    for index, entity in enumerate(robot.env.scene.get_all_actors()):
        if entity.per_scene_id in robot.robot_ids:
            continue
        name = entity.get_name() or f"body{index}"
        try:
            meshes = []
            for component in entity.get_components():
                if not hasattr(component, "get_collision_shapes"):
                    continue
                for shape in component.get_collision_shapes():
                    if type(shape).__name__.endswith("Plane"):
                        pose = (
                            entity.get_pose().to_transformation_matrix()
                            @ shape.get_local_pose().to_transformation_matrix()
                        )
                        planes.append(
                            dict(
                                id=f"{name}:{entity.per_scene_id}",
                                origin=pose[:3, 3].tolist(),
                                normal=pose[:3, 0].tolist(),
                            )
                        )
                        continue
                    mesh = shape_mesh(shape)
                    mesh.apply_transform(shape.get_local_pose().to_transformation_matrix())
                    meshes.append(mesh)
            if not meshes:
                continue
            mesh = trimesh.util.concatenate(meshes)
            local, _ = trimesh.sample.sample_surface(mesh, 1800, seed=0)
            pose = np.asarray(entity.get_pose().to_transformation_matrix())
            points = local @ pose[:3, :3].T + pose[:3, 3]
            objects.append(
                dict(
                    id=f"{name}:{entity.per_scene_id}",
                    label=name,
                    pose=pose.tolist(),
                    points=points.tolist(),
                    lower=points.min(0).tolist(),
                    upper=points.max(0).tolist(),
                    geometry="native_collision_surface",
                )
            )
        except Exception as error:
            errors.append(dict(object=name, error=type(error).__name__, message=str(error)))
    return dict(frame="world", steps=int(robot.env.take_action_cnt), objects=objects, planes=planes, errors=errors)
