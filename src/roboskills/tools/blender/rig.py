# /// script
# requires-python = "==3.11.*"
# dependencies = ["bpy==4.5.14", "numpy==1.26.4"]
# ///
"""A Mixamo character's GLB, and that GLB animated by a saved motion, in Blender (its inline dependencies, as
motion_studio.py's). `import_character` and `animate_character` in blender.py run it.

    python rig.py import CHARACTER.fbx OUT.glb
    python rig.py animate CHARACTER.glb MOTION.npz OUT.glb

MOTION holds fps, turns [T, J, 3, 3] and positions [T, J, 3] of joint_names, and their rest_joints [J, 3] (the
workflow's `retarget_motion`). The character's other bones follow their parents.
"""

import sys

import numpy as np


def rigid(turns: np.ndarray, joints: np.ndarray, rest: np.ndarray) -> np.ndarray:
    """[T, J, 4, 4] what each bone's rest becomes: the turn about its rest joint, carried to its posed joint."""
    out = np.tile(np.eye(4), (*turns.shape[:2], 1, 1))
    out[..., :3, :3], out[..., :3, 3] = turns, joints - np.einsum("tjik,jk->tji", turns, rest)
    return out


def bases(moves: np.ndarray, rests: np.ndarray, parents: np.ndarray, armature: np.ndarray) -> np.ndarray:
    """[T, B, 4, 4] Blender's pose matrices (`matrix_basis`) for `moves` [T, B, 4, 4] in the world, of bones with
    `rests` [B, 4, 4] (`matrix_local`) and `parents` [B] (-1: none) under an `armature` 4x4 world matrix. A bone's
    pose is parent_pose @ parent_rest^-1 @ rest @ basis, and the move is what takes its rest to its pose."""
    moved = np.linalg.inv(armature) @ moves @ armature
    above = np.where(parents[:, None, None] < 0, np.eye(4), np.linalg.inv(moved[:, parents]))
    return np.linalg.inv(rests) @ above @ moved @ rests


def clear() -> None:
    import bpy

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def export(path: str, animations: bool) -> None:
    import bpy

    bpy.ops.export_scene.gltf(filepath=path, export_format="GLB", export_animations=animations)


def import_character(source: str, out: str) -> None:
    import bpy

    clear()
    bpy.ops.import_scene.fbx(filepath=source)
    export(out, False)


def animate_character(character: str, motion: str, out: str) -> None:
    import bpy
    from mathutils import Matrix

    d = np.load(motion)
    fps, names = float(d["fps"]), [str(n) for n in d["joint_names"]]
    clear()
    scene = bpy.context.scene
    scene.render.fps, scene.render.fps_base = round(fps), round(fps) / fps
    scene.frame_start, scene.frame_end = 0, len(d["turns"]) - 1
    bpy.ops.import_scene.gltf(filepath=character)
    rig = next(o for o in scene.objects if o.type == "ARMATURE")
    rig.animation_data_clear()
    bones = sorted(rig.data.bones, key=lambda b: len(b.parent_recursive))
    index = {b.name: i for i, b in enumerate(bones)}
    missing = [n for n in names if n not in index]
    if missing:
        raise ValueError(f"{character} lacks the motion's bones {missing}")
    parents = np.array([index[b.parent.name] if b.parent else -1 for b in bones])
    rests = np.array([np.array(b.matrix_local) for b in bones])
    armature = np.array(rig.matrix_world)
    lift = (armature @ rests[index[names[0]]])[:3, 3] - d["rest_joints"][0]  # the character's floor is not Blender's
    named = rigid(d["turns"], d["positions"] + lift, d["rest_joints"] + lift)
    moves = np.empty((len(named), len(bones), 4, 4))
    for i, bone in enumerate(bones):  # a bone the motion does not name moves as its parent does
        if bone.name in names:
            moves[:, i] = named[:, names.index(bone.name)]
        else:
            moves[:, i] = moves[:, parents[i]] if parents[i] >= 0 else np.eye(4)
    pose = bases(moves, rests, parents, armature)
    for i, bone in enumerate(bones):
        posed = rig.pose.bones[bone.name]
        posed.rotation_mode = "QUATERNION"
        for t, matrix in enumerate(pose[:, i]):
            posed.matrix_basis = Matrix(matrix.tolist())
            posed.keyframe_insert("rotation_quaternion", frame=t)
            posed.keyframe_insert("location", frame=t)
    export(out, True)


if __name__ == "__main__":
    {"import": import_character, "animate": animate_character}[sys.argv[1]](*sys.argv[2:])
