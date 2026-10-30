"""Isaac Sim 6.1 fixes for the pinned HumanoidArena sources, outside them (upstream targets Isaac Sim 5.0).

Adapted from RAX's `services/humanoid/sim/{compat,prim_reset,door_reset,goal_collider}.py` (Apache-2.0, Copyright 2026
RAX contributors), where each is explained at length:

- `install_compat`: Kit 110 renamed CreateShaderPrimFromSdrCommand's `name` to `prim_name` (Isaac Lab's PreviewSurface
  materials call it), and the Arena observation functions' hardware DDS side channel is replaced by a stub (no robot).
- `needs_physics_restart`: PhysX builds a static collider when it parses the stage and does not follow later USD edits;
  when upstream's seeded reset has written one (VisNavi's sign), the episode restarts physics and replays the reset.
- `DoorReset`: OpenDoor's inference profile spawns the door unmanaged (AssetBaseCfg): its physical root follows the
  seeded container pose and its joints return to their initial state at the reset.
- `exact_goal_mesh`: the football goal collides as its exact triangle mesh; 6.1's cooking of upstream's convex
  decomposition puts an invisible 5.4 cm step across the goal mouth.
"""

import inspect
import sys
import types

import numpy as np


def install_compat():
    from isaaclab.sim.spawners.materials import visual_materials
    from pxr import Sdr

    command = visual_materials.CreateShaderPrimFromSdrCommand
    params = inspect.signature(command).parameters
    if "name" not in params and "prim_name" in params:

        def compatible(*args, name=None, **kwargs):
            if name is not None:
                kwargs["prim_name"] = name
            if kwargs.get("identifier") == "UsdPreviewSurface" and "node_type" in params:
                node = Sdr.Registry().GetShaderNodeByIdentifier("UsdPreviewSurface")
                kwargs.setdefault("node_type", str(node.GetSourceType()))
            return command(*args, **kwargs)

        visual_materials.CreateShaderPrimFromSdrCommand = compatible
    if "dds.dds_master" in sys.modules:
        raise RuntimeError("install the DDS stub before importing the Arena tasks")
    stub = types.ModuleType("dds.dds_master")
    stub.dds_manager = types.SimpleNamespace(get_object=lambda name: None, register_object=lambda *a, **k: None)
    sys.modules["dds.dds_master"] = stub


def needs_physics_restart(cfg):
    """Upstream's reset wrote an appended transform on a prim with no rigid body or articulation below it (a static
    collider PhysX parsed at its old pose)."""
    import omni.usd
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    for spec in getattr(cfg, "deterministic_object_resets", None) or []:
        if not (isinstance(spec, dict) and spec.get("prim_paths") and spec.get("pose_range")):
            continue
        for template in spec["prim_paths"]:
            prim = stage.GetPrimAtPath(str(template).replace("{env_idx}", "0"))
            if not (prim and prim.IsValid()):
                continue
            ops = [op.GetOpName() for op in UsdGeom.Xformable(prim).GetOrderedXformOps()]
            static = not any(p.HasAPI(UsdPhysics.RigidBodyAPI) or p.HasAPI(UsdPhysics.ArticulationRootAPI)
                             for p in Usd.PrimRange(prim))  # fmt: skip
            if static and "xformOp:transform" in ops:
                return True
    return False


def _world(path, authored=False):
    """World position and wxyz rotation of a prim (`authored`: its translate/orient/scale ops only, ignoring an
    appended transform)."""
    import omni.usd
    from pxr import Gf, Usd, UsdGeom

    prim = omni.usd.get_context().get_stage().GetPrimAtPath(path)
    ops = {op.GetOpName(): op for op in UsdGeom.Xformable(prim).GetOrderedXformOps()}
    trs = ("xformOp:translate", "xformOp:orient", "xformOp:scale")
    if authored and all(k in ops for k in trs):
        t, o, s = (ops[k].Get() for k in trs)
        local = Gf.Transform()
        local.SetScale(Gf.Vec3d(*s))
        local.SetRotation(Gf.Rotation(Gf.Quatd(float(o.GetReal()), Gf.Vec3d(*o.GetImaginary()))))
        local.SetTranslation(Gf.Vec3d(*t))
        M = local.GetMatrix() * UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(prim.GetParent())
    else:
        M = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    q = M.RemoveScaleShear().ExtractRotationQuat()
    return np.array(M.ExtractTranslation(), float), np.array([q.GetReal(), *q.GetImaginary()], float)


def _qmul(a, b):
    w, x, y, z = a
    v, i, j, k = b
    return np.array([w * v - x * i - y * j - z * k, w * i + x * v + y * k - z * j, w * j - x * k + y * v + z * i,
                     w * k + x * j - y * i + z * v])  # fmt: skip


def _rot(q, p):
    return _qmul(_qmul(q, np.r_[0.0, p]), q * [1, -1, -1, -1])[1:]


class DoorReset:
    """The OpenDoor door's physical state, calibrated against its authored container at boot."""

    PATH = "/World/envs/env_0/Door"

    def __init__(self, env, cfg):
        import torch

        handles = cfg._get_open_door_tensor_proxy_handles(env)
        if not handles or handles["leaf_id"] is None:
            raise RuntimeError("cannot find the physical door")
        self.view, self.sim = handles["view"], env.sim
        self.indices = torch.arange(self.view.count, device=env.device, dtype=torch.int32)
        self.q0 = self.view.get_dof_positions().clone()
        root = self.view.get_root_transforms()[0].cpu().numpy()  # xyz, xyzw
        p, q = _world(self.PATH, authored=True)
        qi = q * [1, -1, -1, -1]
        self.off_p, self.off_q = _rot(qi, root[:3] - p), _qmul(qi, np.r_[root[6], root[3:6]])

    def restore(self):
        import torch

        p, q = _world(self.PATH)
        pos, rot = p + _rot(q, self.off_p), _qmul(q, self.off_q)
        root = self.view.get_root_transforms().clone()
        root[0] = torch.as_tensor([*pos, *rot[1:], rot[0]], device=root.device, dtype=root.dtype)
        self.view.set_root_transforms(root, indices=self.indices)
        self.view.set_root_velocities(torch.zeros_like(self.view.get_root_velocities()), indices=self.indices)
        self.view.set_dof_positions(self.q0, indices=self.indices)
        self.view.set_dof_velocities(torch.zeros_like(self.q0), indices=self.indices)
        self.sim.forward()


def exact_goal_mesh(stage, cfg, ns="/World/envs/env_0"):
    """Author triangle-mesh collision on the football goal's meshes (before the physics reset that parses them)."""
    from pxr import Usd, UsdGeom, UsdPhysics

    changed = 0
    for value in vars(cfg.scene).values():
        asset = str(getattr(getattr(value, "spawn", None), "usd_path", "") or "")
        if "/football_scene/interaction_obj/football_goal/" not in asset:
            continue
        root = stage.GetPrimAtPath(value.prim_path.replace("{ENV_REGEX_NS}", ns).replace("/World/envs/env_.*", ns))
        while instances := [p for p in Usd.PrimRange(root) if p.IsInstance()]:
            for p in instances:
                p.SetInstanceable(False)
        for p in Usd.PrimRange(root):
            if p.IsA(UsdGeom.Mesh) and p.HasAPI(UsdPhysics.MeshCollisionAPI):
                UsdPhysics.MeshCollisionAPI(p).GetApproximationAttr().Set("none")
                changed += 1
    if not changed:
        raise RuntimeError("no football goal collision mesh found")
    return changed
