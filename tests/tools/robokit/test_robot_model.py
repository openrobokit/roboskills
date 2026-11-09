"""The collision spheres generated for a model without a sphere file."""

import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import trimesh
import yaml

from roboskills.tools.robokit.robot_model import fit_spheres, write_collision_spheres


def test_fit_spheres_cover_the_mesh_and_reach_about_the_slack_past_it():
    """A 9 x 2 x 1 cm finger-like box: every surface point inside a sphere, no sphere past the box by much more than
    the slack (the bounding-box slices this replaced reached 1.5 cm past the 1 cm side of so flat a part)."""
    box = trimesh.creation.box((0.09, 0.02, 0.01))
    C, R = fit_spheres(box, slack=0.005)
    S = trimesh.sample.sample_surface(box, 2000, seed=3)[0]
    assert (np.linalg.norm(S[:, None] - C[None], axis=2) <= R[None] + 1e-3).any(1).all()
    reach = np.abs(C)[:, None] + R[:, None, None] * np.eye(3)[None]
    assert (reach.max((0, 1)) - box.extents / 2 <= 0.005 + 0.002).all()


def test_hand_links_without_a_collision_mesh_use_their_visual_mesh():
    """Only the links named in `visual` fall back to their visual mesh; a link left over its budget gets at most
    `budget` spheres; a link no joint moves keeps its bounding-box slices."""
    scene = lambda mesh: SimpleNamespace(geometry={"m": mesh} if mesh is not None else {}, to_mesh=lambda: mesh)
    plate = trimesh.creation.box((0.2, 0.2, 0.004))
    rk = SimpleNamespace(link_names=["base", "arm", "palm", "cover"], spec=SimpleNamespace(
        link_parent_joint_indices=np.array([-1, 0, 1, 1]), joint_names=["j1", "fixed"], nonfixed_joint_names=["j1"],
        link_ancestor_links_mask=np.tril(np.ones((4, 4), bool), -1),
        link_collision_geometries={"base": scene(trimesh.creation.box((0.3, 0.1, 0.1))), "arm": scene(plate),
                                   "palm": scene(None), "cover": scene(None)},
        link_visual_geometries={"arm": scene(plate), "palm": scene(trimesh.creation.box((0.04, 0.04, 0.04))),
                                "cover": scene(trimesh.creation.box((0.04, 0.04, 0.04)))}))  # fmt: skip
    path = write_collision_spheres(rk, Path(tempfile.mkdtemp()) / "s.yml", visual=["palm"], budget=6)
    S = yaml.safe_load(path.read_text())["collision_spheres"]
    assert set(S) == {"base", "arm", "palm"}
    assert len(S["arm"]) <= 6
    assert len(S["base"]) == 6 and len({s["radius"] for s in S["base"]}) == 1  # 9 slices of 3.5 cm, capped at 6


def test_fit_spheres_capped_still_cover_the_mesh():
    """At most `cap` spheres, every surface point still inside one."""
    plate = trimesh.creation.box((0.2, 0.2, 0.004))
    C, R = fit_spheres(plate, slack=0.01, cap=4)
    S = trimesh.sample.sample_surface(plate, 2000, seed=3)[0]
    assert len(R) <= 4 and (np.linalg.norm(S[:, None] - C[None], axis=2) <= R[None] + 5e-3).any(1).all()
