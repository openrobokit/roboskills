"""An object's mesh and camera pose from one video frame and its mask with SAM 3D Objects, in docker/sam3d's image with
its weights mounted: the best of several pose samples by mask and depth agreement, then its mesh. This file is
self-contained: the image runs its text.

    python -m roboskills.tools.sam3d --video V --masks masks/ --depth depth/depth.npz --out shape/ [--frame 0] [--label 1]
      [--no-offload] [--gaussians]
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np


IMAGE = "ghcr.io/openrobokit/roboskills-sam3d:0.3.0"
REPO = "/opt/sam3d"  # the sam-3d-objects checkout in that image
WEIGHTS = Path.home() / ".roboskills/weights/sam3d"  # docker/weights.sh sam3d; mounted read-only at /weights/sam3d


def reconstruct_object(
    video: Path, masks: Path, depth: Path, out: Path, frame: int = 0, label: int = 1, samples: int = 8,
    keep: bool = False, offload: bool = True, gaussians: bool = False,
):  # fmt: skip
    """out/mesh.ply (the object's frame, metres, vertex colours) and out/pose.json (`camera_from_object`, 4x4 in the
    OpenCV camera frame; the mesh's `scale` from SAM 3D's frame; the chosen `seed`; each sample's [seed, mask IoU, depth
    residual]) for label `label` of masks/<frame>.png, on the point map of depth.npz (MoGe: depth, valid, K, frames);
    with `keep`, every sample's too, in out/samples/<seed>/. With `offload`, only the running stage's models are on the
    GPU: 8.4 GiB peak allocation instead of 17.3 GiB, the same samples and pose, about 20 s more per call. Only the mesh
    is decoded unless `gaussians`: then SAM 3D's Gaussians too, in out/gaussians.ply (its own frame: the mesh's before
    pose.json's `scale`)."""
    from roboskills.eval.runtime import contained

    if not WEIGHTS.is_dir():
        raise SystemExit(f"SAM 3D's weights are not in {WEIGHTS}: fill it once with docker/weights.sh sam3d")
    paths = [Path(p).resolve() for p in (video, masks, depth, out)]
    paths[3].mkdir(parents=True, exist_ok=True)
    args = [*paths, frame, label, samples, int(keep), int(offload), int(gaussians)]
    argv = ["python", "-c", Path(__file__).read_text(), *map(str, args)]
    mounts = sorted({str(p.parent) for p in paths}) + [(WEIGHTS.resolve(), "/weights/sam3d", "ro")]
    if (cmd := contained(IMAGE, argv, mounts)) is None:
        raise SystemExit(f"docker cannot run {IMAGE} here")
    subprocess.run(cmd, check=True)


def points(depth, k):
    v, u = np.indices(depth.shape)
    return np.stack([(u - k[0, 2]) / k[0, 0] * depth, (v - k[1, 2]) / k[1, 1] * depth, depth], -1)


def staged(config: str, gaussians: bool):
    """SAM 3D's pipeline as `Inference(config)` builds it, but loaded on the CPU, each stage moving only its own models
    to the GPU as it starts: sparse structure (ss_generator, ss_decoder, their DINOv2 embedder), structured latent
    (slat_generator, its embedder), decoding (the mesh decoder, and the Gaussian one with `gaussians`). MoGe stays on
    the CPU: the call passes its own point map. The peak is in sparse-structure sampling."""
    import torch
    from hydra.utils import instantiate
    from inference import BLACKLIST_FILTERS, WHITELIST_FILTERS, check_hydra_safety
    from omegaconf import OmegaConf

    settings = OmegaConf.load(config)  # Inference's settings, every model on the CPU
    settings.rendering_engine, settings.compile_model = "pytorch3d", False
    settings.workspace_dir, settings.device, settings.depth_model.device = str(Path(config).parent), "cpu", "cpu"
    check_hydra_safety(settings, WHITELIST_FILTERS, BLACKLIST_FILTERS)
    pipeline = instantiate(settings)
    pipeline.device = torch.device("cuda")  # inputs and intermediates
    models, embedders = pipeline.models, pipeline.condition_embedders
    stages = {
        "sample_sparse_structure": [models["ss_generator"], models["ss_decoder"], embedders["ss_condition_embedder"]],
        "sample_slat": [models["slat_generator"], embedders["slat_condition_embedder"]],
        "decode_slat": [models["slat_decoder_mesh"]] + [models["slat_decoder_gs"]] * gaussians,
    }
    stages = {name: [m for m in ms if isinstance(m, torch.nn.Module)] for name, ms in stages.items()}
    everything = [m for ms in stages.values() for m in ms]

    def resident(step, modules):
        def call(*args, **kwargs):
            for m in everything:  # leave, then arrive: the GPU holds one stage's models at a time
                if all(m is not n for n in modules):
                    m.to("cpu")
            torch.cuda.empty_cache()
            for m in modules:
                m.to("cuda")
            return step(*args, **kwargs)

        return call

    for name, modules in stages.items():
        setattr(pipeline, name, resident(getattr(pipeline, name), modules))
    return pipeline


def run(video, masks, depth, out, frame, label, samples, every=0, offload=1, gaussians=0):
    """`reconstruct_object`'s model side."""
    import cv2
    import torch
    from PIL import Image

    sys.path[:0] = [f"{REPO}/notebook", REPO]
    from inference import Inference  # SAM 3D's own entry point
    from pytorch3d.transforms import quaternion_to_matrix
    from sam3d_objects.pipeline.inference_pipeline_pointmap import camera_to_pytorch3d_camera
    from scipy.ndimage import binary_closing
    from scipy.spatial import cKDTree

    frame, label = int(frame), int(label)
    capture = cv2.VideoCapture(video)
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
    rgb = np.ascontiguousarray(capture.read()[1][..., ::-1])
    labels = np.array(Image.open(Path(masks) / f"{frame:05d}.png"))
    obj, other = labels == label, (labels > 0) & (labels != label)  # another object (a hand) may hide part of it
    scene = np.load(depth)
    row = int(np.searchsorted(scene["frames"], frame))
    metres, valid, k = scene["depth"][row].astype(np.float32), scene["valid"][row], scene["K"]
    to_p3d = camera_to_pytorch3d_camera().rotation[0].numpy()  # row vectors: p3d = cv @ to_p3d
    pointmap = points(metres, k) @ to_p3d
    pointmap[~valid] = np.nan
    config = f"{REPO}/checkpoints/pipeline.yaml"
    gaussians = bool(int(gaussians))
    pipeline = staged(config, gaussians) if int(offload) else Inference(config)._pipeline
    formats, decode = ["mesh", "gaussian"] if gaussians else ["mesh"], pipeline.decode_slat
    # to_glb is always handed a Gaussian, which it reads only to bake a texture
    pipeline.decode_slat = lambda slat, formats: dict(gaussian=[None]) | decode(slat, formats)
    image = np.concatenate([rgb, obj[..., None].astype(np.uint8) * 255], -1)
    pointmap = torch.from_numpy(pointmap).float()

    def pose_of(result):  # camera(cv) = ((scale * local) @ rotation + translation) @ to_p3d.T, rotation acting on rows
        pose = np.eye(4)
        pose[:3, :3] = to_p3d @ quaternion_to_matrix(result["rotation"][0]).cpu().numpy().T
        pose[:3, 3] = result["translation"][0].cpu().numpy() @ to_p3d.T
        return pose, result["scale"][0].cpu().numpy()

    # pose samples vary widely; keep the one whose coarse voxels best match the mask (IoU off other objects) and depth
    seen = points(metres, k)[obj & valid]
    seen = seen[np.random.default_rng(0).permutation(len(seen))[:3000]]
    target, keep = obj[::2, ::2], ~other[::2, ::2]
    trials = []
    for seed in range(int(samples)):
        result = pipeline.run(image, None, seed, stage1_only=True, pointmap=pointmap)
        pose, scale = pose_of(result)
        cam = (result["voxel"].cpu().numpy() * scale) @ pose[:3, :3].T + pose[:3, 3]
        u, v = ((cam[:, :2] / cam[:, 2:] * np.diag(k)[:2] + k[:2, 2]) / 2).astype(int).T
        drawn = np.zeros_like(target)
        inside = (u >= 0) & (u < drawn.shape[1]) & (v >= 0) & (v < drawn.shape[0])
        drawn[v[inside], u[inside]] = True
        drawn = binary_closing(drawn, iterations=2)
        iou = (drawn & target & keep).sum() / max(((drawn | target) & keep).sum(), 1)
        trials.append([seed, float(iou), float(np.median(cKDTree(cam).query(seen)[0]) / np.ptp(cam, 0).max())])
    ranks = np.argsort(np.argsort([-t[1] for t in trials])) + np.argsort(np.argsort([t[2] for t in trials]))
    seed = trials[int(np.lexsort(([-t[1] for t in trials], ranks))[0])][0]
    for each in [seed] + ([t[0] for t in trials if t[0] != seed] if int(every) else []):  # the chosen one first
        result = pipeline.run(image, None, each, with_mesh_postprocess=False, with_texture_baking=False,
                              use_vertex_color=True, pointmap=pointmap, decode_formats=formats)  # fmt: skip
        mesh, (pose, scale) = result["glb"], pose_of(result)
        mesh.vertices = (
            np.asarray(mesh.vertices) @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]).T * scale
        )  # undo GLB's y-up
        record = dict(camera_from_object=pose.tolist(), scale=np.atleast_1d(scale).tolist(), seed=seed, trials=trials)
        for folder in [Path(out) / "samples" / str(each)] * int(every) + [Path(out)] * (each == seed):
            folder.mkdir(parents=True, exist_ok=True)
            mesh.export(folder / "mesh.ply")
            gaussians and result["gs"].save_ply(str(folder / "gaussians.ply"))
            (folder / "pose.json").write_text(json.dumps(dict(record, sample=each), indent=2) + "\n")
        del result, mesh
        torch.cuda.empty_cache()  # one sample's decoders at a time
    print(f"sam3d: peak GPU allocation {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", file=sys.stderr)


if __name__ == "__main__":
    run(*sys.argv[1:])
