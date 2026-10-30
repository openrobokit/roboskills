# Images

Every image this repository uses, benchmark simulators and models alike, is `docker/<name>/Dockerfile`, pushed to GHCR
as `ghcr.io/openrobokit/roboskills-<name>:<version>`. Public images are `libero`, `robotwin`, `robosuite`, `dexjoco`,
`robocasa365`, `sam3d` and `hawor`; `robodojo`, `embodiedswe` and `humanoidarena` remain private because their Isaac Sim
runtime requires an [NVIDIA AI Enterprise license for redistribution](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/common/license-faq.html).
Users and runtimes only
pull: [`eval/runtime.py`](../src/roboskills/eval/runtime.py) pulls a missing image into docker, or builds its Apptainer
sandbox (`~/.roboskills/images/roboskills-<name>-<version>`) from the registry.

**One model or tool per image, never mixed.** Each Dockerfile starts from a digest-pinned upstream runtime and owns its
dependencies. There is no project base image and no image inherits another tool. Prefer straightforward installs
and existing pinned dependencies; use builder stages when they remove substantial build environments.
Benchmark images bundle their assets and target 15 GB compressed, with [documented exceptions and real-call
results](validation.md) for required assets or vendor runtimes. Model images bake no weights: they hold only code and
dependencies, and their weights live once per home, outside every image version ([Weights](#weights)).
Avoid dependency pruning, extra stages or custom tooling for marginal savings.
Isaac Sim images retain NVIDIA's runtime, which sets a substantial lower bound on their size.

**A tool gets an image only when `uv` cannot make its environment on a plain host.** SAM 3D and HaWoR compile CUDA
extensions (pytorch3d, nvdiffrast, gsplat) against a CUDA toolkit; every other tool (SAM 3, MoGe, GeoCalib, CoACD,
GVHMR, Kimodo, WiLoR, Blender) runs in `uv run` of its own file's inline dependencies, its weights downloaded once on
first use, and needs no image.

An image holds one simulator, model or tool with what it alone needs (its
environment and, for a simulator, its assets), so a caller pulls only the image it uses; two tools a caller uses
separately get two images, even when one pipeline runs both. A model or tool's image puts its python first on `PATH`
(`/opt/venv`) and its weights mounted, read offline; licensed models (SMPL-X, MANO) never go in, callers mount them.
Its
callers pass their worker script in and run it in a fresh container per process (`eval.runtime.contained`, or
`python -m roboskills.eval.runtime IMAGE COMMAND...` from a shell), as the caller, seeing only what they mount. The
[tools table](../README.md#tools) lists what each model and tool does; each has its `SKILL.md` in `src/roboskills/tools/`.

| Image | Holds | Used by |
|---|---|---|
| `libero` 0.2.0 | LIBERO / LIBERO-Pro SDK checkouts and venv | [`eval/libero`](../src/roboskills/eval/libero/SKILL.md) |
| `robosuite` 0.2.0 | CaP-X's robosuite fork and venv | [`eval/robosuite`](../src/roboskills/eval/robosuite/SKILL.md) |
| `robotwin` 0.2.0 | RoboTwin 2.0 checkout, assets and venv | [`eval/robotwin`](../src/roboskills/eval/robotwin/SKILL.md) |
| `dexjoco` 0.2.0 | DexJoCo checkout and venv | [`eval/dexjoco`](../src/roboskills/eval/dexjoco/SKILL.md) |
| `robocasa365` 0.2.0 | RoboCasa checkout, 23 GB of kitchen assets and venv | [`eval/robocasa365`](../src/roboskills/eval/robocasa365/SKILL.md) |
| `robodojo` 0.2.0 | Isaac Sim 5.1 with RoboDojo's IsaacLab and Python deps | [`eval/robodojo`](../src/roboskills/eval/robodojo/SKILL.md) |
| `embodiedswe` 0.2.0 | Isaac Lab 2.3.2 and its pinned assets package on Isaac Sim 5.1 | [`eval/embodiedswe`](../src/roboskills/eval/embodiedswe/SKILL.md) |
| `humanoidarena` 0.2.0 | Isaac Sim 6.1, Isaac Lab patched for it, onnxruntime | [`eval/humanoidarena`](../src/roboskills/eval/humanoidarena/SKILL.md) |
| `sam3d` 0.3.0 | SAM 3D Objects (its checkout, verified environment and CUDA extensions); its gated weights, MoGe and DINOv2 mounted | [`tools/sam3d`](../src/roboskills/tools/sam3d/SKILL.md) |
| `hawor` 0.3.0 | HaWoR (its checkout and verified environment); its hand checkpoint and MANO mounted | [`tools/hawor`](../src/roboskills/tools/hawor/SKILL.md) |

BEHAVIOR runs its official image (`docker.io/stanfordvl/behavior`). Code never goes in an image (runtimes mount the
host's `roboskills` package, workers go in as arguments), and neither do licensed datasets or models, or benchmark
checkouts a runtime mounts. Baselines that are never released, such as EmbodiedSWE's reference agent, keep their
Dockerfile beside their artifact and build locally.

## Pull

Public images need no registry login:

```bash
docker pull ghcr.io/openrobokit/roboskills-libero:0.2.0   # optional: runtimes pull a missing image themselves
```

For private images, log in once per host with a GitHub token that can read packages
(Apptainer: `$APPTAINER_DOCKER_USERNAME` and `$APPTAINER_DOCKER_PASSWORD`):

```bash
gh auth token | docker login ghcr.io -u <github user> --password-stdin
```

## Redistribution

Public visibility permits anonymous pulls; each bundled component retains its upstream license and notices.
The release review covers the pinned upstream revisions used by these images, including dependencies and simulator
assets. SAM 3D and HaWoR publish only their weight-free `0.3.0` images; MANO and other separately licensed weights
stay outside the images.

| Image | Redistribution verdict |
|---|---|
| `libero`, `robotwin` | Allowed with the upstream [LIBERO MIT](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/LICENSE) and [RoboTwin MIT](https://github.com/RoboTwin-Platform/RoboTwin/blob/main/LICENSE) notices retained. |
| `robosuite` | Allowed: [CaP-X fork, MIT](https://github.com/uynitsuj/robosuite/blob/97292732ed909ac3ae116579fb768607034a4dbd/LICENSE); retain copyright and license notices. |
| `dexjoco` | Allowed: [MIT code](https://github.com/brave-eai/dexjoco/blob/8d23b0fab23b17a58c4b55f3942e17013aaf8267/LICENSE), with the bundled [Panda Apache-2.0](https://github.com/brave-eai/dexjoco/blob/8d23b0fab23b17a58c4b55f3942e17013aaf8267/dexjoco/dexjoco/sim/envs/xmls/franka_emika_panda/LICENSE) and [Allegro BSD](https://github.com/brave-eai/dexjoco/blob/8d23b0fab23b17a58c4b55f3942e17013aaf8267/dexjoco/dexjoco/sim/envs/xmls/wonik_allegro/LICENSE) notices retained. |
| `robocasa365` | Allowed: [MIT code](https://github.com/robocasa/robocasa/blob/4f8a2980def75a55dff96b990745b83540425f09/LICENSE) and [CC BY 4.0 assets](https://github.com/robocasa/robocasa/blob/4f8a2980def75a55dff96b990745b83540425f09/README.md#license); retain attribution, license links and modification notices. |
| `sam3d` | Allowed under the [SAM License](https://github.com/facebookresearch/sam-3d-objects/blob/f91db411c50efee93d8db7aeb323885650f6f722/LICENSE), including its use restrictions and a bundled copy of the agreement. The [mip-splatting rasterizer license](https://github.com/autonomousvision/mip-splatting/blob/dda02ab5ecf45d6edb8c540d9bb65c7e451345a9/submodules/diff-gaussian-rasterization/LICENSE.md) limits the built image to noncommercial research/evaluation; retain its license and notices. |
| `hawor` | Allowed for noncommercial redistribution of the unmodified checkout under [CC BY-NC-ND 4.0](https://github.com/ThunderVVV/HaWoR/blob/de9027265c0dd6abac42b11ce80f237107dd36ad/license.txt), with attribution and license retained. Section 2(a)(4) permits necessary format/technical changes; sharing adapted HaWoR code is not cleared. |

## Weights

SAM 3D's and HaWoR's images (0.3.0 on) hold only code and dependencies, so a new version no longer ships their
unchanged weights again. The weights, the same files at the same pinned revisions their 0.2.0 images bundled, live
once per home in `~/.roboskills/weights/<name>` (hosts on a shared home keep one copy): [`tools/sam3d`](../src/roboskills/tools/sam3d/SKILL.md)
and [`tools/hawor`](../src/roboskills/tools/hawor/SKILL.md) mount it read-only at `/weights/<name>`, where the image's
own paths link, and stop with the command to run when it is missing. Fill it once, with uv, git and curl on `PATH`:

```bash
docker/weights.sh sam3d   # facebook/sam-3d-objects is gated: a token with access in ~/.cache/huggingface/token
docker/weights.sh hawor
```

[`weights.sh`](weights.sh) fetches through [`hf.sh`](hf.sh), never prints the token, and leaves the files
world-readable, as they were in the images. Sizes in decimal GB (compressed: the registry's layers; unpacked: Docker's):

| Image | Compressed 0.2.0 → 0.3.0 | Unpacked 0.2.0 → 0.3.0 | `~/.roboskills/weights/<name>` |
|---|---:|---:|---|
| `sam3d` | 20.600 → 6.274 | 30.799 → 15.211 | 15.584: `checkpoints/` 13.106 (pipeline.yaml), `hf/` 1.257 (MoGe ViT-L), `torch/` 1.222 (DINOv2 ViT-L/14 reg, its source) |
| `hawor` | 7.839 → 4.862 | 12.540 → 9.270 | 3.267: `hawor/checkpoints/hawor.ckpt`, `hawor/model_config.yaml` |

`sam3d` 0.3.0 keeps 0.2.0's eight code layers byte for byte; `hawor`'s final stage holds the same files as 0.2.0's,
without the weights and the stdlib bytecode their download compiled. Validated 2026-10-28 on a Quadro RTX 6000 (driver 580), one real call per tool and image
on DexYCB `s01-mug-2010` (`runs/mounted_weights_20261028` there): SAM 3D's `pose.json` (seed, `camera_from_object`,
every trial) and HaWoR's `hands.npz` are byte-identical to 0.2.0's and to the original run's. SAM 3D's mesh differs
between any two calls; 0.3.0's is within 0.06 mm of 0.2.0's (99th percentile of nearest-vertex distance; mean
0.003 mm, max 0.84 mm, 0.05% fewer vertices), and 0.2.0 reading the mounted folder shifts the same way, so the shift
follows the folder (byte-identical files, read from the home's NFS) rather than the image. Time and memory are
unchanged: SAM 3D 220–250 s and 22.0 GiB in use on the card, HaWoR 20 s and 6.4 GiB.

## Change an image

Edit its Dockerfile, bump its version here and in the constants that name it (`IMAGE` in the runtime's `launch.py`,
`IMAGE` in `sam3d.py` and `hawor.py`), then build and push
it once from a docker host, logged in with a token holding `write:packages`:

```bash
docker/build.sh NAME VERSION [CONTEXT [docker build args...]]
```

Use `docker/build.sh --build-only NAME VERSION [CONTEXT [docker build args...]]` to build without publishing.
Validate that local image through its actual worker or simulator adapter, including rendering and oracle replay
where applicable, then `docker push ghcr.io/openrobokit/roboskills-NAME:VERSION` to publish the tested image without
rebuilding it. Record the image ID, compressed and unpacked sizes, hardware, commands and results under `runs/`.
For validation on another host, publish a unique candidate tag after local import checks, pull and validate it
there, then promote that same image to its release tag. Keep callers on the previous release until validation passes.

`build.sh` refuses a version that is already pushed. Every image's context is the repository root; RoboDojo fetches
its pinned Isaac Lab source during the build. The Isaac Sim images accept NVIDIA's Isaac
Sim EULA. Inputs are pinned (base images by digest; venvs by `uv.lock`, by a `uv pip compile` of the image's direct
dependencies, or by the verified environment's freeze, each `requirements.txt` beside its Dockerfile; checkouts by
commit; downloads by revision or SHA-256), and every layer stays under GHCR's 10 GB: large weights get a layer each.
