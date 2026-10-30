# Image validation, 2026-10-23

Each image has an independent Dockerfile and retains its model/simulator versions. Model calls run as the caller
with networking disabled and bundled weights. Licensed SMPL-X/MANO models and benchmark vendor assets are mounted.
Simulator checks use the existing adapters, recorded actions and native success checks.

Sizes below are decimal GB, measured from registry manifest layer sizes (compressed) and Docker image inspection
(unpacked). Shared layers count in each image; adding these numbers does not give unique storage or transfer totals.

| Image | Compressed before → after | Unpacked after | Real call |
|---|---:|---:|---|
| LIBERO | 5.847 → 5.002 | 9.798 | Docker and Apptainer: two successful recorded episodes, failing no-ops, RGB-D and calibration. Exact outputs. |
| Robosuite | 4.799 → 4.544 | 9.021 | Five oracle/no-op pairs in Docker with learned and geometric grasp setup; exact outputs. Apptainer also passes. |
| RoboTwin | 9.162 → 8.838 | 15.343 | Five oracle/no-op pairs, cuRobo initialization and RGB-D in Docker and Apptainer; Docker outputs match exactly. |
| DexJoCo | 4.902 → 4.512 | 8.723 | Four oracle/no-op pairs in Docker and Apptainer. Exact joints, depth and calibration; RGB differs by at most 1/255. |
| RoboCasa365 | 15.548 → 15.227 | 32.612 | Apptainer: all recorded oracle/no-op pairs, RGB-D and calibration pass in both images. |
| RoboDojo | 8.350 → 8.313 | 16.901 | Five recorded episodes (three successes, two failures) replay with exact outcomes, joints and object positions in both images. Candidate wire adapter: no-op, RGB-D and calibration pass. |
| EmbodiedSWE | 8.363 → 8.326 | 16.942 | Fixed-G1 and Panda oracles succeed; walking oracle fails in both images. Three no-ops fail as expected. Exact final joints, depth and calibration; RGB varies. |
| HumanoidArena | 15.526 → 15.332 | 29.679 | Apptainer: five failing no-ops, two successful recorded episodes, RGB-D and calibration. Final joints agree within 1e-4. |
| SAM 3 | 7.769 → 7.431 | 11.077 | Detection, tracking and real2sim masks; exact outputs and peak memory. |
| MoGe | 6.641 → 6.370 | 9.912 | MoGe-2 depth worker and real2sim MoGe-1 video depth; exact outputs and peak memory. |
| GeoCalib | 4.244 → 3.973 | 7.117 | Real2sim gravity, two repeats; exact output and peak memory. |
| SAM 3D | 18.449 → 20.600 | 30.799 | Offline real2sim mesh call: 777,692 vertices and 1,555,376 faces; 19.63 GB peak GPU allocation. Auxiliary MoGe/DINO weights and source load without networking. |
| HaWoR | 8.211 → 7.839 | 12.540 | Eight-frame left-hand recovery: identical 21-joint/778-vertex outputs and peak memory. |
| Kimodo | 18.462 → 18.313 | 26.060 | Offline text encoder plus G1 diffusion generation: 60 frames, NPZ/CSV, finite orthonormal rotations and forward motion. |
| GVHMR | 9.743 → 9.604 | 14.323 | Thirty-frame body recovery and a candidate repeat; finite SMPL-X output and identical peak memory. |
| Blender | 0.816 → 0.546 | 1.321 | CUDA Cycles render; maximum pixel difference 1/255. |

The 15 GB compressed target has four exceptions:

- **RoboCasa365:** 23 GB of bundled kitchen assets. Keep their existing quality and layout.
- **HumanoidArena:** NVIDIA's Isaac Sim 6.1 runtime and the verified SONIC dependencies.
- **Kimodo:** the bundled Llama 3 8B text encoder, its adapters and the G1 model.
- **SAM 3D:** its checkpoints, verified environment, and newly bundled MoGe/DINOv2 dependencies. The earlier image
  needed those auxiliary models downloaded at runtime. Keep the verified package list rather than maintaining an
  aggressively pruned environment. Since 0.3.0 its weights, like HaWoR's, are mounted instead
  ([Weights](README.md#weights)).

Real calls identified two caller fixes: LIBERO calibration now reads joint limits through the current robot
model API, and real2sim shape adds the SAM3D checkout root to Python's import path. GVHMR retains GCC and libc
headers because Triton compiles a launcher during inference.

Kimodo was compared across RTX 3090/driver 570 and RTX A5000/driver 580; its generated trajectory varies slightly
(about 1.3 cm at the final root position). GVHMR also varies between repeated runs of the same candidate: maximum
body-pose differences were 0.008 rad against baseline and 0.006 rad between repeats, with translation differences
under 2 mm. These calls preserve the package and weight versions.

EmbodiedSWE’s walking `fruit_delivery` seed-1 replay returns native failure and score 0 in both images on the
RTX A5000, despite an earlier successful Quadro recording. The image comparison preserves that verdict and all final
joints exactly; it does not establish walking-policy success. Both front depth maps have the same one invalid pixel
out of 307,200. A probe requiring every pixel to be finite failed in both images; the additional camera call verifies
the valid-depth mask and calibration. RGB RMSE is 4.4–5.6 out of 255 between images.

Blender EEVEE and Workbench abort EGL initialization in both old and candidate images on the tested hosts
(drivers 570 and 580); Cycles CUDA passes. RoboTwin's old image reports CUDA error 803 under Apptainer on driver
555; the independent candidate passes. RoboDojo's old image did not complete Apptainer viewport initialization on
that host; its Docker validation runs on driver 580. A separate old-image single-environment Docker boot also stalled
in rendering and was stopped; the recorded five-environment replay completed.

GPU calls ran on RTX 3090/A5000 hosts (drivers 550, 555, 570 and 580),
with an additional LIBERO Apptainer pair on a Quadro RTX 6000/driver 580 and the SAM3D mesh call on a
48 GB RTX 6000 Ada. Its measured allocation plus CUDA overhead exceeded the 24 GB hosts’ shared-GPU headroom budget.
Raw logs, manifests, image IDs, fixtures, outputs and comparisons are retained under `runs/slim-images/` in the validation worktree. Wall times include cold
pulls, sandbox creation and shared-host load, so they are not performance comparisons.

Repository checks: `make lint check test check-package` passes (585 tests, 11 skips); Ruff formatting also passes.
Build with `docker/build.sh --build-only`, exercise the real worker
or adapter, then publish the tested image without rebuilding; see [the image workflow](README.md#change-an-image).

## 2026-10-27: WiLoR 0.2.0, Kimodo 0.3.0, SAM 3 on motion_gen props

Built on an RTX 3090 host (driver 570) whose Docker has no buildx: WiLoR's Dockerfile uses no BuildKit-only
syntax and was built with `DOCKER_BUILDKIT=0 docker/build.sh --build-only wilor 0.2.0`. Kimodo's Dockerfile appends
one step (Kimodo-SOMA-RP-v1.1 at a pinned revision) after the Llama shards. Its build secret needs BuildKit, so the
local 0.3.0 tag is that same step built on the pulled 0.2.0 image. Rebuild it with `docker/build.sh` on a BuildKit
host before publishing. Neither image is pushed. Compressed sizes are pigz -6 of the saved layers, an estimate of the
registry's gzip.

| Image | Compressed | Unpacked | Real call |
|---|---:|---:|---|
| WiLoR | 5.757 | 8.850 | 25 frames of a 1080p clip (48 hands) through motion_gen's `wilor-python`: boxes, scores, left/right, `global_orient`, `hand_pose`, `betas`, 3D keypoints and `pred_cam` match the host environment exactly (largest rotation difference 3e-13°). An offline (`--network none`) repeat also passes. |
| Kimodo 0.3.0 | 19.36 | 27.194 | Same prompt and seed with Kimodo-SOMA-RP-v1.1 and `--bvh`, through the container's encoder service and on the host environment (torch 2.6, local encoder): 120 frames, the same BVH hierarchy; joint positions differ by 2.5 mm mean and 6.3 mm max over a 5.9 m walk. |

WiLoR on an RTX 3090 shared with other jobs, batched (detector 16 frames, model 32 crops): detector 20-40 ms per
frame, model about 50 ms per hand, 3.2 GB peak allocation, about 10 s to load. Cutting crops on the CPU
(`ViTDetDataset`) took 0.7-3 s per hand on that loaded host, in both environments. Kimodo's 15 GB target exception now
also covers the 1.05 GB human model.

SAM 3 0.3.0 (unchanged image) replaces motion_gen's host SAM 2 for prop masks. Ten frames per clip, each including
the fastest-moving frames, were run through `roboskills.tools.sam3.sam.Sam` at threshold 0.3 with two phrasings each.
For "umbrella"/"parasol", 9/10 frames gave a clean whole-canopy mask, including blurred spins. The miss was the canopy
seen from inside at the start; "parasol" finds it at 0.23. A small "fan"/"folding fan" (about 100-200 px in 1080p)
was the top detection in 8/10 frames. In one frame it ranked third (0.33) behind wrong detections. In one fast,
blurred frame it scored 0.23. About 1 s per 1080p frame for two prompts, including the PNG round trip.

## 2026-10-28: Kimodo 0.4.0 (adds Kimodo-SMPLX-RP-v1)

`nvidia/Kimodo-SMPLX-RP-v1` is gated (accept its license on Hugging Face; the build takes the same `hf` secret as the
Llama shards) and is pinned at revision `1419ba56b734c48bbafb41fefa84088ca94583b5`. Kimodo writes `STEM_amass.npz`
only for this model. The Dockerfile appends one step. The local and published 0.4.0 are that step built with buildx on
the pulled 0.3.0 (`docker build --network host --ulimit nofile=65536:65536 --secret id=hf,src=$HOME/.cache/huggingface/token`,
a `FROM roboskills-kimodo:0.3.0` Dockerfile holding just that step; a layer of 1.18 GB), because the full rebuild
(about 27 GB plus the Llama download) does not fit this host's disk. Digest
`sha256:8934a65c3ab610e12a2a2ca74b007efeffdb54afbe3b36936b634b35800ec3b8`.

Generation on an RTX 3090 (GPU 4, shared), against the host's already running text encoder at 127.0.0.1:9571, prompt
"a person walks forward and picks up a box from a table", `--duration 4 --seed 0`:

    docker run --rm --gpus '"device=4"' --network host --user $(id -u):$(id -g) -e USER=kimodo -e HOME=/tmp \
      -v $OUT:$OUT -w $OUT ghcr.io/openrobokit/roboskills-kimodo:0.4.0 \
      python -m kimodo.scripts.generate "PROMPT" --model MODEL --duration 4 --seed 0 --output $OUT/STEM [--bvh]

| Model | Files | Result |
|---|---|---|
| Kimodo-SMPLX-RP-v1 | `smplx.npz`, `smplx_amass.npz` | 120 frames; 100 diffusion steps in 19 s |
| Kimodo-G1-RP-v1 | `g1.npz`, `g1.csv` | 120 frames |
| Kimodo-SOMA-RP-v1.1 (`--bvh`) | `soma.npz`, `soma.bvh` | 120 frames |

The AMASS file's keys: `trans (120,3)`, `root_orient (120,3)`, `pose_body (120,63)`, `pose_hand (120,90)`,
`pose_jaw (120,3)`, `pose_eye (120,6)`, `betas (16,)`, `num_betas`, `gender` ('neutral'), `surface_model_type` ('smplx'),
`mocap_frame_rate` (30.0) and `mocap_time_length` (4.0).

RoboKit (the repository environment, `CUDA_VISIBLE_DEVICES=4`, body models in `~/.robokit/body_models/smplx`):
`robokit.helpers.humanoid_retargeting.loaders.load_smplx_file("smplx_amass.npz")` loads it with no conversion:
120 frames at 30 fps; `joints (120,55,3)`, `T_world_joint_xyz_wxyz (120,55,7)`, `vertices (120,10475,3)`; the joint
names start `pelvis, left_hip, right_hip, spine1, left_knee, right_knee, spine2, left_ankle, right_ankle, spine3,
left_foot, right_foot, neck, left_collar, right_collar, head, left_shoulder, right_shoulder, left_elbow, right_elbow,
left_wrist, right_wrist` (the file's 22 body joints, then jaw, eyes and 30 hand joints). `get_smplx_motion` with those
22 names returns `(120,22,7)` at 30 fps. The pelvis starts at (0.03, 0.03, 1.01) m, ends at (0.30, 1.60, 1.02) m and
stays between 0.91 and 1.02 m high: Z-up, walking along +Y. `fetch_smplx_clip` is not used because it takes a
`datasets/<name>/<clip>` path on the RoboKit hub; `load_smplx_file` takes any local path.
