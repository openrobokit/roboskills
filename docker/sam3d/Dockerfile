# syntax=docker/dockerfile:1.7
# SAM 3D Objects, an object's mesh and pose from one image and its mask: the verified environment's packages
# (requirements.txt, torch 2.5.1+cu121; pytorch3d, nvdiffrast, gsplat and mip-splatting's diff-gaussian-rasterization
# built in the first stage for Turing to Ada, kaolin and flash-attn from their release wheels) in /opt/venv, its python
# first on PATH; the sam-3d-objects checkout at a pinned commit in /opt/sam3d (hydra patched as its setup says). The
# weights are not in the image: callers mount ~/.roboskills/weights/sam3d (docker/weights.sh sam3d fills it once;
# facebook/sam-3d-objects is gated) read-only at /weights/sam3d, read offline: checkpoints/ (pipeline.yaml;
# /opt/sam3d/checkpoints links there), and hf/ and torch/ ($HF_HOME, $TORCH_HOME) with MoGe and DINOv2.
# Callers pass their worker in (src/roboskills/tools/sam3d/SKILL.md).
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04@sha256:7012e535a47883527d402da998384c30b936140c05e2537158c80b8143ee7425 AS build
COPY --from=ghcr.io/astral-sh/uv:0.9.5@sha256:f459f6f73a8c4ef5d69f4e6fbbdb8af751d6fa40ec34b39a1ab469acd6e289b7 /uv /uvx /usr/local/bin/
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates git && rm -rf /var/lib/apt/lists/*
ENV PATH=/opt/build/bin:$PATH UV_PYTHON_INSTALL_DIR=/opt/python FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9+PTX" MAX_JOBS=8 CMAKE_BUILD_PARALLEL_LEVEL=8 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
RUN --mount=type=cache,target=/root/.cache/uv uv venv -q --python 3.11 /opt/build \
    && uv pip install -q --python /opt/build/bin/python --index-strategy unsafe-best-match \
        --extra-index-url https://download.pytorch.org/whl/cu121 torch==2.5.1+cu121 numpy==1.26.4 setuptools==75.8.2 \
        wheel pip ninja jaxtyping rich
RUN git clone -q --recursive https://github.com/autonomousvision/mip-splatting.git /src/mip-splatting \
    && git -C /src/mip-splatting checkout -q dda02ab5ecf45d6edb8c540d9bb65c7e451345a9 \
    && git -C /src/mip-splatting submodule -q update --init --recursive \
    && /opt/build/bin/python -m pip wheel -q --no-build-isolation --no-deps -w /wheels \
        "git+https://github.com/facebookresearch/pytorch3d.git@75ebeeaea0908c5527e7b1e305fbc7681382db47" \
        "git+https://github.com/NVlabs/nvdiffrast.git@253ac4fcea7de5f396371124af597e6cc957bfae" \
        "git+https://github.com/nerfstudio-project/gsplat.git@2323de5905d5e90e035f792fe65bad0fedd413e7" \
        /src/mip-splatting/submodules/diff-gaussian-rasterization

FROM ubuntu:22.04@sha256:98510ac07bb4545e389be502a8918b385c018bfe4ceeb5ac7e0375c782268b04
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates git build-essential libegl1 libgl1 libgles2 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /usr/share/glvnd/egl_vendor.d \
    && echo '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}' \
        > /usr/share/glvnd/egl_vendor.d/10_nvidia.json
COPY --from=ghcr.io/astral-sh/uv:0.9.5@sha256:f459f6f73a8c4ef5d69f4e6fbbdb8af751d6fa40ec34b39a1ab469acd6e289b7 /uv /uvx /usr/local/bin/
ENV UV_PYTHON_INSTALL_DIR=/opt/python UV_LINK_MODE=copy MAX_JOBS=8 CMAKE_BUILD_PARALLEL_LEVEL=8 \
    OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
# CONDA_PREFIX: notebook/inference.py, SAM3D's entry point, takes CUDA_HOME from it
ENV PATH=/opt/venv/bin:$PATH CONDA_PREFIX=/opt/venv NVIDIA_REQUIRE_CUDA="cuda>=12.0"
# hf.sh is unused since 0.3.0; it stays so the layers below remain 0.2.0's, shared in the registry
COPY docker/sam3d/requirements.txt docker/hf.sh /opt/build/
COPY --from=build /wheels /opt/build/wheels
# torch and its CUDA libraries first, a layer of their own (GHCR takes layers up to 10 GB)
RUN --mount=type=cache,target=/root/.cache/uv uv venv -q --python 3.11 /opt/venv \
    && grep -E '^(torch|torchvision|torchaudio|triton|nvidia-[a-z0-9-]+)==' /opt/build/requirements.txt \
        > /opt/build/torch.txt \
    && uv pip install -q --compile-bytecode --no-deps --python /opt/venv/bin/python --index-strategy unsafe-best-match \
        --extra-index-url https://download.pytorch.org/whl/cu121 -r /opt/build/torch.txt \
    && chmod -R a+rX /opt/python /opt/venv
RUN --mount=type=cache,target=/root/.cache/uv uv pip install -q --compile-bytecode --no-deps \
        --python /opt/venv/bin/python --index-strategy unsafe-best-match \
        --extra-index-url https://download.pytorch.org/whl/cu121 \
        --find-links https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html \
        -r /opt/build/requirements.txt kaolin==0.17.0 /opt/build/wheels/*.whl \
        "flash-attn @ https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl" \
    && find /opt/venv \( ! -perm -a+r -o -type d ! -perm -a+x \) -exec chmod a+rX {} +
RUN git init -q /opt/sam3d && cd /opt/sam3d \
    && git fetch -q --depth 1 https://github.com/facebookresearch/sam-3d-objects.git \
        f91db411c50efee93d8db7aeb323885650f6f722 \
    && git checkout -q FETCH_HEAD && rm -rf .git \
    && /opt/venv/bin/python patching/hydra \
    && chmod -R a+rX /opt/sam3d /opt/venv/lib/python3.11/site-packages/hydra/core
# The weights' mount point: the checkpoints, and MoGe's and DINOv2's (and DINOv2's source), which the pipeline also loads
ENV HF_HOME=/weights/sam3d/hf HF_HUB_OFFLINE=1 TORCH_HOME=/weights/sam3d/torch
RUN rm -r /opt/sam3d/checkpoints && mkdir -p /weights/sam3d && ln -s /weights/sam3d/checkpoints /opt/sam3d/checkpoints

ENV PATH=/opt/venv/bin:$PATH HOME=/tmp NVIDIA_REQUIRE_CUDA="cuda>=12.0" \
    PYOPENGL_PLATFORM=egl EGL_PLATFORM=device OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
WORKDIR /opt/sam3d
