# syntax=docker/dockerfile:1.7-labs
# HaWoR, hand motion from video (MANO): the verified environment's packages (requirements.txt, torch 2.9.0+cu128;
# pytorch3d and chumpy built in the first stage) in /opt/venv, its python first on PATH; the HaWoR checkout at a pinned
# commit in /opt/hawor (the working directory: HaWoR resolves its weights and MANO under it). The weights are not in the
# image: callers mount ~/.roboskills/weights/hawor (docker/weights.sh hawor fills it once with the released hawor
# checkpoint and model config) read-only at /weights/hawor, where /opt/hawor/weights links. MANO is licensed, so it is
# never in the image either: callers mount MANO_RIGHT.pkl at /opt/hawor/_DATA/data/mano/ and MANO_LEFT.pkl at
# /opt/hawor/_DATA/data_left/mano_left/. Callers pass their worker in (src/roboskills/tools/hawor/SKILL.md).
FROM nvidia/cuda:12.8.1-devel-ubuntu22.04@sha256:a99a1860ba8e2916e5c3e73b72ec4c4301653a84586e05bfc9a2aa2d58027e97 AS build
COPY --from=ghcr.io/astral-sh/uv:0.9.5@sha256:f459f6f73a8c4ef5d69f4e6fbbdb8af751d6fa40ec34b39a1ab469acd6e289b7 /uv /uvx /usr/local/bin/
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates git && rm -rf /var/lib/apt/lists/*
ENV PATH=/opt/build/bin:$PATH UV_PYTHON_INSTALL_DIR=/opt/python FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9+PTX" MAX_JOBS=8 CMAKE_BUILD_PARALLEL_LEVEL=8 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
RUN --mount=type=cache,target=/root/.cache/uv uv venv -q --python 3.10 /opt/build \
    && uv pip install -q --python /opt/build/bin/python --index-strategy unsafe-best-match \
        --extra-index-url https://download.pytorch.org/whl/cu128 torch==2.9.0+cu128 numpy==1.26.4 setuptools==80.10.2 \
        wheel pip ninja \
    && /opt/build/bin/python -m pip wheel -q --no-build-isolation --no-deps -w /wheels \
        "git+https://github.com/facebookresearch/pytorch3d.git@75ebeeaea0908c5527e7b1e305fbc7681382db47" \
        "git+https://github.com/mattloper/chumpy@580566eafc9ac68b2614b64d6f7aaa84eebb70da"

FROM ubuntu:22.04@sha256:98510ac07bb4545e389be502a8918b385c018bfe4ceeb5ac7e0375c782268b04 AS packages
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates git build-essential libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.9.5@sha256:f459f6f73a8c4ef5d69f4e6fbbdb8af751d6fa40ec34b39a1ab469acd6e289b7 /uv /uvx /usr/local/bin/
ENV UV_PYTHON_INSTALL_DIR=/opt/python UV_LINK_MODE=copy MAX_JOBS=8 CMAKE_BUILD_PARALLEL_LEVEL=8 \
    OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
ENV PATH=/opt/venv/bin:$PATH TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 NVIDIA_REQUIRE_CUDA="cuda>=12.0"
# hf.sh is unused since 0.3.0; it stays so the layers below remain 0.2.0's, shared in the registry
COPY docker/hawor/requirements.txt docker/hf.sh /opt/build/
COPY --from=build /wheels /opt/build/wheels
# torch and its CUDA libraries first, a layer of their own (GHCR takes layers up to 10 GB)
RUN --mount=type=cache,target=/root/.cache/uv uv venv -q --python 3.10 /opt/venv \
    && grep -E '^(torch|torchvision|triton|nvidia-[a-z0-9-]+)==' /opt/build/requirements.txt > /opt/build/torch.txt \
    && uv pip install -q --compile-bytecode --no-deps --python /opt/venv/bin/python --index-strategy unsafe-best-match \
        --extra-index-url https://download.pytorch.org/whl/cu128 -r /opt/build/torch.txt \
    && chmod -R a+rX /opt/python /opt/venv
RUN --mount=type=cache,target=/root/.cache/uv uv pip install -q --compile-bytecode --no-deps \
        --python /opt/venv/bin/python --index-strategy unsafe-best-match \
        --extra-index-url https://download.pytorch.org/whl/cu128 --find-links https://data.pyg.org/whl/torch-2.9.0+cu128.html \
        -r /opt/build/requirements.txt torch-scatter==2.1.2+pt29cu128 /opt/build/wheels/*.whl \
    && find /opt/venv \( ! -perm -a+r -o -type d ! -perm -a+x \) -exec chmod a+rX {} +
RUN git init -q /opt/hawor && cd /opt/hawor \
    && git fetch -q --depth 1 https://github.com/ThunderVVV/HaWoR.git de9027265c0dd6abac42b11ce80f237107dd36ad \
    && git checkout -q FETCH_HEAD && rm -rf .git && chmod -R a+rX /opt/hawor

FROM ubuntu:22.04@sha256:98510ac07bb4545e389be502a8918b385c018bfe4ceeb5ac7e0375c782268b04
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates libegl1 libgl1 libgles2 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /usr/share/glvnd/egl_vendor.d \
    && echo '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}' \
        > /usr/share/glvnd/egl_vendor.d/10_nvidia.json
COPY --from=packages /opt/python /opt/python
COPY --from=packages --exclude=lib/python3.10/site-packages/nvidia /opt/venv /opt/venv
COPY --from=packages /opt/venv/lib/python3.10/site-packages/nvidia /opt/venv/lib/python3.10/site-packages/nvidia
COPY --from=packages /opt/hawor /opt/hawor
RUN mkdir -p /weights/hawor && ln -s /weights/hawor /opt/hawor/weights
ENV TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
ENV PATH=/opt/venv/bin:$PATH HOME=/tmp NVIDIA_REQUIRE_CUDA="cuda>=12.0" \
    PYOPENGL_PLATFORM=egl EGL_PLATFORM=device OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
WORKDIR /opt/hawor
