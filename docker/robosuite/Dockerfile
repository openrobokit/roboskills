# syntax=docker/dockerfile:1.7-labs
# CaP-X Robosuite's simulator image (eval/serve.py): setup.sh's venv, with CaP-X's robosuite fork and its assets
# installed from git. Build tools stay in the first stage; the runtime uses the host's mounted roboskills package.
FROM ubuntu:22.04@sha256:98510ac07bb4545e389be502a8918b385c018bfe4ceeb5ac7e0375c782268b04 AS build
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates git build-essential \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.9.5@sha256:f459f6f73a8c4ef5d69f4e6fbbdb8af751d6fa40ec34b39a1ab469acd6e289b7 /uv /usr/local/bin/
ENV UV_PYTHON_INSTALL_DIR=/opt/python UV_LINK_MODE=copy MAX_JOBS=8 CMAKE_BUILD_PARALLEL_LEVEL=8
WORKDIR /opt/roboskills
COPY pyproject.toml ./
COPY src/roboskills/__init__.py src/roboskills/
COPY src/roboskills/eval/robosuite/ src/roboskills/eval/robosuite/
RUN --mount=type=cache,target=/root/.cache/uv HOME=/root \
    bash src/roboskills/eval/robosuite/setup.sh && chmod -R a+rX /opt

FROM ubuntu:22.04@sha256:98510ac07bb4545e389be502a8918b385c018bfe4ceeb5ac7e0375c782268b04
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates libegl1 libgl1 libgles2 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /usr/share/glvnd/egl_vendor.d \
    && echo '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}' \
        > /usr/share/glvnd/egl_vendor.d/10_nvidia.json
COPY --from=build /opt/python /opt/python
# Keep the CUDA libraries in their own layer, below the registry's per-layer limit.
ARG VENV=/opt/roboskills/src/roboskills/eval/robosuite/.venv
COPY --from=build --exclude=lib/python3.10/site-packages/nvidia $VENV $VENV
COPY --from=build $VENV/lib/python3.10/site-packages/nvidia $VENV/lib/python3.10/site-packages/nvidia
ENV PATH=/opt/roboskills/src/roboskills/eval/robosuite/.venv/bin:$PATH PYTHONPATH=/src HOME=/tmp \
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_PLATFORM=device NVIDIA_REQUIRE_CUDA="cuda>=12.0" \
    OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
WORKDIR /tmp
