# syntax=docker/dockerfile:1.7-labs
# robocasa365: pinned setup in a builder; only its Python environment and runtime assets ship.
FROM ubuntu:22.04@sha256:98510ac07bb4545e389be502a8918b385c018bfe4ceeb5ac7e0375c782268b04 AS build
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates git build-essential curl \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.9.5@sha256:f459f6f73a8c4ef5d69f4e6fbbdb8af751d6fa40ec34b39a1ab469acd6e289b7 /uv /usr/local/bin/
ENV UV_PYTHON_INSTALL_DIR=/opt/python UV_LINK_MODE=copy MAX_JOBS=8 CMAKE_BUILD_PARALLEL_LEVEL=8 \
    OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
WORKDIR /opt/roboskills
COPY pyproject.toml ./
COPY src/roboskills/__init__.py src/roboskills/
COPY src/roboskills/eval/robocasa365/setup.sh src/roboskills/eval/robocasa365/pyproject.toml \
    src/roboskills/eval/robocasa365/uv.lock src/roboskills/eval/robocasa365/
RUN --mount=type=cache,target=/root/.cache/uv HOME=/root \
    ROBOCASA=/opt/roboskills/runs/vendor/robocasa bash src/roboskills/eval/robocasa365/setup.sh \
    && rm -rf runs/vendor/robocasa/.git && chmod -R a+rX /opt
# RoboCasa creates temporary XML beside its assets in the container writable layer.
RUN find runs/vendor/robocasa/robocasa/models/assets -type d -exec chmod a+w {} +

FROM ubuntu:22.04@sha256:98510ac07bb4545e389be502a8918b385c018bfe4ceeb5ac7e0375c782268b04
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates libegl1 libgl1 libgles2 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /usr/share/glvnd/egl_vendor.d \
    && echo '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}' \
        > /usr/share/glvnd/egl_vendor.d/10_nvidia.json
COPY --from=build /opt/python /opt/python
# Keep the CUDA libraries in their own layer, below the registry's per-layer limit.
ARG VENV=/opt/roboskills/src/roboskills/eval/robocasa365/.venv
COPY --from=build --exclude=lib/python3.11/site-packages/nvidia $VENV $VENV
COPY --from=build $VENV/lib/python3.11/site-packages/nvidia $VENV/lib/python3.11/site-packages/nvidia
# Split bundled kitchen assets into layers below the registry limit.
ARG RC=/opt/roboskills/runs/vendor/robocasa
ARG ASSETS=$RC/robocasa/models/assets
COPY --from=build --exclude=robocasa/models/assets $RC $RC
COPY --from=build --exclude=objects $ASSETS $ASSETS
COPY --from=build --exclude=aigen_objs $ASSETS/objects $ASSETS/objects
COPY --from=build --exclude=[i-z]* $ASSETS/objects/aigen_objs $ASSETS/objects/aigen_objs
COPY --from=build --exclude=[a-h]* $ASSETS/objects/aigen_objs $ASSETS/objects/aigen_objs
ENV PATH=/opt/roboskills/src/roboskills/eval/robocasa365/.venv/bin:$PATH PYTHONPATH=/src HOME=/tmp ROBOCASA=/opt/roboskills/runs/vendor/robocasa \
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_PLATFORM=device NVIDIA_REQUIRE_CUDA="cuda>=12.0" \
    OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
WORKDIR /tmp
