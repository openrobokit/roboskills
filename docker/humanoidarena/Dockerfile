# syntax=docker/dockerfile:1.7
# The HumanoidArena simulator image: Isaac Sim 6.1 with Isaac Lab patched for 6.1 and the packages HumanoidArena's SONIC
# action provider imports. Adapted from RAX's docker/humanoid/Dockerfile (Apache-2.0, Copyright 2026 RAX contributors);
# cuRobo and the CUDA toolkit it compiles against are left out. Built from the repository root (docker/build.sh); the
# HumanoidArena checkout, its assets and the SONIC models are mounted at run time (launch.py), not copied.
ARG ISAACSIM_IMAGE=nvcr.io/nvidia/isaac-sim:6.1.0@sha256:0c16dd67d09a70ea474f2c809a13c4d09bd23c738184cf1bf28af487ecd39080
FROM ubuntu:22.04@sha256:98510ac07bb4545e389be502a8918b385c018bfe4ceeb5ac7e0375c782268b04 AS source
ARG ISAACLAB_URL=https://github.com/yuechen0614/IsaacLab.git
ARG ISAACLAB_COMMIT=afca7b09d60d8beb9c1cb28b43066499940b969b
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates git \
    && rm -rf /var/lib/apt/lists/*
COPY docker/humanoidarena/isaaclab-isaacsim61.patch /tmp/isaaclab.patch
RUN git init -q /opt/IsaacLab && git -C /opt/IsaacLab fetch -q --depth 1 "$ISAACLAB_URL" "$ISAACLAB_COMMIT" \
    && git -C /opt/IsaacLab checkout -q FETCH_HEAD \
    && git -C /opt/IsaacLab apply /tmp/isaaclab.patch

FROM ${ISAACSIM_IMAGE}
USER root
SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PRIVACY_CONSENT=Y \
    OMNI_KIT_ALLOW_ROOT=1 PIP_USER=0 PYTHONNOUSERSITE=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates libglu1-mesa libglib2.0-0 \
        libxrender1 libsm6 libice6 libxrandr2 libxi6 libxcursor1 libxinerama1 libvulkan1 \
    && rm -rf /var/lib/apt/lists/*
COPY docker/humanoidarena/requirements.txt /tmp/runtime-requirements.txt
# Keep CUDA/PyTorch below the registry's per-layer size limit; do not resolve or upgrade other packages here.
RUN grep -E '^(torch|torchvision|torchaudio|triton|nvidia-[a-z0-9-]+|cuda-[a-z-]+)==' /tmp/runtime-requirements.txt \
        >/tmp/torch-requirements.txt \
    && /isaac-sim/python.sh -m pip install --no-cache-dir --no-deps \
        --extra-index-url https://download.pytorch.org/whl/cu128 -r /tmp/torch-requirements.txt \
    && rm /tmp/torch-requirements.txt
RUN /isaac-sim/python.sh -m pip install --no-cache-dir --no-deps \
        --extra-index-url https://download.pytorch.org/whl/cu128 -r /tmp/runtime-requirements.txt \
    && rm /tmp/runtime-requirements.txt
COPY --from=source /opt/IsaacLab/source /opt/IsaacLab/source
COPY --from=source /opt/IsaacLab/apps /opt/IsaacLab/apps
COPY --from=source /opt/IsaacLab/VERSION /opt/IsaacLab/LICENSE /opt/IsaacLab/LICENSE-mimic /opt/IsaacLab/
RUN ln -sfn /isaac-sim /opt/IsaacLab/_isaac_sim \
    && for p in isaaclab isaaclab_assets isaaclab_contrib isaaclab_mimic isaaclab_rl isaaclab_tasks; do \
        /isaac-sim/python.sh -m pip install --no-cache-dir --no-deps --no-build-isolation \
            -e /opt/IsaacLab/source/$p || exit 1; \
    done
RUN /isaac-sim/python.sh -c "\
import onnxruntime as ort, msgpack, numpy, torch, zmq, redis, toml; \
assert 'CUDAExecutionProvider' in ort.get_available_providers(), ort.get_available_providers(); \
print('HUMANOID_IMAGE_OK', ort.__version__, torch.__version__, numpy.__version__)"
ENTRYPOINT ["/bin/bash"]
