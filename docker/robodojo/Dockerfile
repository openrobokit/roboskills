# syntax=docker/dockerfile:1.7
# Fetch only the pinned Isaac Lab source; Git and its cache stay in the builder.
FROM ubuntu:22.04@sha256:98510ac07bb4545e389be502a8918b385c018bfe4ceeb5ac7e0375c782268b04 AS source
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates git \
    && rm -rf /var/lib/apt/lists/*
RUN git init -q /opt/IsaacLab && git -C /opt/IsaacLab fetch -q --depth 1 \
        https://github.com/yuechen0614/IsaacLab.git afca7b09d60d8beb9c1cb28b43066499940b969b \
    && git -C /opt/IsaacLab checkout -q FETCH_HEAD

FROM nvcr.io/nvidia/isaac-sim@sha256:f3563cb2ba0c18af0b2fb321360dcb73a917b899f879e3213623d6bee484fa54
USER root
COPY --from=source /opt/IsaacLab/source /opt/robodojo-deps/IsaacLab/source
COPY --from=source /opt/IsaacLab/apps /opt/robodojo-deps/IsaacLab/apps
COPY --from=source /opt/IsaacLab/VERSION /opt/IsaacLab/LICENSE /opt/IsaacLab/LICENSE-mimic /opt/robodojo-deps/IsaacLab/
COPY docker/robodojo/requirements.txt /tmp/runtime-requirements.txt
# Only additions and replacements relative to this exact vendor image, at the previous release's versions.
RUN /isaac-sim/python.sh -m pip install --no-cache-dir --no-deps -r /tmp/runtime-requirements.txt \
    && L=/opt/robodojo-deps/IsaacLab/source && P="/isaac-sim/python.sh -m pip install --no-cache-dir --no-deps --no-build-isolation" \
    && $P -e $L/isaaclab -e $L/isaaclab_assets -e $L/isaaclab_tasks \
    && rm /tmp/runtime-requirements.txt
RUN mkdir -p /isaac-sim/kit/cache /isaac-sim/kit/data /isaac-sim/kit/logs /var/tmp/OptixCache_robodojo \
    && chmod -R a+rwX /isaac-sim/kit/cache /isaac-sim/kit/data /isaac-sim/kit/logs /var/tmp/OptixCache_robodojo \
    && chmod o+rx /isaac-sim
