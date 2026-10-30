#!/usr/bin/env bash
# Build one image, docker/<name>/Dockerfile, as ghcr.io/openrobokit/roboskills-<name>:<version> and push it. For
# maintainers: everyone else, and every runtime, pulls. A pushed version is never rebuilt over; bump it instead.
#
#   gh auth token | docker login ghcr.io -u <github user> --password-stdin   # a token holding write:packages
#   docker/build.sh libero 0.1.0
#   docker/build.sh --build-only libero 0.2.0   # validate locally before publishing
#   docker/build.sh robodojo 0.2.0   # every maintained image uses this repository as its context
#   docker/build.sh sam3 0.2.0 . --secret id=hf,src="$HOME/.cache/huggingface/token"   # facebook/sam3 is gated
#
# CONTEXT (default: the repository root) and any further arguments go to `docker build`. --network host: some dockerds
# resolve no names otherwise.
set -euo pipefail
publish=true
if [ "${1:-}" = --build-only ]; then publish=false; shift; fi
[ $# -ge 2 ] || { echo "usage: $0 [--build-only] NAME VERSION [CONTEXT [docker build args...]]" >&2; exit 2; }
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
name=$1 version=$2 context=${3:-$repo}
image=ghcr.io/openrobokit/roboskills-$name:$version
if docker manifest inspect "$image" >/dev/null 2>&1; then
  echo "$image is already pushed; bump the version" >&2
  exit 1
fi
docker build --network host -t "$image" -f "$repo/docker/$name/Dockerfile" "${@:4}" "$context"
if $publish; then docker push "$image"; fi
