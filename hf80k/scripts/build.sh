#!/usr/bin/env bash
# Build the hf80k image.
#
#   ./scripts/build.sh
#   IMAGE_TAG=fr3-hf80k:2026-08-12 ./scripts/build.sh
#   NO_CACHE=1 ./scripts/build.sh          # force a clean pip/apt layer
#
# The build context is the repository root, because the Dockerfile copies
# hf80k/src and hf80k/assets. docker/Dockerfile.dockerignore trims that context
# back to hf80k/ — BuildKit reads <dockerfile>.dockerignore in preference to the
# context root's .dockerignore, which lets us do this without adding a file
# outside hf80k/. That is also why DOCKER_BUILDKIT=1 is not optional here.
set -euo pipefail

IMAGE_TAG="${IMAGE_TAG:-fr3-hf80k:latest}"
HF80K_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONTEXT="$(cd "$HF80K_DIR/.." && pwd)"

command -v docker > /dev/null 2>&1 || { echo "docker not on PATH" >&2; exit 1; }

# Assets are fetched, not committed. Building without them yields an image that
# dies in preflight on the first chunk, an hour after someone walked away.
missing=0
for p in assets/fwd_annotated.hdf5 \
         assets/fr3_cube_system_calibration_bundle_v1 \
         assets/fr3_visual_randomization_v1 ; do
  if [[ ! -e "$HF80K_DIR/$p" ]]; then
    echo "missing $p" >&2
    missing=1
  fi
done
if [[ "$missing" -ne 0 ]]; then
  echo "run ./scripts/fetch_assets.sh <host-or-path> first" >&2
  exit 1
fi

echo "building $IMAGE_TAG"
echo "  dockerfile: $HF80K_DIR/docker/Dockerfile"
echo "  context:    $CONTEXT (trimmed by docker/Dockerfile.dockerignore)"

DOCKER_BUILDKIT=1 docker build \
  -f "$HF80K_DIR/docker/Dockerfile" \
  -t "$IMAGE_TAG" \
  ${NO_CACHE:+--no-cache} \
  --progress "${BUILD_PROGRESS:-auto}" \
  "$CONTEXT"

echo
docker image ls "$IMAGE_TAG"
echo
echo "next: cp .env.example .env, fill HF_TOKEN and HF_REPO_ID, then ./scripts/run.sh"
