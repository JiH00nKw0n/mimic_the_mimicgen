#!/usr/bin/env bash
# Auto-annotate the lab peg-insert demos with the official isaaclab_mimic annotate_demos.py,
# with our lab FR3 peg Mimic task registered. Runs INSIDE the isaac-lab 3.0 container (the peg
# demos were recorded there). We DON'T edit the container's Isaac Lab source: we copy
# annotate_demos.py to /tmp inside the container and inject one `import peg_register` line after
# `import isaaclab_mimic.envs` (which runs after Isaac Sim has launched).
#
# Usage:
#   bash run_peg_annotate.sh [input.hdf5] [output.hdf5] [device] [extra args...]
#     input   defaults to datasets/peg_source.hdf5   (paths are relative to THIS dir = /peg in-container)
#     output  defaults to datasets/peg_annotated.hdf5
#     device  defaults to cuda
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT="${1:-datasets/peg_source.hdf5}"
OUTPUT="${2:-datasets/peg_annotated.hdf5}"
DEVICE="${3:-cuda}"
shift 3 2>/dev/null || shift $# 2>/dev/null || true
EXTRA="$*"

IMAGE="${IMAGE:-nvcr.io/nvidia/isaac-lab:3.0.0-beta2-post1}"
CACHE="${CACHE:-/home/ubuntu/docker/isaac-sim}"
# Where peg_hole_env.usd / hole_01.usd live on the HOST (mounted at /work/assets in-container,
# matching the cfg's LAB_PEG_ENV_USD / LAB_PEG_HOLE_USD defaults). Override for your box.
ASSETS_DIR="${ASSETS_DIR:-/home/ubuntu/peg_in_hole/assets}"
TASK="Isaac-PegInsert-LabFR3-IK-Rel-Mimic-v0"

mkdir -p "$CACHE/cache/kit" "$CACHE/cache/ov"
TTY_FLAGS=""; [ -t 0 ] && TTY_FLAGS="-it"

echo "[run_peg_annotate] task=$TASK device=$DEVICE input=$INPUT output=$OUTPUT assets=$ASSETS_DIR"

docker run --rm $TTY_FLAGS --gpus all \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y -e OMNI_KIT_ACCEPT_EULA=YES \
  -e TASK="$TASK" -e DEVICE="$DEVICE" -e INPUT="$INPUT" -e OUTPUT="$OUTPUT" -e EXTRA="$EXTRA" \
  -e LAB_PEG_ENV_USD=/work/assets/peg_hole_env.usd -e LAB_PEG_HOLE_USD=/work/assets/hole_01.usd \
  -v "$HERE":/peg \
  -v "$ASSETS_DIR":/work/assets:ro \
  -v "$CACHE/cache/kit":/isaac-sim/kit/cache:rw \
  -v "$CACHE/cache/ov":/root/.cache/ov:rw \
  --entrypoint bash "$IMAGE" -lc '
    set -e
    ISAACLAB=/workspace/isaaclab
    MIMIC=$ISAACLAB/scripts/imitation_learning/isaaclab_mimic
    P=/tmp/annotate_peg.py
    cp "$MIMIC/annotate_demos.py" "$P"
    sed -i "s|^import isaaclab_mimic.envs.*|&\nimport peg_register|" "$P"
    grep -q "^import peg_register" "$P" || { echo "ERROR: failed to inject peg_register import"; exit 1; }
    export PYTHONPATH=/peg:${PYTHONPATH:-}
    "$ISAACLAB/isaaclab.sh" -p "$P" \
      --task "$TASK" --auto --headless --device "$DEVICE" \
      --input_file "/peg/$INPUT" --output_file "/peg/$OUTPUT" $EXTRA
  '
