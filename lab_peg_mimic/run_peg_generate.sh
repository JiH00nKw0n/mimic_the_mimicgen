#!/usr/bin/env bash
# Generate synthetic peg-insert demos with MimicGen from an annotated lab dataset, with our lab
# FR3 peg Mimic task registered. Runs INSIDE the isaac-lab 3.0 container (copy generate_dataset.py
# + inject `import peg_register`; the container's Isaac Lab source is left untouched).
#
# Usage:
#   bash run_peg_generate.sh [annotated.hdf5] [generated.hdf5] [device] [num_trials] [num_envs]
#     annotated  defaults to datasets/peg_annotated.hdf5   (relative to THIS dir = /peg in-container)
#     generated  defaults to datasets/peg_generated.hdf5
#     device     defaults to cuda
#     num_trials defaults to 10   (number of CLEAN demos to keep; generation_guarantee retries failures)
#     num_envs   defaults to 4    (parallel envs; raise to saturate the GPU — see gpu_saturate.sh)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT="${1:-datasets/peg_annotated.hdf5}"
OUTPUT="${2:-datasets/peg_generated.hdf5}"
DEVICE="${3:-cuda}"
NUM_TRIALS="${4:-10}"
NUM_ENVS="${5:-4}"

IMAGE="${IMAGE:-nvcr.io/nvidia/isaac-lab:3.0.0-beta2-post1}"
CACHE="${CACHE:-/home/ubuntu/docker/isaac-sim}"
ASSETS_DIR="${ASSETS_DIR:-/home/ubuntu/peg_in_hole/assets}"
TASK="Isaac-PegInsert-LabFR3-IK-Rel-Mimic-v0"

# generation IK-rel scale 0.5 (gentler than the teleop-faithful 1.0; avoids the FR3 wrist
# singularity that pins DGR at 0% — same fix as the stack). Override with LAB_ARM_SCALE.
ARM_SCALE="${LAB_ARM_SCALE:-0.5}"

mkdir -p "$CACHE/cache/kit" "$CACHE/cache/ov"
TTY_FLAGS=""; [ -t 0 ] && TTY_FLAGS="-it"

echo "[run_peg_generate] task=$TASK device=$DEVICE trials=$NUM_TRIALS num_envs=$NUM_ENVS scale=$ARM_SCALE"
echo "[run_peg_generate] input=$INPUT output=$OUTPUT assets=$ASSETS_DIR"

docker run --rm $TTY_FLAGS --gpus all \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y -e OMNI_KIT_ACCEPT_EULA=YES \
  -e TASK="$TASK" -e DEVICE="$DEVICE" -e INPUT="$INPUT" -e OUTPUT="$OUTPUT" \
  -e NUM_TRIALS="$NUM_TRIALS" -e NUM_ENVS="$NUM_ENVS" \
  -e LAB_ARM_SCALE="$ARM_SCALE" \
  -e LAB_PEG_ENV_USD=/work/assets/peg_hole_env.usd -e LAB_PEG_HOLE_USD=/work/assets/hole_01.usd \
  -v "$HERE":/peg \
  -v "$ASSETS_DIR":/work/assets:ro \
  -v "$CACHE/cache/kit":/isaac-sim/kit/cache:rw \
  -v "$CACHE/cache/ov":/root/.cache/ov:rw \
  --entrypoint bash "$IMAGE" -lc '
    set -e
    ISAACLAB=/workspace/isaaclab
    MIMIC=$ISAACLAB/scripts/imitation_learning/isaaclab_mimic
    P=/tmp/generate_peg.py
    cp "$MIMIC/generate_dataset.py" "$P"
    sed -i "s|^import isaaclab_mimic.envs.*|&\nimport peg_register|" "$P"
    grep -q "^import peg_register" "$P" || { echo "ERROR: failed to inject peg_register import"; exit 1; }
    export PYTHONPATH=/peg:${PYTHONPATH:-}
    "$ISAACLAB/isaaclab.sh" -p "$P" \
      --task "$TASK" --headless --device "$DEVICE" --num_envs "$NUM_ENVS" \
      --generation_num_trials "$NUM_TRIALS" \
      --input_file "/peg/$INPUT" --output_file "/peg/$OUTPUT"
  '
