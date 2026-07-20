#!/usr/bin/env bash
# Run peg generation while logging GPU utilization + memory to a CSV, so you can size num_envs to
# actually saturate the L40S rather than leaving it idle. nvidia-smi runs on the HOST (the docker
# --gpus all container shares the device), sampled every 2 s for the lifetime of the run.
#
# Usage:
#   LAB_NUM_ENVS=16 NUM_TRIALS=200 bash gpu_saturate.sh
#   bash gpu_saturate.sh                      # defaults below
#
# Knobs (env vars):
#   LAB_NUM_ENVS  parallel envs passed to run_peg_generate.sh   (default 8)
#   NUM_TRIALS    clean demos to keep                           (default 100)
#   DEVICE        cuda|cpu                                       (default cuda)
#   INPUT/OUTPUT  hdf5 paths (relative to this dir)             (defaults: annotated -> generated)
#   LOG           where to write the utilization CSV            (default /tmp/peg_gpu_util.csv)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAB_NUM_ENVS="${LAB_NUM_ENVS:-8}"
NUM_TRIALS="${NUM_TRIALS:-100}"
DEVICE="${DEVICE:-cuda}"
INPUT="${INPUT:-datasets/peg_annotated.hdf5}"
OUTPUT="${OUTPUT:-datasets/peg_generated.hdf5}"
LOG="${LOG:-/tmp/peg_gpu_util.csv}"

echo "[gpu_saturate] num_envs=$LAB_NUM_ENVS trials=$NUM_TRIALS device=$DEVICE -> util log: $LOG"

# start the sampler in the background; stop it when this script exits (normally or on error).
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total --format=csv -l 2 > "$LOG" &
SMI_PID=$!
cleanup() { kill "$SMI_PID" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

# num_envs is the 5th positional arg of run_peg_generate.sh.
bash "$HERE/run_peg_generate.sh" "$INPUT" "$OUTPUT" "$DEVICE" "$NUM_TRIALS" "$LAB_NUM_ENVS"

echo "[gpu_saturate] done. GPU utilization samples -> $LOG"
echo "[gpu_saturate] peak: $(awk -F',' 'NR>1{gsub(/ %/,"",$2); if($2+0>m)m=$2+0} END{print m" % GPU"}' "$LOG" 2>/dev/null || echo n/a)"
