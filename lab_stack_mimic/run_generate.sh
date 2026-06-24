#!/usr/bin/env bash
# Generate synthetic demos with MimicGen from an annotated lab dataset (fwd or rev),
# with our lab FR3 Mimic task registered (copy generate_dataset.py + inject lab_register;
# the colleague's Isaac Lab source is left untouched).
#
# Usage:
#   bash run_generate.sh <fwd|rev> <annotated.hdf5> <generated.hdf5> [device] [num_trials] [num_envs]
set -eu

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GROUP="${1:?usage: run_generate.sh <fwd|rev> <annotated> <generated> [device] [num_trials] [num_envs]}"
INPUT="${2:?need annotated.hdf5}"
OUTPUT="${3:?need generated.hdf5}"
DEVICE="${4:-cpu}"
NUM_TRIALS="${5:-10}"
NUM_ENVS="${6:-4}"

case "$GROUP" in
  fwd) TASK="Isaac-Stack-Cube-LabFR3-Fwd-IK-Rel-Mimic-v0" ;;
  rev) TASK="Isaac-Stack-Cube-LabFR3-Rev-IK-Rel-Mimic-v0" ;;
  *)   echo "ERROR: group must be 'fwd' or 'rev'"; exit 1 ;;
esac

R=/home/ubuntu/jake/UWLab/_isaaclab/IsaacLab
cd "$R"
export OMNI_KIT_ACCEPT_EULA=YES
export UV_CACHE_DIR=/home/ubuntu/jake/.uv-cache
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH=/home/ubuntu/jake/syslibs/extracted/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}
source /home/ubuntu/jake/env_uwlab/bin/activate

PATCHED=/tmp/generate_lab.py
cp "$R/scripts/imitation_learning/isaaclab_mimic/generate_dataset.py" "$PATCHED"
sed -i 's|^import isaaclab_mimic.envs.*|&\nimport lab_register|' "$PATCHED"
grep -q "^import lab_register" "$PATCHED" || { echo "ERROR: failed to inject lab_register import"; exit 1; }

echo "[run_generate] group=$GROUP task=$TASK device=$DEVICE trials=$NUM_TRIALS num_envs=$NUM_ENVS"
PYTHONPATH="$HERE:${PYTHONPATH:-}" python "$PATCHED" \
  --task "$TASK" --headless --device "$DEVICE" --num_envs "$NUM_ENVS" \
  --generation_num_trials "$NUM_TRIALS" \
  --input_file "$INPUT" --output_file "$OUTPUT"
