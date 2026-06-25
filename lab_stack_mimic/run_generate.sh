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
# inject our task registration + clean-success gating + source-usage provenance capture
# (clean_success_hook BEFORE provenance_hooks so provenance counts the gated successes;
# shared source untouched)
sed -i 's|^import isaaclab_mimic.envs.*|&\nimport lab_register\nimport clean_success_hook\nimport provenance_hooks|' "$PATCHED"
grep -q "^import lab_register" "$PATCHED" || { echo "ERROR: failed to inject lab_register import"; exit 1; }
grep -q "^import clean_success_hook" "$PATCHED" || { echo "ERROR: failed to inject clean_success_hook import"; exit 1; }
grep -q "^import provenance_hooks" "$PATCHED" || { echo "ERROR: failed to inject provenance_hooks import"; exit 1; }

# generation IK-rel scale: 0.5 (official; gentler than the teleop-faithful 1.0 → higher DGR)
export LAB_ARM_SCALE="${LAB_ARM_SCALE:-0.5}"
# provenance output: which source seed fed each subtask of each kept demo
export LAB_PROVENANCE_INPUT="$INPUT"
export LAB_PROVENANCE_OUT="${LAB_PROVENANCE_OUT:-${OUTPUT%.hdf5}.provenance.json}"

echo "[run_generate] group=$GROUP task=$TASK device=$DEVICE trials=$NUM_TRIALS num_envs=$NUM_ENVS scale=$LAB_ARM_SCALE"
echo "[run_generate] provenance -> $LAB_PROVENANCE_OUT"
PYTHONPATH="$HERE:${PYTHONPATH:-}" python "$PATCHED" \
  --task "$TASK" --headless --device "$DEVICE" --num_envs "$NUM_ENVS" \
  --generation_num_trials "$NUM_TRIALS" \
  --input_file "$INPUT" --output_file "$OUTPUT"
