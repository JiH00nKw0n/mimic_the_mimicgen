#!/usr/bin/env bash
# Generate synthetic demos with SKILLGEN (MimicGen + cuRobo motion-planned transitions) from an
# annotated lab dataset, inside the Docker container with the pinned NVIDIA cuRobo.
#
# Injects `import skillgen_register` after `import isaaclab_mimic.envs` (registers the FR3 SkillGen
# task, installs the warp.torch shim, and routes cuRobo's from_task_name to the FR3 planner config).
# For the full run we also inject clean_success_hook + provenance_hooks (set LAB_SKILLGEN_FULL=1) so
# only clean canonical stacks count as success and source-usage is recorded, matching the vanilla run.
#
# Usage (run inside the container):
#   bash run_skillgen_generate.sh <annotated.hdf5> <generated.hdf5> [num_trials] [num_envs] [device]
set -eu

HERE=/workspace/skillgen_work/lab_stack_mimic
R=/workspace/isaaclab
TASK="Isaac-Stack-Cube-LabFR3-Skillgen-IK-Rel-v0"

INPUT="${1:?usage: run_skillgen_generate.sh <annotated> <generated> [num_trials] [num_envs] [device]}"
OUTPUT="${2:?need generated.hdf5}"
NUM_TRIALS="${3:-5}"
NUM_ENVS="${4:-1}"
DEVICE="${5:-cuda}"

PATCHED=/tmp/generate_skillgen.py
cp "$R/scripts/imitation_learning/isaaclab_mimic/generate_dataset.py" "$PATCHED"
INJECT='import skillgen_register'
if [ "${LAB_SKILLGEN_FULL:-0}" = "1" ]; then
  INJECT='import skillgen_register\nimport clean_success_hook\nimport provenance_hooks'
  export LAB_PROVENANCE_INPUT="$INPUT"
  export LAB_PROVENANCE_OUT="${LAB_PROVENANCE_OUT:-${OUTPUT%.hdf5}.provenance.json}"
fi
sed -i "s|^import isaaclab_mimic.envs.*|&\n${INJECT}|" "$PATCHED"
grep -q "^import skillgen_register" "$PATCHED" || { echo "ERROR: failed to inject skillgen_register"; exit 1; }

# IK-rel scale for the replayed contact-skill segments (cuRobo plans the transitions itself).
# 0.5 matches the proven vanilla generation regime (gentler deltas -> higher success).
export LAB_ARM_SCALE="${LAB_ARM_SCALE:-0.5}"
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONUNBUFFERED=1
export PYTHONPATH="$HERE:${PYTHONPATH:-}"
# lab table USD lives on jake's host path by default; in the container use the copied asset.
export LAB_TABLE_USD="${LAB_TABLE_USD:-/workspace/skillgen_work/assets/table_scene.usdc}"

echo "[run_skillgen_generate] task=$TASK device=$DEVICE trials=$NUM_TRIALS num_envs=$NUM_ENVS scale=$LAB_ARM_SCALE full=${LAB_SKILLGEN_FULL:-0}"
cd "$R"
./isaaclab.sh -p "$PATCHED" \
  --task "$TASK" --use_skillgen --headless --device "$DEVICE" --num_envs "$NUM_ENVS" \
  --generation_num_trials "$NUM_TRIALS" \
  --input_file "$INPUT" --output_file "$OUTPUT"
