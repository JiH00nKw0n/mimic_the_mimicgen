#!/bin/bash
# Isaac Lab Mimic generation for the LabFR3 stack task on aidas (docker) —
# faithful port of lab_stack_mimic/run_generate.sh (arpa/UWLab prologue
# replaced by the isaac-lab container; table USD falls back to the slab).
#
#   ./run_lab_generate_docker.sh <fwd|rev> <annotated(container path)> \
#       <generated(container path)> [device=cpu] [num_trials=10] [num_envs=4]
set -eu
GROUP="${1:?fwd|rev}"; INPUT="${2:?annotated}"; OUTPUT="${3:?generated}"
DEVICE="${4:-cpu}"; NUM_TRIALS="${5:-10}"; NUM_ENVS="${6:-4}"
case "$GROUP" in
  fwd) TASK="Isaac-Stack-Cube-LabFR3-Fwd-IK-Rel-Mimic-v0" ;;
  rev) TASK="Isaac-Stack-Cube-LabFR3-Rev-IK-Rel-Mimic-v0" ;;
  *) echo "group must be fwd|rev"; exit 1 ;;
esac
IMAGE="nvcr.io/nvidia/isaac-lab:3.0.0-beta2-post1"
REPO=/home/ubuntu/mimicgen_jihoonkwon/mimic_the_mimicgen
mkdir -p /home/ubuntu/contract_out
docker run --rm --gpus all --network host \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y -e OMNI_KIT_ACCEPT_EULA=YES \
  -e PYTHONUNBUFFERED=1 \
  -e LAB_ARM_SCALE="${LAB_ARM_SCALE:-0.5}" \
  -e LAB_TABLE_USD=/nonexistent.usdc \
  -e LAB_KEEP_FAILED="${LAB_KEEP_FAILED:-1}" \
  -v "$REPO":/repo \
  -v /home/ubuntu/rl_demos:/rl_demos \
  -v /home/ubuntu/contract_out:/out \
  -v /home/ubuntu/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
  -v /home/ubuntu/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
  --entrypoint bash "$IMAGE" -c '
set -eu
PATCHED=/tmp/generate_lab.py
cp /workspace/isaaclab/scripts/imitation_learning/isaaclab_mimic/generate_dataset.py "$PATCHED"
sed -i "s|^import isaaclab_mimic.envs.*|&\nimport lab_register\nimport clean_success_hook\nimport provenance_hooks|" "$PATCHED"
grep -q "^import lab_register" "$PATCHED"
export LAB_PROVENANCE_INPUT="'"$INPUT"'"
export LAB_PROVENANCE_OUT="'"${OUTPUT%.hdf5}.provenance.json"'"
export PYTHONPATH=/repo/lab_stack_mimic:${PYTHONPATH:-}
/workspace/isaaclab/isaaclab.sh -p "$PATCHED" \
  --task "'"$TASK"'" --headless --device "'"$DEVICE"'" --num_envs "'"$NUM_ENVS"'" \
  --generation_num_trials "'"$NUM_TRIALS"'" \
  --input_file "'"$INPUT"'" --output_file "'"$OUTPUT"'"
'
