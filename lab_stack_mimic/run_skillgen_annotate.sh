#!/usr/bin/env bash
# Auto-annotate the canonical lab 3-cube demos for SKILLGEN, inside the Docker container
# (isaac-lab-skillgen) that has the pinned NVIDIA cuRobo. Unlike the MimicGen path this also
# records per-subtask START signals (--annotate_subtask_start_signals), which SkillGen needs to
# split each subtask into a cuRobo-planned free-space transition + a replayed contact skill.
#
# We don't edit the container's Isaac Lab source: copy annotate_demos.py to /tmp and inject one
# `import skillgen_register` line after `import isaaclab_mimic.envs` (registers our FR3 SkillGen
# task, installs the warp.torch shim, and patches cuRobo's from_task_name to the FR3 config).
#
# Usage (run inside the container):
#   bash run_skillgen_annotate.sh <input.hdf5> <output.hdf5> [device]
set -eu

HERE=/workspace/skillgen_work/lab_stack_mimic
R=/workspace/isaaclab
TASK="Isaac-Stack-Cube-LabFR3-Skillgen-IK-Rel-v0"

INPUT="${1:?usage: run_skillgen_annotate.sh <input.hdf5> <output.hdf5> [device]}"
OUTPUT="${2:?need output.hdf5}"
DEVICE="${3:-cuda}"

PATCHED=/tmp/annotate_skillgen.py
cp "$R/scripts/imitation_learning/isaaclab_mimic/annotate_demos.py" "$PATCHED"
sed -i 's|^import isaaclab_mimic.envs.*|&\nimport skillgen_register|' "$PATCHED"
grep -q "^import skillgen_register" "$PATCHED" || { echo "ERROR: failed to inject skillgen_register"; exit 1; }

# Fix an upstream bug: the auto-mode START-signal check calls torch.any() on a raw python
# list (the term-signal loop converts to a tensor first, the start loop forgot to). Insert the
# same tensor conversion so --annotate_subtask_start_signals works in auto mode.
sed -i '/subtask_start_signal_dict\.items():/a\
                signal_flags = torch.tensor(signal_flags, device=env.device)' "$PATCHED"
grep -q "signal_flags = torch.tensor(signal_flags, device=env.device)" "$PATCHED" || { echo "ERROR: start-signal tensor patch failed"; exit 1; }

export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONUNBUFFERED=1
export PYTHONPATH="$HERE:${PYTHONPATH:-}"
# lab table USD lives on jake's host path by default; in the container use the copied asset.
export LAB_TABLE_USD="${LAB_TABLE_USD:-/workspace/skillgen_work/assets/table_scene.usdc}"

echo "[run_skillgen_annotate] task=$TASK device=$DEVICE input=$INPUT output=$OUTPUT"
cd "$R"
./isaaclab.sh -p "$PATCHED" \
  --task "$TASK" --auto --annotate_subtask_start_signals --headless --device "$DEVICE" \
  --input_file "$INPUT" --output_file "$OUTPUT"
