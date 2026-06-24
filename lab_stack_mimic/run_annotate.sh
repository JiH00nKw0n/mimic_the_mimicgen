#!/usr/bin/env bash
# Auto-annotate the lab 3-cube demos with the official annotate_demos.py, with our lab
# FR3 Mimic task registered (forward or reverse order). We DON'T edit the colleague's
# Isaac Lab source: we copy annotate_demos.py to /tmp and inject one `import lab_register`
# line after `import isaaclab_mimic.envs` (which runs after Isaac Sim has launched).
#
# Usage:
#   bash run_annotate.sh <fwd|rev> <input.hdf5> <output.hdf5> [device] [extra args]
#     device defaults to cpu; pass cuda to run on GPU.
set -eu

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GROUP="${1:?usage: run_annotate.sh <fwd|rev> <input> <output> [device]}"
INPUT="${2:?need input.hdf5}"
OUTPUT="${3:?need output.hdf5}"
DEVICE="${4:-cpu}"
shift 4 2>/dev/null || shift $#

case "$GROUP" in
  fwd) TASK="Isaac-Stack-Cube-LabFR3-Fwd-IK-Rel-Mimic-v0" ;;
  rev) TASK="Isaac-Stack-Cube-LabFR3-Rev-IK-Rel-Mimic-v0" ;;
  *)   echo "ERROR: group must be 'fwd' or 'rev', got '$GROUP'"; exit 1 ;;
esac

R=/home/ubuntu/jake/UWLab/_isaaclab/IsaacLab
cd "$R"
export OMNI_KIT_ACCEPT_EULA=YES
export UV_CACHE_DIR=/home/ubuntu/jake/.uv-cache
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH=/home/ubuntu/jake/syslibs/extracted/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}
source /home/ubuntu/jake/env_uwlab/bin/activate

PATCHED=/tmp/annotate_lab.py
cp "$R/scripts/imitation_learning/isaaclab_mimic/annotate_demos.py" "$PATCHED"
sed -i 's|^import isaaclab_mimic.envs.*|&\nimport lab_register|' "$PATCHED"
grep -q "^import lab_register" "$PATCHED" || { echo "ERROR: failed to inject lab_register import"; exit 1; }

echo "[run_annotate] group=$GROUP task=$TASK device=$DEVICE input=$INPUT output=$OUTPUT"
PYTHONPATH="$HERE:${PYTHONPATH:-}" python "$PATCHED" \
  --task "$TASK" --auto --headless --device "$DEVICE" \
  --input_file "$INPUT" --output_file "$OUTPUT" "$@"
