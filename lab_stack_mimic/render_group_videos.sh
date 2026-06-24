#!/usr/bin/env bash
# Render grouped demo videos: ORIGINAL (recorded states) and REPLAY (actions, fixed
# dataset) for the fwd-success / rev-success / both-fail groups, in two Isaac sessions.
set -eu
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R=/home/ubuntu/jake/UWLab/_isaaclab/IsaacLab
D=/home/ubuntu/jake/aidas/3cube_stack/datasets

G="fwd_success=demo_1,demo_2,demo_3,demo_6,demo_9,demo_12,demo_13,demo_14,demo_15,demo_16,demo_19,demo_20,demo_21;rev_success=demo_4,demo_5,demo_8,demo_10,demo_18,demo_23,demo_24,demo_25,demo_26,demo_27,demo_28;fail=demo_0,demo_7,demo_11,demo_17,demo_22"

cd "$R"
export OMNI_KIT_ACCEPT_EULA=YES
export UV_CACHE_DIR=/home/ubuntu/jake/.uv-cache
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH=/home/ubuntu/jake/syslibs/extracted/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}
source /home/ubuntu/jake/env_uwlab/bin/activate

echo "=== ORIGINAL (recorded states) ==="
PYTHONPATH="$HERE" python "$HERE/record_video.py" --mode states \
  --dataset "$D/teleop_dataset_success.hdf5" --groups "$G" --out_prefix /tmp/demo --device cpu

echo "=== REPLAY (actions, fixed start) ==="
PYTHONPATH="$HERE" python "$HERE/record_video.py" --mode replay \
  --dataset "$D/teleop_dataset_fixed.hdf5" --groups "$G" --out_prefix /tmp/replay --device cpu

echo ALLDONE
