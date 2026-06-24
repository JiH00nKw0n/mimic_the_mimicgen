#!/usr/bin/env bash
# Run replay_count.py inside the UWLab native Isaac Lab env on the GPU server.
# (The lab FR3 / desk assets + UWLab live here, NOT in the isaac-lab-base docker.)
#
# Usage:
#   bash run_replay.sh <dataset.hdf5> [extra args for replay_count.py]
# Example:
#   bash run_replay.sh /home/ubuntu/jake/aidas/3cube_stack/datasets/teleop_dataset_success.hdf5 \
#                       --report /tmp/replay_success.json
set -eu

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET="${1:?usage: run_replay.sh <dataset.hdf5> [extra args]}"; shift || true

# Same runtime env as aidas/3cube_stack/run_live.sh
cd /home/ubuntu/jake/UWLab
export OMNI_KIT_ACCEPT_EULA=YES
export UV_CACHE_DIR=/home/ubuntu/jake/.uv-cache
export LD_LIBRARY_PATH=/home/ubuntu/jake/syslibs/extracted/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}
export PYTHONUNBUFFERED=1   # so per-demo progress shows up live in piped logs
source /home/ubuntu/jake/env_uwlab/bin/activate

python "$HERE/replay_count.py" --dataset_file "$DATASET" "$@"
