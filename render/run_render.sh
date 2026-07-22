#!/usr/bin/env bash
# Render demos from the 4 calibrated FR3 camera viewpoints (arpa, UWLab env).
# Usage:
#   bash run_render.sh <dataset.hdf5> [extra args for render_viewpoints.py]
# Example:
#   bash run_render.sh /home/ubuntu/jake/aidas/3cube_stack/datasets/random_generated_2000_FINAL.hdf5 \
#       --count 25 --preview_video 2
set -eu

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET="${1:?usage: run_render.sh <dataset.hdf5> [extra args]}"; shift || true
# absolutize BEFORE any cd (the script cds twice); other path-valued extra args must be absolute
DATASET="$(cd "$(dirname "$DATASET")" && pwd)/$(basename "$DATASET")"

cd /home/ubuntu/jake/UWLab
export OMNI_KIT_ACCEPT_EULA=YES
export UV_CACHE_DIR=/home/ubuntu/jake/.uv-cache
export LD_LIBRARY_PATH=/home/ubuntu/jake/syslibs/extracted/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}
export PYTHONUNBUFFERED=1
source /home/ubuntu/jake/env_uwlab/bin/activate

cd "$HERE"
python render_viewpoints.py --dataset "$DATASET" "$@"
