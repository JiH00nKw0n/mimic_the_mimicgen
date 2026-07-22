#!/usr/bin/env bash
# Measure the fr3_hand -> fr3v2_hand_tcp binding in the UWLab Isaac Lab env (arpa).
# Usage: bash run_probe.sh [extra args for probe_tcp_binding.py]
set -eu

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd /home/ubuntu/jake/UWLab
export OMNI_KIT_ACCEPT_EULA=YES
export UV_CACHE_DIR=/home/ubuntu/jake/.uv-cache
export LD_LIBRARY_PATH=/home/ubuntu/jake/syslibs/extracted/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}
export PYTHONUNBUFFERED=1
source /home/ubuntu/jake/env_uwlab/bin/activate

cd "$HERE"
python probe_tcp_binding.py "$@"
