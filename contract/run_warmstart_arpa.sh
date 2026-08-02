#!/bin/bash
# Closed-loop contract replay on arpa — UWLab-native Isaac env (same prologue
# as lab_stack_mimic/run_replay.sh). Requires /home/ubuntu/jake/{UWLab,env_uwlab}.
#
#   ./run_warmstart_arpa.sh --contract <c.hdf5> --source <src.hdf5> \
#       --demo demo_0 --output <executed.hdf5>
set -e
cd /home/ubuntu/jake/UWLab
source /home/ubuntu/jake/env_uwlab/bin/activate
export LD_LIBRARY_PATH=/home/ubuntu/jake/syslibs/extracted/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}
export OMNI_KIT_ACCEPT_EULA=YES
REPO=${CONTRACT_REPO:-/home/ubuntu/mimicgen_jihoonkwon/mimic_the_mimicgen}
python "$REPO/contract/warmstart_replay.py" --device cpu "$@"
