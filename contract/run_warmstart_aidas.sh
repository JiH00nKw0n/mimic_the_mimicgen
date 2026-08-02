#!/bin/bash
# Closed-loop contract replay on aidas — Isaac Lab docker + the handoff's
# FROZEN RelCartesianOSC (contract/uwlab_frozen, historical_09f7e5b). No UWLab
# checkout needed. Paths are IN-CONTAINER (repo->/repo, ~/contract_out->/out).
#
#   ./run_warmstart_aidas.sh --contract /out/human_demo0_contract.hdf5 \
#       --source /repo/datasets/fwd_annotated.hdf5 --demo demo_0 \
#       --output /out/human_demo0_executed.hdf5
set -e
IMAGE="nvcr.io/nvidia/isaac-lab:3.0.0-beta2-post1"
REPO=/home/ubuntu/mimicgen_jihoonkwon/mimic_the_mimicgen
mkdir -p /home/ubuntu/contract_out
docker run --rm --gpus all --network host \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y -e OMNI_KIT_ACCEPT_EULA=YES \
  -v "$REPO":/repo \
  -v /home/ubuntu/rl_demos:/rl_demos \
  -v /home/ubuntu/contract_out:/out \
  -v /home/ubuntu/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
  -v /home/ubuntu/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
  --entrypoint /workspace/isaaclab/isaaclab.sh \
  "$IMAGE" -p /repo/contract/warmstart_replay.py --device cpu "$@"
