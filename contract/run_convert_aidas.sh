#!/bin/bash
# Offline contract conversion on aidas — Isaac Lab docker, no UWLab needed.
# All paths are IN-CONTAINER: repo -> /repo, ~/rl_demos -> /rl_demos, ~/contract_out -> /out
#
#   ./run_convert_aidas.sh --dataset /repo/datasets/fwd_annotated.hdf5 \
#       --output /out/human_demo0_contract.hdf5 --count 1 \
#       --reference /rl_demos/fr3_three_cube_fullstack_success_50.hdf5
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
  "$IMAGE" -p /repo/contract/convert_demo.py --device cpu "$@"
