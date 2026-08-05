#!/bin/bash
# Render calibrated FR3 camera views on aidas — Isaac Lab docker, no UWLab needed.
# (render/run_render.sh is the arpa/UWLab-venv variant; this is its docker twin.)
#
# Paths below are IN-CONTAINER:
#   /repo   = mimic_the_mimicgen                /out    = ~/contract_out
#   /vrand  = ~/fr3_visual_randomization_v1     /rl_demos = ~/rl_demos
#
#   ./run_render_aidas.sh --dataset /out/gen_human_25.hdf5 --count 1 \
#       --width 320 --height 180 --preview_video 1
set -e
IMAGE="nvcr.io/nvidia/isaac-lab:3.0.0-beta2-post1"
REPO=/home/ubuntu/mimicgen_jihoonkwon/mimic_the_mimicgen
VRAND=/home/ubuntu/fr3_visual_randomization_v1
mkdir -p /home/ubuntu/contract_out
docker run --rm --gpus all --network host \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y -e OMNI_KIT_ACCEPT_EULA=YES \
  -e LAB_TABLE_USD="${LAB_TABLE_USD:-}" \
  -e LAB_ROBOT_SPAWN_ROT="${LAB_ROBOT_SPAWN_ROT:-0,0,1,0}" \
  -e VRAND_PROFILE="${VRAND_PROFILE:-}" \
  -e VRAND_SEED="${VRAND_SEED:-}" \
  -v "$REPO":/repo \
  -v "$VRAND":/vrand \
  -v /home/ubuntu/rl_demos:/rl_demos \
  -v /home/ubuntu/contract_out:/out \
  -v /home/ubuntu/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
  -v /home/ubuntu/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
  --entrypoint /workspace/isaaclab/isaaclab.sh \
  "$IMAGE" -p /repo/render/render_viewpoints.py --device cpu "$@"
