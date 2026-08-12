#!/usr/bin/env bash
# Start one hf80k container, i.e. one GPU's share of the run.
#
#   ./scripts/run.sh                                   # GPU 0, /data/hf80k/gpu0
#   CUDA_VISIBLE_DEVICES=2 WORK_DIR_HOST=/data/hf80k/gpu2 ./scripts/run.sh
#
# Knobs (all optional): IMAGE_TAG, CONTAINER_NAME, ENV_FILE, WORK_DIR_HOST,
# ISAAC_CACHE_ROOT, and anything you want pushed past the .env file —
# TARGET_EPISODES, SEED_BASE, HF_REPO_ID, HF_TOKEN — which is how the 4-GPU loop
# at the bottom of this file gives each container its own share.
#
# The container is detached and restarts by itself. A Kit crash mid-chunk is
# normal at this scale; with RESUME=1 the orchestrator re-reads the MANIFEST.json
# files and picks up at the first chunk that is not done, so a restart costs at
# most one chunk of work.
set -euo pipefail

IMAGE_TAG="${IMAGE_TAG:-fr3-hf80k:latest}"
HF80K_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

GPU="${CUDA_VISIBLE_DEVICES:-0}"
NAME="${CONTAINER_NAME:-fr3-hf80k-gpu${GPU}}"
ENV_FILE="${ENV_FILE:-$HF80K_DIR/.env}"
WORK_DIR_HOST="${WORK_DIR_HOST:-/data/hf80k/gpu${GPU}}"
# Kit shader/asset caches. Sharing them across containers is fine and saves
# several minutes of startup per restart.
ISAAC_CACHE_ROOT="${ISAAC_CACHE_ROOT:-$HOME/docker/isaac-sim}"

command -v docker > /dev/null 2>&1 || { echo "docker not on PATH" >&2; exit 1; }
if [[ ! -f "$ENV_FILE" ]]; then
  echo "no env file at $ENV_FILE (cp $HF80K_DIR/.env.example $HF80K_DIR/.env)" >&2
  exit 1
fi
if ! docker image inspect "$IMAGE_TAG" > /dev/null 2>&1; then
  echo "no image $IMAGE_TAG; run ./scripts/build.sh" >&2
  exit 1
fi
if [[ -n "$(docker ps -a --filter "name=^${NAME}$" --format '{{.Names}}')" ]]; then
  echo "container $NAME already exists; 'docker rm -f $NAME' to replace it" >&2
  exit 1
fi

mkdir -p "$WORK_DIR_HOST" "$ISAAC_CACHE_ROOT/cache/kit" "$ISAAC_CACHE_ROOT/cache/ov"

# --gpus all + CUDA_VISIBLE_DEVICES rather than --gpus "device=$GPU": the
# interface names CUDA_VISIBLE_DEVICES as the variable that picks the GPU, and
# with device= the container would renumber it to 0 and the variable would lie.
# --init puts tini at PID 1 so SIGTERM reaches python and Kit's children get
# reaped instead of piling up as zombies over a week-long run.
docker run -d \
  --name "$NAME" \
  --gpus all \
  --init \
  --restart unless-stopped \
  --network host \
  --shm-size 8g \
  -e CUDA_VISIBLE_DEVICES="$GPU" \
  ${HF_TOKEN:+-e HF_TOKEN="$HF_TOKEN"} \
  ${HF_REPO_ID:+-e HF_REPO_ID="$HF_REPO_ID"} \
  ${TARGET_EPISODES:+-e TARGET_EPISODES="$TARGET_EPISODES"} \
  ${SEED_BASE:+-e SEED_BASE="$SEED_BASE"} \
  ${CHUNK_SIZE:+-e CHUNK_SIZE="$CHUNK_SIZE"} \
  ${LOG_LEVEL:+-e LOG_LEVEL="$LOG_LEVEL"} \
  -v "$WORK_DIR_HOST":/work \
  -v "$ENV_FILE":/work/.env:ro \
  -v "$ISAAC_CACHE_ROOT/cache/kit":/isaac-sim/kit/cache:rw \
  -v "$ISAAC_CACHE_ROOT/cache/ov":/root/.cache/ov:rw \
  "$IMAGE_TAG"

echo "started $NAME on GPU $GPU"
echo "  work dir: $WORK_DIR_HOST  (chunks/, logs/, state.json)"
echo "  follow:   docker logs -f $NAME"
echo "  stop:     docker stop $NAME   # SIGTERM; the orchestrator finishes the"
echo "                                # current step and leaves a resumable state"

# ---------------------------------------------------------------------------
# 4-GPU launch, 20,000 episodes each. Distinct WORK_DIR_HOST and SEED_BASE are
# not optional: shared ones make four containers fight over the same chunk
# directories and generate the same demos four times.
#
#   for gpu in 0 1 2 3; do
#     CUDA_VISIBLE_DEVICES=$gpu \
#     WORK_DIR_HOST=/data/hf80k/gpu$gpu \
#     TARGET_EPISODES=20000 \
#     SEED_BASE=$((42000 + gpu * 10000)) \
#     ./scripts/run.sh
#   done
#
# Watch all four:      docker logs -f fr3-hf80k-gpu0   (etc.)
# Progress at a glance: grep -h '"status"' /data/hf80k/gpu*/chunks/*/MANIFEST.json | sort | uniq -c
# Stop all four:       docker stop fr3-hf80k-gpu{0,1,2,3}
# Resume all four:     docker start fr3-hf80k-gpu{0,1,2,3}
