#!/usr/bin/env bash
# 기어 조립 파이프라인이나 진단 도구 하나를 띄운다. 기계마다 다른 경로를 한 군데에
# 모아 두어서, 쓰는 쪽은 기계 이름과 작업 폴더만 고르면 된다.
#
#   SERVER=aws  WORK=/opt/dlami/nvme/gear_run1 ./run_gear.sh pipeline
#   SERVER=lab  WORK="$HOME/workspace/<사람>/gear_run1" ./run_gear.sh pipeline
#   SERVER=lab  WORK=... ./run_gear.sh tool gear_replay_probe.py --demos 20 --mode closed
#
# pipeline은 이미지의 기본 시작 스크립트를 돌린다. 그 스크립트가 작업 폴더의 .env를
# 읽어 생성과 합치기와 올리기를 이어서 한다. 떼어 놓고 도니까 로그는 docker logs로 본다.
# tool은 그 시작 스크립트를 덮어쓰고 도구 하나만 돌린다. 끝나면 컨테이너가 사라진다.
#
# 연구실 서버에서 이 스크립트는 GPU 예약 장부를 거치지 않는다. 띄우기 전에 isaac-status로
# 내 표딱지가 붙은 장을 확인하고 그 번호를 GPU로 준다. 그러지 않으면 같은 장을 두 사람이
# 모르고 나눠 쓰다 메모리가 터진다.
set -euo pipefail

MODE="${1:-}"
case "$MODE" in
  pipeline|tool) shift ;;
  *) echo "쓰는 법: SERVER=aws|lab WORK=<작업폴더> $0 pipeline" >&2
     echo "         SERVER=aws|lab WORK=<작업폴더> $0 tool <도구이름>.py [인자...]" >&2
     exit 2 ;;
esac

SERVER="${SERVER:?SERVER를 aws 또는 lab으로 준다}"
WORK="${WORK:?WORK에 이 실행의 작업 폴더를 준다}"
IMAGE="${IMAGE:-fr3-hf80k:16}"
GPU="${GPU:-0}"
NAME="${NAME:-gear_$(basename "$WORK")}"

# 기계마다 다른 것은 이 네 경로뿐이다.
case "$SERVER" in
  aws)
    ASSETS="${ASSETS:-/opt/dlami/nvme/gear_assets}"
    SRC_ROOT="${SRC_ROOT:-$HOME/hf80k_live}"
    CACHE="${CACHE:-/opt/dlami/nvme/cache}"
    ;;
  lab)
    # 연구실 서버는 지워지는 디스크가 없어서 전부 작업 폴더 아래에 둔다.
    PROJ="${PROJ:-$(cd "$WORK/.." && pwd)}"
    ASSETS="${ASSETS:-$PROJ/gear_assets}"
    SRC_ROOT="${SRC_ROOT:-$PROJ/hf80k_live}"
    CACHE="${CACHE:-$PROJ/cache}"
    ;;
  *) echo "SERVER는 aws 또는 lab이다 (받은 값: $SERVER)" >&2; exit 2 ;;
esac

for d in "$ASSETS" "$SRC_ROOT/src" "$SRC_ROOT/tools"; do
  [[ -d "$d" ]] || { echo "없는 경로: $d" >&2; exit 1; }
done
mkdir -p "$WORK" "$CACHE/kit" "$CACHE/ov"

MOUNTS=(
  -v "$ASSETS:/opt/hf80k/assets:ro"
  -v "$SRC_ROOT/src:/opt/hf80k/src:ro"
  -v "$SRC_ROOT/tools:/opt/hf80k/tools:ro"
  -v "$CACHE/kit:/isaac-sim/kit/cache"
  -v "$CACHE/ov:/root/.cache/ov"
  -v "$WORK:/work"
)

if [[ "$MODE" == "pipeline" ]]; then
  [[ -f "$WORK/.env" ]] || { echo "작업 폴더에 .env가 없다: $WORK/.env" >&2; exit 1; }
  docker run -d --name "$NAME" --gpus all --network host --shm-size=8g \
    -e CUDA_VISIBLE_DEVICES="$GPU" "${MOUNTS[@]}" "$IMAGE"
  echo "띄웠다: $NAME  (작업 폴더 $WORK)"
  echo "  따라가기: docker logs -f $NAME"
  echo "  세우기:   docker stop $NAME   # 지금 단계를 마치고 이어할 수 있게 남긴다"
else
  TOOL="${1:?돌릴 도구 이름을 준다}"; shift
  # 환경변수는 기본 시작 스크립트가 넣어 주던 것이라서 덮어쓸 때는 직접 줘야 한다.
  docker run --rm --gpus all --network host --shm-size=8g \
    -e CUDA_VISIBLE_DEVICES="$GPU" -e HEADLESS=1 \
    -e PYTHONPATH=/opt/hf80k/src/env_gear:/opt/hf80k/src/env:/opt/hf80k/src \
    -e LAB_GEAR_TABLE_USD=/opt/hf80k/assets/legacy_multi_gear/assets/lab/table_scene.usdc \
    -e LAB_GEAR_ASSET_ROOT=/opt/hf80k/assets/legacy_multi_gear/assets/gears \
    -e LAB_ARM_SCALE=1.0 "${MOUNTS[@]}" \
    --entrypoint bash "$IMAGE" -lc \
    "/workspace/isaaclab/isaaclab.sh -p /opt/hf80k/tools/$TOOL $*"
fi
