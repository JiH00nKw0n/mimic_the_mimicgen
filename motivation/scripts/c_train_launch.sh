#!/usr/bin/env bash
# C단계 학습 런처 — aidas에서 검증된 실행 조건 (2026-07-19 밤 확정).
#
# 왜 이 형태인가 (전부 실제로 겪은 것):
#  1) LD_LIBRARY_PATH를 venv의 nvidia 라이브러리로만 제한:
#     시스템 /usr/local/cuda(13.2)가 앞서면 pip torch(cu13.0/13.1 세트)와
#     버전이 섞여 cuDNN LSTM에서 "Cannot load symbol cublasLtCreate" 크래시.
#  2) stdin은 반드시 /dev/null: `yes |`로 물리면 첫 배치에서 코어덤프.
#     (출력 폴더가 이미 있으면 robomimic이 프롬프트를 띄우므로 폴더를 먼저 정리)
#  3) CUDA_LAUNCH_BLOCKING=1 + PYTHONFAULTHANDLER=1: 검증된 조합 그대로.
#     (低차원 BC-RNN은 ~1.8s/epoch라 동기 실행 비용이 무시 가능)
#
# 사용:  CONFIG=/path/to/bc_config.json ./c_train_launch.sh
set -euo pipefail

CONFIG=${CONFIG:?robomimic 학습 config json 경로}
NV=${NV:-"$HOME/mimicgen_jihoonkwon/robosuite_mimicgen/venv/lib/python3.10/site-packages/nvidia"}
V=${V:-"$HOME/mimicgen_jihoonkwon/robosuite_mimicgen/venv/bin/python"}
ROBOMIMIC=${ROBOMIMIC:-"$HOME/mimicgen_jihoonkwon/robosuite_mimicgen/robomimic"}
LOG=${LOG:-"/tmp/train_$(basename "$CONFIG" .json).log"}

# 기존 출력 폴더가 있으면 robomimic이 stdin 프롬프트를 띄운다 — 미리 확인
NAME=$("$V" -c "import json,sys; print(json.load(open(\"$CONFIG\"))[\"experiment\"][\"name\"])")
OUT=$("$V" -c "import json,sys; print(json.load(open(\"$CONFIG\"))[\"train\"][\"output_dir\"])")
if [ -d "$OUT/$NAME" ]; then
  echo "ERROR: $OUT/$NAME 이미 존재 — 지우거나 experiment.name을 바꿀 것" >&2
  exit 1
fi

cd "$ROBOMIMIC"
setsid env LD_LIBRARY_PATH="$NV/cu13/lib:$NV/cudnn/lib" CUDA_LAUNCH_BLOCKING=1 PYTHONFAULTHANDLER=1 \
  "$V" robomimic/scripts/train.py --config "$CONFIG" > "$LOG" 2>&1 < /dev/null &
echo "launched pid $! -> $LOG"
