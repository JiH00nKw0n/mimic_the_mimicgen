#!/usr/bin/env bash
# B1/B2: 한 (task, variant, seed) pool 생성. gen-config가 만든 mg_*.json을
# 커스텀 변형 등록 후 공식 generate_dataset에 위임한다. 프로세스 병렬은 이
# 스크립트를 seed만 바꿔 여러 개 띄우는 것으로 충분 (mimicgen은 단일 스레드).
#
# 사용:  TASK=threading VARIANT=D2E SEED=1 ./b1_generate_pool.sh
set -euo pipefail

MOTIVATION=${MOTIVATION:-"$HOME/mimicgen_jihoonkwon/mimic_the_mimicgen/motivation"}
TASK=${TASK:?e.g. threading}
VARIANT=${VARIANT:?e.g. D2E}
SEED=${SEED:-1}

cd "$MOTIVATION"
python -m genaudit gen-config \
  --task-config "configs/tasks/${TASK}.yaml" \
  --experiment-config "configs/experiments/e2_${TASK}.yaml" \
  --variant "$VARIANT" --seed "$SEED"

CONFIG="$(python - <<PY
import yaml, pathlib
spec = yaml.safe_load(pathlib.Path("configs/experiments/e2_${TASK}.yaml").read_text())
print(pathlib.Path(spec["paths"]["out_dir"]) / "mg_${VARIANT}_seed${SEED}.json")
PY
)"
exec python -m genaudit.generation.run_mimicgen --config "$CONFIG"
