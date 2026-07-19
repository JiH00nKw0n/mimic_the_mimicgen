#!/usr/bin/env bash
# B0: source demo 다운로드 + annotation (PLAN.md §4). 서버의 robosuite_mimicgen
# venv 안에서 실행. Phase-0 4개 태스크(square/threading/coffee/three_piece)는
# aidas에 완료본이 있으므로 신규 태스크만 처리하면 된다.
set -euo pipefail

MIMICGEN=${MIMICGEN:-"$HOME/mimicgen_jihoonkwon/robosuite_mimicgen/mimicgen"}
DATA=${DATA:-"$HOME/mimicgen_jihoonkwon/datasets"}

# task -> env interface (mimicgen/scripts/prepare_all_src_datasets.sh와 동일)
declare -A INTERFACES=(
  [stack]=MG_Stack
  [stack_three]=MG_StackThree
)

for task in "${!INTERFACES[@]}"; do
  echo "=== ${task}: download source demos ==="
  python "$MIMICGEN/mimicgen/scripts/download_datasets.py" \
    --dataset_type source --tasks "$task" --download_dir "$DATA"
  echo "=== ${task}: annotate (prepare_src_dataset) ==="
  python "$MIMICGEN/mimicgen/scripts/prepare_src_dataset.py" \
    --dataset "$DATA/source/${task}.hdf5" \
    --env_interface "${INTERFACES[$task]}" \
    --env_interface_type robosuite
  python "$MIMICGEN/mimicgen/scripts/get_source_info.py" \
    --dataset "$DATA/source/${task}.hdf5"
done
