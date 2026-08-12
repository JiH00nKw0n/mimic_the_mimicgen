#!/usr/bin/env bash
# 자산 세 가지를 재익님께 보낼 하나의 tar.gz로 묶는다.
#
# 자산은 git으로 나르지 않으므로(.gitignore 참고) 코드 저장소만으로는 이미지를 구울 수
# 없다. 이 스크립트가 그 빈자리를 채우는 묶음을 만든다. 손으로 tar를 치지 않고 여기에
# 둔 이유는 안내문(assets_bundle_README.txt)이 저장소와 같이 갱신되게 하기 위해서다.
#
#   scripts/pack_assets.sh [출력경로]
#
# 기본 출력은 ~/Downloads/hf80k_assets.tar.gz 다.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HF80K="$(dirname "$HERE")"
ASSETS="$HF80K/assets"
OUT="${1:-$HOME/Downloads/hf80k_assets.tar.gz}"

ITEMS=(fwd_annotated.hdf5 fr3_cube_system_calibration_bundle_v1 fr3_visual_randomization_v1)

missing=0
for item in "${ITEMS[@]}"; do
  if [ ! -e "$ASSETS/$item" ]; then
    echo "없음: $ASSETS/$item" >&2
    missing=1
  fi
done
if [ "$missing" -ne 0 ]; then
  echo "먼저 scripts/fetch_assets.sh 로 자산을 채운다." >&2
  exit 1
fi

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
mkdir -p "$STAGE/hf80k_assets"

cp "$HERE/assets_bundle_README.txt" "$STAGE/hf80k_assets/README.txt"
for item in "${ITEMS[@]}"; do
  cp -R "$ASSETS/$item" "$STAGE/hf80k_assets/$item"
done

mkdir -p "$(dirname "$OUT")"
tar czf "$OUT" -C "$STAGE" hf80k_assets

echo "만들었다: $OUT ($(du -h "$OUT" | cut -f1))"
echo "받는 쪽은 압축을 풀어 세 항목을 hf80k/assets/ 아래로 옮기면 된다."
