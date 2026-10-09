#!/usr/bin/env bash
# 태스크 하나를 돌리는 데 필요한 자산을 받는 쪽에 보낼 tar.gz 하나로 묶는다.
#
# 자산은 git으로 나르지 않으므로(.gitignore 참고) 코드 저장소만으로는 이미지를 구울 수
# 없다. 이 스크립트가 그 빈자리를 채우는 묶음을 만든다. 손으로 tar를 치지 않고 여기에
# 둔 이유는 안내문이 저장소와 같이 갱신되게 하기 위해서다. 묶음마다 안내문이 다르고,
# 태스크 프로필의 assets.required에 적힌 항목과 아래 목록이 일치해야 한다.
#
#   scripts/pack_assets.sh                      큐브 쌓기 (기본)
#   scripts/pack_assets.sh peg                  핀 꽂기
#   scripts/pack_assets.sh peg ~/Desktop/a.tar.gz
#
# 기본 출력은 ~/Downloads/hf80k_<태스크>_assets.tar.gz 다.
# 자산이 이 기계에 없으면 scripts/fetch_assets.sh 로 먼저 채운다.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HF80K="$(dirname "$HERE")"
ASSETS="${ASSETS_DIR:-$HF80K/assets}"

TASK="${1:-cube}"
case "$TASK" in
  cube)
    ITEMS=(fwd_annotated.hdf5
           fr3_cube_system_calibration_bundle_v1
           fr3_visual_randomization_v1)
    README="$HERE/assets_bundle_README.txt"
    TOP="hf80k_assets"
    ;;
  peg)
    ITEMS=(peg_annotated.hdf5
           peg_hole_env.usd
           hole_01.usd
           fr3_peg_in_hole_physics_bundle_v1_normalized
           fr3_visual_randomization_v1)
    README="$HERE/assets_bundle_peg_README.txt"
    TOP="hf80k_peg_assets"
    ;;
  *)
    echo "태스크는 cube 또는 peg 다. 받은 값: $TASK" >&2
    exit 2
    ;;
esac

OUT="${2:-$HOME/Downloads/hf80k_${TASK}_assets.tar.gz}"

missing=0
for item in "${ITEMS[@]}"; do
  if [ ! -e "$ASSETS/$item" ]; then
    echo "없음: $ASSETS/$item" >&2
    missing=1
  fi
done
if [ "$missing" -ne 0 ]; then
  echo "먼저 scripts/fetch_assets.sh 로 자산을 채운다." >&2
  echo "이미지에 구워진 것을 꺼내 쓰려면 ASSETS_DIR로 그 경로를 준다." >&2
  exit 1
fi

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
mkdir -p "$STAGE/$TOP"

cp "$README" "$STAGE/$TOP/README.txt"
for item in "${ITEMS[@]}"; do
  cp -R "$ASSETS/$item" "$STAGE/$TOP/$item"
done

# 체크섬을 내는 명령은 맥과 리눅스에서 이름이 다르다. 맥에는 shasum만 있고 리눅스에는
# sha256sum이 있다. 둘 중 있는 것을 쓴다.
if command -v sha256sum >/dev/null 2>&1; then
  SHA=(sha256sum)
elif command -v shasum >/dev/null 2>&1; then
  SHA=(shasum -a 256)
else
  echo "sha256sum도 shasum도 없다. 체크섬을 낼 수 없다." >&2
  exit 1
fi

# 받는 쪽이 전송 중 깨짐을 확인할 수 있게 항목별 체크섬을 같이 넣는다.
( cd "$STAGE/$TOP" && find . -type f ! -name SHA256SUMS -print0 \
    | sort -z | xargs -0 "${SHA[@]}" > SHA256SUMS )

mkdir -p "$(dirname "$OUT")"
tar czf "$OUT" -C "$STAGE" "$TOP"

echo "만들었다: $OUT ($(du -h "$OUT" | cut -f1))"
echo "항목 ${#ITEMS[@]}개, 파일 $(grep -c . "$STAGE/$TOP/SHA256SUMS")개"
echo "묶음 자체의 체크섬: $("${SHA[@]}" "$OUT" | cut -d' ' -f1)"
echo "받는 쪽은 압축을 풀어 ${#ITEMS[@]}개 항목을 hf80k/assets/ 아래로 옮기면 된다."
