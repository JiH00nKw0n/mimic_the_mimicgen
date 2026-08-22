#!/usr/bin/env python3
"""핀 삽입 판정이 실패를 실패라고 말하는지 확인한다.

큐브 쪽 test_success_gates.py와 같은 이유로 있다. 성공만 들어오는 자료로는 판정이
제대로 도는지와 무조건 통과시키는지를 구분할 수 없다.

    python3 src/tests/test_peg_success.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "render"))

from peg_success_criteria import insertion_status  # noqa: E402

HOLE = (0.4, 0.0)
RIM_Z = 0.80
PEG_LEN = 0.06
UP = (0.0, 0.0, 1.0)          # 똑바로 선 핀
TILTED = (0.0, 0.6, 0.8)      # 기울어진 핀. 수직도 0.8

failures = []


def check(name, expected, *args, **kwargs):
    got = insertion_status(*args, **kwargs)
    ok = got["ok"] == expected
    print(f"  {'통과' if ok else '어긋남'}  {name:44s} 기대 {expected!s:5s} 결과 {got['ok']!s:5s} "
          f"(반경 {got['radial']}, 깊이 {got['depth']}, 수직도 {got['upright']})")
    if not ok:
        failures.append(name)


print("성공해야 하는 배치")
# 깊이 3 cm: 바닥이 rim 아래 3 cm. 중심 z = 바닥 + 길이/2
check("구멍 위에 똑바로 3 cm 들어갔다", True,
      (0.4, 0.0, RIM_Z - 0.03 + PEG_LEN / 2), UP, HOLE, RIM_Z, PEG_LEN)
check("살짝 치우쳤지만 1 cm 안이다", True,
      (0.406, 0.002, RIM_Z - 0.03 + PEG_LEN / 2), UP, HOLE, RIM_Z, PEG_LEN)

print("\n실패해야 하는 배치")
check("구멍에서 5 cm 떨어져 있다", False,
      (0.45, 0.0, RIM_Z - 0.03 + PEG_LEN / 2), UP, HOLE, RIM_Z, PEG_LEN)
check("구멍 위지만 아직 얹혀만 있다", False,
      (0.4, 0.0, RIM_Z + PEG_LEN / 2), UP, HOLE, RIM_Z, PEG_LEN)
check("깊이가 1 cm뿐이다", False,
      (0.4, 0.0, RIM_Z - 0.01 + PEG_LEN / 2), UP, HOLE, RIM_Z, PEG_LEN)
check("깊이는 되지만 기울어져 있다", False,
      (0.4, 0.0, RIM_Z - 0.03 + PEG_LEN / 2), TILTED, HOLE, RIM_Z, PEG_LEN)
check("반경 1.2 cm로 성공 문턱 밖이다", False,
      (0.412, 0.0, RIM_Z - 0.03 + PEG_LEN / 2), UP, HOLE, RIM_Z, PEG_LEN)

print("\n판정이 사라지는 경우")
try:
    insertion_status((0.4, 0.0, 0.8), None, HOLE, RIM_Z, PEG_LEN)
    print("  어긋남  회전이 없는데 그냥 통과했다")
    failures.append("빈 회전")
except ValueError as exc:
    print(f"  통과   회전이 없으면 오류를 낸다: {exc}")

print()
if failures:
    print(f"어긋난 항목 {len(failures)}개: {', '.join(failures)}")
    sys.exit(1)
print("핀 삽입 판정이 기대대로 동작한다")
