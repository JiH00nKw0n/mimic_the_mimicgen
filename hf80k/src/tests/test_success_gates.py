#!/usr/bin/env python3
"""성공 판정이 실패를 실제로 실패라고 말하는지 확인한다.

왜 필요한가. 스모크 시험 100편에서 렌더 재실행 판정이 100편 모두 "쌓였다"로 나왔다.
전부 진짜로 성공한 것일 수도 있고, 판정이 무조건 성공을 돌려주는 것일 수도 있는데,
성공만 들어오는 자료로는 그 둘을 구분할 수 없다. 그래서 실패해야 하는 배치를 손으로
만들어 넣고 판정이 거부하는지 본다. 8만 편을 만들기 전에 이것을 확인해 두지 않으면,
학습이 안 되는 이유를 나중에 데이터에서 찾아야 한다.

    python3 src/tests/test_success_gates.py

한 줄도 실패하지 않으면 0을 반환한다.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "render"))

from success_criteria import GAP_HI, GAP_LO, GRIPPER_OPEN, tower_status  # noqa: E402

CUBE = (GAP_LO + GAP_HI) / 2      # 큐브 한 개 높이의 대표값. 두 창의 한가운데다
OPEN = GRIPPER_OPEN + 0.01        # 확실히 놓은 상태
SHUT = GRIPPER_OPEN - 0.01        # 확실히 쥐고 있는 상태

failures = []


def check(name: str, expected_ok: bool, cubes, fingers, **kwargs):
    status = tower_status(cubes, fingers, **kwargs)
    got = status["ok"]
    mark = "통과" if got == expected_ok else "어긋남"
    print("  %-46s 기대 %-5s 결과 %-5s %s  %s"
          % (name, expected_ok, got, mark,
             "" if got == expected_ok else status))
    if got != expected_ok:
        failures.append(name)


print("성공해야 하는 배치")
check("세 개가 제대로 쌓이고 그리퍼를 놓았다", True,
      [(0.0, 0.0, 0.0), (0.0, 0.0, CUBE), (0.0, 0.0, 2 * CUBE)], [OPEN, OPEN])
check("쌓인 순서가 색 순서와 달라도 된다", True,
      [(0.0, 0.0, 2 * CUBE), (0.0, 0.0, 0.0), (0.0, 0.0, CUBE)], [OPEN, OPEN])
check("살짝 어긋나도 4 cm 안이면 탑이다", True,
      [(0.0, 0.0, 0.0), (0.02, 0.0, CUBE), (0.03, 0.01, 2 * CUBE)], [OPEN, OPEN])

print("\n실패해야 하는 배치")
check("셋 다 책상에 나란히 놓여 있다", False,
      [(0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.2, 0.0, 0.0)], [OPEN, OPEN])
check("두 개만 쌓이고 하나는 옆에 있다", False,
      [(0.0, 0.0, 0.0), (0.0, 0.0, CUBE), (0.2, 0.0, 0.0)], [OPEN, OPEN])
check("탑은 맞지만 그리퍼가 아직 쥐고 있다", False,
      [(0.0, 0.0, 0.0), (0.0, 0.0, CUBE), (0.0, 0.0, 2 * CUBE)], [SHUT, SHUT])
check("손가락 한쪽만 열려 있다", False,
      [(0.0, 0.0, 0.0), (0.0, 0.0, CUBE), (0.0, 0.0, 2 * CUBE)], [OPEN, SHUT])
check("층 간격이 너무 넓다 (공중에 떠 있다)", False,
      [(0.0, 0.0, 0.0), (0.0, 0.0, CUBE), (0.0, 0.0, CUBE + GAP_HI + 0.01)], [OPEN, OPEN])
check("층 간격이 너무 좁다 (겹쳐 있다)", False,
      [(0.0, 0.0, 0.0), (0.0, 0.0, GAP_LO - 0.005), (0.0, 0.0, 2 * CUBE)], [OPEN, OPEN])
check("세 개가 계단처럼 어긋나 있다", False,
      [(0.0, 0.0, 0.0), (0.06, 0.0, CUBE), (0.12, 0.0, 2 * CUBE)], [OPEN, OPEN])
check("색 순서를 요구하면 뒤바뀐 탑은 실패다", False,
      [(0.0, 0.0, 2 * CUBE), (0.0, 0.0, 0.0), (0.0, 0.0, CUBE)], [OPEN, OPEN],
      canonical=True)

print("\n판정 자체가 사라지는 경우")
try:
    tower_status([(0.0, 0.0, 0.0)] * 3, [])
    print("  %-46s 어긋남  손가락 목록이 비었는데 그냥 통과했다" % "손가락 목록이 비었다")
    failures.append("빈 손가락 목록")
except ValueError as exc:
    print("  %-46s 통과   %s" % ("손가락 목록이 비었다", exc))

print()
if failures:
    print("어긋난 항목 %d개: %s" % (len(failures), ", ".join(failures)))
    raise SystemExit(1)
print("모두 기대대로 판정했다")
