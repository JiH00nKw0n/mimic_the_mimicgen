#!/usr/bin/env python3
"""렌더가 매기는 재생 성공 판정이 태스크마다 제대로 동작하는지 확인한다.

왜 필요한가. 렌더는 궤적을 다시 재생한 뒤 마지막 상태를 보고 성공 여부를 파일에 적는다.
기록 단계는 그 표시가 있는 에피소드만 데이터셋에 넣는다. 표시가 없으면 그 에피소드는
"render replay wrote no success verdict"라는 이유로 전부 버려진다. 실제로 핀 삽입에서
그렇게 됐다. 렌더가 큐브 판정 함수만 알고 있어서 핀 장면에서는 아무것도 적지 않았고,
생성과 변환과 렌더를 다 통과한 10편이 기록 단계에서 0편이 됐다.

두 태스크의 판정 함수가 같은 이름과 같은 인자 모양을 갖는지, 그리고 성공과 실패를
실제로 구분하는지 본다.

    python3 src/tests/test_replay_verdict.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "render"))

from peg_success_criteria import replay_verdict as peg_verdict  # noqa: E402
from success_criteria import replay_verdict as cube_verdict  # noqa: E402

failures = []
FINGERS_OPEN = [0.04, 0.04]        # 그리퍼를 놓은 상태
IDENT = [0.0, 0.0, 0.0, 1.0]       # XYZW 항등 회전


def check(label, ok, detail=""):
    print(("  통과  " if ok else "  실패  ") + label + ((" | " + detail) if detail else ""))
    if not ok:
        failures.append(label)


print("큐브 쌓기")
tower = {f"cube_{i}": [0.3, 0.0, 0.7454 + 0.0507 * (i - 1)] + IDENT for i in (1, 2, 3)}
v = cube_verdict(tower, FINGERS_OPEN)
check("세 개가 쌓이면 성공이다", v is not None and v["ok"] is True)
check("쌓인 순서를 함께 적는다", v is not None and "stack_order" in v.get("attrs", {}),
      str(v.get("attrs") if v else None))

spread = {f"cube_{i}": [0.3 + 0.1 * i, 0.0, 0.7454] + IDENT for i in (1, 2, 3)}
v = cube_verdict(spread, FINGERS_OPEN)
check("흩어져 있으면 실패다", v is not None and v["ok"] is False)
check("큐브가 없는 장면에서는 판정하지 않는다", cube_verdict({"peg": [0, 0, 0.75] + IDENT},
                                                            FINGERS_OPEN) is None)

print("핀 삽입")
# 사람 시연 4편의 마지막 핀 위치가 모두 (0.0917, 0.103, 0.7525) 부근이다.
v = peg_verdict({"peg": [0.0917, 0.103, 0.7525] + IDENT}, FINGERS_OPEN)
check("사람 시연의 마지막 상태를 성공으로 본다", v is not None and v["ok"] is True)
check("삽입 깊이가 사람 시연 값 3.81 cm와 같다",
      v is not None and abs(v["attrs"]["insert_depth_m"] - 0.0381) < 1e-4,
      "%.4f m" % v["attrs"]["insert_depth_m"] if v else "")

v = peg_verdict({"peg": [0.185, 0.033, 0.750] + IDENT}, FINGERS_OPEN)
check("책상에 그대로 서 있으면 실패다", v is not None and v["ok"] is False,
      "구멍에서 %.3f m" % v["attrs"]["insert_radial_m"] if v else "")

# x축 90도(XYZW). 구멍 자리에 있어도 누워 있으면 꽂힌 것이 아니다.
v = peg_verdict({"peg": [0.0917, 0.103, 0.7525, 0.7071068, 0.0, 0.0, 0.7071068]}, FINGERS_OPEN)
check("구멍 자리에 누워 있으면 실패다", v is not None and v["ok"] is False,
      "수직도 %.3f" % v["attrs"]["peg_upright"] if v else "")
check("핀이 없는 장면에서는 판정하지 않는다",
      peg_verdict({"cube_1": [0, 0, 0.75] + IDENT}, FINGERS_OPEN) is None)

print("두 태스크의 진입점 모양")
check("두 함수 이름이 replay_verdict로 같다",
      cube_verdict.__name__ == peg_verdict.__name__ == "replay_verdict")
import inspect  # noqa: E402
check("두 함수의 인자 이름이 같다",
      list(inspect.signature(cube_verdict).parameters)
      == list(inspect.signature(peg_verdict).parameters),
      str(list(inspect.signature(peg_verdict).parameters)))

print()
if failures:
    print("실패 %d건: %s" % (len(failures), ", ".join(failures)))
    sys.exit(1)
print("모두 통과했다.")
