#!/usr/bin/env python3
"""핀이 구멍에 꽂혔는지 판정한다. 렌더가 궤적을 다시 재생하며 매기는 두 번째 관문이다.

큐브의 `success_criteria.tower_status`와 짝을 이룬다. 판정 기준은
`lab_peg_mimic/peg_mdp.py`의 `peg_inserted`에서 그대로 가져왔다. 세 조건을 모두 만족해야
성공이다.

  1. 핀 중심이 구멍 축에서 1 cm 안에 있다. 구멍 위에 있지 않으면 깊이를 재는 것이
     의미가 없으므로, 먼저 구멍 입구 반경 1.6 cm 안에 있는지로 거른다.
  2. 구멍 테두리 높이에서 핀 바닥까지의 깊이가 2 cm를 넘는다.
  3. 핀이 서 있다. 핀의 위쪽 축과 수직축이 이루는 방향 일치도가 0.9를 넘는다.

Isaac 없이 도는 순수 파이썬이라 시험에서 그대로 부를 수 있다.
"""
from __future__ import annotations

import math

# lab_peg_mimic/peg_mdp.py:34-41 과 같은 값
HOLE_INNER = 0.032          # 사각 구멍 한 변
RADIAL_INSIDE = HOLE_INNER / 2   # 0.016. 구멍 위에 있다고 볼 반경
RADIAL_SUCCESS = 0.010      # 성공으로 볼 반경
DEPTH_SUCCESS = 0.020       # 성공으로 볼 삽입 깊이
UPRIGHT_SUCCESS = 0.9       # 성공으로 볼 수직도


def insertion_status(peg_position, peg_up_axis, hole_position, rim_z,
                     peg_length, *, radial_success: float = RADIAL_SUCCESS,
                     depth_success: float = DEPTH_SUCCESS,
                     upright_success: float = UPRIGHT_SUCCESS):
    """한 시점을 판정한다.

    Args:
        peg_position: 핀 중심의 (x, y, z).
        peg_up_axis: 핀의 위쪽 방향 단위 벡터 (x, y, z). 회전에서 계산해 넘긴다.
        hole_position: 구멍 중심의 (x, y).
        rim_z: 구멍 테두리 높이.
        peg_length: 핀의 길이. 바닥 높이를 중심에서 빼는 데 쓴다.

    Returns:
        ok, radial, depth, upright, inside를 담은 사전.
    """
    if peg_up_axis is None or len(peg_up_axis) != 3:
        raise ValueError("peg_up_axis가 없다. 핀의 회전을 읽지 못했다는 뜻이고, "
                         "이대로면 서 있는지 검사가 통째로 사라진다.")
    radial = math.hypot(peg_position[0] - hole_position[0],
                        peg_position[1] - hole_position[1])
    inside = radial < RADIAL_INSIDE
    bottom_z = peg_position[2] - peg_length / 2.0
    depth = (rim_z - bottom_z) if inside else 0.0
    upright = float(peg_up_axis[2])
    ok = (radial < radial_success) and (depth > depth_success) and (upright > upright_success)
    return {
        "ok": bool(ok),
        "radial": round(radial, 5),
        "depth": round(depth, 5),
        "upright": round(upright, 4),
        "inside": bool(inside),
    }


# 장면 상수. src/env_peg/peg_mdp.py와 같은 값이다. 렌더는 Isaac 장면을 만들지만 판정은
# 기록된 마지막 상태만 보므로, 여기서 같은 숫자를 들고 있어야 한다.
HOLE_XY = (0.091, 0.104)
DESK_Z = 0.720
HOLE_HEIGHT = 0.0406
PEG_LEN = 0.060


def replay_verdict(objects: dict, fingers) -> dict:
    """렌더가 부르는 통일된 진입점. 큐브 쪽 success_criteria.replay_verdict와 짝이다.

    Args:
        objects: 마지막 프레임의 강체 자세. 이름을 키로 하고 값은 (x, y, z, qx, qy, qz, qw)
            일곱 숫자다. 쿼터니언 순서는 XYZW다. 이 컨테이너가 그 순서로 기록한다.
        fingers: 마지막 프레임의 그리퍼 손가락 관절 값. 핀 판정에는 쓰지 않는다.

    Returns:
        ok에 성공 여부를, attrs에 삽입 깊이와 반경을 담은 사전. 핀이 장면에 없으면 None.
    """
    if "peg" not in objects:
        return None
    pose = list(objects["peg"])
    x, y, z, qx, qy, qz, qw = (float(v) for v in pose[:7])
    # 핀 몸통 z축의 월드 성분. XYZW 쿼터니언에서 회전행렬 세 번째 열이다.
    up = (2.0 * (qx * qz + qw * qy),
          2.0 * (qy * qz - qw * qx),
          1.0 - 2.0 * (qx * qx + qy * qy))
    status = insertion_status((x, y, z), up, HOLE_XY, DESK_Z + HOLE_HEIGHT, PEG_LEN)
    return {
        "ok": bool(status["ok"]),
        "attrs": {"insert_depth_m": float(status["depth"]),
                  "insert_radial_m": float(status["radial"]),
                  "peg_upright": float(status["upright"])},
        "label": "삽입",
    }
