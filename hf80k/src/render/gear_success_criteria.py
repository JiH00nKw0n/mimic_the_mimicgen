#!/usr/bin/env python3
"""기어 셋이 받침의 축에 앉았는지 판정한다. 렌더가 궤적을 다시 재생하며 매기는 두 번째 관문이다.

핀 꽂기의 `peg_success_criteria`, 큐브 쌓기의 `success_criteria`와 같은 자리에 같은 방식으로
붙는다. 판정 기준은 생성 중에 쓰는 `src/env_gear/gear_geom.py`의 값을 그대로 가져왔다.
기어 세 개가 모두 아래 세 조건을 만족하고 그리퍼가 열려 있어야 성공이다.

  1. 기어의 축 구멍 중심이 받침의 해당 축 중심에서 가로세로로 1 mm 안에 있다. 두 물체의
     원점을 비교하지 않는 이유는, 기어와 받침의 USD 원점이 축에서 떨어져 있어서 기어가
     제 축 위에 똑바로 앉아 있어도 원점끼리는 크게 벌어지기 때문이다.
  2. 기어 원점과 받침 원점의 높이 차이가 2.5 mm 안이다.
  3. 기어 몸통의 z축이 월드 z축에서 0.12 rad(약 6.9도) 안으로만 기울어져 있다.

그리퍼가 열려 있어야 한다는 조건을 더한 이유는, 아직 들고 있는 상태를 앉은 것으로 세지
않기 위해서다. 핀 꽂기와 큐브 쌓기에서 같은 이유로 같은 조건을 넣었다.

생성 중 판정과 다른 점이 하나 있다. 생성 쪽 `gear_success_hook`은 기어가 아직 움직이는
속도도 보고 0.025 m/s 미만을 요구하는데, 여기서는 그 조건을 보지 않는다. 계약 형식에
기록되는 것이 물체의 자세뿐이고 속도는 들어가지 않기 때문이다. 속도 조건은 생성 단계에서
이미 걸러진 뒤이므로, 여기서 빠지는 것이 판정을 느슨하게 만들지는 않는다.

Isaac 없이 도는 순수 파이썬이라 시험에서 그대로 부를 수 있다.
"""
from __future__ import annotations

import math
import os

# src/env_gear/gear_geom.py의 AXIS_X, AXIS_Z와 같은 값이다. 기어마다 축 구멍이 물체
# 원점에서 x축으로 얼마나 떨어져 있는지, 미터. 받침의 세 축도 같은 값만큼 떨어져 있다.
AXIS_X = {"large": -0.03025, "small": 0.05075, "medium": 0.02025}
# 축 맞음을 보는 높이, 미터. 시연을 만든 코드가 축 오차를 이 높이에서 재고 판정했다.
AXIS_Z = 0.005

# 조립 순서. 시연이 이 순서로만 만들어져 있다.
GEARS = ("large", "small", "medium")

# 기준값 세 개. gear_geom.py와 같다.
AXIS_XY_TOL_M = 0.001        # 축 중심이 받침 축에서 가로세로로 벗어난 거리
Z_TOL_M = 0.0025             # 기어 원점과 받침 원점의 높이 차이
TILT_TOL_RAD = 0.12          # 기어 몸통 z축이 월드 z축에서 기울어진 각

# 그리퍼가 열렸다고 볼 손가락 관절 값, 미터. gear_success_hook과 같은 환경변수를 읽어서
# 두 관문이 서로 다른 기준으로 판정하는 일이 생기지 않게 한다.
GRIPPER_OPEN = float(os.environ.get("LAB_SUCCESS_GRIPPER_OPEN", "0.03"))


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def quat_apply_xyzw(quat, vec):
    """XYZW 순서 쿼터니언으로 벡터를 회전한다. gear_geom.quat_apply_xyzw와 같은 식이다.

    식은 v' = v + 2w(q x v) + 2(q x (q x v))이고 q는 쿼터니언의 벡터 성분이다.
    라이브러리를 거치지 않는 이유는 회전 규약을 한 군데서만 정하기 위해서다.
    """
    q = (float(quat[0]), float(quat[1]), float(quat[2]))
    w = float(quat[3])
    t = tuple(2.0 * c for c in _cross(q, vec))
    qt = _cross(q, t)
    return tuple(vec[i] + w * t[i] + qt[i] for i in range(3))


def upright_z_from_quat(quat) -> float:
    """물체의 몸통 z축이 월드 z축과 얼마나 같은 방향인지를 -1에서 1 사이로 돌려준다.

    1이면 똑바로 서 있고 0이면 옆으로 누웠고 -1이면 거꾸로다. XYZW에서 회전행렬의
    (2,2) 성분이 ``1 - 2(x^2 + y^2)``이다.
    """
    return 1.0 - 2.0 * (float(quat[0]) ** 2 + float(quat[1]) ** 2)


def tilt_rad_from_quat(quat) -> float:
    """몸통 z축이 월드 z축에서 기울어진 각을 라디안으로 돌려준다."""
    return math.acos(max(-1.0, min(1.0, upright_z_from_quat(quat))))


def axis_xy_error(gear_pos, gear_quat, base_pos, base_quat, gear_name: str) -> float:
    """기어의 축 중심과 받침의 해당 축 중심이 가로세로로 얼마나 벗어났는지, 미터.

    기어가 축 주위로 돌아가 있어도 축 중심은 제자리이므로, 돌아간 각과 무관하게 잰다.
    """
    offset = (AXIS_X[gear_name], 0.0, AXIS_Z)
    g = quat_apply_xyzw(gear_quat, offset)
    b = quat_apply_xyzw(base_quat, offset)
    dx = (gear_pos[0] + g[0]) - (base_pos[0] + b[0])
    dy = (gear_pos[1] + g[1]) - (base_pos[1] + b[1])
    return math.hypot(dx, dy)


def assembly_status(objects: dict) -> dict:
    """기어 세 개의 네 수치를 한 번에 구한다. 그리퍼는 보지 않는다.

    Args:
        objects: 강체 자세. 이름을 키로 하고 값은 (x, y, z, qx, qy, qz, qw) 일곱 숫자다.
            쿼터니언 순서는 XYZW다.

    Returns:
        기어 이름을 키로 하고 값이 축 오차와 높이 차이와 기울기와 통과 여부인 사전.
        받침이나 기어가 하나라도 없으면 None.
    """
    if "base" not in objects or any(g not in objects for g in GEARS):
        return None
    base = [float(v) for v in list(objects["base"])[:7]]
    base_pos, base_quat = base[:3], base[3:7]
    rows = {}
    for name in GEARS:
        pose = [float(v) for v in list(objects[name])[:7]]
        pos, quat = pose[:3], pose[3:7]
        axis_m = axis_xy_error(pos, quat, base_pos, base_quat, name)
        dz_m = abs(pos[2] - base_pos[2])
        tilt_rad = tilt_rad_from_quat(quat)
        rows[name] = {
            "axis_m": axis_m,
            "dz_m": dz_m,
            "tilt_rad": tilt_rad,
            "ok": (axis_m < AXIS_XY_TOL_M and dz_m < Z_TOL_M
                   and tilt_rad < TILT_TOL_RAD),
        }
    return rows


def replay_verdict(objects: dict, fingers) -> dict:
    """렌더가 부르는 통일된 진입점. 핀 꽂기의 peg_success_criteria.replay_verdict와 짝이다.

    Args:
        objects: 마지막 프레임의 강체 자세. 이름을 키로 하고 값은 (x, y, z, qx, qy, qz, qw)
            일곱 숫자다. 쿼터니언 순서는 XYZW다. 이 컨테이너가 그 순서로 기록한다.
            기어 태스크는 base, large, small, medium 넷을 기록한다.
        fingers: 마지막 프레임의 그리퍼 손가락 관절 값. 둘 중 작은 값이 열림 기준을
            넘어야 한다.

    Returns:
        ok에 성공 여부를, attrs에 기어마다의 축 오차와 높이 차이와 기울기를 담은 사전.
        받침이나 기어가 장면에 없으면 None.
    """
    rows = assembly_status(objects)
    if rows is None:
        return None
    try:
        grip_min = min(float(v) for v in fingers)
    except (TypeError, ValueError):
        # 손가락 값을 읽지 못하면 성공이 아닌 쪽으로 판정한다. 조용히 통과시키는 쪽이
        # 더 나쁘다. gear_success_hook도 읽기 실패를 실패로 처리한다.
        grip_min = -1.0
    ok = grip_min > GRIPPER_OPEN and all(r["ok"] for r in rows.values())
    attrs = {"grip_min_m": float(grip_min)}
    for name, r in rows.items():
        attrs[f"{name}_axis_m"] = float(r["axis_m"])
        attrs[f"{name}_dz_m"] = float(r["dz_m"])
        attrs[f"{name}_tilt_rad"] = float(r["tilt_rad"])
    return {"ok": bool(ok), "attrs": attrs, "label": "조립"}
