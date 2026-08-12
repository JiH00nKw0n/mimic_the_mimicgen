#!/usr/bin/env python3
"""Order-agnostic 3-cube stacking success criterion.

Ported verbatim (same thresholds) from the lab teleop project's
`teleop/filter_success.py`, so the replay / annotation success judgement matches
exactly how the 29 seed demos were filtered.

Success = the 3 cubes form a single vertical tower, in ANY colour/identity order:
  - the gripper has RELEASED the stack (both fingers open), AND
  - sorting the cubes bottom->top by height, both consecutive z gaps are about one
    cube tall (rejects 2-stacks and flat layouts), AND
  - consecutive cubes are aligned in x-y (a real tower, not a staircase).

`--canonical` instead requires the identity order cube_1 < cube_2 < cube_3.

This module is pure Python (no Isaac Sim / numpy required) so it can be imported
from a replay loop, an annotation hook, or an offline checker alike.
"""

from __future__ import annotations

import math

# Thresholds (identical to teleop/filter_success.py) -------------------------
XY_THRESHOLD = 0.04            # consecutive cubes aligned within 4 cm in x-y
# hf80k 변경: 원래 0.038-0.052였다. 생성 단계 판정(clean_success_hook)이 쓰는 창은
# 0.0368-0.0568이라 두 창이 서로 포함 관계가 아니었고, 간격이 0.0368-0.038이나
# 0.052-0.0568에 떨어진 멀쩡한 탑이 생성에는 통과했다가 렌더에서 버려졌다. GPU 시간을
# 다 쓰고 버리는 것이라 손해가 크다. 생성 창과 같게 맞춰 이쪽이 간격을 이유로 버리지
# 않게 한다. 렌더 판정은 대신 그리퍼를 놓았는지를 보는 역할을 맡는다.
GAP_LO, GAP_HI = 0.0368, 0.0568
GRIPPER_OPEN = 0.03            # finger joint > this => released


def _xy_dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def tower_status(cube_positions, finger_positions, *, canonical: bool = False):
    # 손가락 값이 비어 있으면 아래 all(...)이 무조건 참이 되어 "그리퍼를 놓았다"는
    # 조건이 사라진다. 관절 이름이 바뀌어 목록이 비는 경우가 실제로 가능하므로 막는다.
    if not list(finger_positions):
        raise ValueError("finger_positions가 비었다. 그리퍼 관절을 찾지 못했다는 뜻이고, "
                         "이대로면 그리퍼가 놓였는지 검사가 통째로 사라진다.")
    """Judge a single frame.

    Args:
        cube_positions: list of 3 (x, y, z) for cube_1, cube_2, cube_3 (any frame,
            world or env-relative — only relative geometry is used).
        finger_positions: iterable of the gripper finger joint values.
        canonical: if True, also require the identity order cube_1 < cube_2 < cube_3.

    Returns:
        dict with keys: ok (bool), released (bool), order (list bottom->top indices,
        0-based), gaps (g1, g2), xy_ok (bool), gaps_ok (bool).
    """
    released = all(f > GRIPPER_OPEN for f in finger_positions)

    order = sorted(range(3), key=lambda i: cube_positions[i][2])  # bottom -> top by z
    b, m, t = (cube_positions[i] for i in order)
    g1, g2 = m[2] - b[2], t[2] - m[2]
    gaps_ok = (GAP_LO < g1 < GAP_HI) and (GAP_LO < g2 < GAP_HI)
    xy_ok = _xy_dist(m, b) < XY_THRESHOLD and _xy_dist(t, m) < XY_THRESHOLD

    ok = released and gaps_ok and xy_ok
    if canonical and order != [0, 1, 2]:
        ok = False

    return {
        "ok": ok,
        "released": released,
        "order": order,           # e.g. [2, 0, 1] => cube_3 bottom, cube_1 mid, cube_2 top
        "gaps": (round(g1, 4), round(g2, 4)),
        "xy_ok": xy_ok,
        "gaps_ok": gaps_ok,
    }
