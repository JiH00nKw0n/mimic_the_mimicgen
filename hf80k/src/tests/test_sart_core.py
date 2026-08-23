#!/usr/bin/env python3
"""SART 증강의 기하 계산을 시뮬레이터 없이 확인한다.

왜 필요한가. SART는 정밀 구간 바로 앞의 손 자세를 하나 뽑아 거기서 원래 자세로
되돌아오는 접근 경로를 새로 만드는 증강이다. 그 "뽑는다"가 무너져 늘 같은 자세가
나오면, 증강한 에피소드는 원본의 복사본이 되고 학습에는 아무 도움이 되지 않는다.
그런데 성공률만 보면 그 상태가 오히려 좋아 보인다. 원본이 성공한 궤적이니 복사본도
전부 성공하기 때문이다.

실제로 한 번 그렇게 됐다. 뽑는 반지름을 손이 물체 위에 떠 있는 높이의 절반으로
제한했더니, 수렴 시점에는 손이 이미 물체 바로 위라 그 값이 거의 0이 되었고 같은
소스에서 나온 접근 경로의 표준편차가 0.0025 m, 즉 2.5 mm에 그쳤다. 여기서 막는다.

    python3 src/tests/test_sart_core.py
"""
import math
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(SRC, "sart"))

import sart_core  # noqa: E402

failures = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(("  통과  " if ok else "  실패  ") + label + ((" | " + detail) if detail else ""))
    if not ok:
        failures.append(label)


def pose(x: float, y: float, z: float) -> np.ndarray:
    out = np.eye(4)
    out[:3, 3] = (x, y, z)
    return out


# ---------------------------------------------------------------- 1) 뽑은 자세
print("1) 접근 자세를 뽑는 방식")
RADIUS = 0.06
CENTER = pose(0.10, 0.10, 0.85)
FLOOR = 0.7606                      # 소켓 테두리 높이. 이 아래로는 내려가면 안 된다
rng = np.random.default_rng(7)
samples = np.array([sart_core.sample_offset(CENTER, FLOOR, RADIUS, math.radians(10.0), rng)[:3, 3]
                    for _ in range(10000)])
offsets = samples - CENTER[:3, 3]
radii = np.linalg.norm(offsets, axis=1)

check("뽑은 자세가 모두 반지름 %.3f m 안에 있다" % RADIUS,
      bool(radii.max() <= RADIUS + 1e-9), "최대 %.4f m" % float(radii.max()))
check("뽑은 자세가 모두 바닥 높이 %.4f m 위에 있다" % FLOOR,
      bool(samples[:, 2].min() >= FLOOR - 1e-12), "최저 %.4f m" % float(samples[:, 2].min()))

# 부피 기준으로 고르게 차면 평균 반지름이 최대 반지름의 0.75배 근처가 된다. 반지름을
# 그냥 균등하게 뽑으면 0.5배가 되므로 둘을 구분할 수 있다. 바닥 자르기 때문에 위쪽이
# 조금 더 남지만 평균 반지름 자체는 거의 바뀌지 않는다.
mean_ratio = float(radii.mean() / RADIUS)
check("반지름 평균이 최대의 0.75배 근처다(부피 기준 균등)",
      0.70 <= mean_ratio <= 0.80, "비율 %.3f" % mean_ratio)

per_axis_std = offsets.std(axis=0)
check("축마다 흩어짐이 0.0025 m를 크게 넘는다(복사본이 아니다)",
      bool(per_axis_std.min() > 0.010),
      "축별 표준편차 %s m" % np.round(per_axis_std, 4).tolist())

# 바닥을 중심보다 훨씬 위에 두면 조건을 만족하는 점이 없다. 그때 중심을 그대로 돌려주면
# 그 시도는 원본의 복사본이 된다. 그래서 오류를 던져야 한다.
raised = False
try:
    sart_core.sample_offset(CENTER, CENTER[2, 3] + 1.0, RADIUS, 0.0,
                            np.random.default_rng(0))
except sart_core.DegenerateOffset:
    raised = True
check("뽑을 것이 없으면 중심을 돌려주지 않고 오류를 던진다", raised)

# 위치를 고정하면 위치는 중심 그대로이고 회전만 달라진다. 원본 RoboManipAug의 뜻이다.
fixed = sart_core.sample_offset(CENTER, FLOOR, RADIUS, math.radians(10.0),
                                np.random.default_rng(3), fix_position=True)
check("위치 고정을 켜면 위치가 수렴 자세와 같다",
      bool(np.allclose(fixed[:3, 3], CENTER[:3, 3])))
check("위치 고정을 켜도 회전은 달라진다",
      not bool(np.allclose(fixed[:3, :3], np.eye(3))))

# ---------------------------------------------------------------- 2) 회전 행렬
print("\n2) 작은 무작위 회전")
max_angle = math.radians(10.0)
rng = np.random.default_rng(11)
ortho_ok, det_ok, angle_ok = True, True, True
worst_angle = 0.0
for _ in range(2000):
    R = sart_core.rand_rot(max_angle, rng)
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-9):
        ortho_ok = False
    if abs(float(np.linalg.det(R)) - 1.0) > 1e-9:
        det_ok = False
    angle = math.acos(max(-1.0, min(1.0, (float(np.trace(R)) - 1.0) / 2.0)))
    worst_angle = max(worst_angle, angle)
    if angle > max_angle + 1e-9:
        angle_ok = False
check("회전 행렬이 직교한다", ortho_ok)
check("회전 행렬의 행렬식이 1이다(뒤집기가 아니다)", det_ok)
check("회전 각도가 정한 상한 10도를 넘지 않는다", angle_ok,
      "최대 %.2f도" % math.degrees(worst_angle))

# ---------------------------------------------------------------- 3) 쥔 시점
print("\n3) 물체를 쥔 시점")
fingers = np.full((60, 2), 0.04)
fingers[25:, :] = 0.02
check("손가락이 닫힌 첫 스텝을 찾는다",
      sart_core.grasp_step(fingers, 0.035) == 25,
      "찾은 스텝 %d" % sart_core.grasp_step(fingers, 0.035))
check("끝까지 닫히지 않으면 0을 돌려준다",
      sart_core.grasp_step(np.full((60, 2), 0.04), 0.035) == 0)

# ---------------------------------------------------------------- 4) 수렴 시점
print("\n4) 수렴 시점을 고르는 세 규칙")
T = 100
T_GRASP = 20
TARGET_XY = (0.091, 0.104)
# 합성 궤적을 만든다. 20스텝에 물체를 쥐고, 55스텝까지 목표 축 쪽으로 수평으로 다가가고,
# 60스텝부터 손이 내려가기 시작한다. 내려가는 동안 물체는 목표 축에서 1 mm에서 3 mm
# 사이로 조금씩 흔들린다. 실제 삽입 궤적이 이 모양이다.
eef_z = np.full(T, 0.90)
eef_z[61:] = 0.90 - 0.004 * np.arange(1, T - 60)
dxy_track = np.empty(T)
dxy_track[:20] = 0.25
dxy_track[20:56] = np.linspace(0.25, 0.004, 36)
dxy_track[56:] = np.linspace(0.001, 0.003, T - 56)
# 목표에서 그만큼 떨어진 물체 위치를 x축 방향으로 만든다.
obj_xy = np.tile(np.array(TARGET_XY), (T, 1))
obj_xy[:, 0] += dxy_track
first_inside = int(np.argmax((np.arange(T) >= T_GRASP) & (dxy_track < 0.016)))

t_radial, fb_radial = sart_core.converge_step("radial_gate", eef_z, obj_xy, TARGET_XY,
                                              T_GRASP, 25, 0.016)
check("radial_gate가 목표 축 0.016 m 안에 드는 첫 스텝을 고른다",
      t_radial == first_inside and not fb_radial,
      "고른 스텝 %d, 기대 %d" % (t_radial, first_inside))

t_desc, fb_desc = sart_core.converge_step("descent_onset", eef_z, obj_xy, TARGET_XY,
                                          T_GRASP, 25, 0.016)
check("descent_onset이 손이 내려가기 시작하는 60스텝을 고른다",
      t_desc == 60 and not fb_desc, "고른 스텝 %d" % t_desc)

t_tail, fb_tail = sart_core.converge_step("tail_offset", eef_z, obj_xy, TARGET_XY,
                                          T_GRASP, 25, 0.016)
check("tail_offset이 끝에서 25스텝 앞선 75를 고른다",
      t_tail == 75 and not fb_tail, "고른 스텝 %d" % t_tail)

# 아무 규칙도 찾지 못하는 자료를 준다. 높이가 계속 올라가고 물체는 목표에서 멀다.
rising = np.linspace(0.80, 0.95, T)
far_xy = np.tile(np.array([1.0, 1.0]), (T, 1))
t_none, fb_none = sart_core.converge_step("radial_gate", rising, far_xy, TARGET_XY,
                                          T_GRASP, 25, 0.016)
check("radial_gate가 못 찾으면 끝에서 25스텝 앞선 값을 대체로 쓴다",
      t_none == 75 and fb_none, "고른 스텝 %d, 대체 %s" % (t_none, fb_none))
t_none2, fb_none2 = sart_core.converge_step("descent_onset", rising, far_xy, TARGET_XY,
                                            T_GRASP, 25, 0.016)
check("descent_onset이 못 찾으면 같은 대체값을 쓴다",
      t_none2 == 75 and fb_none2, "고른 스텝 %d, 대체 %s" % (t_none2, fb_none2))

bad = False
try:
    sart_core.converge_step("descent", eef_z, obj_xy, TARGET_XY, T_GRASP, 25, 0.016)
except ValueError as exc:
    bad = "radial_gate" in str(exc) and "tail_offset" in str(exc)
check("모르는 규칙 이름은 받을 수 있는 이름을 적어 거부한다", bool(bad))

everything = sart_core.converge_step_all(eef_z, obj_xy, TARGET_XY, T_GRASP, 25, 0.016)
check("세 규칙을 한 번에 계산해 모두 돌려준다",
      set(everything) == set(sart_core.CONVERGE_RULES),
      str({k: v[0] for k, v in everything.items()}))

# ---------------------------------------------------------------- 5) 구간 계획
print("\n5) 구간 나누기")
plan = sart_core.plan_segments(n_steps=100, t_conv=60, divert_steps=10,
                               converge_steps=20, settle_steps=5)
check("갈라지는 스텝이 수렴 스텝에서 되돌아오는 스텝 수만큼 앞이다",
      plan["t_branch"] == 40, "t_branch %d" % plan["t_branch"])
check("전체 길이가 원본 길이에 벗어남과 머무름만 더한 값이다",
      plan["expected_waypoints"] == 100 + 10 + 5,
      "%d개" % plan["expected_waypoints"])
check("수렴 스텝이 되돌아오는 스텝 수보다 작으면 갈라지는 스텝이 1이다",
      sart_core.plan_segments(100, 5, 10, 20, 5)["t_branch"] == 1)
check("구간이 이송, 벗어남, 되돌아옴, 머무름, 삽입 순서다",
      [s["name"] for s in plan["segments"]]
      == ["transport", "divert", "converge", "settle", "insert"])

# ---------------------------------------------------------------- 6) 그대로 재생
print("\n6) 이송 구간과 삽입 구간이 원본과 한 숫자도 다르지 않다")
rng = np.random.default_rng(5)
T = 100
source_pose = np.zeros((T, 4, 4))
for t in range(T):
    source_pose[t] = np.eye(4)
    source_pose[t][:3, 3] = rng.normal(size=3)
    source_pose[t][:3, :3] = sart_core.rand_rot(0.5, rng)
source_grip = rng.choice([-1.0, 1.0], size=(T, 1))
offset = sart_core.sample_offset(source_pose[60], -10.0, 0.06, math.radians(10.0),
                                 np.random.default_rng(1))
poses, grips = sart_core.assemble_segments(source_pose, source_grip, plan,
                                           offset, source_grip[60])
check("만들어진 길이가 계획한 길이와 같다",
      poses.shape[0] == plan["expected_waypoints"],
      "%d개" % poses.shape[0])
check("앞의 이송 구간 40스텝이 원본과 완전히 같다",
      bool(np.array_equal(poses[:40], source_pose[:40])))
check("이송 구간의 그리퍼 명령도 원본과 완전히 같다",
      bool(np.array_equal(grips[:40], source_grip[:40])))

# ---------------------------------------------------------------- 6-1) 한 방향 수렴
# 이것이 SART를 그냥 "돌아가는 길"과 구별하는 성질이다. 가운데에 끼워 넣은 구간은 뽑은
# 자세까지 벗어난 다음, 그 뒤로는 한 번도 멀어지지 않고 수렴 자세로 다가와야 한다.
# 가까워졌다가 다시 멀어지는 움직임을 기록하면 수렴을 방해하는 행동이 학습 자료에 들어간다.
# 참고 구현이 세 번의 적대적 검토에서 모두 지적받은 결함이 정확히 이것이다.
#
# 위에서 쓴 무작위 궤적으로는 이 성질을 볼 수 없다. 무작위 자세는 수렴 자세에서 이미 2 m쯤
# 떨어져 있어서, 벗어나는 구간마저 처음부터 가까워지는 것으로 보인다. 그래서 여기서는
# 실제 시연을 닮은 궤적을 만든다. 손끝이 수렴 자세로 매끄럽게 다가갔다가 그대로 내려간다.
print("\n6-1) 가운데 구간이 한 방향으로만 수렴한다")
T2 = 100
conv_xyz = np.array([0.30, 0.10, 0.80])
smooth_pose = np.zeros((T2, 4, 4))
for t in range(T2):
    smooth_pose[t] = np.eye(4)
    if t <= 60:                       # 0.25 m 밖에서 수렴 자세까지 곧게 다가간다
        a = t / 60.0
        smooth_pose[t][:3, 3] = conv_xyz + (1.0 - a) * np.array([0.25, 0.0, 0.10])
    else:                             # 수렴 자세에서 아래로 3 cm 내려가며 꽂는다
        smooth_pose[t][:3, 3] = conv_xyz - np.array([0.0, 0.0, 0.03 * (t - 60) / 39.0])
smooth_grip = np.full((T2, 1), -1.0)
plan2 = sart_core.plan_segments(n_steps=T2, t_conv=60, divert_steps=10,
                                converge_steps=20, settle_steps=5)
offset2 = sart_core.sample_offset(smooth_pose[60], conv_xyz[2] - 0.05, 0.06,
                                  math.radians(10.0), np.random.default_rng(3))
poses2, _ = sart_core.assemble_segments(smooth_pose, smooth_grip, plan2,
                                        offset2, smooth_grip[60])
mid_start = plan2["t_branch"]
mid_stop = mid_start + 10 + 20 + 5
conv_pose2 = smooth_pose[60]
mid_dist = np.linalg.norm(poses2[mid_start:mid_stop, :3, 3] - conv_pose2[:3, 3], axis=1)
divert_end = 10 - 1                # 벗어나는 구간의 마지막 웨이포인트 자리
offset_dist = float(np.linalg.norm(offset2[:3, 3] - conv_pose2[:3, 3]))
check("벗어나는 구간이 뽑은 자세에서 정확히 끝난다",
      abs(mid_dist[divert_end] - offset_dist) < 1e-9,
      "뽑은 자세까지 %.4f m, 구간 끝에서 %.4f m" % (offset_dist, mid_dist[divert_end]))
check("뽑은 자세가 수렴 자세와 다른 자리다",
      offset_dist > 1e-3, "%.4f m 떨어져 있다" % offset_dist)
after = mid_dist[divert_end:]
worsened = np.nonzero(np.diff(after) > 1e-9)[0]
check("뽑은 자세를 지난 뒤로는 한 번도 멀어지지 않는다",
      worsened.size == 0, "멀어진 지점 %d곳" % worsened.size)
check("뽑은 자세에 닿기 전에 수렴 자세를 먼저 지나가지 않는다",
      float(np.min(mid_dist[:divert_end + 1])) >= offset_dist - 1e-9,
      "벗어나는 동안 가장 가까웠던 거리 %.4f m"
      % float(np.min(mid_dist[:divert_end + 1])))
check("가운데 구간의 마지막 웨이포인트가 원본의 수렴 자세와 같다",
      bool(np.allclose(poses2[mid_stop - 1], conv_pose2)),
      "차이 %.2e" % float(np.abs(poses2[mid_stop - 1] - conv_pose2).max()))
check("삽입 구간이 수렴 자세에서 이어서 시작한다",
      bool(np.array_equal(poses2[mid_stop], smooth_pose[60])))

# 구간 순서를 뒤바꾼 계획으로는 위 검사가 반드시 깨져야 한다. 검사가 실제로 무언가를 보고
# 있는지 확인하는 것이다. 순서를 뒤집으면 수렴 자세에 먼저 갔다가 거기서 다시 벗어난다.
flipped = {k: (v if k != "segments" else [dict(seg) for seg in v]) for k, v in plan2.items()}
flipped["segments"][1]["pose"] = "conv"
flipped["segments"][2]["pose"] = "offset"
bad_poses, _ = sart_core.assemble_segments(smooth_pose, smooth_grip, flipped,
                                           offset2, smooth_grip[60])
bad_dist = np.linalg.norm(bad_poses[mid_start:mid_stop, :3, 3] - conv_pose2[:3, 3], axis=1)
check("구간 순서를 뒤집으면 위 검사가 실제로 깨진다",
      np.any(np.diff(bad_dist[divert_end:]) > 1e-9)
      or float(np.min(bad_dist[:divert_end + 1])) < offset_dist - 1e-9,
      "뒤집으면 벗어나는 동안 수렴 자세까지 %.4f m까지 가까워졌다가 다시 멀어진다"
      % float(np.min(bad_dist[:divert_end + 1])))

# ---------------------------------------------------------------- 7) 회전 표현
print("\n7) 회전 표현을 이 파일에서 다루지 않는다")
core_src = open(os.path.join(SRC, "sart", "sart_core.py"), encoding="utf-8").read()
check("sart_core.py에 네 숫자 회전 표현을 다루는 코드가 없다",
      "quat" not in core_src.lower())

# ---------------------------------------------------------------- 8) 프로필 대조
print("\n8) 프로필 값이 환경 코드의 상수와 어긋나지 않는다")
try:
    import yaml
except ImportError:
    print("  건너뜀  PyYAML이 없다")
else:
    mdp_src = open(os.path.join(SRC, "env_peg", "peg_mdp.py"), encoding="utf-8").read()
    m = re.search(r"^HOLE_INNER\s*=\s*([0-9.]+)", mdp_src, re.M)
    with open(os.path.join(SRC, "profiles", "peg_insert_fr3.yaml"), encoding="utf-8") as fh:
        peg_doc = yaml.safe_load(fh)
    block = ((peg_doc.get("generate") or {}).get("sart") or {})
    if m is None:
        check("peg_mdp.py에서 구멍 안지름을 읽는다", False)
    else:
        want = float(m.group(1)) / 2.0
        got = block.get("converge_radius_m")
        check("프로필의 수렴 반지름이 구멍 안지름의 절반과 같다",
              got is not None and abs(float(got) - want) < 1e-9,
              "프로필 %s, 구멍 안지름의 절반 %.4f" % (got, want))
    check("핀 삽입 프로필이 SART를 켜 둔다", block.get("enable") is True)

# ---------------------------------------------------------------- 9) 다양성 지표
# 지표가 실제로 무너진 증강과 살아 있는 증강을 구분하는지 본다. 구분하지 못하면 이 지표는
# 있으나 마나다. 실제로 한 번 그런 적이 있다. 에피소드 끝에서 90스텝만 보는 창을 쓰다가,
# 삽입 구간이 그보다 긴 핀 삽입에서 멀쩡한 증강을 0.99 mm로 보고했다.
print("\n9) 다양성 지표가 무너진 증강을 구분한다")
sys.path.insert(0, os.path.join(SRC, "sart"))
import sart_metrics  # noqa: E402

rng8 = np.random.default_rng(11)
base = np.zeros((200, 3))
base[:, 2] = np.linspace(0.9, 0.75, 200)          # 아래로 내려가는 공통 궤적
def make(spread):
    """앞 절반에 spread만큼 흩어지고 뒤 절반은 똑같은 궤적을 다섯 편 만든다."""
    out = []
    for _ in range(5):
        a = base.copy()
        a[80:110, :2] += rng8.normal(scale=spread, size=(30, 2))
        out.append(a)
    return out

alive = sart_metrics.approach_std_profile(make(0.02), tail=30)
dead = sart_metrics.approach_std_profile(make(0.0), tail=30)
check("살아 있는 증강은 접근 구간 다양성이 삽입 구간보다 훨씬 크다",
      alive["peak_over_tail"] is not None and alive["peak_over_tail"] > 3.0,
      "접근 %.4f m, 삽입 %.6f m, 대비 %s배"
      % (alive["peak_m"], alive["tail_m"], alive["peak_over_tail"]))
check("무너진 증강은 대비가 크지 않다",
      dead["peak_over_tail"] is not None and dead["peak_over_tail"] <= 3.0,
      "접근 %.6f m, 대비 %s배" % (dead["peak_m"], dead["peak_over_tail"]))
check("다양성이 있는 자리가 에피소드 중간이어도 놓치지 않는다",
      80 <= alive["peak_index"] < 110, "가장 큰 자리 %d번째" % alive["peak_index"])
check("잴 수 없으면 0이 아니라 None을 돌려준다",
      sart_metrics.approach_std_profile([base], tail=30)["peak_m"] is None)

print()
if failures:
    print("어긋난 항목 %d개: %s" % (len(failures), ", ".join(failures)))
    sys.exit(1)
print("SART 기하 계산이 기대대로 동작한다")
