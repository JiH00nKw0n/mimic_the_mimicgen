#!/usr/bin/env python3
"""SART 증강에서 시뮬레이터가 없어도 계산할 수 있는 부분만 모은 모듈.

SART(Self-Augmented Robot Trajectory)는 사람 시연 한 편에서 여러 편을 스스로 만들어 내는
증강 방법이다. 정밀하게 맞춰야 하는 순간, 예를 들어 핀을 구멍 바로 위까지 가져온 순간의
손 자세를 하나 정해 두고, 그 자세 주변의 공 모양 영역에서 손 자세를 하나 뽑아 거기서
출발해 원래 자세로 되돌아오게 만든다. 되돌아오는 구간만 새로 만들고 그 앞뒤는 원본을
그대로 재생하므로, 물체를 옮기는 구간과 구멍에 밀어 넣는 구간은 원본과 한 스텝도 다르지
않고 접근 경로만 달라진다.

이 파일에 Isaac Lab을 부르는 코드가 한 줄도 없는 이유는 검사 때문이다. 자세 계산이
틀리면 증강이 통째로 무너지는데, GPU 서버에 올려 보기 전에 노트북에서 그 계산만 따로
확인할 수 있어야 한다. src/env_peg/peg_geom.py를 같은 이유로 떼어 놓은 것과 같다.

여기서 자세는 4x4 행렬 하나다. 왼쪽 위 3x3이 회전이고 오른쪽 위 3x1이 위치다. 회전은
3x3 행렬로만 다루고 네 숫자로 줄여 쓰는 표현은 쓰지 않는다. 그 표현은 순서 규약이
버전마다 달라서 한 번 잘못 읽었다가 생성 697회가 모두 실패한 적이 있고, 그 규약을 다루는
코드는 전부 Isaac Lab 쪽 파일에 모아 두었다.

    python3 src/tests/test_sart_core.py
"""
from __future__ import annotations

import numpy as np

# 수렴 시점을 고르는 세 가지 규칙. 무엇이 맞는지는 태스크마다 다르므로 이름으로 고른다.
#   radial_gate    물체를 쥔 뒤 목표 축까지의 수평 거리가 정해진 값보다 작아지는 첫 스텝.
#                  핀 꽂기는 구멍 안지름의 절반인 0.016 m를 쓴다. 태스크가 스스로 정한
#                  공차라서 에피소드 길이가 달라져도 값이 흔들리지 않는다.
#   descent_onset  손이 세 스텝 연속으로 내려가기 시작하면서 물체가 이미 목표 가까이에
#                  있는 첫 스텝. robosuite Square에서 실제로 측정해 55%의 증강 성공률을
#                  낸 규칙이지만, 그 태스크는 움직이는 물체와 고정 목표가 핀 꽂기와
#                  반대라서 그대로 옮겨 오면 맞지 않을 수 있다.
#   tail_offset    끝에서 tail_steps만큼 앞선 스텝. 앞의 두 규칙이 아무것도 찾지 못했을
#                  때 쓰는 값이기도 하다.
CONVERGE_RULES = ("radial_gate", "descent_onset", "tail_offset")


class DegenerateOffset(RuntimeError):
    """샘플링한 접근 자세가 모두 바닥면 아래로 떨어져 쓸 것이 없다는 뜻이다."""


# ------------------------------------------------------------------ 두 시점 찾기
def grasp_step(finger_pos, closed_m: float) -> int:
    """물체를 쥔 첫 스텝을 돌려준다.

    finger_pos는 (T, 2) 배열이고 두 손가락 관절의 열림 정도가 미터 단위로 들어 있다.
    두 값의 절댓값 평균이 closed_m 이하로 내려간 첫 스텝을 쥔 순간으로 본다. 끝까지
    그런 스텝이 없으면 0을 돌려준다.

    생성된 에피소드 파일에는 구간 경계 표시가 들어 있지 않다. 그 표시는 시뮬레이터가
    도는 동안에만 존재하는 관측값이라 파일에 남지 않는다. 그래서 기록된 관절 각도에서
    다시 계산한다.
    """
    fp = np.asarray(finger_pos, dtype=float)
    if fp.ndim != 2 or fp.shape[1] != 2:
        raise ValueError(f"finger_pos는 (T, 2)여야 하는데 {fp.shape}이다")
    if fp.shape[0] == 0:
        return 0
    closed = np.abs(fp).mean(axis=1) <= float(closed_m)
    index = int(np.argmax(closed))
    return index if bool(closed[index]) else 0


def _tail_value(n_steps: int, t_grasp: int, tail_steps: int) -> int:
    """끝에서 tail_steps만큼 앞선 스텝. 항상 1 이상 n_steps - 1 이하로 잘라 준다."""
    n = int(n_steps)
    value = max(int(t_grasp) + 1, n - int(tail_steps))
    top = max(1, n - 1)
    return int(min(max(value, 1), top))


def converge_step(rule: str, eef_z, obj_xy, target_xy, t_grasp: int,
                  tail_steps: int, converge_radius_m: float,
                  percentile: float = 35.0) -> tuple:
    """수렴 시점 한 개와 그 값이 대체값인지 여부를 (t_conv, used_fallback)로 돌려준다.

    수렴 시점이란 증강한 접근 경로가 되돌아와야 하는 목표 스텝이다. 이 스텝부터 끝까지는
    원본을 그대로 재생한다.

    eef_z는 (T,) 손끝 높이, obj_xy는 (T, 2) 옮기는 물체의 수평 위치, target_xy는 고정
    목표의 수평 위치다. used_fallback이 참이면 규칙이 아무것도 찾지 못해 끝에서
    tail_steps만큼 앞선 스텝을 대신 썼다는 뜻이다.
    """
    if rule not in CONVERGE_RULES:
        raise ValueError(f"모르는 수렴 시점 규칙 {rule!r}이다. "
                         f"{', '.join(CONVERGE_RULES)} 중 하나를 적는다")
    z = np.asarray(eef_z, dtype=float).reshape(-1)
    xy = np.asarray(obj_xy, dtype=float).reshape(-1, 2)
    target = np.asarray(target_xy, dtype=float).reshape(2)
    n = int(min(z.shape[0], xy.shape[0]))
    start = max(int(t_grasp), 0)
    fallback = _tail_value(n, t_grasp, tail_steps)

    if rule == "tail_offset":
        return fallback, False
    if n < 4 or start >= n:
        return fallback, True

    dxy = np.linalg.norm(xy[:n] - target[None, :], axis=1)

    if rule == "radial_gate":
        for t in range(start, n):
            if dxy[t] < float(converge_radius_m):
                return int(min(max(t, 1), n - 1)), False
        return fallback, True

    # descent_onset
    window = dxy[start:n]
    if window.size == 0:
        return fallback, True
    thresh = float(np.percentile(window, float(percentile)))
    for t in range(max(start, 1), n - 3):
        if (z[t + 1] < z[t] and z[t + 2] < z[t + 1] and z[t + 3] < z[t + 2]
                and dxy[t] < thresh):
            return int(t), False
    return fallback, True


def converge_step_all(eef_z, obj_xy, target_xy, t_grasp: int, tail_steps: int,
                      converge_radius_m: float, percentile: float = 35.0) -> dict:
    """세 규칙을 모두 계산해 규칙 이름을 키로 하는 사전으로 돌려준다.

    계산 비용이 거의 없으므로 실행에 쓰는 규칙과 무관하게 전부 구해 보고서에 적는다.
    그러면 시범 실행 한 번으로 이 태스크에 어떤 규칙이 맞는지 눈으로 확인할 수 있고,
    추측으로 고르지 않아도 된다.
    """
    return {rule: converge_step(rule, eef_z, obj_xy, target_xy, t_grasp,
                                tail_steps, converge_radius_m, percentile)
            for rule in CONVERGE_RULES}


# --------------------------------------------------------------- 접근 자세 뽑기
def rand_rot(max_angle_rad: float, rng) -> np.ndarray:
    """작은 무작위 회전 3x3 행렬을 만든다.

    축은 3차원 정규분포에서 뽑아 길이를 1로 맞추므로 구면 위에서 고르게 나온다. 각도는
    -max_angle_rad 이상 +max_angle_rad 이하에서 고르게 뽑는다. 로드리게스 공식으로 축과
    각도를 바로 행렬로 바꾼다.
    """
    axis = np.asarray(rng.standard_normal(3), dtype=float)
    axis = axis / (float(np.linalg.norm(axis)) + 1e-9)
    angle = float(rng.uniform(-float(max_angle_rad), float(max_angle_rad)))
    K = np.array([[0.0, -axis[2], axis[1]],
                  [axis[2], 0.0, -axis[0]],
                  [-axis[1], axis[0], 0.0]], dtype=float)
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def sample_offset(center_pose, floor_z: float, radius_m: float, max_angle_rad: float,
                  rng, fix_position: bool = False, max_tries: int = 20) -> np.ndarray:
    """수렴 자세 주변에서 접근을 시작할 자세 하나를 뽑아 4x4 행렬로 돌려준다.

    중심은 수렴 스텝에서 **명령한** 손 자세다. 실제로 도달한 자세가 아니라 명령한
    자세인 이유는, 증강 궤적도 같은 방식으로 명령을 넣어 실행하기 때문이다.

    방향은 3차원 정규분포를 길이 1로 맞춰 구면 위에서 고르게 뽑는다. 반지름은
    radius_m 곱하기 균등난수의 세제곱근이라 공 안에서 부피 기준으로 고르게 찬다.
    뽑은 점의 높이가 floor_z 이상일 때만 받아들인다.

    이 조건은 높이 방향으로 자른 반평면 하나가 전부다. 옆으로 얼마나 벌어지는지에는
    아무 제한이 없고, 충돌 검사도 도달 가능성 검사도 하지 않는다. 걸러 내는 것은 오직
    태스크 성공 여부다.

    예전 판은 반지름을 손이 물체 위에 떠 있는 높이의 절반으로 제한했다. 수렴 시점에는
    손이 이미 물체 바로 위에 있어서 그 값이 거의 0이 되었고, 증강한 에피소드가 전부
    원본의 복사본이 되었다. 같은 소스에서 나온 접근 경로의 표준편차가 0.0025 m,
    즉 2.5 mm에 그쳤다. 높이로 자르는 지금 방식으로 바꾸고 약 15 mm를 회복했다.

    max_tries번 뽑아도 조건을 만족하는 점이 없으면 DegenerateOffset을 던진다. 중심을
    그대로 돌려주지 않는 이유는, 그러면 그 시도가 원본을 그대로 복사하게 되어 이
    조건을 넣은 목적이 사라지기 때문이다.

    fix_position이 참이면 위치를 수렴 자세 그대로 두고 회전만 흔든 자세를 만든다.
    RoboManipAug 원본의 --position_fix가 하는 일이 이것이다. robosuite 판
    sart_mimicgen.py는 같은 이름의 값이 반대로 회전을 고정하게 되어 있는데, 그 판을
    옮겨 오지 않는다. 측정에 쓴 실행은 이 값이 꺼진 상태였으므로 측정치는 그대로다.
    """
    pose = np.asarray(center_pose, dtype=float)
    if pose.shape != (4, 4):
        raise ValueError(f"center_pose는 (4, 4)여야 하는데 {pose.shape}이다")
    center = pose[:3, 3].copy()
    rotation = pose[:3, :3].copy()

    if fix_position:
        position = center
    else:
        position = None
        for _ in range(int(max_tries)):
            direction = np.asarray(rng.standard_normal(3), dtype=float)
            direction = direction / (float(np.linalg.norm(direction)) + 1e-9)
            radius = float(radius_m) * (float(rng.random()) ** (1.0 / 3.0))
            candidate = center + radius * direction
            if candidate[2] >= float(floor_z):
                position = candidate
                break
        if position is None:
            raise DegenerateOffset(
                f"{int(max_tries)}번 뽑는 동안 바닥 높이 {float(floor_z):.4f} m 위로 올라온 "
                f"자세가 없다. 반지름 {float(radius_m):.4f} m가 수렴 자세의 높이 여유 "
                f"{float(center[2]) - float(floor_z):.4f} m에 비해 크다")

    out = np.eye(4)
    out[:3, 3] = position
    out[:3, :3] = rotation @ rand_rot(float(max_angle_rad), rng)
    return out


# ------------------------------------------------------------------ 구간 나누기
def plan_segments(n_steps: int, t_conv: int, divert_steps: int,
                  converge_steps: int, settle_steps: int) -> dict:
    """증강 궤적을 다섯 구간으로 나눈 계획을 돌려준다.

    다섯 구간은 순서대로 이렇다. 원본 그대로 재생하는 이송 구간, 뽑은 자세까지 벗어나는
    구간, 거기서 수렴 자세로 한 방향으로 돌아오는 구간, 수렴 자세에 머무르며 제어기가
    실제로 그 자세에 도달하게 두는 구간, 그리고 원본 그대로 재생하는 삽입 구간이다.

    핵심은 t_branch를 max(1, t_conv - converge_steps)로 잡는다는 점이다. 원본의 마지막
    접근 스텝 converge_steps개를 버리고 그 자리에 벗어남과 되돌아옴을 끼워 넣는다.
    그래서 기록된 데이터에는 "수렴 자세에 도착했다가 다시 나갔다가 돌아온다"는 움직임이
    없다. 그런 움직임을 기록하면 수렴을 방해하는 행동이 학습 자료에 들어간다.

    전체 길이는 t_conv가 converge_steps보다 클 때 n_steps + divert_steps + settle_steps다.
    converge_steps개가 서로 상쇄되기 때문이다. 기본값에서 원본보다 15스텝 길어질 뿐이고
    두 배가 되지 않는다.

    expected_waypoints는 이 계산에 따른 웨이포인트 개수다. Isaac Lab의 보간 함수가
    스텝 수를 세는 방식이 한 개 다를 수 있으므로, 실행기는 실제로 만들어진 개수를 따로
    세어 보고서에 적는다.
    """
    n = int(n_steps)
    if n < 2:
        raise ValueError(f"소스 에피소드가 {n}스텝이라 구간을 나눌 수 없다")
    tc = int(min(max(int(t_conv), 1), n - 1))
    divert = int(divert_steps)
    converge = int(converge_steps)
    settle = int(settle_steps)
    t_branch = max(1, tc - converge)
    segments = [
        {"name": "transport", "kind": "verbatim", "start": 0, "stop": t_branch},
        {"name": "divert", "kind": "target", "pose": "offset", "grip": "conv",
         "steps": divert},
        {"name": "converge", "kind": "target", "pose": "conv", "grip": "conv",
         "steps": converge},
        {"name": "settle", "kind": "hold", "pose": "conv", "grip": "conv",
         "steps": settle},
        {"name": "insert", "kind": "verbatim", "start": tc, "stop": n},
    ]
    return {
        "t_branch": t_branch,
        "t_conv": tc,
        "transport": (0, t_branch),
        "insert": (tc, n),
        "expected_waypoints": t_branch + divert + converge + settle + (n - tc),
        "segments": segments,
    }


def linear_pose_interp(pose_a, pose_b, num_steps: int) -> np.ndarray:
    """두 자세 사이를 성분마다 직선으로 이어 num_steps개를 만든다. 시작 자세는 뺀다.

    검사와 형상 확인에만 쓰는 기본 보간이다. 실제 실행은 Isaac Lab의 보간을 넘겨받아
    쓴다. 회전을 성분별 직선으로 이으면 정확히 회전 행렬이 되지 않기 때문이다.
    """
    a = np.asarray(pose_a, dtype=float)
    b = np.asarray(pose_b, dtype=float)
    steps = int(num_steps)
    if steps < 1:
        return np.zeros((0,) + a.shape, dtype=float)
    weights = np.linspace(0.0, 1.0, steps + 1)[1:]
    return np.stack([(1.0 - w) * a + w * b for w in weights], axis=0)


def assemble_segments(target_pose, gripper, plan: dict, offset_pose, grip_conv,
                      interp=linear_pose_interp) -> tuple:
    """계획대로 자세와 그리퍼 명령을 이어 붙여 (자세 배열, 그리퍼 배열)을 돌려준다.

    이송 구간과 삽입 구간은 원본 배열의 값을 그대로 복사하므로 한 숫자도 달라지지 않는다.
    검사가 확인하는 것이 그 점이다. 실행기는 같은 plan["segments"]를 읽어 Isaac Lab의
    웨이포인트로 만든다. 그래서 구간 순서와 첨자 계산이 두 곳에 따로 있지 않다.
    """
    tp = np.asarray(target_pose, dtype=float)
    gp = np.asarray(gripper, dtype=float)
    if gp.ndim == 1:
        gp = gp.reshape(-1, 1)
    conv_pose = tp[int(plan["t_conv"])]
    named = {"offset": np.asarray(offset_pose, dtype=float), "conv": conv_pose}
    grip_c = np.asarray(grip_conv, dtype=float).reshape(-1)

    poses, grips, last = [], [], None
    for seg in plan["segments"]:
        if seg["kind"] == "verbatim":
            for t in range(int(seg["start"]), int(seg["stop"])):
                poses.append(tp[t])
                grips.append(gp[t])
        elif seg["kind"] == "target":
            goal = named[seg["pose"]]
            base = last if last is not None else tp[0]
            for pose in interp(base, goal, int(seg["steps"])):
                poses.append(np.asarray(pose, dtype=float))
                grips.append(grip_c)
        elif seg["kind"] == "hold":
            goal = named[seg["pose"]]
            for _ in range(int(seg["steps"])):
                poses.append(goal)
                grips.append(grip_c)
        else:
            raise ValueError(f"모르는 구간 종류 {seg['kind']!r}이다")
        if poses:
            last = poses[-1]
    if not poses:
        raise ValueError("이어 붙일 구간이 하나도 없다")
    return np.stack(poses, axis=0), np.stack(grips, axis=0)
