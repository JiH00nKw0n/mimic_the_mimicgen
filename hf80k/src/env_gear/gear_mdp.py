"""기어 조립 MimicGen용 mdp 함수들.

핀 꽂기의 peg_mdp와 같은 구실이다. 세 가지를 만든다.

  * ``gear_grasped``       -> 집기 구간의 완료 신호 여섯 개 중 세 개
  * ``gear_seated``        -> 꽂기 구간의 완료 신호
  * ``all_gears_assembled``-> ``success`` 종료 조건
  * ``randomize_gear_layout`` -> 리셋 때 받침과 기어 세 개의 자리를 새로 뽑는 이벤트

모든 좌표는 환경 기준이다. 즉 월드 좌표에서 그 환경의 원점을 뺀 값이다. MimicGen은 환경을
여러 개 띄워 바닥에 늘어놓으므로, 원점을 빼지 않으면 0번 환경에서만 맞는다.

기준값과 기하는 시연을 만든 코드(multigear_cap_demos.py, teleop_multigear_lab_v3.py)와
배치 파일(legacy_multi_gear/config/scene_layout.json)에서 그대로 가져왔다. 숫자를 새로
정한 곳은 무작위화 범위뿐이고, 그것도 시연 20편의 실제 배치에서 뽑은 최솟값과 최댓값이다.
"""

from __future__ import annotations

import os

import torch

from isaaclab.managers import SceneEntityCfg

from gear_geom import (AXIS_X, AXIS_XY_TOL_M, AXIS_Z, GEARS, SPEED_TOL_M_S,  # noqa: F401
                       TILT_TOL_RAD, Z_TOL_M, axis_xy_error, quat_apply_xyzw,
                       tilt_rad_from_quat, upright_z_from_quat)

# ---------------------------------------------------------------------------- 장면 기하
DESK_Z = 0.720                 # 작업면 높이. scene_layout.json의 desk_top_z다
GEAR_SPAWN_Z = 0.715           # 기어 원점이 놓이는 높이. 시연 20편 전부 이 값이다
BASE_SPAWN_Z = 0.720           # 받침 원점이 놓이는 높이. 시연 20편 전부 이 값이다

# 손가락을 닫을 때 기어마다 주는 목표 반폭, 미터. 기어 위쪽 돌출부의 반지름에서 살짝
# 깎은 값이다. 집기 판정에는 쓰지 않고 기록용으로 둔다.
HALF_GAP = {"large": 0.0178, "small": 0.0078, "medium": 0.0140}

# 집기 판정에서 손끝과 기어 원점 사이에 허용하는 거리, 미터. 기어는 위쪽 돌출부를
# 집는데 그 자리가 원점에서 떨어져 있어서 핀(0.13)보다 넉넉하게 둔다. 이 값이 너무
# 작으면 집고 나서 들어 올리는 동안 신호가 꺼져 구간 경계가 흔들린다.
GRASP_DIFF_THRESHOLD = 0.15

# FR3 이진 그리퍼. gear_mimic_cfg가 설정에 넣는 값과 같아야 한다.
GRIPPER_OPEN_VAL = 0.04
GRIPPER_THRESHOLD = 0.005

# ---------------------------------------------------------------------------- 무작위화 범위
# 시연 20편의 실제 배치에서 뽑은 최솟값과 최댓값이다. 다른 태스크의 값을 옮겨 쓰면 안 된다.
#
#   받침  x 0.120에서 0.230, y -0.230에서 0.500, z 0.720 고정, 회전 -0.488에서 0.535 rad
#   기어  x 0.035에서 0.292, y -0.288에서 0.464, z 0.715 고정
#   기어끼리 최소 거리 0.126 m, 받침과 기어 최소 거리 0.091 m
#
# 범위를 시연보다 넓히지 않았다. 시연이 닿지 않은 자리에서는 MimicGen이 옮겨 붙일 궤적의
# 근거가 없어서, 넓히면 실패만 늘고 수율의 뜻이 흐려진다.
BASE_REGION_X = (0.12, 0.23)
BASE_REGION_Y = (-0.23, 0.50)
BASE_YAW_RANGE = (-0.49, 0.54)
GEAR_REGION_X = (0.035, 0.292)
GEAR_REGION_Y = (-0.288, 0.464)
GEAR_MIN_SEP = 0.126           # 기어끼리
GEAR_BASE_MIN_SEP = 0.091      # 기어와 받침

# 시연이 시작하는 팔 자세. 관절 일곱 개와 손가락 두 개의 각도이고 단위는 라디안이다.
# 손가락의 0.04는 완전히 열린 값이다. multigear_annotated.hdf5의
# initial_state/articulation/robot/joint_position을 20편 모두 읽어 보면 전부 이 값이다.
# 이 자세에서 손끝은 환경 기준 (0.413, 0.138, 1.209)에 있다.
#
# 큐브 쌓기가 쓰는 FR3_HOME_JOINT_POSE와 다른 값이므로 그것을 가져다 쓰면 안 된다.
# 큐브 쪽은 (0, -0.569, 0, -2.810, 0, 2.25, 0.741)이고 기어 시연은 아래 값이다.
GEAR_HOME_JOINT_POSE = (0.0, -0.7854, 0.0, -2.3562, 0.0, 1.5708, 0.7854, 0.04, 0.04)

# 리셋마다 팔 관절에 더하는 잡음의 표준편차, 라디안. 공식 Franka 리셋이 쓰는 0.02와 같다.
# 시작 자세에 다양성을 주려고 넣는다. 손가락에는 더하지 않는다.
ARM_JITTER_STD = float(os.environ.get("LAB_ARM_JITTER", "0.02"))


def reset_arm_to_home(env, env_ids, pose=GEAR_HOME_JOINT_POSE,
                      arm_jitter_std=ARM_JITTER_STD):
    """리셋 때 팔을 시연이 시작하는 집 자세에 즉시 가져다 둔다.

    없으면 무슨 일이 생기는가. 2026-10-22에 이 이벤트가 없는 상태로 생성을 돌려 보았다.
    설정에 적어 둔 초기 관절값은 제대로 전해져 있었고(물체의 default_joint_pos가 집
    자세였다) 그런데도 리셋 직후 관절은 (-0.03, -0.03, 0.05, -0.15, -0.15, 0.29, 0.20)
    이었다. 즉 기본값이 시뮬레이터에 써지지 않는다. 기본값을 관절에 써넣는 일은 리셋
    이벤트가 하는데, 기어 환경은 물려받은 이벤트를 전부 None으로 비우고 기어 배치
    이벤트 하나만 달아 두었기 때문에 그 일을 하는 것이 아무것도 없었다.

    그 결과 팔은 거의 수직으로 뻗은 자세에서 시작한다. 손끝이 z 1.600에 서는데 시연은
    z 1.209에서 시작하므로 0.39 m 차이다. MimicGen은 구간의 첫 목표까지를 다섯 스텝에
    이어 붙이므로 한 스텝에 0.08 m짜리 명령이 나가고, 뻗은 자세는 야코비안이 나빠서
    팔이 명령의 20%만 따라간다. 팔은 작업대 반대쪽(x 1.3에서 1.6 m)을 헤매다 끝났고
    기어 세 개는 처음 놓인 자리에 그대로 남았다. 생성 329번 모두 실패했다.

    큐브 쌓기와 핀 꽂기는 같은 일을 하는 init_franka_arm_pose 이벤트를 달고 있다.
    기어만 빠져 있었다.

    PD 목표를 주는 것(set_joint_position_target)이 아니라 관절 상태를 직접 써넣는다
    (write_joint_state_to_sim). 목표만 주면 리셋 직후 첫 스텝에서 팔이 아직 옛 자리에
    있고, 생성은 바로 그 첫 스텝의 손끝 자리를 기준으로 명령을 만든다.
    """
    robot = env.scene["robot"]
    if not isinstance(env_ids, torch.Tensor):
        env_ids = torch.tensor(env_ids, device=env.device, dtype=torch.long)
    n = int(env_ids.numel())
    if n == 0:
        return
    p = torch.tensor(pose, device=env.device, dtype=torch.float32).repeat(n, 1)
    if arm_jitter_std > 0.0:
        noise = arm_jitter_std * torch.randn((n, p.shape[1]), device=env.device)
        noise[:, -2:] = 0.0            # 손가락은 열린 값 그대로 둔다
        p = p + noise
    robot.write_joint_state_to_sim(p, torch.zeros_like(p), env_ids=env_ids)


def _as_torch(value):
    """이 Isaac 판본은 물리 값을 warp 배열로 돌려줄 때가 있다. 파이토치 텐서로 바꾼다.

    ``.data.root_quat_w``가 ProxyArray로 오면 뒤의 계산이
    "Expected a value of type 'Tensor' ... but instead found type 'ProxyArray'"로 죽는다.
    핀 꽂기에서 같은 문제를 겪었고 같은 방식으로 푼다.
    """
    if isinstance(value, torch.Tensor):
        return value
    try:
        import warp as wp
        return wp.to_torch(value).clone()
    except Exception:                                   # noqa: BLE001
        return torch.as_tensor(value)


def _root(env, name: str):
    """물체 하나의 환경 기준 위치와 XYZW 자세를 돌려준다."""
    obj = env.scene[name]
    pos = _as_torch(obj.data.root_pos_w) - _as_torch(env.scene.env_origins)
    quat = _as_torch(obj.data.root_quat_w)
    if quat.ndim == 1 or quat.shape[-1] != 4:
        quat = quat.reshape(-1, 4)
    return pos, quat


# ---------------------------------------------------------------------------- 구간 완료 신호
def gear_grasped(
    env,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("large"),
    diff_threshold: float = GRASP_DIFF_THRESHOLD,
    gripper_open_val: float = GRIPPER_OPEN_VAL,
    gripper_threshold: float = GRIPPER_THRESHOLD,
) -> torch.Tensor:
    """기어 하나를 쥐었는지. 손끝이 기어 원점에서 diff_threshold 안에 있고 손가락 둘이
    모두 열린 값에서 gripper_threshold 넘게 닫혀 있으면 참이다.

    공식 스택 환경의 ``mdp.object_grasped``와 같은 구조이고 대상만 기어로 바꿨다.
    손가락은 관절 배열의 마지막 두 개다. FR3에서 fr3_finger_joint1과 2가 마지막 둘이다.
    """
    robot = env.scene[robot_cfg.name]
    ee_frame = env.scene[ee_frame_cfg.name]
    obj = env.scene[object_cfg.name]

    object_pos = _as_torch(obj.data.root_pos_w)
    ee_pos = _as_torch(ee_frame.data.target_pos_w)[:, 0, :]
    pose_diff = torch.linalg.vector_norm(object_pos - ee_pos, dim=1)

    joint_pos = _as_torch(robot.data.joint_pos)
    grasped = pose_diff < diff_threshold
    grasped = torch.logical_and(
        grasped, torch.abs(joint_pos[:, -1] - gripper_open_val) > gripper_threshold)
    grasped = torch.logical_and(
        grasped, torch.abs(joint_pos[:, -2] - gripper_open_val) > gripper_threshold)
    return grasped


def gear_seated(
    env,
    gear_name: str = "large",
    base_cfg: SceneEntityCfg = SceneEntityCfg("base"),
    axis_tol: float = AXIS_XY_TOL_M,
    z_tol: float = Z_TOL_M,
    tilt_tol: float = TILT_TOL_RAD,
    speed_tol: float = SPEED_TOL_M_S,
) -> torch.Tensor:
    """기어 하나가 제 축에 제대로 앉았는지. 네 조건을 모두 만족해야 참이다.

    첫째로 기어의 축 중심이 받침의 해당 축 중심에서 가로세로로 axis_tol 안에 있어야 한다.
    둘째로 기어 원점과 받침 원점의 높이 차이가 z_tol 안이어야 한다.
    셋째로 기어가 tilt_tol보다 덜 기울어져 있어야 한다.
    넷째로 기어가 speed_tol보다 느려야 한다. 아직 떨어지는 중인 기어를 성공으로 세지
    않기 위해서다.

    기준값 네 개는 시연을 만든 코드의 assembly_status에서 그대로 가져왔다. 그 코드가
    시연 20편의 기어 60개 전부를 이 기준으로 다시 확인했다고 적고 있다.
    """
    gear_pos, gear_quat = _root(env, gear_name)
    base_pos, base_quat = _root(env, base_cfg.name)

    axis_err = axis_xy_error(gear_pos, gear_quat, base_pos, base_quat, gear_name)
    z_err = torch.abs(gear_pos[:, 2] - base_pos[:, 2])
    tilt = tilt_rad_from_quat(gear_quat)
    vel = _as_torch(env.scene[gear_name].data.root_vel_w)
    if vel.ndim == 1 or vel.shape[-1] < 3:
        vel = vel.reshape(-1, 6)
    speed = torch.linalg.vector_norm(vel[:, :3], dim=-1)

    result = ((axis_err < axis_tol) & (z_err < z_tol)
              & (tilt < tilt_tol) & (speed < speed_tol))
    if os.environ.get("LAB_GEAR_DEBUG", "") == "1":
        print(f"[gear_seated:{gear_name}] 축오차={axis_err[0]:.5f}(<{axis_tol}) "
              f"높이차={z_err[0]:.5f}(<{z_tol}) 기울기={tilt[0]:.4f}(<{tilt_tol}) "
              f"속도={speed[0]:.4f}(<{speed_tol}) -> {bool(result[0])}", flush=True)
    return result


def all_gears_assembled(
    env,
    base_cfg: SceneEntityCfg = SceneEntityCfg("base"),
) -> torch.Tensor:
    """``success`` 종료 조건. 기어 셋이 모두 제 축에 앉아 있으면 참이다."""
    out = None
    for name in GEARS:
        ok = gear_seated(env, gear_name=name, base_cfg=base_cfg)
        out = ok if out is None else (out & ok)
    return out


# ---------------------------------------------------------------------------- 리셋 무작위화
def randomize_gear_layout(
    env,
    env_ids: torch.Tensor,
    base_x_range: tuple = BASE_REGION_X,
    base_y_range: tuple = BASE_REGION_Y,
    base_yaw_range: tuple = BASE_YAW_RANGE,
    gear_x_range: tuple = GEAR_REGION_X,
    gear_y_range: tuple = GEAR_REGION_Y,
    gear_min_sep: float = GEAR_MIN_SEP,
    gear_base_min_sep: float = GEAR_BASE_MIN_SEP,
    base_z: float = BASE_SPAWN_Z,
    gear_z: float = GEAR_SPAWN_Z,
    max_tries: int = 200,
) -> None:
    """리셋 때 받침 하나와 기어 세 개의 자리를 새로 뽑아 써넣는다.

    생성에서는 이 이벤트가 있어야 새 배치가 나온다. 어노테이션 재생에서는 반드시 꺼야
    한다. 기록된 초기 배치를 덮어 버리면 재생하는 로봇이 빈 자리를 집는다. 핀 꽂기에서
    실제로 그렇게 돼서 12편 전부 실패로 끝난 일이 있다. ``LAB_DISABLE_RESET_RANDOMIZATION``을
    1로 두면 gear_mimic_cfg가 이 이벤트를 아예 붙이지 않는다.

    뽑는 방식은 거절 표집이다. 받침을 먼저 뽑고, 기어를 하나씩 뽑으면서 이미 뽑은 것들과
    gear_min_sep보다 가깝거나 받침과 gear_base_min_sep보다 가까우면 버리고 다시 뽑는다.
    max_tries 안에 못 뽑으면 시연 20편의 평균 자리로 되돌린다. 그 자리는 겹치지 않는다.
    """
    device = env.device
    if not isinstance(env_ids, torch.Tensor):
        env_ids = torch.tensor(env_ids, device=device, dtype=torch.long)
    n = int(env_ids.numel())
    if n == 0:
        return
    origins = _as_torch(env.scene.env_origins)[env_ids]

    def draw(x_range, y_range, m):
        return (torch.empty(m, device=device).uniform_(*x_range),
                torch.empty(m, device=device).uniform_(*y_range))

    bx, by = draw(base_x_range, base_y_range, n)
    yaw = torch.empty(n, device=device).uniform_(*base_yaw_range)

    placed = []                     # 이미 자리를 정한 기어들의 (x, y)
    gear_xy = {}
    for name in GEARS:
        gx = torch.empty(n, device=device)
        gy = torch.empty(n, device=device)
        todo = torch.arange(n, device=device)
        for _ in range(max_tries):
            if todo.numel() == 0:
                break
            m = int(todo.numel())
            cx, cy = draw(gear_x_range, gear_y_range, m)
            keep = torch.hypot(cx - bx[todo], cy - by[todo]) > gear_base_min_sep
            for px, py in placed:
                keep &= torch.hypot(cx - px[todo], cy - py[todo]) > gear_min_sep
            sel = todo[keep]
            gx[sel] = cx[keep]
            gy[sel] = cy[keep]
            todo = todo[~keep]
        if todo.numel() > 0:
            # 시연 20편의 기본 배치로 되돌린다. scene_layout.json의
            # initial_gear_root_positions 값이고 서로 겹치지 않는다.
            fallback = {"large": (0.33025, -0.06), "small": (0.24925, 0.04),
                        "medium": (0.27975, 0.13)}[name]
            gx[todo] = fallback[0]
            gy[todo] = fallback[1]
        gear_xy[name] = (gx, gy)
        placed.append((gx, gy))

    # 받침을 써넣는다. 회전은 z축 기준이고 반각 공식을 쓴다.
    #
    # 순서는 XYZW다. 쓰는 쪽과 읽는 쪽이 둘 다 XYZW다. 2026-10-21에 실험으로 정했다.
    # 같은 z축 20도 회전을 두 순서로 써넣고 결과를 비교했다
    # (tools/gear_quat_order_probe.py).
    #
    #   (cos, 0, 0, sin)로 넣으면   기울기 2.79 rad이 되고 기어가 쓰러져 바닥을 뚫는다
    #   (0, 0, sin, cos)로 넣으면   기울기 0.00 rad이 되고 기어가 제자리에 가만히 있다
    #
    # 라이브러리 설명도 같다. write_root_pose_to_sim은 write_root_pose_to_sim_index의
    # 옛 이름이고, 그 함수의 설명에 "quaternion orientation in (x, y, z, w)"라고 적혀
    # 있다. 읽는 쪽 .data.root_quat_w도 XYZW다.
    #
    # HDF5 파일 안의 root_pose만 WXYZ다. 파일을 읽는 HDF5DatasetFileHandler가 꺼낼 때
    # XYZW로 바꿔서 reset_to에 넘긴다. 그래서 변환기는 파일에 WXYZ로 적고, 이 함수는
    # 시뮬레이터에 XYZW로 쓴다. 두 자리의 순서가 다른 것이 맞다.
    #
    # 전에 이 자리를 WXYZ로 바꾼 적이 있는데 그것이 틀렸다. 그 상태에서 무작위 배치로
    # 리셋하면 기어 세 개가 168도 뒤집혀 놓이고, 24번 리셋에서 기어 72개 중 50개(69%)가
    # 작업면을 뚫고 받침보다 1.74 m 아래로 떨어졌다. 생성 시도가 전부 실패한 원인이다.
    half = yaw * 0.5
    base_quat = torch.zeros((n, 4), device=device)
    base_quat[:, 2] = torch.sin(half)      # z
    base_quat[:, 3] = torch.cos(half)      # w
    base_pos = torch.stack([bx, by, torch.full((n,), base_z, device=device)], dim=1) + origins
    base = env.scene[base_cfg_name()]
    base.write_root_pose_to_sim(torch.cat([base_pos, base_quat], dim=1), env_ids=env_ids)
    # 받침에는 속도를 쓰지 않는다. kinematic 물체라 PhysX가
    # "PxRigidDynamic::setLinearVelocity: Body must be non-kinematic!"로 거부한다.
    # 거부당해도 생성은 계속 돌지만 로그가 그 오류로 가득 차서 다른 문제를 못 본다.
    # kinematic 물체는 애초에 속도를 가지지 않으므로 0으로 둘 필요도 없다.

    # 기어를 써넣는다. 기어도 받침과 같은 각으로 돌려 둔다. 시연이 그렇게 만들어졌다.
    # 시연 20편의 기어 초기 자세를 열어 보면 받침과 같은 z축 회전이 들어 있다.
    for name in GEARS:
        gx, gy = gear_xy[name]
        pos = torch.stack([gx, gy, torch.full((n,), gear_z, device=device)], dim=1) + origins
        obj = env.scene[name]
        obj.write_root_pose_to_sim(torch.cat([pos, base_quat], dim=1), env_ids=env_ids)
        obj.write_root_velocity_to_sim(torch.zeros((n, 6), device=device), env_ids=env_ids)


def base_cfg_name() -> str:
    """받침 물체의 장면 이름. 한 군데서만 정해 두고 쓴다."""
    return "base"
