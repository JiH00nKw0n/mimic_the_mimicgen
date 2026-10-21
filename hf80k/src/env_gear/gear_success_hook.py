"""생성된 데모를 최종 상태로 다시 판정한다. 기어 셋이 실제로 앉아 있어야 성공으로 센다.

핀 꽂기의 peg_success_hook과 같은 자리에 같은 방식으로 붙는다. MimicGen은 한 번이라도
성공 신호가 서면 그 시도를 성공으로 기록하는데, 그러면 앉았다가 튀어나온 것도 성공으로
남는다. 그래서 마지막 순간의 상태를 직접 보고 두 곳에서 같은 판정을 건다.

  1. RecorderManager.set_success_to_episodes 에서 파일로 나가는 에피소드를 거른다.
  2. DataGenerator.generate 의 반환값에서 성공 개수 세는 것을 바로잡는다.

판정 기준은 gear_mdp.gear_seated와 같다. 기어의 축 중심이 받침 축에서 가로세로 1 mm 안,
높이 차이 2.5 mm 안, 기울기 0.12 rad 안, 속도 0.025 m/s 안이다. 여기에 한 가지를 더한다.
그리퍼가 열려 있어야 한다. 아직 들고 있는 상태를 성공으로 세지 않기 위해서다. 핀 꽂기와
큐브 쌓기에서 같은 이유로 같은 조건을 더했다.

읽기에 실패하면 성공이 아닌 쪽으로 판정한다. 그리고 몇 번 났는지 세어 계속 실패하면
장면 구성이 바뀐 것으로 보고 실행을 멈춘다. 조용히 전부 실패로 처리되는 쪽이 더 나쁘다.
"""
from __future__ import annotations

import os

import torch
from isaaclab.managers.recorder_manager import RecorderManager
from isaaclab_mimic.datagen.data_generator import DataGenerator

GRIPPER_OPEN = float(os.environ.get("LAB_SUCCESS_GRIPPER_OPEN", "0.03"))
MAX_READ_ERRORS = 50

_read_errors = 0


def _finger_indices(env):
    """그리퍼 손가락 관절을 이름으로 찾는다.

    뒤에서 두 개로 자르면 관절 순서가 바뀌는 순간 엉뚱한 관절을 보고도 아무 표시가
    나지 않는다. 큐브 쪽에서 같은 이유로 이름으로 바꿨다.
    """
    names = env.scene["robot"].data.joint_names
    idx = [i for i, n in enumerate(names) if "finger" in n.lower()]
    if not idx:
        raise RuntimeError(f"그리퍼 손가락 관절을 찾지 못했다. 있는 이름: {names}")
    return idx


def _final_assembly_ok(env, env_id: int) -> bool:
    """지금 이 순간 env_id의 기어 셋이 모두 제 축에 앉아 있는지 판정한다."""
    global _read_errors
    try:
        import gear_mdp
        from gear_geom import (AXIS_XY_TOL_M, GEARS, SPEED_TOL_M_S, TILT_TOL_RAD, Z_TOL_M,
                               axis_xy_error, tilt_rad_from_quat)

        org = env.scene.env_origins[env_id]
        base = env.scene["base"].data
        base_pos = (base.root_pos_w[env_id] - org).reshape(1, 3)
        base_quat = base.root_quat_w[env_id].reshape(1, 4)       # XYZW

        for name in GEARS:
            g = env.scene[name].data
            pos = (g.root_pos_w[env_id] - org).reshape(1, 3)
            quat = g.root_quat_w[env_id].reshape(1, 4)
            speed = float(torch.linalg.vector_norm(g.root_lin_vel_w[env_id]))

            if float(axis_xy_error(pos, quat, base_pos, base_quat, name)[0]) >= AXIS_XY_TOL_M:
                return False
            if abs(float(pos[0, 2]) - float(base_pos[0, 2])) >= Z_TOL_M:
                return False
            if float(tilt_rad_from_quat(quat)[0]) >= TILT_TOL_RAD:
                return False
            if speed >= SPEED_TOL_M_S:
                return False

        fingers = env.scene["robot"].data.joint_pos[env_id, _finger_indices(env)]
        if float(fingers.min()) <= GRIPPER_OPEN:
            return False           # 아직 쥐고 있으면 놓은 것이 아니다
        return True
    except Exception as exc:                       # noqa: BLE001
        _read_errors += 1
        if _read_errors <= 5 or _read_errors % 100 == 0:
            print(f"[gear_success_hook] 최종 상태를 읽지 못해 실패로 처리 "
                  f"({_read_errors}번째): {type(exc).__name__}: {exc}", flush=True)
        if _read_errors >= MAX_READ_ERRORS:
            raise RuntimeError(
                f"성공 판정이 {_read_errors}번 실패했다. 장면 구성이 바뀐 것으로 보고 "
                f"실행을 멈춘다. 마지막 오류: {type(exc).__name__}: {exc}") from exc
        return False


def _ids(env, env_ids):
    if env_ids is None:
        return list(range(env.num_envs))
    if isinstance(env_ids, torch.Tensor):
        return env_ids.tolist()
    return list(env_ids)


_orig_set = RecorderManager.set_success_to_episodes


def _set_success(self, env_ids, success_values):
    for vi, eid in enumerate(_ids(self._env, env_ids)):
        if (vi < len(success_values) and bool(success_values[vi].item())
                and not _final_assembly_ok(self._env, eid)):
            success_values[vi] = False
    return _orig_set(self, env_ids, success_values)


_orig_gen = DataGenerator.generate


async def _gen(self, env_id, *args, **kwargs):
    res = await _orig_gen(self, env_id, *args, **kwargs)
    if isinstance(res, dict) and res.get("success") and not _final_assembly_ok(self.env, env_id):
        res["success"] = False
    return res


RecorderManager.set_success_to_episodes = _set_success
DataGenerator.generate = _gen
print("[gear_success_hook] 성공을 최종 상태의 기어 세 개 착좌로 다시 판정한다")
