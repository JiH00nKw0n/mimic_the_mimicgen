"""생성된 데모를 최종 상태로 다시 판정한다. 핀이 실제로 꽂혀 있어야 성공으로 센다.

큐브의 `clean_success_hook.py`와 같은 자리에 같은 방식으로 붙는다. MimicGen은 한 번이라도
성공 신호가 서면 그 시도를 성공으로 기록하는데, 그러면 꽂혔다가 빠진 것도 성공으로 남는다.
그래서 마지막 순간의 상태를 직접 보고 두 곳에서 같은 판정을 건다.

  1. RecorderManager.set_success_to_episodes 에서, 파일로 나가는 에피소드를 거른다.
  2. DataGenerator.generate 의 반환값에서, 성공 개수 세는 것을 바로잡는다.

판정 기준은 lab_peg_mimic/peg_mdp.py의 peg_inserted와 같다. 핀 중심이 구멍 축에서 1 cm
안, 삽입 깊이 2 cm 이상, 수직도 0.9 이상이다. 여기에 두 가지를 더한다. 핀이 거의 멈춰
있어야 하고, 그리퍼가 열려 있어야 한다. 큐브 쪽에서 배운 것이다. 무너지는 도중의 한
프레임이나 아직 들고 있는 상태가 성공으로 새는 것을 막는다.

읽기에 실패하면 성공이 아닌 쪽으로 판정한다. 그리고 몇 번 났는지 세어, 계속 실패하면
씬 구성이 바뀐 것으로 보고 실행을 멈춘다.
"""
from __future__ import annotations

import os

import torch
from isaaclab.managers.recorder_manager import RecorderManager
from isaaclab_mimic.datagen.data_generator import DataGenerator

# peg_mdp.py 와 같은 값
RADIAL_SUCCESS = float(os.environ.get("LAB_PEG_RADIAL_SUCCESS", "0.010"))
DEPTH_SUCCESS = float(os.environ.get("LAB_PEG_DEPTH_SUCCESS", "0.020"))
UPRIGHT_SUCCESS = float(os.environ.get("LAB_PEG_UPRIGHT_SUCCESS", "0.9"))
REST_SPEED = float(os.environ.get("LAB_SUCCESS_REST_SPEED", "0.02"))
GRIPPER_OPEN = float(os.environ.get("LAB_SUCCESS_GRIPPER_OPEN", "0.03"))
MAX_READ_ERRORS = 50

_read_errors = 0


def _finger_indices(env):
    """그리퍼 손가락 관절을 이름으로 찾는다.

    위치로 자르면(뒤에서 두 개) 관절 순서가 바뀌는 순간 엉뚱한 관절을 보고도 아무 표시가
    나지 않는다. 큐브 쪽에서 같은 이유로 이름으로 바꿨다.
    """
    names = env.scene["robot"].data.joint_names
    idx = [i for i, n in enumerate(names) if "finger" in n.lower()]
    if not idx:
        raise RuntimeError(f"그리퍼 손가락 관절을 찾지 못했다. 있는 이름: {names}")
    return idx


def _final_insert_ok(env, env_id: int) -> bool:
    """지금 이 순간 env_id의 핀이 구멍에 꽂혀 있는지 판정한다."""
    global _read_errors
    try:
        import peg_mdp

        org = env.scene.env_origins[env_id]
        peg = env.scene["peg"].data
        pos = peg.root_pos_w[env_id] - org
        quat = peg.root_quat_w[env_id]          # XYZW (이 컨테이너에서 확인함)
        speed = float(torch.linalg.vector_norm(peg.root_lin_vel_w[env_id]))

        # 쿼터니언에서 핀의 위쪽 축을 뽑는다. 순서는 XYZW이므로 x가 0번, y가 1번이다.
        x, y, z, w = (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))
        up_z = 1.0 - 2.0 * (x * x + y * y)

        hole_xy = getattr(peg_mdp, "HOLE_XY", (0.0, 0.0))
        desk_z = getattr(peg_mdp, "DESK_Z", 0.0)
        hole_h = getattr(peg_mdp, "HOLE_HEIGHT", getattr(peg_mdp, "HOLE_H", 0.0))
        peg_len = getattr(peg_mdp, "PEG_LEN", getattr(peg_mdp, "PEG_SIZE", (0, 0, 0.06))[2])

        radial = float(torch.linalg.vector_norm(
            pos[:2] - torch.tensor(hole_xy, device=pos.device, dtype=pos.dtype)))
        rim_z = desk_z + hole_h
        depth = rim_z - (float(pos[2]) - peg_len / 2.0)

        if radial >= RADIAL_SUCCESS:
            return False
        if depth <= DEPTH_SUCCESS:
            return False
        if up_z <= UPRIGHT_SUCCESS:
            return False
        if speed > REST_SPEED:
            return False           # 아직 움직이는 중이면 꽂혔다고 보지 않는다
        fingers = env.scene["robot"].data.joint_pos[env_id, _finger_indices(env)]
        if float(fingers.min()) <= GRIPPER_OPEN:
            return False           # 아직 쥐고 있으면 놓은 것이 아니다
        return True
    except Exception as exc:                       # noqa: BLE001
        _read_errors += 1
        if _read_errors <= 5 or _read_errors % 100 == 0:
            print(f"[peg_success_hook] 최종 상태를 읽지 못해 실패로 처리 "
                  f"({_read_errors}번째): {type(exc).__name__}: {exc}", flush=True)
        if _read_errors >= MAX_READ_ERRORS:
            raise RuntimeError(
                f"성공 판정이 {_read_errors}번 실패했다. 씬 구성이 바뀐 것으로 보고 "
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
                and not _final_insert_ok(self._env, eid)):
            success_values[vi] = False
    return _orig_set(self, env_ids, success_values)


_orig_gen = DataGenerator.generate


async def _gen(self, env_id, *args, **kwargs):
    res = await _orig_gen(self, env_id, *args, **kwargs)
    if isinstance(res, dict) and res.get("success") and not _final_insert_ok(self.env, env_id):
        res["success"] = False
    return res


RecorderManager.set_success_to_episodes = _set_success
DataGenerator.generate = _gen
print("[peg_success_hook] success gated on final-state peg insertion")
