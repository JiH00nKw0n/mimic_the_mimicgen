"""Count a generated demo as success ONLY if its FINAL state is a clean canonical stack.

MimicGen marks success stickily inside DataGenerator.generate:
    generated_success = generated_success or exec_results["success"]
so a demo where the tower momentarily forms (success term fires) and then collapses
during post-placement settling is still exported as a success. On our lab FR3 setup
~20% of "successes" end with a toppled / partial tower (cubes back on the desk),
which is bad supervision.

This hook re-evaluates the stack at the FINAL state (directly from cube poses) and gates
success on it, at two consistent points:
  1. RecorderManager.set_success_to_episodes  -> the EXPORTED episode is success only if
     the final state is a clean stack (so collapsed demos are not written to the kept file).
  2. DataGenerator.generate (return value)     -> the env_loop success counter / the
     generation_guarantee target count only clean demos, so requesting N trials yields N
     clean demos (not N sticky-successes of which ~20% are junk).

Both read the SAME final-state check. Imported (after Isaac launches) by run_generate.sh,
BEFORE provenance_hooks, so provenance counts the corrected (clean) successes.
Shared Isaac Lab source is left untouched.

깨끗한 탑의 조건은 다섯 가지이고, 다섯을 모두 만족해야 성공으로 센다.
  1. 높이 순서가 cube_1 < cube_2 < cube_3
  2. 이웃한 두 높이 차가 큐브 한 변 근처(0.0368부터 0.0568 m)
  3. 위 큐브가 아래 큐브 위에 xy로 0.02 m 안에 정렬
  4. 세 큐브가 모두 거의 멈춰 있음(초당 0.02 m 미만). 무너지는 도중의 한 프레임이
     조건을 만족해 성공으로 새는 것을 막는다
  5. 그리퍼가 열려 있음. 아직 쥐고 있으면 완성된 탑이 아니라 들고 있는 상태다

읽기에 실패하면 성공이 아닌 쪽으로 판정한다. 예전에는 반대로 통과시켰는데, 그러면
구조가 바뀌어 필터가 통째로 무력해져도 아무 표시가 나지 않는다.
"""

from __future__ import annotations

import torch

from isaaclab.managers.recorder_manager import RecorderManager
from isaaclab_mimic.datagen.data_generator import DataGenerator

import os

GAP_LO, GAP_HI = 0.0368, 0.0568   # one cube height 0.0468 +/- 0.01
# 마지막 순간에 아직 움직이고 있으면 곧 무너질 탑이다. 정지 판정을 넣지 않으면
# 무너지는 도중의 한 프레임이 기준을 만족해 성공으로 기록된다.
REST_SPEED = float(os.environ.get("LAB_SUCCESS_REST_SPEED", "0.02"))   # m/s
# 그리퍼가 아직 잡고 있으면 완성된 탑이 아니라 들고 있는 상태다. 렌더 단계 판정
# (success_criteria.tower_status)은 이 조건을 보는데 생성 단계에는 없었다.
GRIPPER_OPEN = float(os.environ.get("LAB_SUCCESS_GRIPPER_OPEN", "0.03"))
# 0.04 was LOOSER than the physical topple limit (half-cube edge ~0.0235 m): a stack whose COM sits
# past the supporting cube's edge (~2.9 cm off) still read "clean", got marked success stickily, then
# toppled. Tighten to 0.02 so only physically stable stacks pass. (Toppled demos already fail the
# z-order/gap check — cubes settle flat to z~0.745 — so this mainly rejects metastable near-edge ones.)
XY_TOL = float(os.environ.get("LAB_SKILLGEN_XY_TOL", "0.02"))


_read_errors = 0
_finger_idx = None
# 읽기 오류가 이만큼 쌓이면 구조가 깨진 것으로 보고 실행을 멈춘다. 멈추지 않으면
# 도달할 수 없는 목표를 향해 GPU를 계속 태운다.
MAX_READ_ERRORS = int(os.environ.get("LAB_SUCCESS_MAX_READ_ERRORS", "50"))


def _finger_indices(env):
    """그리퍼 손가락 관절을 이름으로 찾는다.

    예전에는 joint_pos[:, -2:]로 위치를 가정했다. 관절 순서가 [팔 7개, 손가락 2개]가
    아니면 팔 관절값이 걸리는데, 그 값들은 열림 기준(0.03)보다 훨씬 커서 "그리퍼를
    놓았다"가 항상 참이 된다. 조용히 검사 하나가 사라지는 것이라 이름으로 찾는다.
    렌더 단계(success_criteria)도 이름으로 찾으므로 두 판정이 같은 관절을 본다.
    """
    global _finger_idx
    if _finger_idx is None:
        names = env.scene["robot"].data.joint_names
        _finger_idx = [i for i, n in enumerate(names) if "finger" in n]
        if len(_finger_idx) != 2:
            raise RuntimeError(
                f"그리퍼 손가락 관절 2개를 찾지 못했다: {_finger_idx} / {names}")
        print(f"[clean_success_hook] 손가락 관절 인덱스 {_finger_idx} "
              f"({[names[i] for i in _finger_idx]})", flush=True)
    return _finger_idx


def _final_stack_ok(env, env_id: int) -> bool:
    """지금 이 순간 env_id의 큐브가 깨끗한 3단 탑(cube_1<cube_2<cube_3)인지 판정한다."""
    global _read_errors
    try:
        org = env.scene.env_origins[env_id]
        z, xy, speed = {}, {}, {}
        for i in (1, 2, 3):
            data = env.scene[f"cube_{i}"].data
            p = data.root_pos_w[env_id] - org
            z[i] = float(p[2])
            xy[i] = p[:2]
            speed[i] = float(torch.linalg.vector_norm(data.root_lin_vel_w[env_id]))
        if not (z[1] < z[2] < z[3]):
            return False
        g1, g2 = z[2] - z[1], z[3] - z[2]
        if not (GAP_LO < g1 < GAP_HI and GAP_LO < g2 < GAP_HI):
            return False
        if float(torch.linalg.vector_norm(xy[2] - xy[1])) > XY_TOL:
            return False
        if float(torch.linalg.vector_norm(xy[3] - xy[2])) > XY_TOL:
            return False
        if max(speed.values()) > REST_SPEED:
            return False          # 아직 움직이는 중이면 완성된 탑으로 보지 않는다
        idx = _finger_indices(env)
        fingers = env.scene["robot"].data.joint_pos[env_id, idx]
        if float(fingers.min()) <= GRIPPER_OPEN:
            return False          # 아직 쥐고 있으면 놓은 것이 아니다
        return True
    except Exception as exc:
        # 닫히는 쪽으로 실패한다. 예전에는 여기서 True를 돌려줬는데, 그러면 큐브
        # 이름이 바뀌는 것 같은 구조적 변화가 생겼을 때 필터가 통째로 무력해지고도
        # 아무 표시가 나지 않는다. 8만 개를 만드는 동안 그걸 모르면 데이터가 통째로
        # 오염된다. 그래서 떨어뜨리고, 몇 번 났는지 세어 로그에 남긴다.
        _read_errors += 1
        if _read_errors <= 5 or _read_errors % 100 == 0:
            print(f"[clean_success_hook] 최종 상태를 읽지 못해 실패로 처리 "
                  f"({_read_errors}번째): {type(exc).__name__}: {exc}", flush=True)
        if _read_errors >= MAX_READ_ERRORS:
            raise RuntimeError(
                f"성공 판정이 {_read_errors}번 연속으로 실패했다. 씬 구성이 바뀐 것으로 "
                f"보고 실행을 멈춘다. 마지막 오류: {type(exc).__name__}: {exc}") from exc
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
        if vi < len(success_values) and bool(success_values[vi].item()) and not _final_stack_ok(self._env, eid):
            success_values[vi] = False
    return _orig_set(self, env_ids, success_values)


_orig_gen = DataGenerator.generate


async def _gen(self, env_id, *args, **kwargs):
    res = await _orig_gen(self, env_id, *args, **kwargs)
    if isinstance(res, dict) and res.get("success") and not _final_stack_ok(self.env, env_id):
        res["success"] = False
    return res


RecorderManager.set_success_to_episodes = _set_success
DataGenerator.generate = _gen
print("[clean_success_hook] success gated on clean final-state canonical stack")
