"""기어 조립 MimicGen 환경 클래스.

핀 꽂기의 LabFR3PegInsertIKRelMimicEnv와 같은 짜임이다. 공식 큐브 쌓기 Mimic 환경을
물려받아 네 가지를 바꾼다.

  * target_eef_pose_to_action   손끝 목표를 상대 명령으로 바꾼다
  * action_to_target_eef_pose   상대 명령을 손끝 목표로 되돌린다
  * get_object_poses            MimicGen이 궤적을 옮길 때 기준으로 쓸 물체 자세를 넘긴다
  * get_subtask_term_signals    구간 완료 신호 다섯 개를 넘긴다

앞의 두 개는 핀 꽂기에서 그대로 가져온다. 받침이 z축으로 180도 돌아간 FR3에서 상대
명령이 적용되는 좌표계를 맞추는 수정이고, 이것이 없으면 생성이 수렴하지 않는다. 큐브
쌓기에서 먼저 발견해 핀 꽂기로 옮긴 것이고 기어도 같은 로봇 배치라 같은 수정이 필요하다.

2026-10-21에 이 좌표계 규약을 다시 확인했다. 핀 꽂기 시연의 손끝 자세에서 같은 식으로
명령을 다시 계산해 기록된 명령과 최대 오차 1.8e-7로 재현되는 것을 보았다. 사원수는
XYZW 순서이고 받침 회전은 (0, 0, 1, 0)이다.
"""

from __future__ import annotations

import os

import torch

import isaaclab.utils.math as PoseUtils
from isaaclab_mimic.envs.franka_stack_ik_rel_mimic_env import FrankaCubeStackIKRelMimicEnv

from gear_geom import GEARS

_DEBUG = os.environ.get("LAB_GEAR_DEBUG", "") == "1"
_DEBUG_MAX = 40
_DEBUG_PATH = os.environ.get("LAB_GEAR_DEBUG_PATH", "/work/gear_action_debug.log")


class LabFR3MultiGearIKRelMimicEnv(FrankaCubeStackIKRelMimicEnv):
    """받침이 180도 돌아간 FR3에서 상대 명령 좌표계를 맞추고, 받침과 기어 세 개의 자세와
    구간 신호 다섯 개를 넘긴다."""

    _dbg_n = 0

    def _root_quat(self, env_ids) -> torch.Tensor:
        """로봇 받침의 월드 기준 자세. 모양은 (환경 개수, 4)."""
        return self.scene["robot"].data.root_quat_w[env_ids]

    # ------------------------------------------------------------------ 상대 명령 좌표계 수정
    def target_eef_pose_to_action(
        self,
        target_eef_pose_dict: dict,
        gripper_action_dict: dict,
        action_noise_dict: dict | None = None,
        env_id: int = 0,
    ) -> torch.Tensor:
        eef_name = list(self.cfg.subtask_configs.keys())[0]

        (target_eef_pose,) = target_eef_pose_dict.values()
        target_pos, target_rot = PoseUtils.unmake_pose(target_eef_pose)
        curr_pose = self.get_robot_eef_pose(eef_name, env_ids=[env_id])[0]
        curr_pos, curr_rot = PoseUtils.unmake_pose(curr_pose)

        # 월드 기준 차이
        delta_position = target_pos - curr_pos
        delta_rot_mat = target_rot.matmul(curr_rot.transpose(-1, -2))
        delta_quat = PoseUtils.quat_from_matrix(delta_rot_mat)
        delta_rotation = PoseUtils.axis_angle_from_quat(delta_quat)

        # 월드 기준 차이를 로봇 받침 좌표계로 돌린다. 상대 명령이 실제로 적용되는 좌표계다.
        root_quat = self._root_quat([env_id])
        delta_position = PoseUtils.quat_apply_inverse(
            root_quat, delta_position.unsqueeze(0)).squeeze(0)
        delta_rotation = PoseUtils.quat_apply_inverse(
            root_quat, delta_rotation.unsqueeze(0)).squeeze(0)

        (gripper_action,) = gripper_action_dict.values()
        pose_action = torch.cat([delta_position, delta_rotation], dim=0)
        if action_noise_dict is not None:
            noise = action_noise_dict[eef_name] * torch.randn_like(pose_action)
            pose_action += noise
            pose_action = torch.clamp(pose_action, -1.0, 1.0)

        if _DEBUG and env_id == 0 and LabFR3MultiGearIKRelMimicEnv._dbg_n < _DEBUG_MAX:
            i = LabFR3MultiGearIKRelMimicEnv._dbg_n
            LabFR3MultiGearIKRelMimicEnv._dbg_n += 1
            dpw = float(torch.linalg.vector_norm(target_pos - curr_pos))
            drw = float(torch.linalg.vector_norm(delta_rotation))
            with open(_DEBUG_PATH, "a") as fh:
                fh.write(f"{i:3d} curr={[round(float(v),3) for v in curr_pos]} "
                         f"target={[round(float(v),3) for v in target_pos]} "
                         f"|dpos_w|={dpw:.4f} |drot_base|={drw:.4f} "
                         f"grip={float(gripper_action.reshape(-1)[0]):+.1f}\n")

        return torch.cat([pose_action, gripper_action], dim=0)

    def action_to_target_eef_pose(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        eef_name = list(self.cfg.subtask_configs.keys())[0]

        delta_position = action[:, :3]
        delta_rotation = action[:, 3:6]

        # 명령은 받침 좌표계에 있다. 나머지 계산이 쓰는 월드 좌표계로 되돌린다.
        root_quat = self._root_quat(slice(None))
        delta_position = PoseUtils.quat_apply(root_quat, delta_position)
        delta_rotation = PoseUtils.quat_apply(root_quat, delta_rotation)

        curr_pose = self.get_robot_eef_pose(eef_name, env_ids=None)
        curr_pos, curr_rot = PoseUtils.unmake_pose(curr_pose)

        target_pos = curr_pos + delta_position

        delta_rotation_angle = torch.linalg.norm(delta_rotation, dim=-1, keepdim=True)
        delta_rotation_axis = delta_rotation / delta_rotation_angle
        is_zero = torch.isclose(delta_rotation_angle,
                                torch.zeros_like(delta_rotation_angle)).squeeze(1)
        delta_rotation_axis[is_zero] = torch.zeros_like(delta_rotation_axis)[is_zero]

        delta_quat = PoseUtils.quat_from_angle_axis(
            delta_rotation_angle.squeeze(1), delta_rotation_axis).squeeze(0)
        delta_rot_mat = PoseUtils.matrix_from_quat(delta_quat)
        target_rot = torch.matmul(delta_rot_mat, curr_rot)

        return {eef_name: PoseUtils.make_pose(target_pos, target_rot).clone()}

    # ------------------------------------------------------------------ 기어 쪽 바꿔 끼우기
    def get_object_poses(self, env_ids=None):
        """MimicGen이 궤적을 옮길 때 기준으로 쓸 물체 자세. 환경 기준 좌표계다.

        즉 월드 좌표에서 그 환경의 원점을 뺀 값이다. 공식 큐브 쌓기 환경이 큐브에 쓰는
        규약과 같다. 원점을 빼지 않으면 환경을 여러 개 띄울 때 0번에서만 맞는다.

        돌려주는 이름 네 개가 변환기로 만든 소스 파일의 object_pose 이름과 같아야 한다.
        받침과 기어 셋이다.
        """
        if env_ids is None:
            env_ids = slice(None)
        origins = self.scene.env_origins[env_ids]
        out = {}
        for name in ("base",) + GEARS:
            obj = self.scene[name]
            pos = obj.data.root_pos_w[env_ids] - origins
            quat = obj.data.root_quat_w[env_ids]
            out[name] = PoseUtils.make_pose(pos, PoseUtils.matrix_from_quat(quat))
        return out

    def get_subtask_term_signals(self, env_ids=None):
        """구간 완료 신호 다섯 개. 이름이 설정의 subtask_terms 항 이름과 같아야 한다."""
        if env_ids is None:
            env_ids = slice(None)
        terms = self.obs_buf["subtask_terms"]
        names = ("grasp_large", "seat_large", "grasp_small", "seat_small", "grasp_medium")
        return {n: terms[n][env_ids] for n in names}
