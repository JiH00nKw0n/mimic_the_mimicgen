"""Lab FR3 peg-insert Mimic env — base-frame-correct IK-rel action conversion + peg/socket poses.

Subclasses the official ``FrankaCubeStackIKRelMimicEnv`` and:

  1. copies lab_stack_mimic/lab_mimic_env.py's ``target_eef_pose_to_action`` /
     ``action_to_target_eef_pose`` VERBATIM — the base-frame IK-rel fix for the yaw-180 FR3
     base (world-frame delta must be rotated into the robot base frame before it is handed to
     the relative-mode DifferentialInverseKinematics controller). It is read generically from
     ``root_quat_w`` so it stays correct for any base yaw / any robot. See that file's docstring
     for the full derivation.
  2. overrides ``get_object_poses`` to return the tracked peg pose (and a CONSTANT ``socket``
     frame — the static collider fixed at HOLE_XY, whose MimicGen transform is therefore the
     identity, so an ``object_ref="socket"`` subtask replays verbatim).
  3. overrides ``get_subtask_term_signals`` to expose just the ``grasp_peg`` signal.

Imported (after Isaac Sim launches) via peg_register.py.

ON-BOX TODO(base class): confirm the base import path
``isaaclab_mimic.envs.franka_stack_ik_rel_mimic_env.FrankaCubeStackIKRelMimicEnv`` and that
``get_object_poses(self, env_ids=None)`` / ``get_subtask_term_signals(self, env_ids=None)`` are
the exact method names/signatures in this container's isaaclab_mimic. They match the stack
env used by lab_stack_mimic, which is the same package family.
"""

from __future__ import annotations

import os

import torch

import isaaclab.utils.math as PoseUtils
from isaaclab_mimic.envs.franka_stack_ik_rel_mimic_env import FrankaCubeStackIKRelMimicEnv

# socket / desk geometry, kept in sync with peg_mdp (the constant socket frame below).
from peg_mdp import HOLE_XY, DESK_Z, HOLE_HEIGHT

# Optional per-step debug dump of the first N target_eef_pose_to_action calls.
_DEBUG = os.environ.get("LAB_MIMIC_DEBUG")
_DEBUG_PATH = os.environ.get("LAB_MIMIC_DEBUG_PATH", "/tmp/tea_debug.txt")
_DEBUG_MAX = int(os.environ.get("LAB_MIMIC_DEBUG_MAX", "80"))

# constant socket reference frame (env-local): the insertion target. z at the socket rim.
SOCKET_REF = (HOLE_XY[0], HOLE_XY[1], DESK_Z + HOLE_HEIGHT)


class LabFR3PegInsertIKRelMimicEnv(FrankaCubeStackIKRelMimicEnv):
    """FR3 peg-insert Mimic env: converts IK-rel deltas in the robot base frame, and returns
    peg + constant-socket object poses for the MimicGen object-frame transform."""

    _dbg_n = 0

    def _root_quat(self, env_ids) -> torch.Tensor:
        """Robot base orientation in world (w,x,y,z), shape (len(env_ids), 4)."""
        return self.scene["robot"].data.root_quat_w[env_ids]

    # ------------------------------------------------------------------ IK-rel base-frame fix
    # (VERBATIM from lab_stack_mimic/lab_mimic_env.py; only the class name in the debug counter
    #  differs. Do not "improve" this — it is the fix that makes generation converge on the
    #  yaw-180 FR3.)
    def target_eef_pose_to_action(
        self,
        target_eef_pose_dict: dict,
        gripper_action_dict: dict,
        action_noise_dict: dict | None = None,
        env_id: int = 0,
    ) -> torch.Tensor:
        eef_name = list(self.cfg.subtask_configs.keys())[0]

        # target / current eef pose (both in the world / env-origin frame)
        (target_eef_pose,) = target_eef_pose_dict.values()
        target_pos, target_rot = PoseUtils.unmake_pose(target_eef_pose)
        curr_pose = self.get_robot_eef_pose(eef_name, env_ids=[env_id])[0]
        curr_pos, curr_rot = PoseUtils.unmake_pose(curr_pose)

        # world-frame deltas
        delta_position = target_pos - curr_pos
        delta_rot_mat = target_rot.matmul(curr_rot.transpose(-1, -2))
        delta_quat = PoseUtils.quat_from_matrix(delta_rot_mat)
        delta_rotation = PoseUtils.axis_angle_from_quat(delta_quat)

        # --- LAB FIX: rotate world-frame deltas into the robot BASE frame (the frame
        # the IK-rel controller applies the command in). See module docstring.
        root_quat = self._root_quat([env_id])  # (1, 4)
        delta_position = PoseUtils.quat_apply_inverse(root_quat, delta_position.unsqueeze(0)).squeeze(0)
        delta_rotation = PoseUtils.quat_apply_inverse(root_quat, delta_rotation.unsqueeze(0)).squeeze(0)
        # --- end LAB FIX

        (gripper_action,) = gripper_action_dict.values()
        pose_action = torch.cat([delta_position, delta_rotation], dim=0)
        if action_noise_dict is not None:
            noise = action_noise_dict[eef_name] * torch.randn_like(pose_action)
            pose_action += noise
            pose_action = torch.clamp(pose_action, -1.0, 1.0)

        if _DEBUG and env_id == 0 and LabFR3PegInsertIKRelMimicEnv._dbg_n < _DEBUG_MAX:
            i = LabFR3PegInsertIKRelMimicEnv._dbg_n
            LabFR3PegInsertIKRelMimicEnv._dbg_n += 1
            tp = [round(float(v), 3) for v in target_pos]
            cp = [round(float(v), 3) for v in curr_pos]
            dpw = float(torch.linalg.vector_norm(target_pos - curr_pos))
            drw = float(torch.linalg.vector_norm(delta_rotation))
            with open(_DEBUG_PATH, "a") as fh:
                fh.write(
                    f"{i:3d} curr={cp} target={tp} |dpos_w|={dpw:.3f} |drot_base|={drw:.3f} "
                    f"act_pos={[round(float(v),2) for v in pose_action[:3]]} "
                    f"act_rot={[round(float(v),2) for v in pose_action[3:6]]} "
                    f"grip={float(gripper_action.reshape(-1)[0]):+.1f}\n"
                )

        return torch.cat([pose_action, gripper_action], dim=0)

    def action_to_target_eef_pose(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        eef_name = list(self.cfg.subtask_configs.keys())[0]

        delta_position = action[:, :3]
        delta_rotation = action[:, 3:6]

        # --- LAB FIX: action deltas are in the BASE frame; rotate back to WORLD to
        # match the world-frame eef poses used everywhere else in the pipeline.
        root_quat = self._root_quat(slice(None))  # (num_envs, 4)
        delta_position = PoseUtils.quat_apply(root_quat, delta_position)
        delta_rotation = PoseUtils.quat_apply(root_quat, delta_rotation)
        # --- end LAB FIX

        curr_pose = self.get_robot_eef_pose(eef_name, env_ids=None)
        curr_pos, curr_rot = PoseUtils.unmake_pose(curr_pose)

        target_pos = curr_pos + delta_position

        delta_rotation_angle = torch.linalg.norm(delta_rotation, dim=-1, keepdim=True)
        delta_rotation_axis = delta_rotation / delta_rotation_angle
        is_close_to_zero_angle = torch.isclose(delta_rotation_angle, torch.zeros_like(delta_rotation_angle)).squeeze(1)
        delta_rotation_axis[is_close_to_zero_angle] = torch.zeros_like(delta_rotation_axis)[is_close_to_zero_angle]

        delta_quat = PoseUtils.quat_from_angle_axis(delta_rotation_angle.squeeze(1), delta_rotation_axis).squeeze(0)
        delta_rot_mat = PoseUtils.matrix_from_quat(delta_quat)
        target_rot = torch.matmul(delta_rot_mat, curr_rot)

        target_poses = PoseUtils.make_pose(target_pos, target_rot).clone()
        return {eef_name: target_poses}

    # ------------------------------------------------------------------ peg-specific overrides
    def get_object_poses(self, env_ids=None):
        """Object frames for the MimicGen transform, in the env-local frame (world minus env
        origin), matching the base class's convention for the cubes.

        Returns the tracked ``peg`` pose and a CONSTANT ``socket`` pose. Because the socket is a
        static collider fixed at HOLE_XY, its frame is identical at annotation and generation
        time, so for an ``object_ref="socket"`` subtask MimicGen's O_0 * (O'_0)^-1 * C_t reduces
        to the identity and the insert segment replays verbatim.
        """
        if env_ids is None:
            env_ids = slice(None)

        peg = self.scene["peg"]
        peg_pos = peg.data.root_pos_w[env_ids] - self.scene.env_origins[env_ids]
        peg_rot = PoseUtils.matrix_from_quat(peg.data.root_quat_w[env_ids])  # WXYZ; see peg_mdp TODO(quat)
        poses = {"peg": PoseUtils.make_pose(peg_pos, peg_rot)}

        n = peg_pos.shape[0]
        socket_pos = torch.tensor(SOCKET_REF, device=self.device, dtype=peg_pos.dtype).repeat(n, 1)
        socket_rot = torch.eye(3, device=self.device, dtype=peg_pos.dtype).unsqueeze(0).repeat(n, 1, 1)
        poses["socket"] = PoseUtils.make_pose(socket_pos, socket_rot)
        return poses

    def get_subtask_term_signals(self, env_ids=None):
        """Per-step boolean subtask-termination signals, read from the ``subtask_terms`` obs
        group. Only ``grasp_peg`` exists (2 subtasks need N-1 = 1 signal; the final insert
        subtask has none)."""
        if env_ids is None:
            env_ids = slice(None)
        signals = dict()
        subtask_terms = self.obs_buf["subtask_terms"]
        signals["grasp_peg"] = subtask_terms["grasp_peg"][env_ids]
        return signals
