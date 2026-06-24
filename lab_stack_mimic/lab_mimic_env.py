"""Lab FR3 Mimic env — base-frame-correct IK-rel action conversion.

WHY THIS EXISTS
---------------
The stock `FrankaCubeStackIKRelMimicEnv` computes generation actions like this:

    delta_position = target_eef_pos_world - curr_eef_pos_world      # WORLD frame
    action = [delta_position, delta_rotation, gripper]

and hands `action` straight to the IK-rel controller. But the controller
(`DifferentialInverseKinematicsAction`, relative mode) interprets the delta in the
robot **base/root** frame: it computes the current EE pose in the root frame via
`subtract_frame_transforms(root_pose_w, ee_pose_w)` and then does
`target_pos_base = ee_pos_base + delta`. So the action delta must be expressed in
the BASE frame, not the world frame.

For the official Franka the robot base sits at the world origin with identity
rotation, so world == base and the stock code works by coincidence. Our lab FR3 is
mounted at `ROBOT_ROT = (0,0,0,1)` = **yaw 180°** (it faces -x toward the desk).
With a 180° base yaw, a world-frame delta `(dx, dy, dz)` is applied by the
controller as `(-dx, -dy, dz)` in the world — the x/y commands are sign-flipped.
The robot drives away from the target, the error grows, the IK-rel action saturates
at ±1, and generation diverges → 0% DGR. (Annotation/replay are unaffected because
they replay the *recorded* actions, which were already produced in the base frame
by the teleop IK pipeline.)

THE FIX
-------
Everything else (eef poses from `get_robot_eef_pose`, object poses from
`get_object_poses`) is consistently in the world/env-origin frame, so the MimicGen
object-frame transform is valid as-is. The only frame error is at the action
boundary. So we rotate just the delta at that boundary:

    base-frame delta = R_root^{-1} * world-frame delta      (quat_apply_inverse)

applied to BOTH the position delta and the axis-angle rotation delta (conjugating a
rotation by R rotates its axis-angle vector by R, so the same op works for both).
The inverse method `action_to_target_eef_pose` does the opposite rotation
(quat_apply) so the world<->base round-trip stays consistent.

This is read generically from `root_quat_w`, so it stays correct for any base yaw.
"""

from __future__ import annotations

import os

import torch

import isaaclab.utils.math as PoseUtils
from isaaclab_mimic.envs.franka_stack_ik_rel_mimic_env import FrankaCubeStackIKRelMimicEnv

# Optional per-step debug dump of the first N target_eef_pose_to_action calls.
_DEBUG = os.environ.get("LAB_MIMIC_DEBUG")
_DEBUG_PATH = os.environ.get("LAB_MIMIC_DEBUG_PATH", "/tmp/tea_debug.txt")
_DEBUG_MAX = int(os.environ.get("LAB_MIMIC_DEBUG_MAX", "80"))


class LabFR3CubeStackIKRelMimicEnv(FrankaCubeStackIKRelMimicEnv):
    """FR3 stack Mimic env that converts IK-rel deltas in the robot base frame."""

    _dbg_n = 0

    def _root_quat(self, env_ids) -> torch.Tensor:
        """Robot base orientation in world (w,x,y,z), shape (len(env_ids), 4)."""
        return self.scene["robot"].data.root_quat_w[env_ids]

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

        if _DEBUG and env_id == 0 and LabFR3CubeStackIKRelMimicEnv._dbg_n < _DEBUG_MAX:
            i = LabFR3CubeStackIKRelMimicEnv._dbg_n
            LabFR3CubeStackIKRelMimicEnv._dbg_n += 1
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
