# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import re
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers.action_manager import ActionTerm

from . import actions_cfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class RelCartesianOSCAction(ActionTerm):
    """Relative Cartesian OSC action term using a geometric Jacobian and task-space PD control.

    Control law (no inertial decoupling, no mass matrix):
        tau = J^T (Kp * pose_error + Kd * vel_error)  [+ null-space regulation for redundant arms]

    The flow per policy step:
        1. process_actions: scale raw 6-DOF delta, compute desired EE pose
        2. apply_actions (every physics step): compute current state, Jacobian, PD torques, clamp,
           and apply as joint effort targets

    Jacobian source (``cfg.use_physx_jacobian``):
        * False (default): the calibrated UR5e analytical Jacobian (6-DOF, sim2real-aligned). Backward
          compatible -- the UR5e configs are unchanged.
        * True: the PhysX articulation geometric Jacobian, rotated into the robot base frame. Robot-generic,
          supports any DOF count; required for the 7-DOF Franka FR3.

    For redundant arms (DOF > 6) set ``cfg.nullspace_stiffness`` to regulate the self-motion of the arm toward
    a rest posture in the null space of the task, removing the uncontrolled drift of the extra DOF at zero
    action. For a non-redundant arm the null-space projector is the zero map, so leaving it ``None`` keeps the
    UR5e behaviour byte-for-byte.

    Frame convention: both EE pose and Jacobian are expressed in the robot base frame (REP-103).
    """

    cfg: actions_cfg.RelCartesianOSCActionCfg
    """The configuration of the action term."""
    _asset: Articulation
    """The articulation asset on which the action term is applied."""

    def __init__(self, cfg: actions_cfg.RelCartesianOSCActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        # Resolve joints
        self._joint_ids, self._joint_names = self._asset.find_joints(self.cfg.joint_names)
        self._num_dof = len(self._joint_ids)
        # Avoid slice-vs-list indexing overhead when all joints match
        if self._num_dof == self._asset.num_joints:
            self._joint_ids = slice(None)

        # Resolve EE body
        body_ids, body_names = self._asset.find_bodies(self.cfg.body_name)
        if len(body_ids) != 1:
            raise ValueError(
                f"Expected one match for body_name '{self.cfg.body_name}', got {len(body_ids)}: {body_names}"
            )
        self._ee_body_idx = body_ids[0]

        # PhysX Jacobian indexing (mirrors isaaclab DifferentialInverseKinematicsAction): a fixed-base
        # articulation's get_jacobians() omits the (non-existent) floating base, shifting body/joint indices.
        if self._asset.is_fixed_base:
            self._jacobi_body_idx = self._ee_body_idx - 1
            self._jacobi_joint_ids = self._joint_ids
        else:
            self._jacobi_body_idx = self._ee_body_idx
            self._jacobi_joint_ids = (
                slice(None) if isinstance(self._joint_ids, slice) else [i + 6 for i in self._joint_ids]
            )

        # Controller gains (per-env for domain randomization): Kd = 2 * sqrt(Kp) * damping_ratio
        kp = torch.tensor(cfg.motion_stiffness, device=self.device, dtype=torch.float32)
        damping_ratio = torch.tensor(cfg.motion_damping_ratio, device=self.device, dtype=torch.float32)
        kd = 2.0 * torch.sqrt(kp) * damping_ratio
        # Store defaults (1D) and expand to per-env (N, 6)
        self._kp_default = kp
        self._kd_default = kd
        self._damping_ratio_default = damping_ratio
        self._kp = kp.unsqueeze(0).expand(self.num_envs, -1).clone()
        self._kd = kd.unsqueeze(0).expand(self.num_envs, -1).clone()
        self._torque_max = torch.tensor(cfg.torque_limit, device=self.device, dtype=torch.float32)

        # Action scaling
        self._scale = torch.tensor(cfg.scale_xyz_axisangle, device=self.device, dtype=torch.float32)
        if cfg.input_clip is not None:
            self._input_clip = torch.tensor(cfg.input_clip, device=self.device, dtype=torch.float32)
        else:
            self._input_clip = None

        # Null-space regulation (redundant arms only)
        self._has_nullspace = cfg.nullspace_stiffness is not None
        if self._has_nullspace:
            self._kp_ns = float(cfg.nullspace_stiffness)
            self._kd_ns = 2.0 * (self._kp_ns**0.5) * cfg.nullspace_damping_ratio
            self._jac_damping = float(cfg.jacobian_damping)
            self._regulate_to_reset = bool(cfg.nullspace_regulate_to_reset)
            if self._regulate_to_reset:
                # Per-env null-space target = the arm config captured on the first apply after each reset.
                # The action term's reset() runs BEFORE the reset events place the joints, so the placed config
                # cannot be captured there; instead flag the envs and capture in the next apply_actions.
                self._q_default = self._asset.data.joint_pos[:, self._joint_ids].clone()  # (N, num_dof)
                self._need_capture = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
            else:
                self._q_default = self._resolve_default_joint_pos()  # (1 or N, num_dof)
                self._need_capture = None
            self._eye_task = torch.eye(6, device=self.device).unsqueeze(0)
            self._eye_dof = torch.eye(self._num_dof, device=self.device).unsqueeze(0)

        # Buffers
        self._raw_actions = torch.zeros(self.num_envs, 6, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, 6, device=self.device)
        self._ee_pos_des = torch.zeros(self.num_envs, 3, device=self.device)
        self._ee_quat_des = torch.zeros(self.num_envs, 4, device=self.device)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def action_dim(self) -> int:
        return 6

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    # ------------------------------------------------------------------
    # Operations
    # ------------------------------------------------------------------

    def process_actions(self, actions: torch.Tensor):
        """Scale raw 6-DOF deltas and compute desired EE pose for the PD tracker.

        Called once per policy step. The desired pose is held fixed while
        apply_actions recomputes torques at each physics step.
        """
        self._raw_actions[:] = actions
        scaled = actions * self._scale
        if self._input_clip is not None:
            scaled = torch.clamp(scaled, min=self._input_clip[0], max=self._input_clip[1])
        self._processed_actions[:] = scaled

        # Current EE pose in root (base_link) frame
        ee_pos_b, ee_quat_b = self._get_ee_pose_root_frame()

        # Desired position = current + delta
        self._ee_pos_des[:] = ee_pos_b + scaled[:, :3]

        # Desired orientation: axis-angle delta -> quaternion -> compose
        delta_rot = scaled[:, 3:6]
        angle = torch.norm(delta_rot, dim=-1, keepdim=True)
        safe_angle = torch.where(angle > 1e-6, angle, torch.ones_like(angle))
        axis = delta_rot / safe_angle
        axis = torch.where(angle > 1e-6, axis, torch.zeros_like(axis))
        half = angle / 2.0
        delta_quat = torch.cat([torch.cos(half), axis * torch.sin(half)], dim=-1)
        self._ee_quat_des[:] = math_utils.quat_mul(delta_quat, ee_quat_b)

    def apply_actions(self):
        """Compute PD torques using the geometric Jacobian and apply as joint efforts.

        Called every physics step (decimation times per policy step).
        """
        # Current state
        ee_pos_b, ee_quat_b = self._get_ee_pose_root_frame()
        joint_pos = self._asset.data.joint_pos[:, self._joint_ids]
        joint_vel = self._asset.data.joint_vel[:, self._joint_ids]

        # Geometric Jacobian (base frame, matching EE pose frame). (N, 6, num_dof)
        jacobian = self._compute_jacobian(joint_pos)

        # EE velocity from J @ dq (consistent with the Jacobian)
        ee_vel = torch.bmm(jacobian, joint_vel.unsqueeze(-1)).squeeze(-1)  # (N, 6)

        # Pose error
        pos_error = self._ee_pos_des - ee_pos_b
        quat_error = math_utils.quat_mul(self._ee_quat_des, math_utils.quat_inv(ee_quat_b))
        axis_angle_error = math_utils.axis_angle_from_quat(quat_error)
        pose_error = torch.cat([pos_error, axis_angle_error], dim=-1)  # (N, 6)

        # PD control: tau = J^T (Kp * err + Kd * (-vel))
        vel_error = -ee_vel
        task_force = self._kp * pose_error + self._kd * vel_error
        joint_torques = torch.bmm(jacobian.transpose(-1, -2), task_force.unsqueeze(-1)).squeeze(-1)

        # Null-space regulation for redundant arms (zero map for non-redundant arms; skipped when disabled)
        if self._has_nullspace:
            if self._regulate_to_reset and self._need_capture.any():
                # First apply after a reset: latch the IK-placed arm config as the null-space target so the
                # redundant DOF is held where the reset events put it (not dragged toward home).
                cap = self._need_capture
                self._q_default[cap] = joint_pos[cap]
                self._need_capture[cap] = False
            joint_torques = joint_torques + self._nullspace_torque(jacobian, joint_pos, joint_vel)

        joint_torques = torch.clamp(joint_torques, -self._torque_max, self._torque_max)

        self._asset.set_joint_effort_target(joint_torques, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Reset targets to current EE pose to avoid transients."""
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        ee_pos_b, ee_quat_b = self._get_ee_pose_root_frame()
        self._ee_pos_des[env_ids] = ee_pos_b[env_ids]
        self._ee_quat_des[env_ids] = ee_quat_b[env_ids]
        # Re-capture the null-space target on the next apply (reset events have not placed the joints yet here).
        if self._has_nullspace and self._regulate_to_reset:
            self._need_capture[env_ids] = True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_ee_pose_root_frame(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get EE pose in root (base_link) frame from sim state."""
        ee_pos_w = self._asset.data.body_pos_w[:, self._ee_body_idx]
        ee_quat_w = self._asset.data.body_quat_w[:, self._ee_body_idx]
        ee_pos_b, ee_quat_b = math_utils.subtract_frame_transforms(
            self._asset.data.root_pos_w,
            self._asset.data.root_quat_w,
            ee_pos_w,
            ee_quat_w,
        )
        return ee_pos_b, ee_quat_b

    def _compute_jacobian(self, joint_pos: torch.Tensor) -> torch.Tensor:
        """Geometric Jacobian of the EE body w.r.t. the arm joints, in the robot base frame. (N, 6, num_dof)."""
        if self.cfg.use_physx_jacobian:
            # PhysX articulation Jacobian (world frame) -> rotate to base frame.
            jac_w = self._asset.root_physx_view.get_jacobians()[:, self._jacobi_body_idx, :, self._jacobi_joint_ids]
            base_rot = math_utils.matrix_from_quat(math_utils.quat_inv(self._asset.data.root_quat_w))
            return torch.cat([torch.bmm(base_rot, jac_w[:, :3, :]), torch.bmm(base_rot, jac_w[:, 3:, :])], dim=1)
        # Calibrated UR5e analytical Jacobian (lazy import keeps the omnireset mdp decoupled from the robot asset).
        from uwlab_assets.robots.ur5e_robotiq_gripper.kinematics import compute_jacobian_analytical

        return compute_jacobian_analytical(joint_pos, device=str(self.device))

    def _nullspace_torque(
        self, jacobian: torch.Tensor, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> torch.Tensor:
        """Secondary-task torque projected into the null space of the task Jacobian.

        N = I - J^T (J J^T + lam I)^-1 J   (kinematic null-space projector, damped for robustness)
        tau_ns = N (kp_ns (q_default - q) - kd_ns qdot)
        """
        jt = jacobian.transpose(-1, -2)  # (N, dof, 6)
        jjt = torch.bmm(jacobian, jt)  # (N, 6, 6)
        jjt_inv = torch.linalg.solve(jjt + self._jac_damping * self._eye_task, self._eye_task.expand_as(jjt))
        null_proj = self._eye_dof - torch.bmm(torch.bmm(jt, jjt_inv), jacobian)  # (N, dof, dof)
        tau0 = self._kp_ns * (self._q_default - joint_pos) - self._kd_ns * joint_vel  # (N, dof)
        return torch.bmm(null_proj, tau0.unsqueeze(-1)).squeeze(-1)

    def _resolve_default_joint_pos(self) -> torch.Tensor:
        """Rest posture for the null-space target, in arm-joint order. (1, num_dof) or (N, num_dof)."""
        if self.cfg.default_joint_pos is None:
            return self._asset.data.default_joint_pos[:, self._joint_ids].clone()
        values = []
        for name in self._joint_names:
            match = next((ang for pat, ang in self.cfg.default_joint_pos.items() if re.fullmatch(pat, name)), None)
            if match is None:
                raise ValueError(f"default_joint_pos has no entry matching arm joint '{name}'")
            values.append(match)
        return torch.tensor(values, device=self.device, dtype=torch.float32).unsqueeze(0)
