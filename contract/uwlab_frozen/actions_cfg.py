# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.managers.action_manager import ActionTerm
from isaaclab.managers.manager_term_cfg import ActionTermCfg
from isaaclab.utils import configclass

from . import task_space_actions


@configclass
class RelCartesianOSCActionCfg(ActionTermCfg):
    """Configuration for Relative Cartesian OSC action term.

    Uses the analytical Jacobian from calibrated UR5e kinematics and a simple
    task-space PD controller matching the real robot's OSC implementation:
        tau = J^T @ (Kp * pose_error + Kd * vel_error)

    No inertial dynamics decoupling, no mass matrix. Designed to work with
    the DelayedDCMotor actuator for sim2real alignment.
    """

    class_type: type[ActionTerm] = task_space_actions.RelCartesianOSCAction

    @configclass
    class OffsetCfg:
        """Offset configuration for body or frame offsets."""

        pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
        """Translation offset."""
        rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
        """Rotation offset as quaternion (w, x, y, z)."""

    joint_names: list[str] = MISSING
    """Joint names for the arm (regex supported)."""

    body_name: str = MISSING
    """End-effector body name (e.g., 'wrist_3_link')."""

    scale_xyz_axisangle: tuple[float, float, float, float, float, float] = MISSING
    """Per-DOF scaling for [x, y, z, rx, ry, rz] action deltas."""

    input_clip: tuple[float, float] | None = None
    """Optional symmetric clip range for scaled actions."""

    motion_stiffness: tuple[float, float, float, float, float, float] = (200.0, 200.0, 200.0, 3.0, 3.0, 3.0)
    """Task-space stiffness Kp for [x, y, z, rx, ry, rz]."""

    motion_damping_ratio: tuple[float, float, float, float, float, float] = (3.0, 3.0, 3.0, 1.0, 1.0, 1.0)
    """Task-space damping ratio. Kd = 2 * sqrt(Kp) * damping_ratio."""

    torque_limit: tuple[float, ...] = (150.0, 150.0, 150.0, 28.0, 28.0, 28.0)
    """Per-joint torque limits (clamped after J^T multiplication). Length must match the number of arm joints
    (6 for UR5e, 7 for a Franka FR3)."""

    # -- Jacobian source (robot-generic generalization) --
    use_physx_jacobian: bool = False
    """If True, source the geometric Jacobian from the PhysX articulation view (robot-agnostic, supports any DOF
    count). If False (default), use the calibrated UR5e analytical Jacobian -- preserved for backward
    compatibility / sim2real on the UR5e. Redundant (7-DOF) arms such as the FR3 must set this True."""

    # -- Null-space regulation (only meaningful for redundant arms) --
    nullspace_stiffness: float | None = None
    """If set, regulate the redundant joints toward ``default_joint_pos`` with this stiffness, projected into the
    null space of the task Jacobian (tau_ns = N (kp_ns (q_default - q) - kd_ns qdot), N = I - J^T (JJ^T + lam I)^-1 J).
    None (default) skips the null-space term entirely, so the UR5e (non-redundant) path is unchanged."""

    nullspace_damping_ratio: float = 1.0
    """Null-space damping ratio. kd_ns = 2 * sqrt(nullspace_stiffness) * nullspace_damping_ratio."""

    nullspace_regulate_to_reset: bool = False
    """If True, the null-space target is the per-env arm configuration captured on the FIRST ``apply_actions``
    after each env reset (i.e. the IK-placed reset posture), instead of the fixed ``default_joint_pos``. This
    holds the redundant DOF at wherever the reset events placed the arm -- required by the FR3 reset-state
    generator so the IK-placed arm settles in place rather than being dragged toward home. Only read when
    ``nullspace_stiffness`` is set. Defaults False (use ``default_joint_pos``), so the UR5e / RL paths are
    unchanged."""

    jacobian_damping: float = 0.01
    """Damping factor (lambda) for the damped pseudo-inverse in the null-space projector, for singularity
    robustness."""

    default_joint_pos: dict[str, float] | None = None
    """Per-arm-joint rest posture (regex -> angle) used as the null-space target. If None, the articulation's
    default joint positions are used. Only read when ``nullspace_stiffness`` is set."""
