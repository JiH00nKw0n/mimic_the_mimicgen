# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.controllers.operational_space_cfg import OperationalSpaceControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import OperationalSpaceControllerActionCfg
from isaaclab.utils import configclass
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg

from uwlab_assets.robots.fr3.actions import FR3_GRIPPER_BINARY_ACTIONS
from uwlab_assets.robots.fr3.fr3_panda_gripper import FR3_DEFAULT_JOINT_POS

from ...mdp.actions.actions_cfg import RelCartesianOSCActionCfg

# Legacy fixed null-space rest posture = the FR3 home arm pose (drop the finger entry).
_FR3_ARM_REST = {k: v for k, v in FR3_DEFAULT_JOINT_POS.items() if k.startswith("fr3_joint")}

# Pre-train gains (soft initial Kp). The FR3 is 7-DOF redundant, so use the PhysX Jacobian and regulate the
# redundant self-motion toward the reset posture captured after each reset. This keeps OmniReset reset states from
# being dragged back toward the fixed home elbow/wrist branch while leaving the 6D EE action unchanged.
FR3_RELATIVE_OSC = RelCartesianOSCActionCfg(
    asset_name="robot",
    joint_names=["fr3_joint.*"],
    body_name="fr3_hand",
    scale_xyz_axisangle=(0.02, 0.02, 0.02, 0.02, 0.02, 0.2),
    # Axis-wise rotational gains calibrated from the same R/G/NG reset probes as
    # the released UR5e controller.  Keeping rotational Kd at 2*sqrt(3)
    # (axis-specific zeta below) matches its 0.1 s and 0.3 s rotation response
    # without the severe wrist saturation caused by a naive scalar Kp sweep.
    # On R resets at raw action 0.5: policy-step ratio 0.0753 vs UR5e 0.0771,
    # 0.3 s ratio 0.2250 vs 0.2219; G/NG relative grasp slip stays sub-mm.
    motion_stiffness=(200.0, 200.0, 200.0, 4.5, 7.5, 4.0),
    motion_damping_ratio=(3.0, 3.0, 3.0, 0.81649658, 0.63245553, 0.86602540),
    torque_limit=(87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0),
    use_physx_jacobian=True,
    # Calibrated on 256 representative Peg reaching resets against the source UR5e controller. Kp=10 saturated
    # fr3_joint7 in 55.1% of zero-hold environments and drifted 1.34 mm mean / 5.53 mm p95. Kp=0.5 reduced those
    # to 4.3% and 0.17 mm / 0.22 mm while preserving all four reset-distribution dynamics gates.
    nullspace_stiffness=0.5,
    nullspace_damping_ratio=1.0,
    nullspace_regulate_to_reset=True,
)

# Immutable model-4500 Cube controller.  Peg/Gear use the calibrated global
# FR3_RELATIVE_OSC above; Cube SysID post-training must begin from the exact
# Stage-1 policy-facing controller and only then ramp gains through ADR.
FR3_CUBE_STAGE1_RELATIVE_OSC = RelCartesianOSCActionCfg(
    asset_name="robot",
    joint_names=["fr3_joint.*"],
    body_name="fr3_hand",
    scale_xyz_axisangle=(0.02, 0.02, 0.02, 0.02, 0.02, 0.2),
    motion_stiffness=(200.0, 200.0, 200.0, 3.0, 3.0, 3.0),
    motion_damping_ratio=(3.0, 3.0, 3.0, 1.0, 1.0, 1.0),
    torque_limit=(87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0),
    use_physx_jacobian=True,
    nullspace_stiffness=10.0,
    nullspace_damping_ratio=1.0,
    nullspace_regulate_to_reset=True,
)

# Eval / sim2real gains (high Kp, end-of-curriculum values).
FR3_RELATIVE_OSC_EVAL = RelCartesianOSCActionCfg(
    asset_name="robot",
    joint_names=["fr3_joint.*"],
    body_name="fr3_hand",
    scale_xyz_axisangle=(0.01, 0.01, 0.002, 0.02, 0.02, 0.2),
    motion_stiffness=(1000.0, 1000.0, 1000.0, 50.0, 50.0, 50.0),
    motion_damping_ratio=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    torque_limit=(87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0),
    use_physx_jacobian=True,
    nullspace_stiffness=10.0,
    nullspace_damping_ratio=1.0,
    nullspace_regulate_to_reset=True,
)

# Unscaled (for sysid scripts).
FR3_RELATIVE_OSC_UNSCALED = RelCartesianOSCActionCfg(
    asset_name="robot",
    joint_names=["fr3_joint.*"],
    body_name="fr3_hand",
    scale_xyz_axisangle=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    motion_stiffness=(1000.0, 1000.0, 1000.0, 50.0, 50.0, 50.0),
    motion_damping_ratio=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    torque_limit=(87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0),
    use_physx_jacobian=True,
    nullspace_stiffness=10.0,
    nullspace_damping_ratio=1.0,
    default_joint_pos=_FR3_ARM_REST,
)

# Reset-state generation gains. The reset generator IK-places the arm at a sampled pose, applies zero arm
# action, and lets it settle in place; the success checker rejects any env whose EE drifts > 0.1 m or never
# reaches velocity stability within the 2 s settle. The FR3 is 7-DOF redundant: the task-space OSC constrains
# only the 6-DOF EE, leaving the redundant self-motion uncontrolled. Disabling the null-space entirely
# (nullspace_stiffness=None) lets that 7th DOF drift ~0.5 rad and the arm never settles (NOT-stable ~100%).
# Regulating the null-space toward the HOME posture instead drags the IK-placed arm back toward home (EE drift
# ~0.16 m). The correct hold: stiff null-space regulation toward the CAPTURED reset posture
# (nullspace_regulate_to_reset=True) -- the redundant DOF is pinned at exactly where IK placed it, so the EE
# stays put AND the arm settles. High eval-grade Kp gives a stiff task-space hold on top.
FR3_RELATIVE_OSC_RESETGEN = RelCartesianOSCActionCfg(
    asset_name="robot",
    joint_names=["fr3_joint.*"],
    body_name="fr3_hand",
    scale_xyz_axisangle=(0.02, 0.02, 0.02, 0.02, 0.02, 0.2),
    # Soft, well-damped gains (the UR5e reset-gen uses these exact training gains, NOT the stiff eval gains).
    # The stiff eval gains (Kp 1000 / damping 1) buzz on the zero-PD effort arm at 120 Hz: positions barely move
    # but joint velocities stay ~8 rad/s (> the checker's jvel-sum<5 stability gate), so every env is rejected
    # as NOT-stable. The high translation damping ratio (3.0) below settles the hold smoothly.
    motion_stiffness=(200.0, 200.0, 200.0, 3.0, 3.0, 3.0),
    motion_damping_ratio=(3.0, 3.0, 3.0, 1.0, 1.0, 1.0),
    torque_limit=(87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0),
    use_physx_jacobian=True,
    nullspace_stiffness=50.0,  # stiff hold of the redundant DOF toward the captured reset posture
    nullspace_damping_ratio=2.0,  # over-damped so the redundant DOF settles without ringing
    nullspace_regulate_to_reset=True,  # target = captured IK-placed reset posture, not home
    # NOTE: velocity stability is handled by the IMPLICIT_FR3_PD reset-gen robot's implicit joint damping
    # (solved in-PhysX, unconditionally stable), NOT by an explicit effort damper -- an explicit -kd*qdot
    # effort term diverges at dt=1/120 on the FR3's low-inertia distal joints (verified: it made jvel WORSE).
)

# Isaac Lab's built-in operational-space controller. This is the closest in-repo equivalent to a Franka/libfranka-style
# impedance controller: it uses the PhysX mass matrix, supports inertial decoupling, and can apply gravity compensation.
# Keep it as a separate action config so existing OmniReset UR5e-compatible RelCartesianOSC runs remain reproducible.
FR3_ISAACLAB_OSC = OperationalSpaceControllerActionCfg(
    asset_name="robot",
    joint_names=["fr3_joint.*"],
    body_name="fr3_hand",
    controller_cfg=OperationalSpaceControllerCfg(
        target_types=["pose_rel"],
        impedance_mode="fixed",
        inertial_dynamics_decoupling=True,
        partial_inertial_dynamics_decoupling=False,
        gravity_compensation=True,
        motion_stiffness_task=(200.0, 200.0, 200.0, 30.0, 30.0, 30.0),
        motion_damping_ratio_task=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        nullspace_control="position",
        nullspace_stiffness=10.0,
        nullspace_damping_ratio=1.0,
    ),
    nullspace_joint_pos_target="default",
    position_scale=0.01,
    orientation_scale=0.02,
)

FR3_GRIPPER_RESETGEN_JOINT_POSITION = JointPositionActionCfg(
    asset_name="robot",
    joint_names=["fr3_finger_joint.*"],
    scale=1.0,
    offset=0.0,
    use_default_offset=False,
)


@configclass
class Fr3CubeRelativeOSCAction:
    """Action config using the generalized (PhysX-Jacobian + null-space) OSC + Franka binary gripper."""

    arm = FR3_RELATIVE_OSC
    gripper = FR3_GRIPPER_BINARY_ACTIONS


@configclass
class Fr3CubeStage1RelativeOSCAction:
    """Exact legacy OSC/action contract used by Cube model 4500."""

    arm = FR3_CUBE_STAGE1_RELATIVE_OSC
    gripper = FR3_GRIPPER_BINARY_ACTIONS


@configclass
class Fr3CubeIsaacLabOSCAction:
    """Action config using Isaac Lab's built-in operational-space controller + Franka binary gripper."""

    arm = FR3_ISAACLAB_OSC
    gripper = FR3_GRIPPER_BINARY_ACTIONS


@configclass
class Fr3CubeRelativeOSCResetGenAction:
    """Reset-state-generation action: PhysX-Jacobian OSC with stiff null-space regulation toward the CAPTURED
    IK-placed reset posture (nullspace_regulate_to_reset=True), so the redundant 7th DOF is pinned where the
    reset events placed it and the arm settles in place instead of drifting. Used only by the record_reset_states
    tool; RL training uses the same reset-posture target with softer gains."""

    arm = FR3_RELATIVE_OSC_RESETGEN
    gripper = FR3_GRIPPER_RESETGEN_JOINT_POSITION


@configclass
class Fr3CubeRelativeOSCEvalAction:
    """Action config with high Kp gains (end-of-curriculum values) for eval / data-collection."""

    arm = FR3_RELATIVE_OSC_EVAL
    gripper = FR3_GRIPPER_BINARY_ACTIONS


@configclass
class Fr3CubeSysidOSCAction:
    """Unscaled arm action (Cartesian delta) + binary gripper. For Sysid env / scripts."""

    arm = FR3_RELATIVE_OSC_UNSCALED
    gripper = FR3_GRIPPER_BINARY_ACTIONS
