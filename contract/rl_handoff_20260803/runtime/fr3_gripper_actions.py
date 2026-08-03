# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import (
    AbsBinaryJointPositionActionCfg,
    BinaryJointPositionActionCfg,
    DifferentialInverseKinematicsActionCfg,
    JointPositionActionCfg,
)
from isaaclab.utils import configclass

"""
FRANKA RESEARCH 3 ACTIONS
"""

# Parallel-jaw Franka hand:
# open = fingers at 0.04 m, close = fingers at 0.0 m. Match the released
# UR5/Robotiq policy contract: positive actions open and negative actions
# close.  Keeping the decision boundary at zero also avoids suppressing
# release/regrasp exploration in acquisition curricula.
FR3_GRIPPER_BINARY_ACTIONS = AbsBinaryJointPositionActionCfg(
    asset_name="robot",
    joint_names=["fr3_finger_joint.*"],
    open_command_expr={"fr3_finger_joint.*": 0.04},
    close_command_expr={"fr3_finger_joint.*": 0.0},
    threshold=0.0,
    positive_threshold=True,
)

FR3_JOINT_POSITION: JointPositionActionCfg = JointPositionActionCfg(
    asset_name="robot",
    joint_names=["fr3_joint.*"],
    scale=0.1,
)

FR3_JOINT_IKDELTA: DifferentialInverseKinematicsActionCfg = DifferentialInverseKinematicsActionCfg(
    asset_name="robot",
    joint_names=["fr3_joint.*"],
    body_name="fr3_hand",
    controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
    scale=0.5,
    body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=(0.0, 0.0, 0.1034), rot=(1.0, 0.0, 0.0, 0.0)),
)

FR3_JOINT_IKABSOLUTE: DifferentialInverseKinematicsActionCfg = DifferentialInverseKinematicsActionCfg(
    asset_name="robot",
    joint_names=["fr3_joint.*"],
    body_name="fr3_hand",
    controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
    scale=1.0,
    body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=(0.0, 0.0, 0.1034), rot=(1.0, 0.0, 0.0, 0.0)),
)


@configclass
class Fr3JointPositionAction:
    jointpos = FR3_JOINT_POSITION


@configclass
class Fr3IkDeltaAction:
    body_joint_pos = FR3_JOINT_IKDELTA
    gripper_joint_pos = FR3_GRIPPER_BINARY_ACTIONS


@configclass
class Fr3IkAbsoluteAction:
    body_joint_pos = FR3_JOINT_IKABSOLUTE
    gripper_joint_pos = FR3_GRIPPER_BINARY_ACTIONS


@configclass
class Fr3BinaryGripperAction:
    gripper = FR3_GRIPPER_BINARY_ACTIONS
