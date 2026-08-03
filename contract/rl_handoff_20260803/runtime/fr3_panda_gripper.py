# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Franka Research 3 robot with its parallel-jaw hand.

Mirrors :mod:`uwlab_assets.robots.ur5e_robotiq_gripper.ur5e_robotiq_2f85_gripper`.

This spawns the NVIDIA FrankaRobotics/FrankaFR3 USD through a local wrapper. The
wrapper keeps robot metadata beside the USD path for the OmniReset metadata
reader while the referenced geometry, joints, and rigid bodies are the real FR3
asset. The arm is driven in pure effort mode so the relative-Cartesian OSC action
is the sole controller.

The following configurations are available:

* :obj:`FR3_ARTICULATION`: Base articulation (USD, init state).
* :obj:`IMPLICIT_FR3`: Full robot with ImplicitActuator arm (stiffness/damping=0,
  pure effort) for RL training.
* :obj:`EXPLICIT_FR3`: Full robot with DelayedPDActuator arm (motor delay) for
  sim2real finetuning.
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

# Local robot asset directory. `fr3_research3.usda` is a thin wrapper that references the NVIDIA FrankaFR3 USD, and
# `metadata.yaml` (gripper_offset, approach axis, aperture, ...) lives beside it so the OmniReset reward /
# grasp-sampler / reset-generator can read the robot metadata from the USD directory (read_metadata_from_usd_directory).
_FR3_ASSET_DIR = os.path.join(os.path.dirname(__file__), "asset")
FR3_HAND_METADATA_PATH = os.path.join(os.path.dirname(__file__), "hand_metadata", "fr3_hand.urdf")

# Canonical Franka "home" pose (matches isaaclab_assets FRANKA_PANDA_CFG). Also used as the
# OSC null-space target posture for the redundant 7th DOF.
FR3_DEFAULT_JOINT_POS = {
    "fr3_joint1": 0.0,
    "fr3_joint2": -0.569,
    "fr3_joint3": 0.0,
    "fr3_joint4": -2.810,
    "fr3_joint5": 0.0,
    "fr3_joint6": 3.037,
    "fr3_joint7": 0.741,
    "fr3_finger_joint.*": 0.04,
}

FR3_HAND_DEFAULT_JOINT_POS = {
    "fr3_finger_joint1": 0.04,
    "fr3_finger_joint2": 0.04,
}

# Franka datasheet torque limits (proximal joints 1-4 stronger than distal 5-7).
FR3_EFFORT_LIMITS = {
    "fr3_joint[1-4]": 87.0,
    "fr3_joint[5-7]": 12.0,
}

# Franka datasheet joint velocity limits (rad/s).
FR3_VELOCITY_LIMITS = {
    "fr3_joint[1-4]": 2.175,
    "fr3_joint[5-7]": 2.61,
}

FR3_ARTICULATION = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=os.path.join(_FR3_ASSET_DIR, "fr3_research3.usda"),
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=36, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0, 0, 0), rot=(1, 0, 0, 0), joint_pos=FR3_DEFAULT_JOINT_POS),
    soft_joint_pos_limit_factor=1,
)

FR3_HAND = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Fr3Hand",
    spawn=sim_utils.UrdfFileCfg(
        asset_path=os.path.join(_FR3_ASSET_DIR, "fr3_hand.urdf"),
        usd_dir=os.path.join(_FR3_ASSET_DIR, "generated"),
        usd_file_name="fr3_hand.usd",
        force_usd_conversion=True,
        make_instanceable=False,
        fix_base=True,
        merge_fixed_joints=False,
        convert_mimic_joints_to_normal_joints=True,
        joint_drive=None,
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=36, solver_velocity_iteration_count=0
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.6),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0, 0, 0.1), rot=(1, 0, 0, 0), joint_pos=FR3_HAND_DEFAULT_JOINT_POS
    ),
    actuators={
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["fr3_finger_joint.*"],
            stiffness=1e4,
            damping=5e2,
            effort_limit_sim=500.0,
        ),
    },
    soft_joint_pos_limit_factor=1,
)

# RL-training robot: zero-PD arm so set_joint_effort_target (the OSC) is the only driver.
IMPLICIT_FR3 = FR3_ARTICULATION.copy()  # type: ignore
IMPLICIT_FR3.actuators = {
    "arm": ImplicitActuatorCfg(
        joint_names_expr=["fr3_joint.*"],
        stiffness=0.0,
        damping=0.0,
        effort_limit_sim=FR3_EFFORT_LIMITS,
        velocity_limit_sim=FR3_VELOCITY_LIMITS,
    ),
    "gripper": ImplicitActuatorCfg(
        joint_names_expr=["fr3_finger_joint.*"],
        stiffness=2e3,
        damping=1e2,
        effort_limit_sim=200.0,
    ),
}

# Reset-state-generation robot: like IMPLICIT_FR3 but the arm actuator adds IMPLICIT joint-space DAMPING
# (stiffness=0, damping>0). The OSC still drives the arm via set_joint_effort_target; the implicit damping is
# solved INSIDE PhysX, so it is unconditionally stable (unlike an explicit -kd*qdot effort term, which at
# dt=1/120 on the FR3's low-inertia distal joints diverges into a ~10 rad/s limit-cycle buzz that fails the
# reset checker's joint-velocity-stability gate). stiffness=0 means NO position bias, so it does not fight the
# IK-placed reset configuration. Used only by record_reset_states; RL training keeps the zero-PD IMPLICIT_FR3.
IMPLICIT_FR3_PD = FR3_ARTICULATION.copy()  # type: ignore
IMPLICIT_FR3_PD.actuators = {
    "arm": ImplicitActuatorCfg(
        joint_names_expr=["fr3_joint.*"],
        stiffness=0.0,
        damping=80.0,
        effort_limit_sim=FR3_EFFORT_LIMITS,
        velocity_limit_sim=FR3_VELOCITY_LIMITS,
    ),
    "gripper": IMPLICIT_FR3.actuators["gripper"],
}

# Sim2real-finetune robot: same zero-PD arm but with a DelayedPDActuator motor delay.
EXPLICIT_FR3 = FR3_ARTICULATION.copy()  # type: ignore
EXPLICIT_FR3.actuators = {
    "arm": DelayedPDActuatorCfg(
        joint_names_expr=["fr3_joint.*"],
        stiffness=0.0,
        damping=0.0,
        effort_limit=FR3_EFFORT_LIMITS,
        effort_limit_sim=FR3_EFFORT_LIMITS,
        velocity_limit=FR3_VELOCITY_LIMITS,
        velocity_limit_sim=FR3_VELOCITY_LIMITS,
        min_delay=0,
        max_delay=1,
    ),
    "gripper": IMPLICIT_FR3.actuators["gripper"],
}
