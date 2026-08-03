# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING
import os

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from uwlab_assets import UWLAB_CLOUD_ASSETS_DIR
from uwlab_assets.robots.fr3 import EXPLICIT_FR3, IMPLICIT_FR3

from .actions import (
    Fr3CubeIsaacLabOSCAction,
    Fr3CubeRelativeOSCAction,
    Fr3CubeRelativeOSCEvalAction,
    Fr3CubeStage1RelativeOSCAction,
)
from .calibrated_sysid import apply_fr3_cube_calibration_bundle

from ... import mdp as task_mdp


_DEFAULT_WORKSPACE_ROOT = os.environ.get("UWLAB_WORKSPACE_ROOT", os.path.expanduser("~/jake"))
LAB_DATASET_DIR = os.environ.get("UWLAB_FR3_CUBE_DATASET_DIR", f"{_DEFAULT_WORKSPACE_ROOT}/datasets/LabOmniResearch3Procedural")
LAB_ASSET_ROOT = os.environ.get("UWLAB_FR3_CUBE_ASSET_ROOT", f"{_DEFAULT_WORKSPACE_ROOT}/datasets/LabAssets")
LAB_SCENE_ROOT = os.environ.get("UWLAB_FR3_SCENE_ROOT", f"{_DEFAULT_WORKSPACE_ROOT}/aidas/3cube_stack")
LAB_RESET_POOL_DIR = os.environ.get("UWLAB_FR3_CUBE_RESET_POOL_DIR", f"{LAB_DATASET_DIR}/ResetPoolsResearch3")
LAB_TABLE_USD = f"{LAB_SCENE_ROOT}/table_scene.usdc"
# Local 5cm cube metadata (see lab_three_cube_reset_states_cfg.py): cloud cube metadata is for
# 4cm cubes; identity/metadata lookup only — cubes spawn procedurally at 0.05.
INSERTIVE_CUBE_USD = f"{LAB_ASSET_ROOT}/Props/Custom/InsertiveCube/insertive_cube.usd"
RECEPTIVE_CUBE_USD = f"{LAB_ASSET_ROOT}/Props/Custom/ReceptiveCube/receptive_cube.usd"

ROBOT_POS = (0.72, 0.138, 0.722)
ROBOT_ROT = (0.0, 0.0, 0.0, 1.0)
DESK_Z = 0.720
CUBE_HALF_SIZE = 0.025
BASE_XY = (0.25, 0.138)
TABLE_FORBIDDEN_BOXES = (
    # Keep moving robot bodies out of the visible tabletop volume and below-table region.
    ((-0.40, -0.90, 0.00), (0.53, 0.90, DESK_Z - 0.005)),
    # The lab table USD's metal support blocks are visual-only, so mirror them with safety boxes.
    ((0.3975, -0.8355, 0.00), (1.0425, -0.1905, 0.702)),
    ((0.3975, -0.1845, 0.00), (1.0425, 0.4605, 0.702)),
)
TABLE_SAFETY_ROBOT_CFG = SceneEntityCfg("robot", body_names=["fr3_link[4-8]", "fr3_hand"])
ACTIVE_CUBE_STATIC_FRICTION_RANGE = (1.0, 2.0)
ACTIVE_CUBE_DYNAMIC_FRICTION_RANGE = (0.9, 1.9)
RECEPTIVE_CUBE_STATIC_FRICTION_RANGE = (0.2, 0.6)
RECEPTIVE_CUBE_DYNAMIC_FRICTION_RANGE = (0.15, 0.5)
TABLE_STATIC_FRICTION = 0.45
TABLE_DYNAMIC_FRICTION = 0.35

# Rank-0 task success rates from model_4800 over iterations 4800--4809 in the
# exact p=0 full-stack finetune environment.  SysID/action-scale ADR advances
# relative to this checkpoint rather than using OmniReset's UR5e-specific 95%.
MODEL_4800_P0_SUCCESS_BASELINE = [
    0.9385806084,
    0.9116128683,
    0.9335483789,
    0.9231935203,
    0.8874838471,
    0.8837419152,
    0.8922257483,
    0.7837741435,
]


def make_lab_cube(name: str, color: tuple[float, float, float], xy: tuple[float, float], usd_path: str):
    spawn = sim_utils.CuboidCfg(
        size=(0.05, 0.05, 0.05),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=False,
            disable_gravity=False,
            max_depenetration_velocity=5.0,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=2.0, dynamic_friction=1.8),
        physics_material_path="/World/SharedPhysicsMaterials/fr3_cube",
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
    )
    spawn.usd_path = usd_path
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        spawn=spawn,
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(xy[0], xy[1], DESK_Z + CUBE_HALF_SIZE), rot=(1.0, 0.0, 0.0, 0.0)
        ),
    )


@configclass
class RlStateSceneCfg(InteractiveSceneCfg):
    """Scene configuration for RL state environment."""

    robot = IMPLICIT_FR3.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot.init_state.pos = ROBOT_POS
    robot.init_state.rot = ROBOT_ROT

    cube_1 = make_lab_cube("Cube_1", (1.0, 0.0, 0.0), (BASE_XY[0], BASE_XY[1] - 0.10), RECEPTIVE_CUBE_USD)
    cube_2 = make_lab_cube("Cube_2", (0.0, 0.0, 1.0), (BASE_XY[0], BASE_XY[1]), INSERTIVE_CUBE_USD)
    cube_3 = make_lab_cube("Cube_3", (0.0, 0.8, 0.0), (BASE_XY[0], BASE_XY[1] + 0.10), INSERTIVE_CUBE_USD)

    # Environment
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=sim_utils.UsdFileCfg(
            usd_path=LAB_TABLE_USD,
        ),
    )

    work_surface = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/WorkSurface",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.18, 0.20, DESK_Z - 0.01)),
        spawn=sim_utils.CuboidCfg(
            size=(0.70, 0.90, 0.02),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=TABLE_STATIC_FRICTION,
                dynamic_friction=TABLE_DYNAMIC_FRICTION,
            ),
            physics_material_path="/World/SharedPhysicsMaterials/work_surface",
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.55, 0.58), opacity=0.0),
        ),
    )
    table_support_1 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TableSupport1",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.72, -0.513, 0.351)),
        spawn=sim_utils.CuboidCfg(
            size=(0.645, 0.645, 0.702),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.1, 0.1), opacity=0.0),
        ),
    )
    table_support_2 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TableSupport2",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.72, 0.138, 0.351)),
        spawn=sim_utils.CuboidCfg(
            size=(0.645, 0.645, 0.702),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.1, 0.1), opacity=0.0),
        ),
    )

    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        spawn=sim_utils.GroundPlaneCfg(),
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=1000.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class BaseEventCfg:
    """Shared events: material/mass randomization, gripper gains, scene reset.

    Does NOT include arm sysid or OSC gain randomization -- those differ
    between finetune (curriculum-ramped) and eval (fixed) stages.  See
    ``FinetuneEventCfg`` and ``FinetuneEvalEventCfg``.
    """

    # mode: startup (randomize dynamics)
    robot_material = EventTerm(
        func=task_mdp.randomize_rigid_body_material,  # type: ignore
        mode="startup",
        params={
            "static_friction_range": (0.3, 1.2),
            "dynamic_friction_range": (0.2, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 256,
            "asset_cfg": SceneEntityCfg("robot"),
            "make_consistent": True,
        },
    )

    cube_1_material = EventTerm(
        func=task_mdp.randomize_rigid_body_material,  # type: ignore
        mode="startup",
        params={
            "static_friction_range": RECEPTIVE_CUBE_STATIC_FRICTION_RANGE,
            "dynamic_friction_range": RECEPTIVE_CUBE_DYNAMIC_FRICTION_RANGE,
            "restitution_range": (0.0, 0.0),
            "num_buckets": 256,
            "asset_cfg": SceneEntityCfg("cube_1"),
            "make_consistent": True,
        },
    )

    cube_2_material = EventTerm(
        func=task_mdp.randomize_rigid_body_material,  # type: ignore
        mode="startup",
        params={
            "static_friction_range": ACTIVE_CUBE_STATIC_FRICTION_RANGE,
            "dynamic_friction_range": ACTIVE_CUBE_DYNAMIC_FRICTION_RANGE,
            "restitution_range": (0.0, 0.0),
            "num_buckets": 256,
            "asset_cfg": SceneEntityCfg("cube_2"),
            "make_consistent": True,
        },
    )

    cube_3_material = EventTerm(
        func=task_mdp.randomize_rigid_body_material,  # type: ignore
        mode="startup",
        params={
            "static_friction_range": ACTIVE_CUBE_STATIC_FRICTION_RANGE,
            "dynamic_friction_range": ACTIVE_CUBE_DYNAMIC_FRICTION_RANGE,
            "restitution_range": (0.0, 0.0),
            "num_buckets": 256,
            "asset_cfg": SceneEntityCfg("cube_3"),
            "make_consistent": True,
        },
    )

    table_material = None

    randomize_robot_mass = EventTerm(
        func=task_mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "mass_distribution_params": (0.7, 1.3),
            "operation": "scale",
            "distribution": "uniform",
            "recompute_inertia": True,
        },
    )

    randomize_cube_1_mass = EventTerm(
        func=task_mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("cube_1"),
            "mass_distribution_params": (0.1, 0.1),
            "operation": "abs",
            "distribution": "uniform",
            "recompute_inertia": True,
        },
    )

    randomize_cube_2_mass = EventTerm(
        func=task_mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("cube_2"),
            "mass_distribution_params": (0.1, 0.1),
            "operation": "abs",
            "distribution": "uniform",
            "recompute_inertia": True,
        },
    )

    randomize_cube_3_mass = EventTerm(
        func=task_mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("cube_3"),
            "mass_distribution_params": (0.1, 0.1),
            "operation": "abs",
            "distribution": "uniform",
            "recompute_inertia": True,
        },
    )

    randomize_gripper_actuator_parameters = EventTerm(
        func=task_mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["fr3_finger_joint.*"]),
            "stiffness_distribution_params": (0.5, 2.0),
            "damping_distribution_params": (0.5, 2.0),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )

    # mode: reset
    reset_everything = EventTerm(func=task_mdp.reset_scene_to_default, mode="reset", params={})


@configclass
class TrainEventCfg(BaseEventCfg):
    """Training events: material/mass randomization + 4-path resets. No sysid or OSC gain randomization."""

    reset_from_reset_states = EventTerm(
        func=task_mdp.lab_three_cube_reset_from_states,
        mode="reset",
        params={
            "dataset_dir": LAB_RESET_POOL_DIR,
            "stages": ["two_stacked"],
            "reset_types": ["reaching", "grasped", "nearobject", "neargoal"],
            "probs": [0.25, 0.25, 0.25, 0.25],
            "min_states_per_pool": 10000,
            "success": "env.reward_manager.get_term_cfg('progress_context').func.success",
        },
    )


@configclass
class FullStackTrainEventCfg(TrainEventCfg):
    """Training events for both one_stacked and two_stacked reset pools."""

    reset_from_reset_states = EventTerm(
        func=task_mdp.lab_three_cube_reset_from_states,
        mode="reset",
        params={
            "dataset_dir": LAB_RESET_POOL_DIR,
            "stages": ["one_stacked", "two_stacked"],
            "reset_types": ["reaching", "grasped", "nearobject", "neargoal"],
            "probs": [0.125] * 8,
            "min_states_per_pool": 10000,
            "success": "env.reward_manager.get_term_cfg('progress_context').func.success",
        },
    )


@configclass
class FullStackEvalEventCfg(BaseEventCfg):
    """Eval events for both stages; reaching-only by default."""

    reset_from_reset_states = EventTerm(
        func=task_mdp.lab_three_cube_reset_from_states,
        mode="reset",
        params={
            "dataset_dir": LAB_RESET_POOL_DIR,
            "stages": ["one_stacked", "two_stacked"],
            "reset_types": ["reaching"],
            "probs": [0.5, 0.5],
            "min_states_per_pool": 10000,
            "success": "env.reward_manager.get_term_cfg('progress_context').func.success",
        },
    )


@configclass
class TrainEvalEventCfg(BaseEventCfg):
    """Eval after Stage 1: no sysid/OSC gain randomization, 1-path resets."""

    reset_from_reset_states = EventTerm(
        func=task_mdp.lab_three_cube_reset_from_states,
        mode="reset",
        params={
            "dataset_dir": LAB_RESET_POOL_DIR,
            "stages": ["two_stacked"],
            "reset_types": ["reaching"],
            "probs": [1.0],
            "min_states_per_pool": 10000,
            "success": "env.reward_manager.get_term_cfg('progress_context').func.success",
        },
    )


@configclass
class FinetuneEvalEventCfg(BaseEventCfg):
    """Eval after Stage 2: fixed sysid + OSC gains (scale_progress=1) + 1-path resets."""

    randomize_arm_sysid = EventTerm(
        func=task_mdp.randomize_arm_from_sysid_fixed,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "joint_names": [
                "fr3_joint1",
                "fr3_joint2",
                "fr3_joint3",
                "fr3_joint4",
                "fr3_joint5",
                "fr3_joint6",
                "fr3_joint7",
            ],
            "actuator_name": "arm",
            "scale_range": (0.8, 1.2),
            "delay_range": (0, 3),
        },
    )

    randomize_osc_gains = EventTerm(
        func=task_mdp.randomize_rel_cartesian_osc_gains_fixed,
        mode="reset",
        params={
            "action_name": "arm",
            "scale_range": (0.8, 1.2),
        },
    )

    reset_from_reset_states = EventTerm(
        func=task_mdp.lab_three_cube_reset_from_states,
        mode="reset",
        params={
            "dataset_dir": LAB_RESET_POOL_DIR,
            "stages": ["two_stacked"],
            "reset_types": ["reaching"],
            "probs": [1.0],
            "min_states_per_pool": 10000,
            "success": "env.reward_manager.get_term_cfg('progress_context').func.success",
        },
    )


@configclass
class FinetuneEventCfg(TrainEventCfg):
    """Finetune events: curriculum-ramped sysid + OSC gains + 4-path resets. Explicit actuator from start."""

    randomize_arm_sysid = EventTerm(
        func=task_mdp.randomize_arm_from_sysid,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "joint_names": [
                "fr3_joint1",
                "fr3_joint2",
                "fr3_joint3",
                "fr3_joint4",
                "fr3_joint5",
                "fr3_joint6",
                "fr3_joint7",
            ],
            "actuator_name": "arm",
            "scale_range": (0.8, 1.2),
            "delay_range": (0, 3),
            "initial_scale_progress": 0.0,
        },
    )


@configclass
class FullStackFinetuneEventCfg(FinetuneEventCfg):
    """SysID ADR events while replaying all one/two-stack reset distributions."""

    reset_from_reset_states = EventTerm(
        func=task_mdp.lab_three_cube_reset_from_states,
        mode="reset",
        params={
            "dataset_dir": LAB_RESET_POOL_DIR,
            "stages": ["one_stacked", "two_stacked"],
            "reset_types": ["reaching", "grasped", "nearobject", "neargoal"],
            "probs": [0.125] * 8,
            # Pools are generated with 10k rows; runtime joint-limit validation may
            # discard a small fraction. Require >=95% usable rather than accepting
            # invalid states or failing on a harmless sub-percent shortfall.
            "min_states_per_pool": 9500,
            # The baseline is a ten-iteration average of 100-sample task
            # windows. Use an equivalent 1000-outcome window so an individual
            # noisy reset pool cannot chatter the relative ADR gate.
            "success_history_len": 1000,
            "success": "env.reward_manager.get_term_cfg('progress_context').func.success",
        },
    )

    randomize_osc_gains = EventTerm(
        func=task_mdp.randomize_rel_cartesian_osc_gains,
        mode="reset",
        params={
            "action_name": "arm",
            "scale_range": (0.8, 1.2),
            "terminal_kp": (1000.0, 1000.0, 1000.0, 50.0, 50.0, 50.0),
            "terminal_damping_ratio": (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
            "initial_scale_progress": 0.0,
        },
    )


@configclass
class FullStackCalibratedRobustEventCfg(FullStackTrainEventCfg):
    """Team SysID bundle overlay; generic OmniReset physics DR is disabled."""

    robot_material = None
    cube_1_material = None
    cube_2_material = None
    cube_3_material = None
    table_material = None
    randomize_robot_mass = None
    randomize_cube_1_mass = None
    randomize_cube_2_mass = None
    randomize_cube_3_mass = None
    randomize_gripper_actuator_parameters = None

    reset_from_reset_states = EventTerm(
        func=task_mdp.lab_three_cube_reset_from_states,
        mode="reset",
        params={
            "dataset_dir": LAB_RESET_POOL_DIR,
            "stages": ["one_stacked", "two_stacked"],
            "reset_types": ["reaching", "grasped", "nearobject", "neargoal"],
            "probs": [0.125] * 8,
            "min_states_per_pool": 9500,
            "success": "env.reward_manager.get_term_cfg('progress_context').func.success",
        },
    )

    calibrated_physics = EventTerm(
        func=apply_fr3_cube_calibration_bundle,
        mode="startup",
        params={
            "bundle_root": os.environ.get(
                "UWLAB_FR3_CUBE_CALIBRATION_BUNDLE",
                "/home/ubuntu/jake/aidas/3cube_stack/fr3_cube_system_calibration_bundle_v1",
            ),
            "profile": os.environ.get("UWLAB_FR3_CUBE_CALIBRATION_PROFILE", "robust_stochastic"),
            "robot_cfg": SceneEntityCfg("robot"),
            "cube_names": ("cube_1", "cube_2", "cube_3"),
            "work_surface_name": "work_surface",
            "arm_actuator_name": "arm",
            "sample_seed_offset": 73000,
            "log_samples": os.environ.get("UWLAB_FR3_CUBE_CALIBRATION_LOG_SAMPLES", "0") == "1",
        },
    )


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    pass


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        prev_actions = ObsTerm(func=task_mdp.last_action)

        joint_pos = ObsTerm(func=task_mdp.joint_pos)

        end_effector_pose = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("robot", body_names="fr3_hand"),
                "root_asset_cfg": SceneEntityCfg("robot"),
                "rotation_repr": "axis_angle",
            },
        )

        active_cube_pose = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("cube_3"),
                "root_asset_cfg": SceneEntityCfg("robot", body_names="fr3_hand"),
                "rotation_repr": "axis_angle",
            },
        )

        receptive_cube_pose = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("cube_2"),
                "root_asset_cfg": SceneEntityCfg("robot", body_names="fr3_hand"),
                "rotation_repr": "axis_angle",
            },
        )

        active_cube_in_receptive_cube_frame: ObsTerm = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("cube_3"),
                "root_asset_cfg": SceneEntityCfg("cube_2"),
                "rotation_repr": "axis_angle",
            },
        )

        cube_2_in_cube_1_frame: ObsTerm = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("cube_2"),
                "root_asset_cfg": SceneEntityCfg("cube_1"),
                "rotation_repr": "axis_angle",
            },
        )

        cube_3_in_cube_2_frame: ObsTerm = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("cube_3"),
                "root_asset_cfg": SceneEntityCfg("cube_2"),
                "rotation_repr": "axis_angle",
            },
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 5

    @configclass
    class CriticCfg(ObsGroup):
        """Critic observations for policy group."""

        prev_actions = ObsTerm(func=task_mdp.last_action)

        joint_pos = ObsTerm(func=task_mdp.joint_pos)

        end_effector_pose = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("robot", body_names="fr3_hand"),
                "root_asset_cfg": SceneEntityCfg("robot"),
                "rotation_repr": "axis_angle",
            },
        )

        active_cube_pose = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("cube_3"),
                "root_asset_cfg": SceneEntityCfg("robot", body_names="fr3_hand"),
                "rotation_repr": "axis_angle",
            },
        )

        receptive_cube_pose = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("cube_2"),
                "root_asset_cfg": SceneEntityCfg("robot", body_names="fr3_hand"),
                "rotation_repr": "axis_angle",
            },
        )

        active_cube_in_receptive_cube_frame: ObsTerm = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("cube_3"),
                "root_asset_cfg": SceneEntityCfg("cube_2"),
                "rotation_repr": "axis_angle",
            },
        )

        cube_2_in_cube_1_frame: ObsTerm = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("cube_2"),
                "root_asset_cfg": SceneEntityCfg("cube_1"),
                "rotation_repr": "axis_angle",
            },
        )

        cube_3_in_cube_2_frame: ObsTerm = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("cube_3"),
                "root_asset_cfg": SceneEntityCfg("cube_2"),
                "rotation_repr": "axis_angle",
            },
        )

        # privileged observations
        time_left = ObsTerm(func=task_mdp.time_left)

        joint_vel = ObsTerm(func=task_mdp.joint_vel)

        end_effector_vel_lin_ang_b = ObsTerm(
            func=task_mdp.asset_link_velocity_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("robot", body_names="fr3_hand"),
                "root_asset_cfg": SceneEntityCfg("robot"),
            },
        )

        robot_material_properties = ObsTerm(
            func=task_mdp.get_material_properties, params={"asset_cfg": SceneEntityCfg("robot")}
        )

        cube_1_material_properties = ObsTerm(
            func=task_mdp.get_material_properties, params={"asset_cfg": SceneEntityCfg("cube_1")}
        )

        cube_2_material_properties = ObsTerm(
            func=task_mdp.get_material_properties, params={"asset_cfg": SceneEntityCfg("cube_2")}
        )

        cube_3_material_properties = ObsTerm(
            func=task_mdp.get_material_properties, params={"asset_cfg": SceneEntityCfg("cube_3")}
        )

        robot_mass = ObsTerm(func=task_mdp.get_mass, params={"asset_cfg": SceneEntityCfg("robot")})

        cube_1_mass = ObsTerm(
            func=task_mdp.get_mass, params={"asset_cfg": SceneEntityCfg("cube_1")}
        )

        cube_2_mass = ObsTerm(
            func=task_mdp.get_mass, params={"asset_cfg": SceneEntityCfg("cube_2")}
        )

        cube_3_mass = ObsTerm(
            func=task_mdp.get_mass, params={"asset_cfg": SceneEntityCfg("cube_3")}
        )

        robot_joint_friction = ObsTerm(func=task_mdp.get_joint_friction, params={"asset_cfg": SceneEntityCfg("robot")})

        robot_joint_armature = ObsTerm(func=task_mdp.get_joint_armature, params={"asset_cfg": SceneEntityCfg("robot")})

        robot_joint_stiffness = ObsTerm(
            func=task_mdp.get_joint_stiffness, params={"asset_cfg": SceneEntityCfg("robot")}
        )

        robot_joint_damping = ObsTerm(func=task_mdp.get_joint_damping, params={"asset_cfg": SceneEntityCfg("robot")})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            self.history_length = 1

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:

    # safety rewards

    action_magnitude = RewTerm(func=task_mdp.action_l2_clamped, weight=-1e-4)

    action_rate = RewTerm(func=task_mdp.action_rate_l2_clamped, weight=-1e-3)

    joint_vel = RewTerm(
        func=task_mdp.joint_vel_l2_clamped,
        weight=-1e-2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["fr3_joint.*"])},
    )

    abnormal_robot = RewTerm(func=task_mdp.abnormal_robot_state, weight=-100.0)

    table_intrusion = None

    # task rewards

    progress_context = RewTerm(
        func=task_mdp.LabThreeCubeStackProgressContext,  # type: ignore
        weight=0.1,
        params={
            "cube_1_cfg": SceneEntityCfg("cube_1"),
            "cube_2_cfg": SceneEntityCfg("cube_2"),
            "cube_3_cfg": SceneEntityCfg("cube_3"),
            "stage": "two_stacked",
            "cube_half_size": CUBE_HALF_SIZE,
            "success_position_threshold": 0.005,
            "success_orientation_threshold": 0.025,
            "reset_event_name": "reset_from_reset_states",
            "sequential_mode": False,
            "stage_switch_success_steps": 3,
        },
    )

    ee_asset_distance = RewTerm(
        func=task_mdp.ee_asset_distance_tanh,
        weight=0.1,
        params={
            "root_asset_cfg": SceneEntityCfg("robot", body_names="fr3_hand"),
            "target_asset_cfg": SceneEntityCfg("cube_3"),
            "root_asset_offset_metadata_key": "gripper_offset",
            "std": 1.0,
        },
    )

    dense_success_reward = RewTerm(func=task_mdp.dense_success_reward, weight=0.1, params={"std": 1.0})

    foundation_cube_stability = RewTerm(
        func=task_mdp.lab_foundation_cube_stability_penalty,
        weight=-0.5,
        params={
            "context": "progress_context",
            "cube_half_size": CUBE_HALF_SIZE,
            "pos_tolerance": 0.0125,
            "ori_tolerance": 0.08,
            "z_drop_tolerance": 0.01,
        },
    )

    success_reward = RewTerm(func=task_mdp.success_reward, weight=1.0)

    one_stack_intermediate_reward = RewTerm(
        func=task_mdp.lab_one_stack_intermediate_reward,
        weight=0.0,
        params={"context": "progress_context"},
    )

    stage_switch_bonus = RewTerm(
        func=task_mdp.lab_stage_switch_bonus,
        weight=0.0,
        params={"context": "progress_context"},
    )

    final_stack_success_reward = RewTerm(
        func=task_mdp.lab_final_stack_success_reward,
        weight=2.0,
        params={"context": "progress_context"},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=task_mdp.time_out, time_out=True)

    abnormal_robot = DoneTerm(func=task_mdp.abnormal_robot_state)

    table_intrusion = None


@configclass
class FinetuneCurriculumsCfg:
    """Finetune curriculum: ADR sysid + action scale ramp. No actuator swap (explicit from start)."""

    adr_sysid = CurrTerm(
        func=task_mdp.adr_sysid_curriculum,
        params={
            "event_term_names": ["randomize_arm_sysid", "randomize_osc_gains"],
            "reset_event_name": "reset_from_reset_states",
            "success_threshold_up": 0.95,
            "success_threshold_down": 0.9,
            "delta": 0.01,
            "update_every_n_steps": 200,
            "initial_scale_progress": 0.0,
            "warmup_success_threshold": None,
            "baseline_success_rates": MODEL_4800_P0_SUCCESS_BASELINE,
            "mean_margin_up": 0.03,
            "mean_margin_down": 0.05,
            "per_task_margin_up": 0.05,
            "per_task_margin_down": 0.10,
        },
    )

    action_scale = CurrTerm(
        func=task_mdp.action_scale_curriculum,
        params={
            "action_name": "arm",
            "reset_event_name": "reset_from_reset_states",
            "initial_scales": [0.02, 0.02, 0.02, 0.02, 0.02, 0.2],
            "target_scales": [0.01, 0.01, 0.002, 0.02, 0.02, 0.2],
            "success_threshold_up": 0.95,
            "success_threshold_down": 0.9,
            "delta": 0.01,
            "update_every_n_steps": 200,
            "initial_progress": 0.0,
            "baseline_success_rates": MODEL_4800_P0_SUCCESS_BASELINE,
            "mean_margin_up": 0.03,
            "mean_margin_down": 0.05,
            "per_task_margin_up": 0.05,
            "per_task_margin_down": 0.10,
        },
    )


@configclass
class NoCurriculumsCfg:
    """No curriculum (eval / data-collection with fixed 0.8--1.2 randomization)."""

    pass


variants = {}


@configclass
class Fr3CubeRlStateCfg(ManagerBasedRLEnvCfg):
    stack_stage: str = "two_stacked"
    scene: RlStateSceneCfg = RlStateSceneCfg(num_envs=32, env_spacing=1.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: Fr3CubeRelativeOSCAction = Fr3CubeRelativeOSCAction()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: NoCurriculumsCfg = NoCurriculumsCfg()
    events: BaseEventCfg = MISSING
    commands: CommandsCfg = CommandsCfg()
    viewer: ViewerCfg = ViewerCfg(eye=(2.0, 0.0, 0.75), origin_type="world", env_index=0, asset_name="robot")
    variants = variants

    def __post_init__(self):
        self._apply_stack_stage()
        self.decimation = 12
        self.episode_length_s = 16.0
        # simulation settings
        self.sim.dt = 1 / 120.0

        # Contact and solver settings
        self.sim.physx.solver_type = 1
        self.sim.physx.max_position_iteration_count = 192
        self.sim.physx.max_velocity_iteration_count = 1
        self.sim.physx.bounce_threshold_velocity = 0.02
        self.sim.physx.friction_offset_threshold = 0.01
        self.sim.physx.friction_correlation_distance = 0.0005

        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 2**23
        self.sim.physx.gpu_max_rigid_contact_count = 2**23
        self.sim.physx.gpu_max_rigid_patch_count = 2**23
        self.sim.physx.gpu_collision_stack_size = 2**31

        # Render settings
        self.sim.render.enable_dlssg = True
        self.sim.render.enable_ambient_occlusion = True
        self.sim.render.enable_reflections = True
        self.sim.render.enable_dl_denoiser = True

    def _apply_stack_stage(self):
        if self.stack_stage not in ("one_stacked", "two_stacked", "full_stack"):
            raise ValueError(f"Unsupported stack_stage: {self.stack_stage}")

        if self.stack_stage == "full_stack":
            self.scene.cube_1.spawn.rigid_props.kinematic_enabled = True
            self.scene.cube_2.spawn.rigid_props.kinematic_enabled = False
            self.scene.cube_3.spawn.rigid_props.kinematic_enabled = False
            self.scene.cube_1.spawn.mass_props.mass = 0.5
            self.scene.cube_2.spawn.mass_props.mass = 0.1
            self.scene.cube_3.spawn.mass_props.mass = 0.1
            self.scene.cube_1.spawn.usd_path = RECEPTIVE_CUBE_USD
            self.scene.cube_2.spawn.usd_path = INSERTIVE_CUBE_USD
            self.scene.cube_3.spawn.usd_path = INSERTIVE_CUBE_USD
            self._set_cube_material_role("cube_1", "receptive")
            self._set_cube_material_role("cube_2", "active")
            self._set_cube_material_role("cube_3", "active")
            self.rewards.progress_context.params["stage"] = "two_stacked"
            self._apply_full_stack_stage_aware_terms()
            return

        active_cube = "cube_2" if self.stack_stage == "one_stacked" else "cube_3"
        receptive_cube = "cube_1" if self.stack_stage == "one_stacked" else "cube_2"

        # Match the pool generator's kinematic-base assumption: only the active cube is dynamic.
        self.scene.cube_1.spawn.rigid_props.kinematic_enabled = self.stack_stage in ("one_stacked", "two_stacked")
        self.scene.cube_2.spawn.rigid_props.kinematic_enabled = self.stack_stage == "two_stacked"
        self.scene.cube_3.spawn.rigid_props.kinematic_enabled = self.stack_stage == "one_stacked"
        self.scene.cube_1.spawn.mass_props.mass = 0.5 if self.scene.cube_1.spawn.rigid_props.kinematic_enabled else 0.1
        self.scene.cube_2.spawn.mass_props.mass = 0.5 if self.scene.cube_2.spawn.rigid_props.kinematic_enabled else 0.1
        self.scene.cube_3.spawn.mass_props.mass = 0.5 if self.scene.cube_3.spawn.rigid_props.kinematic_enabled else 0.1
        for cube_name in ("cube_1", "cube_2", "cube_3"):
            cube_cfg = getattr(self.scene, cube_name)
            cube_cfg.spawn.usd_path = INSERTIVE_CUBE_USD if cube_name == active_cube else RECEPTIVE_CUBE_USD
            self._set_cube_material_role(cube_name, "active" if cube_name == active_cube else "receptive")

        if hasattr(self.events, "reset_from_reset_states"):
            self.events.reset_from_reset_states.params["stages"] = [self.stack_stage]

        self.rewards.progress_context.params["stage"] = self.stack_stage
        self.rewards.ee_asset_distance.params["target_asset_cfg"] = SceneEntityCfg(active_cube)

        for group in (self.observations.policy, self.observations.critic):
            group.active_cube_pose.params["target_asset_cfg"] = SceneEntityCfg(active_cube)
            group.receptive_cube_pose.params["target_asset_cfg"] = SceneEntityCfg(receptive_cube)
            group.active_cube_in_receptive_cube_frame.params["target_asset_cfg"] = SceneEntityCfg(active_cube)
            group.active_cube_in_receptive_cube_frame.params["root_asset_cfg"] = SceneEntityCfg(receptive_cube)

    def _set_cube_material_role(self, cube_name: str, role: str):
        event = getattr(self.events, f"{cube_name}_material", None)
        if event is None:
            return
        if role == "active":
            event.params["static_friction_range"] = ACTIVE_CUBE_STATIC_FRICTION_RANGE
            event.params["dynamic_friction_range"] = ACTIVE_CUBE_DYNAMIC_FRICTION_RANGE
        elif role == "receptive":
            event.params["static_friction_range"] = RECEPTIVE_CUBE_STATIC_FRICTION_RANGE
            event.params["dynamic_friction_range"] = RECEPTIVE_CUBE_DYNAMIC_FRICTION_RANGE
        else:
            raise ValueError(f"Unsupported cube material role: {role}")
        event.params["num_buckets"] = 256

    def _apply_full_stack_stage_aware_terms(self):
        if hasattr(self.events, "reset_from_reset_states"):
            self.events.reset_from_reset_states.params["stages"] = ["one_stacked", "two_stacked"]

        hand_cfg = SceneEntityCfg("robot", body_names="fr3_hand")
        for group in (self.observations.policy, self.observations.critic):
            group.active_cube_pose.func = task_mdp.lab_stage_target_asset_pose_in_root_asset_frame
            group.active_cube_pose.params = {
                "one_target_asset_cfg": SceneEntityCfg("cube_2"),
                "two_target_asset_cfg": SceneEntityCfg("cube_3"),
                "one_root_asset_cfg": hand_cfg,
                "two_root_asset_cfg": hand_cfg,
                "rotation_repr": "axis_angle",
                "reset_event_name": "reset_from_reset_states",
            }
            group.receptive_cube_pose.func = task_mdp.lab_stage_target_asset_pose_in_root_asset_frame
            group.receptive_cube_pose.params = {
                "one_target_asset_cfg": SceneEntityCfg("cube_1"),
                "two_target_asset_cfg": SceneEntityCfg("cube_2"),
                "one_root_asset_cfg": hand_cfg,
                "two_root_asset_cfg": hand_cfg,
                "rotation_repr": "axis_angle",
                "reset_event_name": "reset_from_reset_states",
            }
            group.active_cube_in_receptive_cube_frame.func = task_mdp.lab_stage_target_asset_pose_in_root_asset_frame
            group.active_cube_in_receptive_cube_frame.params = {
                "one_target_asset_cfg": SceneEntityCfg("cube_2"),
                "two_target_asset_cfg": SceneEntityCfg("cube_3"),
                "one_root_asset_cfg": SceneEntityCfg("cube_1"),
                "two_root_asset_cfg": SceneEntityCfg("cube_2"),
                "rotation_repr": "axis_angle",
                "reset_event_name": "reset_from_reset_states",
            }

        self.rewards.ee_asset_distance.func = task_mdp.lab_stage_ee_active_cube_distance_tanh
        self.rewards.ee_asset_distance.params = {
            "root_asset_cfg": hand_cfg,
            "one_target_asset_cfg": SceneEntityCfg("cube_2"),
            "two_target_asset_cfg": SceneEntityCfg("cube_3"),
            "root_asset_offset_metadata_key": "gripper_offset",
            "std": 1.0,
            "reset_event_name": "reset_from_reset_states",
        }


# Training configuration (Stage 1: no curriculum, implicit actuator, no sysid DR)
@configclass
class Fr3CubeRelCartesianOSCTrainCfg(Fr3CubeRlStateCfg):

    events: TrainEventCfg = TrainEventCfg()
    actions: Fr3CubeRelativeOSCAction = Fr3CubeRelativeOSCAction()


@configclass
class Fr3CubeRelCartesianOSCOneStackTrainCfg(Fr3CubeRelCartesianOSCTrainCfg):
    """Optional pretraining config: cube_2 on cube_1 using the same four OmniReset reset distributions."""

    stack_stage = "one_stacked"


@configclass
class Fr3CubeRelCartesianOSCOneStackLiteTrainCfg(Fr3CubeRelCartesianOSCOneStackTrainCfg):
    """One-stack training config with reduced material count for high-env-count runs."""

    def __post_init__(self):
        super().__post_init__()
        self.events.robot_material = None
        self.events.cube_3_material = None
        self.events.randomize_cube_3_mass = None


@configclass
class Fr3CubeIsaacLabOSCTrainCfg(Fr3CubeRlStateCfg):
    """Training config using Isaac Lab's built-in operational-space controller."""

    events: TrainEventCfg = TrainEventCfg()
    actions: Fr3CubeIsaacLabOSCAction = Fr3CubeIsaacLabOSCAction()


@configclass
class Fr3CubeIsaacLabOSCOneStackTrainCfg(Fr3CubeIsaacLabOSCTrainCfg):
    """One-stack training config using Isaac Lab's built-in operational-space controller."""

    stack_stage = "one_stacked"


@configclass
class Fr3CubeRelCartesianOSCFullStackTrainCfg(Fr3CubeRlStateCfg):
    """Full stack config: one_stacked and two_stacked reset distributions mixed uniformly."""

    stack_stage = "full_stack"
    events: FullStackTrainEventCfg = FullStackTrainEventCfg()
    actions: Fr3CubeRelativeOSCAction = Fr3CubeRelativeOSCAction()


# Finetune configuration (Stage 2: explicit actuator, curriculum ramps sysid + gains + scales)
@configclass
class Fr3CubeRelCartesianOSCFinetuneCfg(Fr3CubeRlStateCfg):
    """Finetune config: loads converged Stage 1 policy, explicit actuator from start, curriculum ramps DR."""

    events: FinetuneEventCfg = FinetuneEventCfg()
    actions: Fr3CubeStage1RelativeOSCAction = Fr3CubeStage1RelativeOSCAction()
    curriculum: FinetuneCurriculumsCfg = FinetuneCurriculumsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = EXPLICIT_FR3.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.actuators["arm"].max_delay = 3
        self.scene.robot.init_state.pos = ROBOT_POS
        self.scene.robot.init_state.rot = ROBOT_ROT


@configclass
class Fr3CubeRelCartesianOSCFullStackFinetuneCfg(Fr3CubeRelCartesianOSCFinetuneCfg):
    """Model-4500 post-training over all eight one/two-stack reset pools."""

    stack_stage = "full_stack"
    events: FullStackFinetuneEventCfg = FullStackFinetuneEventCfg()


@configclass
class Fr3CubeRelCartesianOSCFullStackCalibratedRobustEvalCfg(Fr3CubeRlStateCfg):
    """Full-stack evaluation under the team handoff's calibrated physics.

    The Stage-1 OSC/action contract remains frozen.  This config is separate
    from finetuning so existing training and checkpoints are untouched.
    """

    stack_stage = "full_stack"
    events: FullStackCalibratedRobustEventCfg = FullStackCalibratedRobustEventCfg()
    actions: Fr3CubeStage1RelativeOSCAction = Fr3CubeStage1RelativeOSCAction()
    curriculum: NoCurriculumsCfg = NoCurriculumsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = EXPLICIT_FR3.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.actuators["arm"].max_delay = 3
        self.scene.robot.init_state.pos = ROBOT_POS
        self.scene.robot.init_state.rot = ROBOT_ROT

        # The calibrated objects are three dynamic 50.7 mm measured cubes.
        for cube_name in ("cube_1", "cube_2", "cube_3"):
            cube = getattr(self.scene, cube_name)
            cube.spawn.size = (0.0507, 0.0507, 0.0507)
            cube.spawn.rigid_props.kinematic_enabled = False
            cube.spawn.rigid_props.linear_damping = 0.0682659557
            cube.spawn.rigid_props.angular_damping = 0.0766774104
            cube.spawn.physics_material.friction_combine_mode = "multiply"
            cube.spawn.physics_material.restitution_combine_mode = "average"

        # Make the invisible task support a tracked kinematic rigid object so
        # each environment can receive its own calibrated material component.
        self.scene.work_surface = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/WorkSurface",
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.18, 0.20, DESK_Z - 0.01)),
            spawn=sim_utils.CuboidCfg(
                size=(0.70, 0.90, 0.02),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=True,
                    disable_gravity=True,
                    max_depenetration_velocity=5.0,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=100.0),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=1.0,
                    dynamic_friction=1.0,
                    restitution=0.0,
                    friction_combine_mode="multiply",
                    restitution_combine_mode="average",
                ),
                physics_material_path="/World/SharedPhysicsMaterials/calibrated_work_surface",
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.55, 0.55, 0.58), opacity=0.0
                ),
            ),
        )


# Evaluation configuration (after Stage 1: implicit actuator, soft gains, no sysid DR)
@configclass
class Fr3CubeRelCartesianOSCEvalCfg(Fr3CubeRlStateCfg):
    """Eval after Stage 1: implicit actuator, soft gains, large action scale, no sysid DR."""

    events: TrainEvalEventCfg = TrainEvalEventCfg()
    actions: Fr3CubeRelativeOSCAction = Fr3CubeRelativeOSCAction()


@configclass
class Fr3CubeRelCartesianOSCOneStackEvalCfg(Fr3CubeRelCartesianOSCEvalCfg):
    """Optional one_stacked play/eval config."""

    stack_stage = "one_stacked"


@configclass
class Fr3CubeRelCartesianOSCFullStackEvalCfg(Fr3CubeRlStateCfg):
    """Full stack play/eval config with stage-aware observations."""

    stack_stage = "full_stack"
    events: FullStackEvalEventCfg = FullStackEvalEventCfg()
    actions: Fr3CubeRelativeOSCAction = Fr3CubeRelativeOSCAction()


# Evaluation configuration (after Stage 2: explicit actuator, stiff gains, fixed sysid)
@configclass
class Fr3CubeRelCartesianOSCFinetuneEvalCfg(Fr3CubeRlStateCfg):
    """Eval after Stage 2: explicit actuator, stiff gains, small action scale, fixed sysid + OSC gains."""

    events: FinetuneEvalEventCfg = FinetuneEvalEventCfg()
    actions: Fr3CubeRelativeOSCEvalAction = Fr3CubeRelativeOSCEvalAction()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = EXPLICIT_FR3.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.actuators["arm"].max_delay = 3
        self.scene.robot.init_state.pos = ROBOT_POS
        self.scene.robot.init_state.rot = ROBOT_ROT
