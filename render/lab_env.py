"""Shared lab FR3 3-cube-stack scene builder for the fr3_cams tools.

Mirrors ../record_video.py::build_env_cfg (the proven replay scene: lab desk +
Isaac FrankaFR3 + 3 cubes, IK-rel actions, reset events stripped), plus optional
attachment of the calibrated overlay cameras as Isaac Lab Camera sensors.

Import only AFTER AppLauncher has started the sim app.
"""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as mdp
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

TASK = "Isaac-Stack-Cube-Franka-IK-Rel-v0"
DESK_Z, CUBE = 0.720, 0.05
# NOTE Isaac Lab 3.0 has MIXED quaternion conventions: cfg/init-state quats stay
# (w,x,y,z) — verified: re-encoding this rot as xyzw flips the robot upside down —
# while runtime data reads and asset writes are (x,y,z,w). Keep cfg values classic.
ROBOT_POS, ROBOT_ROT = (0.72, 0.138, 0.722), (0.0, 0.0, 0.0, 1.0)
BASE_XY = (0.32, 0.138)

FR3_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/FrankaRobotics/FrankaFR3/fr3.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, max_depenetration_velocity=5.0),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0)),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=ROBOT_POS, rot=ROBOT_ROT,
        joint_pos={"fr3_joint1": 0.0, "fr3_joint2": -0.569, "fr3_joint3": 0.0, "fr3_joint4": -2.810,
                   "fr3_joint5": 0.0, "fr3_joint6": 3.037, "fr3_joint7": 0.741, "fr3_finger_joint.*": 0.04}),
    actuators={
        "a1": ImplicitActuatorCfg(joint_names_expr=["fr3_joint[1-4]"], stiffness=400.0, damping=80.0),
        "a2": ImplicitActuatorCfg(joint_names_expr=["fr3_joint[5-7]"], stiffness=400.0, damping=80.0),
        "h": ImplicitActuatorCfg(joint_names_expr=["fr3_finger_joint.*"], effort_limit_sim=200.0, stiffness=2e3, damping=1e2)},
    soft_joint_pos_limit_factor=1.0,
)


def _cube_cfg(name, color, xy):
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/" + name,
        init_state=RigidObjectCfg.InitialStateCfg(pos=(xy[0], xy[1], DESK_Z + CUBE + 0.01), rot=(1, 0, 0, 0)),
        spawn=sim_utils.CuboidCfg(size=(0.05, 0.05, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, max_depenetration_velocity=5.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05), collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color)))


def build_env_cfg(device: str, table_usd: str, cameras: dict | None = None, num_envs: int = 1):
    cfg = parse_env_cfg(TASK, device=device, num_envs=num_envs)
    # fabric skips the physx->USD writeback, so camera prims parented to robot
    # links never see the spawn yaw or per-step link motion (Isaac Lab 3.0).
    # USD-pipeline mode restores classic composition; the scene is tiny, so the
    # writeback cost is irrelevant next to RTX rendering.
    if hasattr(cfg.sim, "use_fabric"):
        cfg.sim.use_fabric = False
    if os.path.isfile(table_usd):
        cfg.scene.table = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/Table",
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0, 0), rot=(1, 0, 0, 0)),
            spawn=sim_utils.UsdFileCfg(usd_path=table_usd))
    else:
        # servers without the lab desk asset (e.g. aidas): visible stand-in slab,
        # top surface at the same DESK_Z so replayed cube states still sit on it
        print(f"[lab_env] WARNING: table USD not found ({table_usd}); using a stand-in desk slab")
        cfg.scene.table = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/Table",
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.35, BASE_XY[1], DESK_Z - 0.015)),
            spawn=sim_utils.CuboidCfg(size=(1.4, 1.2, 0.03), collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.48, 0.45, 0.42))))
    cfg.scene.work_surface = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/WorkSurface",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.20, BASE_XY[1], DESK_Z - 0.01)),
        spawn=sim_utils.CuboidCfg(size=(0.55, 0.6, 0.02), collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.55, 0.58), opacity=0.0)))
    cfg.scene.robot = FR3_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=FR3_CFG.init_state)
    cfg.scene.cube_1 = _cube_cfg("Cube_1", (1.0, 0.0, 0.0), (BASE_XY[0], BASE_XY[1] - 0.10))
    cfg.scene.cube_2 = _cube_cfg("Cube_2", (0.0, 0.0, 1.0), (BASE_XY[0], BASE_XY[1]))
    cfg.scene.cube_3 = _cube_cfg("Cube_3", (1.0, 1.0, 0.0), (BASE_XY[0], BASE_XY[1] + 0.10))
    cfg.actions.arm_action = DifferentialInverseKinematicsActionCfg(
        asset_name="robot", joint_names=["fr3_joint.*"], body_name="fr3_hand", scale=1.0,
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=(0.0, 0.0, 0.1034)))
    cfg.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot", joint_names=["fr3_finger_joint.*"],
        open_command_expr={"fr3_finger_joint.*": 0.04}, close_command_expr={"fr3_finger_joint.*": 0.0})
    if hasattr(cfg, "gripper_joint_names"):
        cfg.gripper_joint_names = ["fr3_finger_joint.*"]
    for ev in ("init_franka_arm_pose", "randomize_franka_joint_state", "randomize_cube_positions"):
        if hasattr(cfg.events, ev):
            setattr(cfg.events, ev, None)
    if hasattr(cfg.scene, "ee_frame"):
        cfg.scene.ee_frame.prim_path = "{ENV_REGEX_NS}/Robot/fr3_link0"
        for fr in cfg.scene.ee_frame.target_frames:
            fr.prim_path = (fr.prim_path.replace("panda_hand", "fr3_hand")
                            .replace("panda_rightfinger", "fr3_rightfinger").replace("panda_leftfinger", "fr3_leftfinger"))
    cfg.terminations = {}
    cfg.recorders = {}
    cfg.viewer.eye = (1.45, -0.55, 1.25)
    cfg.viewer.lookat = (0.32, 0.14, 0.76)
    cfg.viewer.origin_type = "world"
    if cameras:
        for name, cam_cfg in cameras.items():
            setattr(cfg.scene, name, cam_cfg)
    return cfg
