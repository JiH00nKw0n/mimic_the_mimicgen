"""Lab FR3 3-cube-stack Mimic env configs — forward and reverse stacking order.

The lab demos stack in two observed orders (replay-order based): forward
(cube_1 bottom < cube_2 < cube_3 top) and reverse (cube_3 bottom < cube_2 < cube_1).
We keep the OFFICIAL Franka stack Mimic schema and make TWO configs, one per order,
so each demo can be annotated under whichever it matches (operational grouping). No
relabeling of the demos is needed — the demo self-sorts into the group whose
success + subtask signals its replay satisfies.

Both configs are the official `FrankaCubeStackIKRelMimicEnvCfg` retargeted to the lab
FR3 + desk scene (mirrors aidas/3cube_stack/teleop/lab_teleop.py). The grasp/stack/
success mdp functions read `env.cfg.gripper_joint_names / gripper_open_val /
gripper_threshold`, so pointing those at the FR3 fingers is what makes them work here.

Both subtasks always GRASP THE MIDDLE CUBE FIRST (cube_2), because bottom-up stacking
places the middle on the bottom, then the top on the middle. So the two orders differ
only in which cube is the base (subtask 1 object_ref / stack_1 lower) and which is the
top (subtask 2 object_ref / grasp_2 object), plus the success ordering.

Imported only after Isaac Sim launches (see run_annotate.sh).
"""

from __future__ import annotations

import os

import torch

import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as mdp
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_mimic.envs.franka_stack_ik_rel_mimic_env_cfg import FrankaCubeStackIKRelMimicEnvCfg

# FR3 home joint pose the teleop demos start from (states[0]); joint order is
# fr3_joint1..7 then the two fingers. joint6 is 2.25 (the FR3 soft-limit-clamped
# value of jake's 3.037 init).
FR3_HOME_JOINT_POSE = [0.0, -0.569, 0.0, -2.810, 0.0, 2.25, 0.741, 0.04, 0.04]
# Small per-reset arm-joint jitter (rad std), applied on top of the home pose so generation
# keeps the official Franka reset's start-state diversity (stock std 0.02). Fingers are not
# jittered. Seeded by generate_dataset.py's torch.manual_seed, so it stays reproducible.
ARM_JITTER_STD = float(os.environ.get("LAB_ARM_JITTER", "0.02"))


def reset_arm_to_home(env, env_ids, pose=FR3_HOME_JOINT_POSE, arm_jitter_std=ARM_JITTER_STD):
    """Reset event: TELEPORT the FR3 to the demo home pose via write_joint_state_to_sim.

    The stock franka reset uses set_default_joint_pose (buffer only) + a gaussian event
    that calls set_joint_position_target (a PD *target*, not an instantaneous write). That
    leaves the arm at the USD ~zero pose for the first post-reset step, which is what broke
    generation (arm starts pointing up, EE ~0.7 m too high, can't reach the first waypoint).
    Writing the joint STATE directly puts the arm exactly at the home pose immediately, which
    is where the source trajectories begin. A small gaussian jitter on the 7 arm joints
    recovers the start-state diversity the stock gaussian reset provided.
    """
    robot = env.scene["robot"]
    n = len(env_ids)
    p = torch.tensor(pose, device=env.device, dtype=torch.float32).repeat(n, 1)
    if arm_jitter_std > 0.0:
        noise = arm_jitter_std * torch.randn((n, p.shape[1]), device=env.device)
        noise[:, -2:] = 0.0  # leave the two fingers at the open value
        p = p + noise
    robot.write_joint_state_to_sim(p, torch.zeros_like(p), env_ids=env_ids)

# Lab geometry (mirrors lab_teleop.py). Table USD overridable via env var.
LAB_TABLE_USD = os.environ.get("LAB_TABLE_USD", "/home/ubuntu/jake/aidas/3cube_stack/table_scene.usdc")
DESK_Z = 0.720
CUBE = 0.05
ROBOT_POS = (0.72, 0.138, 0.722)
ROBOT_ROT = (0.0, 0.0, 0.0, 1.0)
BASE_XY = (0.32, 0.138)

# Threshold tweaks for FR3 (Panda defaults are slightly off for the FR3 hand geometry):
#   grasp proximity 0.06 -> 0.08 m  (FR3 ee_frame TCP sits a touch farther from the cube)
#   success gripper-open tolerance isclose 1e-4 -> 1e-2 (FR3 binary gripper settles near, not exactly, 0.04)
GRASP_DIFF_THRESHOLD = 0.08
SUCCESS_GRIPPER_ATOL = 1e-2

# IK-rel action scale. Teleop/annotation use 1.0 (matches the recorded demos: jake's
# lab_teleop.py uses scale=1.0). GENERATION re-derives actions via target_eef_pose_to_action,
# whose raw deltas can be large; at scale=1.0 a saturated command applies a full ~1 rad/step
# rotation, the wrist overshoots and the arm whips into a boundary singularity -> 0% DGR.
# The official MimicGen Franka uses scale=0.5 for exactly this reason. So generation overrides
# this to 0.5 via LAB_ARM_SCALE; annotation leaves it at the teleop-faithful 1.0.
ARM_SCALE = float(os.environ.get("LAB_ARM_SCALE", "1.0"))

FR3_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/FrankaRobotics/FrankaFR3/fr3.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, max_depenetration_velocity=5.0),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=ROBOT_POS, rot=ROBOT_ROT,
        joint_pos={
            "fr3_joint1": 0.0, "fr3_joint2": -0.569, "fr3_joint3": 0.0, "fr3_joint4": -2.810,
            "fr3_joint5": 0.0, "fr3_joint6": 3.037, "fr3_joint7": 0.741, "fr3_finger_joint.*": 0.04,
        },
    ),
    actuators={
        "a1": ImplicitActuatorCfg(joint_names_expr=["fr3_joint[1-4]"], stiffness=400.0, damping=80.0),
        "a2": ImplicitActuatorCfg(joint_names_expr=["fr3_joint[5-7]"], stiffness=400.0, damping=80.0),
        "h": ImplicitActuatorCfg(
            joint_names_expr=["fr3_finger_joint.*"], effort_limit_sim=200.0, stiffness=2e3, damping=1e2
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)


def _cube_cfg(name, color, xy):
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/" + name,
        init_state=RigidObjectCfg.InitialStateCfg(pos=(xy[0], xy[1], DESK_Z + CUBE + 0.01), rot=(1, 0, 0, 0)),
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.05, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, max_depenetration_velocity=5.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
        ),
    )


def _apply_lab_overrides(self):
    """Retarget the Franka stack scene to the lab FR3 + desk (same as lab_teleop.py)."""
    self.scene.table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0, 0), rot=(1, 0, 0, 0)),
        spawn=sim_utils.UsdFileCfg(usd_path=LAB_TABLE_USD),
    )
    self.scene.work_surface = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/WorkSurface",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.20, BASE_XY[1], DESK_Z - 0.01)),
        spawn=sim_utils.CuboidCfg(
            size=(0.55, 0.6, 0.02),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.55, 0.58), opacity=0.0),
        ),
    )
    self.scene.robot = FR3_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    self.scene.cube_1 = _cube_cfg("Cube_1", (1.0, 0.0, 0.0), (BASE_XY[0], BASE_XY[1] - 0.10))
    self.scene.cube_2 = _cube_cfg("Cube_2", (0.0, 0.0, 1.0), (BASE_XY[0], BASE_XY[1]))
    self.scene.cube_3 = _cube_cfg("Cube_3", (1.0, 1.0, 0.0), (BASE_XY[0], BASE_XY[1] + 0.10))

    self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
        asset_name="robot", joint_names=["fr3_joint.*"], body_name="fr3_hand", scale=ARM_SCALE,
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=(0.0, 0.0, 0.1034)),
    )
    self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot", joint_names=["fr3_finger_joint.*"],
        open_command_expr={"fr3_finger_joint.*": 0.04}, close_command_expr={"fr3_finger_joint.*": 0.0},
    )
    self.gripper_joint_names = ["fr3_finger_joint.*"]
    self.gripper_open_val = 0.04
    self.gripper_threshold = 0.005

    if hasattr(self.scene, "ee_frame") and self.scene.ee_frame is not None:
        self.scene.ee_frame.prim_path = "{ENV_REGEX_NS}/Robot/fr3_link0"
        for fr in self.scene.ee_frame.target_frames:
            fr.prim_path = (
                fr.prim_path.replace("panda_hand", "fr3_hand")
                .replace("panda_rightfinger", "fr3_rightfinger")
                .replace("panda_leftfinger", "fr3_leftfinger")
            )

    # Reset the FR3 arm to the demos' START pose on every reset. CRITICAL for generation:
    # the stock reset never writes the arm joint STATE to sim immediately (set_default_joint_pose
    # is buffer-only; the gaussian event only sets a PD target), so the arm stays at the USD
    # ~zero pose (pointing up, EE ~0.7 m too high) and the regenerated IK-rel trajectory can't
    # reach the first waypoint -> 0% DGR. We replace those two events with one that TELEPORTS
    # the arm to the demo home pose. Annotation is unaffected (reset_to overrides the state).
    self.events.init_franka_arm_pose = EventTerm(func=reset_arm_to_home, mode="reset", params={})
    if hasattr(self.events, "randomize_franka_joint_state"):
        self.events.randomize_franka_joint_state = None

    # cube randomization for GENERATION. Matched to the random-spawn source dataset
    # (teleop_random_success.hdf5: measured cube spawn x[0.150,0.380], y[0.008,0.273],
    # yaw=0) so that generation samples the SAME distribution the source demos span
    # (paper-style D0: source covers the generation region -> higher DGR). yaw is kept
    # at 0 because the source has zero rotation; widen LAB_GEN_YAW for rotation
    # diversity (lower DGR, pure extrapolation). Annotation overrides cube poses via
    # reset_to(recorded_state), so this only affects generation.
    _gen_yaw = float(os.environ.get("LAB_GEN_YAW", "0.0"))
    if hasattr(self.events, "randomize_cube_positions"):
        self.events.randomize_cube_positions.params["pose_range"] = {
            "x": (0.150, 0.380), "y": (0.008, 0.273), "z": (0.745, 0.745),
            "yaw": (-_gen_yaw, _gen_yaw),
        }

    # Whether generation also writes the (much larger) failed-attempt file. Keep ON for
    # debugging; turn OFF (LAB_KEEP_FAILED=0) for big production runs so we don't dump
    # thousands of failed demos next to the kept dataset. Provenance still counts all
    # attempts regardless (it hooks generate(), not the file writer).
    self.datagen_config.generation_keep_failed = os.environ.get("LAB_KEEP_FAILED", "1") == "1"


def _apply_threshold_fixes(self):
    """Loosen the Panda-tuned grasp/gripper thresholds for the FR3."""
    self.observations.subtask_terms.grasp_1.params["diff_threshold"] = GRASP_DIFF_THRESHOLD
    self.observations.subtask_terms.grasp_2.params["diff_threshold"] = GRASP_DIFF_THRESHOLD
    self.terminations.success.params["atol"] = SUCCESS_GRIPPER_ATOL
    self.terminations.success.params["rtol"] = SUCCESS_GRIPPER_ATOL


@configclass
class LabFR3CubeStackFwdMimicEnvCfg(FrankaCubeStackIKRelMimicEnvCfg):
    """Forward order: cube_1 bottom < cube_2 middle < cube_3 top (official schema)."""

    def __post_init__(self):
        super().__post_init__()
        _apply_lab_overrides(self)
        _apply_threshold_fixes(self)


@configclass
class LabFR3CubeStackRevMimicEnvCfg(FrankaCubeStackIKRelMimicEnvCfg):
    """Reverse order: cube_3 bottom < cube_2 middle < cube_1 top.

    Same 'grasp middle first' structure, but the base is cube_3 and the top is cube_1:
      subtask 0: grasp cube_2 (middle)        -> grasp_1
      subtask 1: stack cube_2 on cube_3 (base)-> stack_1   (object_ref cube_3)
      subtask 2: grasp cube_1 (top)           -> grasp_2   (object_ref cube_1)
      subtask 3: stack cube_1 on cube_2       -> (final)   (object_ref cube_2)
    success: z(cube_3) < z(cube_2) < z(cube_1).
    """

    def __post_init__(self):
        super().__post_init__()
        _apply_lab_overrides(self)
        _apply_threshold_fixes(self)

        eef = list(self.subtask_configs.keys())[0]
        # subtask object_refs: base -> cube_3, top -> cube_1
        self.subtask_configs[eef][1].object_ref = "cube_3"
        self.subtask_configs[eef][2].object_ref = "cube_1"
        # subtask term signals: stack middle onto base(cube_3); grasp the top(cube_1)
        self.observations.subtask_terms.stack_1.params["lower_object_cfg"] = SceneEntityCfg("cube_3")
        self.observations.subtask_terms.grasp_2.params["object_cfg"] = SceneEntityCfg("cube_1")
        # success: reverse the bottom/top identities (cube_3 bottom, cube_1 top)
        self.terminations.success.params["cube_1_cfg"] = SceneEntityCfg("cube_3")
        self.terminations.success.params["cube_3_cfg"] = SceneEntityCfg("cube_1")
