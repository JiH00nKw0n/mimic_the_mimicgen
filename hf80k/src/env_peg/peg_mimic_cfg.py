"""Lab FR3 peg-insert Mimic env config.

Subclasses the OFFICIAL ``FrankaCubeStackIKRelMimicEnvCfg`` (same base as
lab_stack_mimic/lab_mimic_cfg.py) and, in ``_apply_lab_overrides``, retargets the scene from
the tutorial Franka + 3 cubes to the lab FR3 + peg + static socket, exactly as the peg teleop
(``peg_hole_teleop_webxr.py``) built it. It then swaps the 4-subtask stack schema for the
2-subtask peg schema (MimicGen paper §K.3 Square recipe: grasp -> insert).

What is reused unchanged from the stack template:
  * the FR3 retarget (robot cfg, IK-rel + binary-gripper actions, ee_frame prim renames),
  * the "teleport the arm home on reset" event (critical for generation start state),
  * the base MimicGen ``datagen_config`` + per-subtask selection (nearest-neighbor).

What is peg-specific:
  * scene: peg (RigidObject, MUST be named ``peg``) + hole (static AssetBase collider) +
    invisible desk physics slab + visual table USD; the 3 cubes are dropped.
  * observations: replaced wholesale (the stock policy group reads the now-deleted cubes) with
    a peg policy group + a ``subtask_terms`` group carrying the single ``grasp_peg`` signal.
  * success termination = ``peg_inserted`` (peg_mdp), referencing the CONSTANT HOLE_XY.
  * a ``mode="reset"`` peg xy randomization event (generation only; annotation uses reset_to).

Isaac Lab 3.0 CONFIG note: ``InitialStateCfg.rot`` is XYZW in this container (identity =
``(0,0,0,1)``; the FR3 base yaw-180 is ``(0,0,1,0)``), matching the teleop script. This is the
config convention only — runtime ``.data`` / math is authored WXYZ (see peg_mdp).

Imported only after Isaac Sim launches (see run_peg_annotate.sh / run_peg_generate.sh).
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
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_mimic.envs.franka_stack_ik_rel_mimic_env_cfg import FrankaCubeStackIKRelMimicEnvCfg

# ON-BOX TODO(imports): the stack policy-obs helpers (ee_frame_pos / ee_frame_quat / gripper_pos)
# live in the stack task mdp. This exact path is what the peg teleop script imports.
from isaaclab_tasks.manager_based.manipulation.stack import mdp as stack_mdp

import peg_mdp

# ---------------------------------------------------------------------------- lab geometry
# All from peg_hole_teleop_webxr.py (the scene the demos were recorded in).
DESK_Z = 0.720
PEG_SIZE = (0.030, 0.030, 0.060)      # exact box; spawned, not meshed
PEG_XY = (0.101, 0.006)               # authored peg start xy (annotation overrides via reset_to)
HOLE_XY = (0.091, 0.104)              # socket centre (static collider)
ROBOT_POS = (0.72, 0.138, 0.722)
ROBOT_ROT = (0.0, 0.0, 1.0, 0.0)      # XYZW yaw-180 (CONFIG convention; faces -x toward the desk)

# Asset USDs. In the isaac-lab container the teleop mounts them at /work/assets (see the runners).
LAB_PEG_ENV_USD = os.environ.get("LAB_PEG_ENV_USD", "/work/assets/peg_hole_env.usd")
LAB_PEG_HOLE_USD = os.environ.get("LAB_PEG_HOLE_USD", "/work/assets/hole_01.usd")

# FR3 finger geometry differs slightly from the Panda; loosen the grasp proximity like the stack.
GRASP_DIFF_THRESHOLD = 0.13           # EE-to-peg-centre (peg grasped near its top; see peg_mdp)
GRIPPER_OPEN_VAL = 0.04
GRIPPER_THRESHOLD = 0.005

# MimicGen per-subtask knobs (paper Square/Stack settings; also the stack base defaults).
ACTION_NOISE = float(os.environ.get("LAB_ACTION_NOISE", "0.03"))
NUM_INTERP = int(os.environ.get("LAB_NUM_INTERP", "5"))
SUBTASK_OFFSET = (10, 20)             # +10..20-step boundary jitter for the grasp subtask

# IK-rel action scale. Teleop/annotation use 1.0 (matches the recorded demos). GENERATION
# overrides to 0.5 via LAB_ARM_SCALE (run_peg_generate.sh) — at stock 1.0 the re-derived IK-rel
# deltas saturate and the FR3 wrist whips into a boundary singularity (0% DGR), the same failure
# documented for the stack. See lab_stack_mimic/lab_mimic_cfg.py for the full rationale.
ARM_SCALE = float(os.environ.get("LAB_ARM_SCALE", "1.0"))

# Small per-reset arm-joint jitter (rad std) on top of the home pose (generation start diversity).
ARM_JITTER_STD = float(os.environ.get("LAB_ARM_JITTER", "0.02"))

# FR3 home the demos start from. joint6 uses the stack's soft-limit-clamped 2.25 for the GENERATION
# reset (the teleop init used 3.037 and then runtime-tuned joint6 into ~[2.0,4.2] to point the
# gripper straight down). Annotation resets from the HDF5 initial_state, so this only sets the
# generation start pose. Fingers open at 0.04.
FR3_HOME_JOINT_POSE = [0.0, -0.569, 0.0, -2.810, 0.0, 2.25, 0.741, 0.04, 0.04]


def reset_arm_to_home(env, env_ids, pose=FR3_HOME_JOINT_POSE, arm_jitter_std=ARM_JITTER_STD):
    """Reset event: TELEPORT the FR3 to the demo home pose via write_joint_state_to_sim.

    Identical intent to the stack's reset_arm_to_home: the stock franka reset never writes the
    arm joint STATE to sim immediately (set_default_joint_pose is buffer-only; the gaussian event
    only sets a PD target), so the arm stays at the USD ~zero pose (pointing up, EE ~0.7 m too
    high) and the regenerated IK-rel trajectory can't reach the first waypoint -> 0% DGR. Writing
    the joint state directly puts the arm exactly where the source trajectories begin. A small
    gaussian jitter on the 7 arm joints recovers start-state diversity.

    ON-BOX TODO(write API): uses write_joint_state_to_sim (the stack's proven call, on env_uwlab).
    The beta2 container's teleop needed the split write_joint_position_to_sim_index /
    write_joint_velocity_to_sim_index instead; switch here if write_joint_state_to_sim raises
    NotImplementedError.
    """
    robot = env.scene["robot"]
    n = len(env_ids)
    p = torch.tensor(pose, device=env.device, dtype=torch.float32).repeat(n, 1)
    if arm_jitter_std > 0.0:
        noise = arm_jitter_std * torch.randn((n, p.shape[1]), device=env.device)
        noise[:, -2:] = 0.0  # leave the two fingers at the open value
        p = p + noise
    robot.write_joint_state_to_sim(p, torch.zeros_like(p), env_ids=env_ids)


# ---------------------------------------------------------------------------- FR3 articulation
# Verbatim from the teleop / stack FR3_CFG (init joint6 = 3.037, as the teleop authored it).
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


def _peg_cfg():
    """The single tracked rigid object. MUST be named ``peg`` (states/rigid_object/peg reset_to,
    SceneEntityCfg('peg') in the mdp fns). Verbatim from the teleop peg RigidObjectCfg."""
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Peg",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(PEG_XY[0], PEG_XY[1], DESK_Z + PEG_SIZE[2] / 2 + 0.001),
            rot=(0.0, 0.0, 0.0, 1.0),   # XYZW upright (CONFIG convention)
        ),
        spawn=sim_utils.CuboidCfg(
            size=PEG_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, max_depenetration_velocity=1.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.25, 0.15), metallic=0.2),
        ),
    )


# ---------------------------------------------------------------------------- observations
@configclass
class PegObservationsCfg:
    """Replaces the stock stack observations (which read the deleted cubes). Mirrors the teleop
    PegHoleObservationsCfg policy group and adds the ``subtask_terms`` group MimicGen reads."""

    @configclass
    class PolicyCfg(ObsGroup):
        actions = ObsTerm(func=mdp.last_action)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        eef_pos = ObsTerm(func=stack_mdp.ee_frame_pos)
        eef_quat = ObsTerm(func=stack_mdp.ee_frame_quat)
        gripper_pos = ObsTerm(func=stack_mdp.gripper_pos)
        peg_pos = ObsTerm(func=mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("peg")})
        peg_quat = ObsTerm(func=mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("peg")})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class SubtaskCfg(ObsGroup):
        # Group name (attribute below) MUST be "subtask_terms"; term name MUST be "grasp_peg"
        # (peg_mimic_env.get_subtask_term_signals reads obs_buf["subtask_terms"]["grasp_peg"]).
        grasp_peg = ObsTerm(
            func=peg_mdp.peg_grasped,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("peg"),
                "diff_threshold": GRASP_DIFF_THRESHOLD,
                "gripper_open_val": GRIPPER_OPEN_VAL,
                "gripper_threshold": GRIPPER_THRESHOLD,
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
    subtask_terms: SubtaskCfg = SubtaskCfg()


# ---------------------------------------------------------------------------- overrides
def _apply_lab_overrides(self):
    """Retarget the Franka stack scene to the lab FR3 + peg + static socket (as the teleop built
    it), then swap the 4-subtask stack schema for the 2-subtask peg schema."""

    # --- scene ---------------------------------------------------------------------------
    # visual table (env USD; static collider group so the peg filters against it).
    self.scene.table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0, 0), rot=(0.0, 0.0, 0.0, 1.0)),  # XYZW identity
        collision_group=-1,
        spawn=sim_utils.UsdFileCfg(usd_path=LAB_PEG_ENV_USD),
    )
    # invisible physics slab flush with the desk top — the imported desk meshes don't cook into
    # working static collision, so this box is what actually stops the peg (teleop desk_surface).
    self.scene.desk_surface = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/DeskSurface",
        collision_group=-1,
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.30, 0.10, DESK_Z - 0.01)),
        spawn=sim_utils.CuboidCfg(
            size=(0.9, 0.9, 0.02), visible=False, collision_props=sim_utils.CollisionPropertiesCfg()
        ),
    )
    self.scene.robot = FR3_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    self.scene.peg = _peg_cfg()
    # socket = STATIC triangle-mesh collider (no rigid body): an immovable insertion target.
    self.scene.hole = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Hole",
        collision_group=-1,
        init_state=AssetBaseCfg.InitialStateCfg(pos=(HOLE_XY[0], HOLE_XY[1], DESK_Z), rot=(0.0, 0.0, 0.0, 1.0)),
        spawn=sim_utils.UsdFileCfg(usd_path=LAB_PEG_HOLE_USD),
    )
    # drop the stack's three cubes (this env has none).
    for _c in ("cube_1", "cube_2", "cube_3"):
        if hasattr(self.scene, _c):
            setattr(self.scene, _c, None)

    # --- actions (IK-rel delta EE pose + binary gripper) ---------------------------------
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
    self.gripper_open_val = GRIPPER_OPEN_VAL
    self.gripper_threshold = GRIPPER_THRESHOLD

    # --- ee_frame prim renames (panda_* -> fr3_*), same as the stack -----------------------
    if hasattr(self.scene, "ee_frame") and self.scene.ee_frame is not None:
        self.scene.ee_frame.prim_path = "{ENV_REGEX_NS}/Robot/fr3_link0"
        for fr in self.scene.ee_frame.target_frames:
            fr.prim_path = (
                fr.prim_path.replace("panda_hand", "fr3_hand")
                .replace("panda_rightfinger", "fr3_rightfinger")
                .replace("panda_leftfinger", "fr3_leftfinger")
            )

    # --- events --------------------------------------------------------------------------
    # teleport the arm home on reset (see reset_arm_to_home); drop the stock franka/cube events.
    self.events.init_franka_arm_pose = EventTerm(func=reset_arm_to_home, mode="reset", params={})
    if hasattr(self.events, "randomize_franka_joint_state"):
        self.events.randomize_franka_joint_state = None
    if hasattr(self.events, "randomize_cube_positions"):
        self.events.randomize_cube_positions = None
    # peg xy randomization for GENERATION (annotation overrides the peg via reset_to).
    self.events.randomize_peg = EventTerm(
        func=peg_mdp.randomize_peg_xy,
        mode="reset",
        params={
            "x_range": peg_mdp.PEG_REGION_X,
            "y_range": peg_mdp.PEG_REGION_Y,
            "hole_xy": HOLE_XY,
            "min_sep": peg_mdp.PEG_MIN_SEP,
            "asset_cfg": SceneEntityCfg("peg"),
        },
    )

    # --- observations (replace wholesale) ------------------------------------------------
    self.observations = PegObservationsCfg()

    # --- terminations --------------------------------------------------------------------
    for _t in ("cube_1_dropping", "cube_2_dropping", "cube_3_dropping"):
        if hasattr(self.terminations, _t):
            setattr(self.terminations, _t, None)
    self.terminations.success = DoneTerm(
        func=peg_mdp.peg_inserted, params={"peg_cfg": SceneEntityCfg("peg")}
    )

    # --- datagen config ------------------------------------------------------------------
    # keep-failed ON for debugging (LAB_KEEP_FAILED=0 to suppress the failed-attempt file);
    # generation_guarantee retries failures until generation_num_trials clean demos exist.
    self.datagen_config.generation_keep_failed = os.environ.get("LAB_KEEP_FAILED", "1") == "1"
    self.datagen_config.generation_guarantee = True
    # ON-BOX TODO(datagen fields): confirm these attr names on this container's DataGenConfig.
    if hasattr(self.datagen_config, "generation_select_src_per_subtask"):
        self.datagen_config.generation_select_src_per_subtask = True
    if hasattr(self.datagen_config, "name"):
        self.datagen_config.name = "peg_insert_lab_fr3_d0"

    # --- subtask schema: 2 subtasks (grasp peg -> insert) --------------------------------
    # Reuse the base SubTaskConfig instances (all fields already valid — selection strategy,
    # nn kwargs, etc.) and just retarget: keep the base's FIRST (a grasp) and LAST (the final,
    # no-signal) subtask, drop the two middle stack subtasks. This avoids constructing
    # SubTaskConfig from scratch (whose field set we can't verify off-box).
    eef = list(self.subtask_configs.keys())[0]
    base_list = self.subtask_configs[eef]
    grasp_st = base_list[0]
    insert_st = base_list[-1]

    grasp_st.object_ref = "peg"
    grasp_st.subtask_term_signal = "grasp_peg"
    grasp_st.subtask_term_offset_range = SUBTASK_OFFSET
    grasp_st.action_noise = ACTION_NOISE
    grasp_st.num_interpolation_steps = NUM_INTERP

    # Final subtask = insert. object_ref="socket" -> the constant socket frame in
    # peg_mimic_env.get_object_poses -> identity MimicGen transform -> the insert replays verbatim
    # (the socket is fixed). Alternatives if you prefer: "peg" (align the insert to the grasped
    # peg) or None. subtask_term_signal MUST be None (N subtasks need N-1 signals).
    insert_st.object_ref = "socket"
    insert_st.subtask_term_signal = None
    insert_st.subtask_term_offset_range = (0, 0)
    insert_st.action_noise = ACTION_NOISE
    insert_st.num_interpolation_steps = NUM_INTERP

    self.subtask_configs[eef] = [grasp_st, insert_st]


@configclass
class LabFR3PegInsertMimicEnvCfg(FrankaCubeStackIKRelMimicEnvCfg):
    """FR3 peg-insert Mimic env: official Franka-stack Mimic base, retargeted to the lab FR3 +
    peg + static socket, with the 2-subtask grasp->insert schema."""

    def __post_init__(self):
        super().__post_init__()
        _apply_lab_overrides(self)
