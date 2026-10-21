"""기어 조립 태스크의 MimicGen 환경 설정.

핀 꽂기의 peg_mimic_cfg와 같은 짜임이다. 공식 큐브 쌓기 Mimic 설정을 물려받아 장면과
동작과 관측과 구간을 기어 조립으로 바꿔 끼운다. 바꾼 곳만 보면 된다.

장면이 핀 꽂기와 겹치는 부분이 많다. 로봇 받침의 자리와 방향이 (0.72, 0.138, 0.722)과
z축 180도로 **한 자리까지 같고** 작업면 높이도 0.720으로 같다. 시연을 만든 쪽의 배치
파일(legacy_multi_gear/config/scene_layout.json)과 우리 핀 환경의 상수를 대조해 확인했다.
그래서 FR3 설정은 핀 환경의 것을 그대로 쓴다.

다른 곳은 물체와 구간이다. 핀과 소켓 둘 대신 받침 하나와 기어 세 개가 올라가고, 구간이
둘에서 여섯으로 늘어난다. 큰 기어를 집고 꽂고, 작은 기어를 집고 꽂고, 중간 기어를 집고
꽂는다. MimicGen은 구간이 N개면 완료 신호를 N-1개 요구하므로 신호는 다섯 개다.

실측 물리는 붙이지 않는다. 기어를 재서 얻은 물리 묶음이 없다. 핀 꽂기는
fr3_peg_in_hole_physics_bundle_v1_normalized를 썼지만 기어용으로 잰 값이 아직 없다.
그래서 이 환경의 수율은 손대지 않은 기본 물리에서 잰 값이다.
"""

from __future__ import annotations

import copy
import os

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
from isaaclab_tasks.manager_based.manipulation.stack import mdp as stack_mdp

import gear_mdp
import source_routing
from gear_geom import GEARS

# ---------------------------------------------------------------------------- 장면 기하
# 전부 scene_layout.json과 teleop_multigear_lab_v3.py에서 가져온 값이다.
DESK_Z = 0.720
ROBOT_POS = (0.72, 0.138, 0.722)
ROBOT_ROT = (0.0, 0.0, 1.0, 0.0)      # XYZW z축 180도. 로봇이 -x 방향으로 책상을 본다
BASE_SPAWN = (0.30, 0.28, 0.720)      # 받침의 기본 자리. 생성에서는 리셋마다 다시 뽑는다
GEAR_SPAWN = {"large": (0.33025, -0.06, 0.715),
              "small": (0.24925, 0.04, 0.715),
              "medium": (0.27975, 0.13, 0.715)}

# 자산 경로. 오케스트레이터가 태스크 프로필의 extra_env로 넣어 준다.
LAB_GEAR_TABLE_USD = os.environ.get(
    "LAB_GEAR_TABLE_USD", "/opt/hf80k/assets/legacy_multi_gear/assets/lab/table_scene.usdc")
LAB_GEAR_ASSET_ROOT = os.environ.get(
    "LAB_GEAR_ASSET_ROOT", "/opt/hf80k/assets/legacy_multi_gear/assets/gears")

# IK 상대 제어에서 한 스텝에 허용하는 이동량의 배율. 기어 시연의 한 스텝 이동량은
# 평균 2.9 mm, 최대 15.3 mm다. 핀 꽂기의 13.6 mm와 비슷해서 같은 1.0을 쓴다.
ARM_SCALE = float(os.environ.get("LAB_ARM_SCALE", "1.0"))

# 매 스텝 명령에 더하는 잡음. 기본은 0이다.
#
# 핀 꽂기에서 잡음 0.03이 성공을 0으로 만들었다. 틈이 0.7 mm인 삽입에서 잡음이 그보다
# 훨씬 커서다. 기어는 축에 끼우는 작업이고 성공 판정이 축 오차 1 mm 안쪽을 요구하므로
# 핀보다 더 빡빡하다. 그래서 처음부터 0으로 둔다. 올릴 이유가 생기면 재 보고 올린다.
ACTION_NOISE = float(os.environ.get("LAB_ACTION_NOISE", "0.0"))

# 구간이 바뀔 때 손을 다음 구간의 시작 자세까지 몇 스텝에 걸쳐 옮길지.
NUM_INTERP = int(os.environ.get("LAB_NUM_INTERP", "5"))

# 구간 경계를 뒤로 옮길 범위, 스텝. 소스마다 경계를 조금씩 흔들어 궤적을 다양하게 한다.
# 기어는 한 구간이 90스텝에서 280스텝이라 핀 꽂기와 같은 (10, 20)을 쓸 수 있다.
SUBTASK_OFFSET = (10, 20)

GRIPPER_OPEN_VAL = 0.04
GRIPPER_THRESHOLD = 0.005


# ---------------------------------------------------------------------------- FR3
# 핀 꽂기의 FR3_CFG와 같다. 받침 자리와 방향이 기어 장면과 한 자리까지 같아서 그대로 쓴다.
# 초기 관절 자세만 기어 시연이 쓴 값으로 바꿨다. 시연 20편의 첫 행에서 읽은 값이다.
FR3_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/FrankaRobotics/FrankaFR3/fr3.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=16,
            solver_velocity_iteration_count=2),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=ROBOT_POS, rot=ROBOT_ROT,
        joint_pos={
            "fr3_joint1": 0.0, "fr3_joint2": -0.7854, "fr3_joint3": 0.0,
            "fr3_joint4": -2.3562, "fr3_joint5": 0.0, "fr3_joint6": 1.5708,
            "fr3_joint7": 0.7854, "fr3_finger_joint.*": 0.04,
        },
    ),
    actuators={
        "a1": ImplicitActuatorCfg(joint_names_expr=["fr3_joint[1-4]"],
                                  stiffness=400.0, damping=80.0),
        "a2": ImplicitActuatorCfg(joint_names_expr=["fr3_joint[5-7]"],
                                  stiffness=400.0, damping=80.0),
        "h": ImplicitActuatorCfg(joint_names_expr=["fr3_finger_joint.*"],
                                 effort_limit_sim=200.0, stiffness=2e3, damping=1e2),
    },
    soft_joint_pos_limit_factor=1.0,
)


def _gear_cfg(name: str) -> RigidObjectCfg:
    """기어 하나의 설정. 이름은 반드시 large, small, medium 중 하나여야 한다.

    장면 이름이 그대로 states/rigid_object/<이름> 경로와 SceneEntityCfg(<이름>)에 쓰이고,
    변환기가 만든 파일의 물체 이름과도 맞아야 한다.

    물리 성질은 시연을 만든 쪽의 설정을 그대로 가져왔다. 위치 반복 32회는 기본값보다
    훨씬 높은데, 기어 톱니가 서로 맞물리는 접촉을 풀려면 그만큼 필요하다. 접촉 여유
    2 mm와 정지 여유 0 mm도 같은 이유다.
    """
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/" + {"large": "LargeGear", "small": "SmallGear",
                                       "medium": "MediumGear"}[name],
        init_state=RigidObjectCfg.InitialStateCfg(pos=GEAR_SPAWN[name]),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{LAB_GEAR_ASSET_ROOT}/factory_gear_{name}/factory_gear_{name}.usd",
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled=False),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=32, solver_velocity_iteration_count=1,
                max_depenetration_velocity=1.0, disable_gravity=False,
                kinematic_enabled=False),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002,
                                                             rest_offset=0.0),
        ),
    )


# ---------------------------------------------------------------------------- 관측
@configclass
class GearObservationsCfg:
    """큐브 쌓기의 관측을 통째로 대신한다. 큐브 쌓기 쪽은 지워 버린 큐브를 읽으려 한다.

    subtask_terms 그룹의 이름과 그 안의 항 이름은 바꾸면 안 된다. 환경 클래스의
    get_subtask_term_signals가 이 이름들로 찾는다.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        actions = ObsTerm(func=mdp.last_action)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        eef_pos = ObsTerm(func=stack_mdp.ee_frame_pos)
        eef_quat = ObsTerm(func=stack_mdp.ee_frame_quat)
        gripper_pos = ObsTerm(func=stack_mdp.gripper_pos)
        base_pos = ObsTerm(func=mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("base")})
        base_quat = ObsTerm(func=mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("base")})
        large_pos = ObsTerm(func=mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("large")})
        large_quat = ObsTerm(func=mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("large")})
        small_pos = ObsTerm(func=mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("small")})
        small_quat = ObsTerm(func=mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("small")})
        medium_pos = ObsTerm(func=mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("medium")})
        medium_quat = ObsTerm(func=mdp.root_quat_w,
                              params={"asset_cfg": SceneEntityCfg("medium")})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class SubtaskCfg(ObsGroup):
        # 신호 다섯 개. 구간 여섯 개의 경계다. 마지막 구간은 신호를 갖지 않는다.
        # 이름이 변환기가 만든 파일의 subtask_term_signals 이름과 같아야 한다.
        grasp_large = ObsTerm(
            func=gear_mdp.gear_grasped,
            params={"robot_cfg": SceneEntityCfg("robot"),
                    "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                    "object_cfg": SceneEntityCfg("large"),
                    "diff_threshold": gear_mdp.GRASP_DIFF_THRESHOLD,
                    "gripper_open_val": GRIPPER_OPEN_VAL,
                    "gripper_threshold": GRIPPER_THRESHOLD})
        seat_large = ObsTerm(func=gear_mdp.gear_seated,
                             params={"gear_name": "large",
                                     "base_cfg": SceneEntityCfg("base")})
        grasp_small = ObsTerm(
            func=gear_mdp.gear_grasped,
            params={"robot_cfg": SceneEntityCfg("robot"),
                    "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                    "object_cfg": SceneEntityCfg("small"),
                    "diff_threshold": gear_mdp.GRASP_DIFF_THRESHOLD,
                    "gripper_open_val": GRIPPER_OPEN_VAL,
                    "gripper_threshold": GRIPPER_THRESHOLD})
        seat_small = ObsTerm(func=gear_mdp.gear_seated,
                             params={"gear_name": "small",
                                     "base_cfg": SceneEntityCfg("base")})
        grasp_medium = ObsTerm(
            func=gear_mdp.gear_grasped,
            params={"robot_cfg": SceneEntityCfg("robot"),
                    "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                    "object_cfg": SceneEntityCfg("medium"),
                    "diff_threshold": gear_mdp.GRASP_DIFF_THRESHOLD,
                    "gripper_open_val": GRIPPER_OPEN_VAL,
                    "gripper_threshold": GRIPPER_THRESHOLD})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
    subtask_terms: SubtaskCfg = SubtaskCfg()


# ---------------------------------------------------------------------------- 바꿔 끼우기
def _apply_lab_overrides(self):
    """큐브 쌓기 장면을 기어 조립으로 바꾼다."""

    # --- 장면 ----------------------------------------------------------------------------
    # 눈에 보이는 책상. 충돌 그룹을 -1로 두어 기어가 이 메시와 충돌 걸러내기를 하게 한다.
    self.scene.table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/LabTableVisual",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0, 1.0)),
        collision_group=-1,
        spawn=sim_utils.UsdFileCfg(usd_path=LAB_GEAR_TABLE_USD),
    )
    # 보이지 않는 물리 판. 불러온 책상 메시는 쓸 만한 정적 충돌로 구워지지 않아서 실제로
    # 기어를 받치는 것은 이 판이다. 시연을 만든 쪽도 같은 이유로 같은 판을 썼고 크기도
    # (0.80, 1.80, 0.02)에 높이 0.710으로 같다.
    #
    # AssetBaseCfg가 아니라 kinematic인 RigidObjectCfg로 둔다. Isaac Lab은 RigidObject로
    # 감싼 자산에만 물리 뷰를 만들어 주는데, 뒤에 실측 물리를 붙일 때 작업면 쪽 마찰에
    # 손을 대려면 그 뷰가 있어야 한다. kinematic이라 중력도 접촉 충격량도 받지 않으므로
    # 고정된 책상으로 그대로 동작한다. 핀 꽂기에서 같은 이유로 같은 모양으로 고쳤다.
    self.scene.work_surface = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/WorkSurface",
        collision_group=-1,
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.710)),
        spawn=sim_utils.CuboidCfg(
            size=(0.80, 1.80, 0.02), visible=False,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        ),
    )
    self.scene.robot = FR3_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # 받침. 움직이지 않지만 자세를 읽어야 하므로 kinematic인 RigidObject로 둔다.
    # 정적 충돌 프림으로 두면 자세를 읽을 물리 뷰가 없어서 성공 판정을 할 수 없다.
    self.scene.base = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/GearBase",
        init_state=RigidObjectCfg.InitialStateCfg(pos=BASE_SPAWN),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{LAB_GEAR_ASSET_ROOT}/factory_gear_base/factory_gear_base_kinematic.usda",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True,
                                                         disable_gravity=True)),
    )
    for name in GEARS:
        setattr(self.scene, name, _gear_cfg(name))

    # 큐브 쌓기의 큐브 셋을 지운다. 이 환경에는 없다.
    for _c in ("cube_1", "cube_2", "cube_3"):
        if hasattr(self.scene, _c):
            setattr(self.scene, _c, None)

    # --- 동작 ----------------------------------------------------------------------------
    self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
        asset_name="robot", joint_names=["fr3_joint.*"], body_name="fr3_hand", scale=ARM_SCALE,
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True,
                                               ik_method="dls"),
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=(0.0, 0.0, 0.1034)),
    )
    self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot", joint_names=["fr3_finger_joint.*"],
        open_command_expr={"fr3_finger_joint.*": 0.04},
        close_command_expr={"fr3_finger_joint.*": 0.0},
    )
    self.gripper_joint_names = ["fr3_finger_joint.*"]
    self.gripper_open_val = GRIPPER_OPEN_VAL
    self.gripper_threshold = GRIPPER_THRESHOLD

    # --- 손끝 좌표계 프림 이름 (panda_* -> fr3_*) ------------------------------------------
    if hasattr(self.scene, "ee_frame") and self.scene.ee_frame is not None:
        self.scene.ee_frame.prim_path = "{ENV_REGEX_NS}/Robot/fr3_link0"
        for fr in self.scene.ee_frame.target_frames:
            fr.prim_path = (fr.prim_path.replace("panda_hand", "fr3_hand")
                            .replace("panda_rightfinger", "fr3_rightfinger")
                            .replace("panda_leftfinger", "fr3_leftfinger"))

    # --- 이벤트 --------------------------------------------------------------------------
    for _e in ("init_franka_arm_pose", "randomize_franka_joint_state",
               "randomize_cube_positions"):
        if hasattr(self.events, _e):
            setattr(self.events, _e, None)

    # 리셋 때 받침과 기어 세 개의 자리를 새로 뽑는다. 생성에서는 이것이 있어야 새 배치가
    # 나온다. 어노테이션 재생에서는 반드시 꺼야 한다. 기록된 초기 배치를 덮으면 재생하는
    # 로봇이 빈 자리를 집는다. 핀 꽂기에서 실제로 그렇게 돼 12편 전부 실패로 끝났다.
    if os.environ.get("LAB_DISABLE_RESET_RANDOMIZATION", "") == "1":
        print("[gear_mimic_cfg] 리셋 무작위화를 끈다 (재생용)")
        self.events.randomize_gear_layout = None
    else:
        self.events.randomize_gear_layout = EventTerm(
            func=gear_mdp.randomize_gear_layout, mode="reset", params={})

    # --- 관측 ----------------------------------------------------------------------------
    self.observations = GearObservationsCfg()

    # --- 종료 조건 -----------------------------------------------------------------------
    for _t in ("cube_1_dropping", "cube_2_dropping", "cube_3_dropping"):
        if hasattr(self.terminations, _t):
            setattr(self.terminations, _t, None)
    self.terminations.success = DoneTerm(
        func=gear_mdp.all_gears_assembled, params={"base_cfg": SceneEntityCfg("base")})

    # --- 생성 설정 -----------------------------------------------------------------------
    self.datagen_config.generation_keep_failed = os.environ.get("LAB_KEEP_FAILED", "1") == "1"
    self.datagen_config.generation_guarantee = True
    if hasattr(self.datagen_config, "name"):
        self.datagen_config.name = "multigear_lab_fr3_d0"

    _apply_subtask_schema(self)
    _apply_source_routing(self)


def _apply_subtask_schema(self):
    """구간 여섯 개를 짜 넣는다.

    공식 큐브 쌓기 설정이 들고 있는 SubTaskConfig 인스턴스를 복사해 쓴다. 새로 만들지
    않는 이유는 그 클래스의 필드 목록을 여기서 확인할 수 없기 때문이다. 선택 전략이나
    최근접 이웃 설정 같은 필드가 이미 올바른 값으로 채워져 있다.

    구간마다 기준 물체를 정한다. MimicGen은 그 물체의 자세가 시연과 지금 장면에서 얼마나
    다른지를 보고 손 궤적을 옮긴다. 집는 구간은 집을 기어가 기준이고, 꽂는 구간은 받침이
    기준이다. 꽂는 구간에서 기어를 기준으로 삼으면 안 된다. 그 구간에서 기어는 손에 들려
    손과 같이 움직이므로 기준이 되지 못한다.
    """
    eef = list(self.subtask_configs.keys())[0]
    base_list = self.subtask_configs[eef]
    proto_mid = base_list[0]        # 신호를 가진 중간 구간의 본보기
    proto_last = base_list[-1]      # 신호가 없는 마지막 구간의 본보기

    order = []
    for i, gear in enumerate(GEARS):
        grasp = copy.deepcopy(proto_mid)
        grasp.object_ref = gear
        grasp.subtask_term_signal = f"grasp_{gear}"
        grasp.subtask_term_offset_range = SUBTASK_OFFSET
        grasp.action_noise = ACTION_NOISE
        grasp.num_interpolation_steps = NUM_INTERP
        order.append(grasp)

        last = (i == len(GEARS) - 1)
        seat = copy.deepcopy(proto_last if last else proto_mid)
        seat.object_ref = "base"
        seat.subtask_term_signal = None if last else f"seat_{gear}"
        # 꽂는 구간의 경계는 흔들지 않는다. 축 오차 1 mm를 요구하는 구간이라 경계를 뒤로
        # 밀면 이미 앉은 기어를 다시 건드리는 궤적이 섞인다. 핀 꽂기에서도 마지막 넣기
        # 구간의 범위를 (0, 0)으로 두어야 구간 겹침 검사를 통과했다.
        seat.subtask_term_offset_range = (0, 0)
        seat.action_noise = ACTION_NOISE
        seat.num_interpolation_steps = NUM_INTERP
        order.append(seat)

    self.subtask_configs[eef] = order
    print(f"[gear_mimic_cfg] 구간 {len(order)}개: "
          + " -> ".join((s.subtask_term_signal or "끝") for s in order))


def _apply_source_routing(self):
    """하위 작업마다 소스 시연을 다시 고르는 휴리스틱을 붙인다.

    설정은 태스크 프로필의 generate.heuristic_selection 절이 정하고 환경변수로 덮을 수
    있다. 정순과 역순 설정 클래스가 둘 다 이 함수를 불러야 한다. 한쪽에만 적어 두면
    실제로 쓰는 태스크에서 휴리스틱이 조용히 꺼진다. 큐브 쌓기에서 실제로 그런 일이 있었고
    로그에 아무 줄도 남지 않아 생성 기록을 열어 보고서야 알았다.
    """
    source_routing.apply(self)


@configclass
class LabFR3MultiGearMimicEnvCfg(FrankaCubeStackIKRelMimicEnvCfg):
    """기어 조립 MimicGen 환경 설정."""

    def __post_init__(self):
        super().__post_init__()
        _apply_lab_overrides(self)
