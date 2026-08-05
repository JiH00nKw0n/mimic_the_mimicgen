# Copyright (c) 2024-2026, The UW Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Measured-v2 RGB collection configs for the lab FR3 three-cube task.

The physics, OSC, action-scale, and reset contract matches the fixed-p=1 state
teacher used after model_7800. Camera extrinsics and intrinsics are loaded from
the lab calibration instead of being duplicated here. A separate audit task
enables raw instance-ID segmentation and fails closed on any foreign-environment
pixel.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import yaml

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

from ... import mdp as task_mdp
from .actions import Fr3CubeP1RelativeOSCAction
from .rl_state_cfg import (
    FullStackFinetuneEventCfg,
    Fr3CubeRelCartesianOSCFullStackFinetuneCfg,
    NoCurriculumsCfg,
    ObservationsCfg,
    RlStateSceneCfg,
)


OVERLAY_CAMERA_ROLES = ("third_person_0", "third_person_1", "third_person_2", "wrist")
# third_person_2 remains in the calibration handoff for provenance, but is not
# rendered or stored for RGB distillation.
CAMERA_ROLES = ("third_person_0", "third_person_1", "wrist")
VISUAL_PROFILE = os.environ.get("UWLAB_FR3_VISUAL_PROFILE", "lab_variation")
if VISUAL_PROFILE not in {"nominal_lab", "lab_variation", "stress_tail"}:
    raise ValueError(
        "UWLAB_FR3_VISUAL_PROFILE must be nominal_lab, lab_variation, or stress_tail; "
        f"got {VISUAL_PROFILE!r}"
    )
# Keep the source resolution for later crop/augmentation/downsampling while
# preserving the calibrated 16:9 D435/D405 aspect ratio.  Both values are
# overridable for storage/throughput ablations without editing the task.
RGB_WIDTH = int(os.environ.get("UWLAB_FR3_RGB_WIDTH", "640"))
RGB_HEIGHT = int(os.environ.get("UWLAB_FR3_RGB_HEIGHT", "360"))
if RGB_WIDTH * 9 != RGB_HEIGHT * 16:
    raise ValueError(f"FR3 RGB capture must remain 16:9, got {RGB_WIDTH}x{RGB_HEIGHT}")
ENV_SPACING_M = 4.0

_REPO_ROOT = Path(__file__).resolve().parents[8]
CAMERA_CALIBRATION_PATH = Path(
    os.environ.get(
        "UWLAB_FR3_CAMERA_CALIBRATION",
        str(
            _REPO_ROOT
            / "artifacts/fr3_camera_calibration_measured_v2"
            / "camera_nominal_measured_ranges.yaml"
        ),
    )
)
_ORIGINAL_RGB_RESOURCES = (
    Path(__file__).resolve().parents[1] / "ur5e_robotiq_2f85" / "resources"
)
TEXTURE_CONFIG_PATH = _ORIGINAL_RGB_RESOURCES / "texture_paths.yaml"
HDRI_CONFIG_PATH = _ORIGINAL_RGB_RESOURCES / "hdri_paths.yaml"
FR3_RGB_VISUAL_USD_PATH = Path(
    os.environ.get(
        "UWLAB_FR3_RGB_VISUAL_USD",
        str(
            _REPO_ROOT
            / "source/uwlab_assets/uwlab_assets/robots/fr3/asset/facelift/fr3_facelift_visual.usda"
        ),
    )
)
FR3_RGB_TABLE_USD_PATH = Path(
    os.environ.get(
        "UWLAB_FR3_RGB_TABLE_USD",
        "/home/ubuntu/jake/aidas/3cube_stack/table_scene_rgb_lab.usda",
    )
)
LAB_GRAY_CARPET_TEXTURE_PATH = Path(
    os.environ.get(
        "UWLAB_FR3_LAB_CARPET_TEXTURE",
        "/home/ubuntu/.cache/uwlab/assets/Assets/NVIDIA/Textures/Base/Carpet/Carpet_Gray/Carpet_Gray_BaseColor.png",
    )
)
LAB_INDOOR_HDRI_BY_PROFILE = {
    "nominal_lab": Path(
        "/home/ubuntu/.cache/uwlab/assets/Assets/PolyHaven/HDRIs/indoor/studio_small_04_1k.hdr"
    ),
    "lab_variation": Path(
        "/home/ubuntu/.cache/uwlab/assets/Assets/PolyHaven/HDRIs/indoor/studio_small_07_1k.hdr"
    ),
    "stress_tail": Path(
        "/home/ubuntu/.cache/uwlab/assets/Assets/PolyHaven/HDRIs/indoor/empty_workshop_1k.hdr"
    ),
}
if not FR3_RGB_VISUAL_USD_PATH.is_file():
    raise FileNotFoundError(
        f"FR3 RGB facelift visual layer is missing: {FR3_RGB_VISUAL_USD_PATH}. "
        "Run scripts_v2/tools/build_fr3_facelift_visual_asset.py first."
    )
if not FR3_RGB_TABLE_USD_PATH.is_file():
    raise FileNotFoundError(f"FR3 RGB lab table layer is missing: {FR3_RGB_TABLE_USD_PATH}")
if not LAB_GRAY_CARPET_TEXTURE_PATH.is_file():
    raise FileNotFoundError(f"FR3 RGB lab carpet texture is missing: {LAB_GRAY_CARPET_TEXTURE_PATH}")
if not LAB_INDOOR_HDRI_BY_PROFILE[VISUAL_PROFILE].is_file():
    raise FileNotFoundError(
        f"FR3 RGB indoor HDRI is missing: {LAB_INDOOR_HDRI_BY_PROFILE[VISUAL_PROFILE]}"
    )
def _matrix_to_quaternion_xyzw(rotation: list[list[float]]) -> tuple[float, float, float, float]:
    m = rotation
    trace = m[0][0] + m[1][1] + m[2][2]
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw, qx, qy, qz = 0.25 * s, (m[2][1] - m[1][2]) / s, (m[0][2] - m[2][0]) / s, (m[1][0] - m[0][1]) / s
    elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2.0
        qw, qx, qy, qz = (m[2][1] - m[1][2]) / s, 0.25 * s, (m[0][1] + m[1][0]) / s, (m[0][2] + m[2][0]) / s
    elif m[1][1] > m[2][2]:
        s = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2.0
        qw, qx, qy, qz = (m[0][2] - m[2][0]) / s, (m[0][1] + m[1][0]) / s, 0.25 * s, (m[1][2] + m[2][1]) / s
    else:
        s = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2.0
        qw, qx, qy, qz = (m[1][0] - m[0][1]) / s, (m[0][2] + m[2][0]) / s, (m[1][2] + m[2][1]) / s, 0.25 * s
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    return qx / norm, qy / norm, qz / norm, qw / norm


def _isaac_pinhole(intrinsics: dict) -> dict:
    focal = 20.0
    width, height = float(intrinsics["width"]), float(intrinsics["height"])
    fx, fy = float(intrinsics["fx"]), float(intrinsics["fy"])
    ppx, ppy = float(intrinsics["ppx"]), float(intrinsics["ppy"])
    return {
        "focal_length_mm": focal,
        "horizontal_aperture_mm": focal * width / fx,
        "vertical_aperture_mm": focal * height / fy,
        "horizontal_aperture_offset_mm": -focal * (ppx - width / 2.0) / fx,
        "vertical_aperture_offset_mm": focal * (ppy - height / 2.0) / fy,
        "clipping_range_m": (0.01, 10.0),
    }


def _normalized_camera(role: str, source: dict, *, fixed: bool) -> dict:
    optical = source["parent_T_camera_optical"]
    # OpenCV optical (X right, Y down, Z forward) -> USD/OpenGL camera.
    usd_rotation = [[float(optical[row][col]) * (1.0 if col == 0 else -1.0) for col in range(3)] for row in range(3)]
    return {
        "parent_semantic": "robot_base" if fixed else "hand_tcp",
        "isaac_prim_name": role if fixed else "wrist_d405_color",
        "parent_T_camera_usd": {
            "translation_m": tuple(float(optical[row][3]) for row in range(3)),
            "quaternion_xyzw": _matrix_to_quaternion_xyzw(usd_rotation),
        },
        "intrinsics": source["intrinsics"],
        "isaac_camera_model": _isaac_pinhole(source["intrinsics"]) if fixed else source["isaac_pinhole"],
    }


def _load_camera_calibration() -> tuple[dict, dict]:
    payload = yaml.safe_load(CAMERA_CALIBRATION_PATH.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "stage2.fr3_camera_nominal_measured_ranges.v2":
        raise ValueError(f"Unsupported FR3 camera calibration: {CAMERA_CALIBRATION_PATH}")
    nominal = payload["nominal"]
    fixed = nominal["fixed_d435"]["by_role"]
    cameras = {
        role: _normalized_camera(role, fixed[role], fixed=True)
        for role in OVERLAY_CAMERA_ROLES[:3]
    }
    cameras["wrist"] = _normalized_camera("wrist", nominal["wrist_d405"], fixed=False)
    measured = payload["measured_ranges"]["camera_local"]
    ranges = dict(measured["fixed_d435"]["by_role"])
    ranges["wrist"] = measured["wrist_d405"]
    return cameras, ranges


_CAMERAS, _CAMERA_RANGES = _load_camera_calibration()


def _camera_parent(camera: dict) -> str:
    semantic = camera["parent_semantic"]
    if semantic == "robot_base":
        return "{ENV_REGEX_NS}/Robot/fr3_link0"
    if semantic == "hand_tcp":
        return "{ENV_REGEX_NS}/Robot/fr3_hand_tcp"
    raise ValueError(f"Unsupported camera parent semantic: {semantic}")


def _camera_cfg(role: str, *, audit: bool = False) -> TiledCameraCfg:
    camera = _CAMERAS[role]
    model = camera["isaac_camera_model"]
    transform = camera["parent_T_camera_usd"]
    qx, qy, qz, qw = transform["quaternion_xyzw"]
    cfg = TiledCameraCfg(
        prim_path=f"{_camera_parent(camera)}/{camera['isaac_prim_name']}",
        update_period=0.0,
        height=RGB_HEIGHT,
        width=RGB_WIDTH,
        data_types=["rgb", "instance_id_segmentation_fast"] if audit else ["rgb"],
        offset=TiledCameraCfg.OffsetCfg(
            pos=tuple(float(value) for value in transform["translation_m"]),
            rot=(float(qw), float(qx), float(qy), float(qz)),
            convention="opengl",
        ),
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=float(model["focal_length_mm"]),
            horizontal_aperture=float(model["horizontal_aperture_mm"]),
            vertical_aperture=float(model["vertical_aperture_mm"]),
            horizontal_aperture_offset=float(model["horizontal_aperture_offset_mm"]),
            vertical_aperture_offset=float(model["vertical_aperture_offset_mm"]),
            clipping_range=tuple(float(value) for value in model["clipping_range_m"]),
        ),
    )
    if audit:
        cfg.colorize_instance_id_segmentation = False
    return cfg


def _curtain(name: str, pos: tuple[float, float, float], size: tuple[float, float, float]) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos),
        spawn=sim_utils.CuboidCfg(
            size=size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 0.0)),
        ),
    )


@configclass
class Fr3CubeRGBSceneCfg(RlStateSceneCfg):
    """Per-env cameras inside a four-wall visual enclosure."""

    # Bounds include all fixed cameras after applying the robot base's 180-deg
    # Z rotation, plus their specified pose perturbations.  Four walls are
    # intentional: the FR3 camera layout surrounds the task on both X sides.
    curtain_left = _curtain("CurtainLeft", (0.25, -1.10, 1.30), (2.60, 0.02, 2.60))
    curtain_right = _curtain("CurtainRight", (0.25, 1.10, 1.30), (2.60, 0.02, 2.60))
    curtain_back = _curtain("CurtainBack", (-1.05, 0.0, 1.30), (0.02, 2.20, 2.60))
    curtain_front = _curtain("CurtainFront", (1.55, 0.0, 1.30), (0.02, 2.20, 2.60))

    # Replace Isaac Sim's black/white grid floor with the lab's gray low-pile
    # carpet.  The cuboid top remains at z=0, preserving the original ground
    # collision plane while removing the synthetic grid from RGB.
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.01)),
        spawn=sim_utils.CuboidCfg(
            size=(100.0, 100.0, 0.02),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.42, 0.43, 0.42),
                roughness=0.95,
                metallic=0.0,
            ),
        ),
    )

    third_person_0 = _camera_cfg("third_person_0")
    third_person_1 = _camera_cfg("third_person_1")
    wrist = _camera_cfg("wrist")


@configclass
class Fr3CubeRGBAuditSceneCfg(Fr3CubeRGBSceneCfg):
    third_person_0 = _camera_cfg("third_person_0", audit=True)
    third_person_1 = _camera_cfg("third_person_1", audit=True)
    wrist = _camera_cfg("wrist", audit=True)


def _camera_pose_event(role: str) -> EventTerm:
    camera = _CAMERAS[role]
    transform = camera["parent_T_camera_usd"]
    qx, qy, qz, qw = transform["quaternion_xyzw"]
    fixed = camera["parent_semantic"] == "robot_base"
    measured = _CAMERA_RANGES[role]
    parent = "/World/envs/env_{}/Robot/fr3_link0" if fixed else "/World/envs/env_{}/Robot/fr3_hand_tcp"
    return EventTerm(
        func=task_mdp.randomize_tiled_cameras,
        mode="reset",
        params={
            "camera_path_template": f"{parent}/{camera['isaac_prim_name']}",
            "base_position": tuple(float(value) for value in transform["translation_m"]),
            "base_rotation": (float(qw), float(qx), float(qy), float(qz)),
            "translation_ball_radius": float(measured["translation_uniform_ball_radius_m"]),
            "rotation_vector_ball_radius_deg": float(measured["rotation_uniform_vector_ball_radius_deg"]),
        },
    )


def _camera_focal_event(role: str) -> EventTerm:
    camera = _CAMERAS[role]
    nominal = float(camera["isaac_camera_model"]["focal_length_mm"])
    scale_low, scale_high = _CAMERA_RANGES[role]["focal_length_scale_uniform"]
    parent = (
        "/World/envs/env_{}/Robot/fr3_link0"
        if camera["parent_semantic"] == "robot_base"
        else "/World/envs/env_{}/Robot/fr3_hand_tcp"
    )
    return EventTerm(
        func=task_mdp.randomize_camera_focal_length,
        mode="reset",
        params={
            "camera_path_template": f"{parent}/{camera['isaac_prim_name']}",
            "focal_length_range": (nominal * float(scale_low), nominal * float(scale_high)),
        },
    )


def _appearance_event(asset_name: str, event_name: str, mesh_names: list[str] | None = None) -> EventTerm:
    return EventTerm(
        func=task_mdp.randomize_visual_appearance_multiple_meshes,
        # Appearance is sampled once at episode reset. Temporal history must
        # never contain an unexplained material jump.
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(asset_name),
            "event_name": event_name,
            "mesh_names": [] if mesh_names is None else mesh_names,
            "texture_prob": 0.5,
            "texture_config_path": str(TEXTURE_CONFIG_PATH),
            "diffuse_tint_range": ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            "colors": {"r": (0.0, 1.0), "g": (0.0, 1.0), "b": (0.0, 1.0)},
            "texture_scale_range": (0.7, 5.0),
            "roughness_range": (0.0, 1.0),
            "metallic_range": (0.0, 1.0),
            "specular_range": (0.0, 1.0),
        },
    )


def _solid_material(
    term: EventTerm,
    *,
    colors: list[tuple[float, float, float]],
    roughness: tuple[float, float],
    metallic: tuple[float, float],
    specular: tuple[float, float],
) -> None:
    """Constrain an appearance term to identity-preserving lab materials."""

    term.params.update(
        {
            "texture_prob": 0.0,
            "texture_config_path": None,
            "texture_paths": None,
            "colors": colors,
            "diffuse_tint_range": None,
            "texture_scale_range": (0.8, 1.5),
            "roughness_range": roughness,
            "metallic_range": metallic,
            "specular_range": specular,
        }
    )


def _profile_colors(nominal: tuple[float, float, float]) -> list[tuple[float, float, float]]:
    half_range = {
        "nominal_lab": 0.035,
        "lab_variation": 0.10,
        "stress_tail": 0.18,
    }[VISUAL_PROFILE]
    scales = (1.0 - half_range, 1.0, 1.0 + half_range)
    return [tuple(min(1.0, max(0.0, value * scale)) for value in nominal) for scale in scales]


def _apply_visual_profile(events: "Fr3CubeRGBEventCfg", scene: "Fr3CubeRGBSceneCfg") -> None:
    """Install one process-level visual profile with per-episode materials.

    The 80K collector runs separate profile-labelled shards. This avoids a
    global DomeLight change in one tiled environment altering other episodes.
    """

    common = {
        "roughness": (0.25, 0.75),
        "metallic": (0.0, 0.03),
        "specular": (0.18, 0.50),
    }
    _solid_material(events.randomize_gripper, colors=_profile_colors((0.93, 0.93, 0.91)), **common)
    _solid_material(events.randomize_cube_1_appearance, colors=_profile_colors((0.72, 0.06, 0.04)), **common)
    _solid_material(events.randomize_cube_2_appearance, colors=_profile_colors((0.03, 0.17, 0.55)), **common)
    _solid_material(events.randomize_cube_3_appearance, colors=_profile_colors((0.04, 0.04, 0.04)), **common)
    _solid_material(
        events.randomize_table_appearance,
        colors=_profile_colors((0.84, 0.83, 0.77)),
        roughness=(0.45, 0.88),
        metallic=(0.0, 0.05),
        specular=(0.18, 0.50),
    )
    for name in (
        "randomize_curtain_left_appearance",
        "randomize_curtain_right_appearance",
        "randomize_curtain_back_appearance",
    ):
        _solid_material(
            getattr(events, name),
            colors=_profile_colors((0.48, 0.49, 0.48)),
            roughness=(0.72, 1.0),
            metallic=(0.0, 0.02),
            specular=(0.08, 0.25),
        )
    _solid_material(
        events.randomize_curtain_front_appearance,
        colors=_profile_colors((0.88, 0.87, 0.83)),
        roughness=(0.65, 0.96),
        metallic=(0.0, 0.01),
        specular=(0.08, 0.30),
    )
    scene.sky_light.spawn.texture_file = str(LAB_INDOOR_HDRI_BY_PROFILE[VISUAL_PROFILE])
    scene.sky_light.spawn.intensity = {
        "nominal_lab": 1100.0,
        "lab_variation": 1350.0,
        "stress_tail": 1650.0,
    }[VISUAL_PROFILE]


@configclass
class Fr3CubeRGBEventCfg(FullStackFinetuneEventCfg):
    """Fixed-p=1 teacher physics/OSC plus visual randomization.

    The state teacher's p=1 training config uses these same two manager terms
    with both curriculum progresses pinned to 1.0.  There is deliberately no
    RGB-side curriculum: every reset samples the terminal SysID and OSC ranges.
    """

    randomize_arm_sysid = EventTerm(
        func=task_mdp.randomize_arm_from_sysid,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "joint_names": [f"fr3_joint{index}" for index in range(1, 8)],
            "actuator_name": "arm",
            "scale_range": (0.8, 1.2),
            "delay_range": (0, 3),
            "initial_scale_progress": 1.0,
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
            "initial_scale_progress": 1.0,
        },
    )

    randomize_third_person_0 = _camera_pose_event("third_person_0")
    randomize_third_person_1 = _camera_pose_event("third_person_1")
    randomize_wrist = _camera_pose_event("wrist")
    randomize_wrist_focal = _camera_focal_event("wrist")

    # Randomize only the replacement hand/finger facade.  The arm remains at
    # its official FR3v2.1 appearance, matching the lab robot and excluding
    # full-arm appearance DR.
    randomize_gripper = _appearance_event(
        "robot",
        "fr3_gripper_appearance",
        [
            "fr3_hand/facelift_visual",
            "fr3_leftfinger/facelift_visual",
            "fr3_rightfinger/facelift_visual",
        ],
    )
    randomize_cube_1_appearance = _appearance_event("cube_1", "cube_1_appearance")
    randomize_cube_2_appearance = _appearance_event("cube_2", "cube_2_appearance")
    randomize_cube_3_appearance = _appearance_event("cube_3", "cube_3_appearance")
    randomize_table_appearance = _appearance_event("table", "table_appearance")
    randomize_curtain_left_appearance = _appearance_event("curtain_left", "curtain_left_appearance")
    randomize_curtain_right_appearance = _appearance_event("curtain_right", "curtain_right_appearance")
    randomize_curtain_back_appearance = _appearance_event("curtain_back", "curtain_back_appearance")
    randomize_curtain_front_appearance = _appearance_event("curtain_front", "curtain_front_appearance")
    randomize_floor_appearance = EventTerm(
        func=task_mdp.randomize_visual_appearance_multiple_meshes,
        # The floor is a global prim, so it cannot be randomized independently
        # per tiled environment. Initialize it once for the whole process.
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("ground"),
            "event_name": "lab_gray_carpet_appearance",
            "mesh_names": [],
            "texture_paths": [str(LAB_GRAY_CARPET_TEXTURE_PATH)],
            "texture_config_path": None,
            "texture_prob": 1.0,
            "diffuse_tint_range": ((0.72, 0.72, 0.72), (0.92, 0.92, 0.92)),
            "colors": [(0.42, 0.43, 0.42)],
            "texture_scale_range": (1.5, 2.5),
            "roughness_range": (0.88, 1.0),
            "metallic_range": (0.0, 0.01),
            "specular_range": (0.05, 0.18),
        },
    )

    # A DomeLight is global across tiled environments. Per-reset HDRI changes
    # would therefore corrupt unrelated in-flight episodes. Each collection
    # process uses one profile-specific neutral indoor light instead.
    randomize_sky_light = None


def _rgb_term(role: str, *, processed: bool) -> ObsTerm:
    return ObsTerm(
        func=task_mdp.process_image,
        params={
            "sensor_cfg": SceneEntityCfg(role),
            "data_type": "rgb",
            "process_image": processed,
            # Never resize in the simulator/collector. Distillation chooses its
            # own crop and input resolution after the source dataset is frozen.
            "output_size": (RGB_HEIGHT, RGB_WIDTH),
        },
    )


@configclass
class Fr3CubeRGBObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
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
        third_person_0_rgb = _rgb_term("third_person_0", processed=True)
        third_person_1_rgb = _rgb_term("third_person_1", processed=True)
        wrist_rgb = _rgb_term("wrist", processed=True)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = False

    @configclass
    class DataCollectionCfg(ObsGroup):
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
        third_person_0_rgb = _rgb_term("third_person_0", processed=False)
        third_person_1_rgb = _rgb_term("third_person_1", processed=False)
        wrist_rgb = _rgb_term("wrist", processed=False)
        cube_1_pose = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame,
            params={"target_asset_cfg": SceneEntityCfg("cube_1"), "root_asset_cfg": SceneEntityCfg("robot")},
        )
        cube_2_pose = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame,
            params={"target_asset_cfg": SceneEntityCfg("cube_2"), "root_asset_cfg": SceneEntityCfg("robot")},
        )
        cube_3_pose = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame,
            params={"target_asset_cfg": SceneEntityCfg("cube_3"), "root_asset_cfg": SceneEntityCfg("robot")},
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
    data_collection: DataCollectionCfg = DataCollectionCfg()
    # Keep the state teacher group in the config from construction time so the
    # full-stack stage-aware rewrite is applied before the environment exists.
    # The RGB policy group remains free of privileged cube poses.
    expert_obs: ObservationsCfg.PolicyCfg = ObservationsCfg.PolicyCfg()
    critic: ObservationsCfg.CriticCfg = ObservationsCfg.CriticCfg()


@configclass
class Fr3CubeRGBTerminationsCfg:
    time_out = DoneTerm(func=task_mdp.time_out, time_out=True)
    abnormal_robot = DoneTerm(func=task_mdp.abnormal_robot_state)
    corrupted_camera = DoneTerm(
        func=task_mdp.corrupted_camera_detected,
        params={"camera_names": list(CAMERA_ROLES), "std_threshold": 10.0},
    )
    early_success = DoneTerm(
        func=task_mdp.early_success_termination,
        params={"num_consecutive_successes": 5, "min_episode_length": 10},
    )
    success = DoneTerm(
        func=task_mdp.consecutive_success_state_with_min_length,
        params={"num_consecutive_successes": 5, "min_episode_length": 10},
    )


@configclass
class Fr3CubeRGBAuditTerminationsCfg(Fr3CubeRGBTerminationsCfg):
    foreign_environment_pixels = DoneTerm(
        func=task_mdp.foreign_environment_pixels_detected,
        params={
            "camera_names": list(CAMERA_ROLES),
            "max_foreign_pixels": 0,
            "fail_closed": True,
        },
    )


@configclass
class Fr3CubeRGBRelCartesianOSCBaseCfg(Fr3CubeRelCartesianOSCFullStackFinetuneCfg):
    scene: Fr3CubeRGBSceneCfg = Fr3CubeRGBSceneCfg(
        num_envs=32,
        env_spacing=ENV_SPACING_M,
        replicate_physics=False,
    )
    events: Fr3CubeRGBEventCfg = Fr3CubeRGBEventCfg()
    observations: Fr3CubeRGBObservationsCfg = Fr3CubeRGBObservationsCfg()
    terminations: Fr3CubeRGBTerminationsCfg = Fr3CubeRGBTerminationsCfg()
    actions: Fr3CubeP1RelativeOSCAction = Fr3CubeP1RelativeOSCAction()
    curriculum: NoCurriculumsCfg = NoCurriculumsCfg()

    def _apply_full_stack_stage_aware_terms(self):
        # The state-task helper expects both ``policy`` and ``critic`` groups.
        # Route its policy-side rewrite to the teacher's state observation group
        # while leaving the student's RGB-only policy group unchanged.
        rgb_policy = self.observations.policy
        self.observations.policy = self.observations.expert_obs
        try:
            super()._apply_full_stack_stage_aware_terms()
        finally:
            self.observations.policy = rgb_policy

    def __post_init__(self):
        super().__post_init__()
        _apply_visual_profile(self.events, self.scene)
        # Fr3CubeRelCartesianOSCFinetuneCfg has already installed the explicit
        # SysID/delay articulation here.  Swap only its composed USD path so
        # every actuator, joint setting, and initial state remains untouched.
        self.scene.robot.spawn.usd_path = str(FR3_RGB_VISUAL_USD_PATH)
        # RGB-only table wrapper hides the unused empty pedestal.  Collision
        # and forbidden-volume proxies remain those of the state task.
        self.scene.table.spawn.usd_path = str(FR3_RGB_TABLE_USD_PATH)
        self.episode_length_s = 32.0
        self.sim.render.enable_dlssg = False
        self.sim.render.enable_ambient_occlusion = True
        self.sim.render.enable_reflections = True
        self.sim.render.enable_dl_denoiser = True
        self.sim.render.antialiasing_mode = "DLAA"
        self.sim.render_interval = self.decimation
        self.num_rerenders_on_reset = 1


@configclass
class Fr3CubeRGBDataCollectionCfg(Fr3CubeRGBRelCartesianOSCBaseCfg):
    """Production RGB demonstration collection (RGB annotators only)."""


@configclass
class Fr3CubeRGBEvalCfg(Fr3CubeRGBRelCartesianOSCBaseCfg):
    """RGB policy evaluation under the same visual randomization contract."""


@configclass
class Fr3CubeRGBIsolationAuditCfg(Fr3CubeRGBRelCartesianOSCBaseCfg):
    """Pre-collection audit: segmentation enabled and foreign pixels are fatal."""

    scene: Fr3CubeRGBAuditSceneCfg = Fr3CubeRGBAuditSceneCfg(
        num_envs=32,
        env_spacing=ENV_SPACING_M,
        replicate_physics=False,
    )
    terminations: Fr3CubeRGBAuditTerminationsCfg = Fr3CubeRGBAuditTerminationsCfg()
