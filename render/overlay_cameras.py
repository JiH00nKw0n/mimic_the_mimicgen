"""fr3 camera overlay (stage2.fr3_camera_overlay.v1) -> Isaac Lab CameraCfg.

The overlay bundle (from 주상's fr_sys calibration of the REAL lab FR3) gives,
for each of the 4 real cameras (3x D435 third-person fixed + 1x D405 wrist):

  - ``parent_T_camera_usd``: local pose in the USD camera convention (X right,
    Y up, view along -Z) relative to a semantic parent frame:
        third_person_*  -> fr3v2_link0    (robot base link)
        wrist           -> fr3v2_hand_tcp (fingertip-center TCP, rigid on hand)
  - ``isaac_camera_model``: USD pinhole params (focal length, apertures,
    aperture offsets, clipping) precomputed from the RealSense color
    intrinsics. We author these verbatim, same as the bundle's own
    tools/apply_overlay_to_isaac.py.

Our sim robot is the Isaac ``FrankaRobotics/FrankaFR3/fr3.usd`` whose base link
prim is ``fr3_link0`` and hand body is ``fr3_hand``. ``fr3v2_hand_tcp`` is not
a prim in that USD, so the wrist camera is attached under ``fr3_hand`` with

    fr3_hand_T_cam = fr3_hand_T_hand_tcp @ hand_tcp_T_cam_usd

where ``fr3_hand_T_hand_tcp`` comes from the binding YAML measured in-sim by
``probe_tcp_binding.py`` against the overlay's reference Franka-Home FK
(the bundle's asset-adapter policy: never assume frame identity by name).

Pure-numpy parsing/math here is importable without Isaac; only
``build_camera_cfgs`` imports isaaclab (call it after AppLauncher).
"""

from __future__ import annotations

import math

import numpy as np
import yaml

SCHEMA = "stage2.fr3_camera_overlay.v1"
FIXED_ROLES = ("third_person_0", "third_person_1", "third_person_2")
ALL_ROLES = FIXED_ROLES + ("wrist",)
# intrinsics in the overlay are for this capture resolution
NATIVE_W, NATIVE_H = 1280, 720


# ---------------------------------------------------------------- SE(3) utils
def T_from(entry: dict) -> np.ndarray:
    """4x4 from an overlay transform entry (rows of the matrix, p' = T p)."""
    T = np.asarray(entry["matrix"], dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"bad transform shape {T.shape}")
    # cross-check against the redundant translation field
    if not np.allclose(T[:3, 3], np.asarray(entry["translation_m"]), atol=1e-9):
        raise ValueError("matrix/translation_m mismatch — row/column convention broke")
    return T


def T_inv(T: np.ndarray) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def T_trans(x: float, y: float, z: float) -> np.ndarray:
    T = np.eye(4)
    T[:3, 3] = (x, y, z)
    return T


def R_from_quat_wxyz(q) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def quat_wxyz_from_R(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> unit quaternion (w,x,y,z), Shepperd's method."""
    tr = np.trace(R)
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.asarray(q)
    return q / np.linalg.norm(q)


def rot_angle_deg(R: np.ndarray) -> float:
    return math.degrees(math.acos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)))


# ------------------------------------------------------------------- overlay
def load_overlay(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        ov = yaml.safe_load(f)
    if ov.get("schema_version") != SCHEMA:
        raise ValueError(f"unsupported overlay schema: {ov.get('schema_version')}")
    # self-check every camera: our matrix->quat must reproduce the overlay's own
    # quaternion_xyzw (catches any row/column-major or ordering confusion).
    for role in ALL_ROLES:
        cam = ov["runtime"]["cameras"][role]
        for key in ("parent_T_camera_usd", "parent_T_camera_optical"):
            T = T_from(cam[key])
            q_ours = quat_wxyz_from_R(T[:3, :3])
            qx, qy, qz, qw = cam[key]["quaternion_xyzw"]
            q_theirs = np.asarray([qw, qx, qy, qz])
            if min(np.linalg.norm(q_ours - q_theirs), np.linalg.norm(q_ours + q_theirs)) > 1e-6:
                raise ValueError(f"{role}.{key}: quaternion self-check failed")
    return ov


def reference_home(ov: dict) -> tuple[dict[str, float], np.ndarray]:
    """(arm joint name->rad map remapped fr3v2_->fr3_, base_T_hand_tcp at that pose)."""
    ref = ov["runtime"]["reference_robot_pose"]
    joints = {
        name.replace("fr3v2_", "fr3_"): float(pos)
        for name, pos in zip(ref["arm_joint_names"], ref["arm_joint_pos_rad"])
    }
    return joints, T_from(ref["base_T_hand_tcp"])


def scaled_K(cam: dict, width: int, height: int) -> np.ndarray:
    """Color intrinsics scaled from the native 1280x720 to the render size."""
    intr = cam["intrinsics"]
    sx, sy = width / intr["width"], height / intr["height"]
    return np.array([
        [intr["fx"] * sx, 0.0, intr["ppx"] * sx],
        [0.0, intr["fy"] * sy, intr["ppy"] * sy],
        [0.0, 0.0, 1.0],
    ])


def camera_metadata(ov: dict, width: int, height: int) -> dict:
    """Per-camera provenance/geometry block to stamp into rendered datasets."""
    out = {
        "bundle_id": ov["bundle_id"],
        "calibration_revision": ov["calibration_revision"],
        "status": ov["status"],
        "generated_at": ov["generated_at"],
        "render_width": width,
        "render_height": height,
        "cameras": {},
    }
    for role in ALL_ROLES:
        cam = ov["runtime"]["cameras"][role]
        intr, model = cam["intrinsics"], cam["isaac_camera_model"]
        # what the Omniverse renderer actually realizes: square pixels from the
        # horizontal aperture with a CENTERED principal point (vertical aperture
        # and both aperture offsets are ignored by the RTX renderer, OM-42611).
        f_eff = float(model["focal_length_mm"]) * width / float(model["horizontal_aperture_mm"])
        out["cameras"][role] = {
            "model": cam["model"],
            "serial": cam["serial"],
            "calibration_id": cam["calibration_id"],
            "calibration_status": cam["calibration_status"],
            "parent_semantic": cam["parent_semantic"],
            "parent_T_camera_usd": np.asarray(cam["parent_T_camera_usd"]["matrix"]).tolist(),
            "parent_T_camera_optical": np.asarray(cam["parent_T_camera_optical"]["matrix"]).tolist(),
            "K_calibrated_at_render_size": scaled_K(cam, width, height).tolist(),
            "K_effective_render": [[f_eff, 0.0, width / 2.0], [0.0, f_eff, height / 2.0], [0.0, 0.0, 1.0]],
            "native_intrinsics": {
                "width": intr["width"], "height": intr["height"],
                "fx": intr["fx"], "fy": intr["fy"], "ppx": intr["ppx"], "ppy": intr["ppy"],
                "distortion_model": intr["distortion_model"],
                "coeffs": [float(c) for c in intr["coeffs"]],
            },
            "render_model_note": (
                "Rendered images follow K_effective_render: Omniverse ignores USD vertical aperture and "
                "aperture offsets (OM-42611), so pixels are square (from fx) with a centered principal "
                "point, and lens distortion is not applied. K_calibrated_at_render_size and "
                "native_intrinsics describe the REAL camera for real-vs-sim math."
            ),
        }
    return out


# -------------------------------------------------------------- Isaac config
def build_camera_cfgs(
    ov: dict,
    hand_T_tcp: np.ndarray,
    width: int,
    height: int,
    robot_prim: str = "{ENV_REGEX_NS}/Robot",
    roles: tuple[str, ...] = ALL_ROLES,
) -> dict:
    """role -> isaaclab.sensors.CameraCfg. Import only after AppLauncher."""
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import CameraCfg

    if abs(width / height - NATIVE_W / NATIVE_H) > 1e-3:
        raise ValueError(f"render size {width}x{height} must keep the 16:9 native aspect")

    cfgs = {}
    for role in roles:
        cam = ov["runtime"]["cameras"][role]
        T_usd = T_from(cam["parent_T_camera_usd"])
        if cam["parent_semantic"] == "hand_tcp":
            parent, T_local = f"{robot_prim}/fr3_hand", hand_T_tcp @ T_usd
        else:  # robot_base
            parent, T_local = f"{robot_prim}/fr3_link0", T_usd
        model = cam["isaac_camera_model"]
        cfgs[role] = CameraCfg(
            prim_path=f"{parent}/{cam['isaac_prim_name']}",
            update_period=0.0,
            height=height,
            width=width,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=float(model["focal_length_mm"]),
                horizontal_aperture=float(model["horizontal_aperture_mm"]),
                vertical_aperture=float(model["vertical_aperture_mm"]),
                horizontal_aperture_offset=float(model["horizontal_aperture_offset_mm"]),
                vertical_aperture_offset=float(model["vertical_aperture_offset_mm"]),
                clipping_range=tuple(float(v) for v in model["clipping_range_m"]),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=tuple(T_local[:3, 3]),
                rot=tuple(quat_wxyz_from_R(T_local[:3, :3])),
                convention="opengl",  # rot already IS the USD/OpenGL camera frame
            ),
        )
    return cfgs


def load_binding(path: str, ov: dict) -> tuple[np.ndarray, dict]:
    """(fr3_hand_T_fr3v2_hand_tcp, full binding dict) from probe_tcp_binding.py's YAML.

    Refuses a binding measured against a DIFFERENT overlay revision — the bundle's
    invalidation policy (new wrist mount => new calibration) must force a re-probe.
    """
    with open(path, "r", encoding="utf-8") as f:
        b = yaml.safe_load(f)
    if not isinstance(b, dict) or not b.get("ready_to_apply", False):
        raise ValueError(f"binding {path} is not marked ready_to_apply — rerun/inspect the probe")
    for key, want in (("overlay_bundle_id", ov["bundle_id"]), ("calibration_revision", ov["calibration_revision"])):
        if b.get(key) != want:
            raise ValueError(
                f"binding {path} was measured against {key}={b.get(key)!r} but the overlay is {want!r} "
                f"— rerun probe_tcp_binding.py against the current bundle")
    return np.asarray(b["hand_tcp"]["team_frame_T_calibrated_frame"]["matrix"], dtype=np.float64), b
