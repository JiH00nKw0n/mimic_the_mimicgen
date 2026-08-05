"""Emit an overlay.yaml from the 2026-07-28 measured camera calibration.

Our renderer consumes `render/fr3_camera_overlay_v1/overlay.yaml` (calibration
revision fr3_four_camera_v3_depth_refined, 2026-07-21). The visual
randomization handoff ships a NEWER calibration for the same four serials —
`config/camera_nominal_measured_ranges.yaml`, sourced from
`fixed_camera_recalibration_v4` / `fr3_d405_rigid_mount_v1` (2026-07-28) —
whose fixed cameras differ by 122-139 mm and whose wrist differs by 81 mm and
158 deg (the wrist bracket was replaced, not re-measured).

Rather than hand-editing numbers, this script keeps the old overlay as the
STRUCTURAL template (its non-camera sections — units, conventions, frame
semantics, and the home-pose `reference_robot_pose` the binding probe fits
against — are robot-side and unaffected by a camera recalibration) and
overwrites exactly the per-camera extrinsics and intrinsics.

The renderer's loader re-derives every quaternion from its matrix and rejects
a mismatch, so the emitted file carries matrix, translation_m and
quaternion_xyzw that are consistent by construction.

  python3 build_overlay_from_measured.py \
      --template render/fr3_camera_overlay_v1/overlay.yaml \
      --measured render/fr3_visual_randomization_v1/config/camera_nominal_measured_ranges.yaml \
      --output   render/fr3_camera_overlay_v2/overlay.yaml
"""
from __future__ import annotations

import argparse
import copy
import datetime
import os

import numpy as np
import yaml

BUNDLE_ID = "fr3_four_camera_overlay_measured_20260728"
CALIB_REV = "fr3_four_camera_v4_measured_20260728"
# USD/OpenGL camera axes from the RealSense OpenCV optical axes:
# X stays, Y flips (down -> up), Z flips (forward -> backward).
OPTICAL_TO_USD = np.diag([1.0, -1.0, -1.0])


def quat_xyzw_from_R(R: np.ndarray) -> list[float]:
    """Rotation matrix -> [x, y, z, w] (the overlay schema's order)."""
    t = float(np.trace(R))
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2.0
        q = [0.0, 0.0, 0.0]
        q[i] = 0.25 * s
        q[j] = (R[j, i] + R[i, j]) / s
        q[k] = (R[k, i] + R[i, k]) / s
        w = (R[k, j] - R[j, k]) / s
        x, y, z = q
    v = np.array([x, y, z, w], dtype=float)
    return [float(c) for c in v / np.linalg.norm(v)]


def pose_block(T: np.ndarray) -> dict:
    return {
        "matrix": [[float(v) for v in row] for row in T],
        "translation_m": [float(v) for v in T[:3, 3]],
        "quaternion_xyzw": quat_xyzw_from_R(T[:3, :3]),
    }


def intrinsics_block(src: dict) -> dict:
    out = {}
    for key in ("width", "height", "fx", "fy", "ppx", "ppy",
                "distortion_model", "coeffs"):
        if key in src:
            out[key] = src[key]
    return out


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--template", default=os.path.join(
        here, "fr3_camera_overlay_v1/overlay.yaml"))
    ap.add_argument("--measured", default=os.path.join(
        here, "fr3_visual_randomization_v1/config/camera_nominal_measured_ranges.yaml"))
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    with open(args.template) as fh:
        ov = yaml.safe_load(fh)
    with open(args.measured) as fh:
        meas = yaml.safe_load(fh)
    nominal = meas["nominal"]

    ov = copy.deepcopy(ov)
    ov["bundle_id"] = BUNDLE_ID
    ov["calibration_revision"] = CALIB_REV
    ov["status"] = "measured_team_handoff_20260728"
    ov["generated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    ov.setdefault("provenance", {})
    ov["provenance"]["rebuilt_from"] = {
        "measured_ranges": os.path.basename(args.measured),
        "measured_schema": meas.get("schema_version"),
        "fixed_source": nominal["fixed_d435"].get("source"),
        "wrist_condition_id": nominal["wrist_d405"].get("condition_id"),
        "structural_template": os.path.basename(args.template),
        "note": ("camera extrinsics/intrinsics replaced; robot-side sections "
                 "(conventions, frame semantics, reference_robot_pose) kept "
                 "from the template because a camera recalibration does not "
                 "change them"),
    }

    cams = ov["runtime"]["cameras"]
    report = []
    for role, cam in cams.items():
        if role in nominal["fixed_d435"]["by_role"]:
            src = nominal["fixed_d435"]["by_role"][role]
            T_opt = np.array(src["parent_T_camera_optical"], dtype=float)
            intr = src["intrinsics"]
            pinhole = intr.get("isaac_pinhole") or src.get("isaac_pinhole")
        elif role == "wrist":
            src = nominal["wrist_d405"]
            T_opt = np.array(src["parent_T_camera_optical"], dtype=float)
            intr = src["intrinsics"]
            pinhole = src.get("isaac_pinhole")
        else:
            raise SystemExit(f"no measured entry for role {role!r}")

        T_usd = T_opt.copy()
        T_usd[:3, :3] = T_opt[:3, :3] @ OPTICAL_TO_USD
        old_t = np.array(cam["parent_T_camera_optical"]["matrix"], float)[:3, 3]
        d = float(np.linalg.norm(old_t - T_opt[:3, 3]))

        cam["parent_T_camera_optical"] = pose_block(T_opt)
        cam["parent_T_camera_usd"] = pose_block(T_usd)
        cam["intrinsics"] = intrinsics_block(intr)
        if pinhole:
            cam["isaac_camera_model"] = dict(pinhole)
        cam["calibration_status"] = "measured_20260728"
        cam["calibration_id"] = f"{role}_{CALIB_REV}"
        if "serial" in src:
            cam["serial"] = src["serial"]
        report.append(f"  {role}: moved {d * 1000:.1f} mm vs template")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as fh:
        yaml.safe_dump(ov, fh, sort_keys=False, allow_unicode=True)
    print(f"[overlay] wrote {args.output}  bundle={BUNDLE_ID}")
    print("\n".join(report))


if __name__ == "__main__":
    main()
