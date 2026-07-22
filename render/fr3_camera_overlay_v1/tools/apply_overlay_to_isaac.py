#!/usr/bin/env python3
"""Validate and author a portable FR3 camera overlay into an Isaac USD stage."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import yaml


SCHEMA = "stage2.fr3_camera_overlay.v1"


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"expected YAML mapping: {path}")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--overlay", type=Path, default=Path(__file__).resolve().parents[1] / "overlay.yaml"
    )
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Output USD; input is never overwritten.")
    parser.add_argument("--base-prim", required=True)
    parser.add_argument("--hand-tcp-prim", required=True)
    parser.add_argument("--camera-root-name", default="calibrated_cameras")
    parser.add_argument("--frame-root-name", default="calibration_frames")
    parser.add_argument("--skip-environment-frames", action="store_true")
    parser.add_argument(
        "--on-conflict",
        choices=("error", "reuse", "replace", "rename"),
        default="error",
        help="Policy for an existing target Camera or calibration-frame prim.",
    )
    parser.add_argument(
        "--allow-stage-convention-mismatch",
        action="store_true",
        help="Permit non-meter or non-Z-up stages after an external frame adapter was reviewed.",
    )
    parser.add_argument(
        "--allow-non-articulated-hand",
        action="store_true",
        help="Permit a hand TCP with no ArticulationRootAPI ancestor.",
    )
    parser.add_argument("--plan-only", action="store_true", help="Print paths without opening USD.")
    parser.add_argument(
        "--validate-only", action="store_true", help="Open and inspect the USD without authoring."
    )
    return parser.parse_args()


def planned_paths(args: argparse.Namespace, cameras: dict[str, Any]) -> dict[str, str]:
    return {
        role: (
            f"{args.hand_tcp_prim.rstrip('/')}/{camera['isaac_prim_name']}"
            if camera["parent_semantic"] == "hand_tcp"
            else f"{args.base_prim.rstrip('/')}/{args.camera_root_name}/{camera['isaac_prim_name']}"
        )
        for role, camera in cameras.items()
    }


def set_matrix_xform(prim: Any, transform: list[list[float]]) -> None:
    from pxr import Gf, UsdGeom

    matrix = Gf.Matrix4d(1.0)
    for row in range(4):
        for column in range(4):
            matrix[row, column] = float(transform[column][row])
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.AddTransformOp().Set(matrix)


def define_camera(stage: Any, path: Any, camera: dict[str, Any]) -> None:
    from pxr import Gf, UsdGeom

    usd_camera = UsdGeom.Camera.Define(stage, path)
    set_matrix_xform(usd_camera.GetPrim(), camera["parent_T_camera_usd"]["matrix"])
    model = camera["isaac_camera_model"]
    usd_camera.CreateFocalLengthAttr(float(model["focal_length_mm"]))
    usd_camera.CreateHorizontalApertureAttr(float(model["horizontal_aperture_mm"]))
    usd_camera.CreateVerticalApertureAttr(float(model["vertical_aperture_mm"]))
    usd_camera.CreateHorizontalApertureOffsetAttr(float(model["horizontal_aperture_offset_mm"]))
    usd_camera.CreateVerticalApertureOffsetAttr(float(model["vertical_aperture_offset_mm"]))
    usd_camera.CreateClippingRangeAttr(
        Gf.Vec2f(*[float(value) for value in model["clipping_range_m"]])
    )
    prim = usd_camera.GetPrim()
    prim.SetCustomDataByKey("frSysCameraRole", camera["role"])
    prim.SetCustomDataByKey("frSysCameraModel", camera["model"])
    prim.SetCustomDataByKey("frSysCameraSerial", camera["serial"])
    prim.SetCustomDataByKey("frSysCalibrationId", camera["calibration_id"])


def articulation_ancestor(stage: Any, path: Any) -> str | None:
    from pxr import Sdf, UsdPhysics

    current = path
    while current != Sdf.Path.absoluteRootPath:
        prim = stage.GetPrimAtPath(current)
        if prim.IsValid() and (
            prim.HasAPI(UsdPhysics.ArticulationRootAPI)
            or any("ArticulationRootAPI" in item for item in prim.GetAppliedSchemas())
        ):
            return str(current)
        current = current.GetParentPath()
    return None


def next_available_path(stage: Any, path: Any) -> Any:
    parent = path.GetParentPath()
    stem = path.name
    index = 1
    while True:
        suffix = "_overlay" if index == 1 else f"_overlay_{index}"
        candidate = parent.AppendChild(f"{stem}{suffix}")
        if not stage.GetPrimAtPath(candidate).IsValid():
            return candidate
        index += 1


def resolve_camera_paths(
    stage: Any,
    requested: dict[str, Any],
    policy: str,
) -> tuple[dict[str, Any], list[dict[str, str]], list[str]]:
    from pxr import UsdGeom

    resolved: dict[str, Any] = {}
    conflicts: list[dict[str, str]] = []
    errors: list[str] = []
    for role, path in requested.items():
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            resolved[role] = path
            continue
        existing_type = prim.GetTypeName() or "untyped"
        action = policy
        if policy == "error":
            errors.append(f"camera target already exists: {path} ({existing_type})")
            resolved[role] = path
        elif policy == "reuse":
            if not prim.IsA(UsdGeom.Camera):
                errors.append(f"cannot reuse non-Camera prim: {path} ({existing_type})")
            resolved[role] = path
        elif policy == "replace":
            resolved[role] = path
        else:
            resolved[role] = next_available_path(stage, path)
            action = f"rename_to:{resolved[role]}"
        conflicts.append(
            {"role": role, "path": str(path), "existing_type": existing_type, "action": action}
        )
    return resolved, conflicts, errors


def resolve_frame_root(stage: Any, requested_root: Any, policy: str) -> tuple[Any, list[str]]:
    from pxr import UsdGeom

    child_names = ("table_task", "threaded_plate")
    occupied = [
        requested_root.AppendChild(name)
        for name in child_names
        if stage.GetPrimAtPath(requested_root.AppendChild(name)).IsValid()
    ]
    if not occupied:
        return requested_root, []
    if policy == "error":
        return requested_root, [f"calibration frame target already exists: {path}" for path in occupied]
    if policy == "rename":
        return next_available_path(stage, requested_root), []
    if policy == "reuse":
        errors = []
        for path in occupied:
            if not stage.GetPrimAtPath(path).IsA(UsdGeom.Xform):
                errors.append(f"cannot reuse non-Xform calibration frame: {path}")
        return requested_root, errors
    return requested_root, []


def inspect_stage(
    stage: Any,
    stage_path: Path,
    args: argparse.Namespace,
    cameras: dict[str, Any],
) -> dict[str, Any]:
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdUtils

    errors: list[str] = []
    warnings: list[str] = []
    base_path = Sdf.Path(args.base_prim)
    hand_path = Sdf.Path(args.hand_tcp_prim)
    base_prim = stage.GetPrimAtPath(base_path)
    hand_prim = stage.GetPrimAtPath(hand_path)
    if not base_prim.IsValid():
        errors.append(f"base prim does not exist: {base_path}")
    if not hand_prim.IsValid():
        errors.append(f"hand TCP prim does not exist: {hand_path}")

    meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
    up_axis = str(UsdGeom.GetStageUpAxis(stage))
    convention_ok = abs(meters_per_unit - 1.0) <= 1e-9 and up_axis.upper() == "Z"
    if not convention_ok and not args.allow_stage_convention_mismatch:
        errors.append(
            f"stage convention mismatch: meters_per_unit={meters_per_unit}, up_axis={up_axis}; expected 1.0 and Z"
        )
    if hand_prim.IsValid() and base_prim.IsValid() and not hand_path.HasPrefix(base_path):
        warnings.append("hand TCP prim is not a descendant of the supplied base prim")
    articulation = articulation_ancestor(stage, hand_path) if hand_prim.IsValid() else None
    physics_joint_count = 0
    if base_prim.IsValid():
        physics_joint_count = sum(
            1 for prim in Usd.PrimRange(base_prim) if prim.IsA(UsdPhysics.Joint)
        )
    articulation_evidence = "ArticulationRootAPI" if articulation is not None else None
    if articulation is None and physics_joint_count >= 7:
        articulation_evidence = f"physics_joint_descendants:{physics_joint_count}"
        warnings.append(
            "no ArticulationRootAPI ancestor is composed at the hand TCP; "
            f"accepted the robot-base hierarchy with {physics_joint_count} physics joints"
        )
    elif articulation is None and not args.allow_non_articulated_hand:
        errors.append(
            "hand TCP has no ArticulationRootAPI ancestor and the robot base has fewer than 7 physics joints"
        )

    unresolved: list[str] = []
    try:
        _, _, unresolved_paths = UsdUtils.ComputeAllDependencies(str(stage_path))
        unresolved = [str(path) for path in unresolved_paths]
    except Exception as exc:
        warnings.append(f"dependency scan unavailable: {exc}")
    if unresolved:
        errors.append(f"USD has unresolved dependencies: {unresolved}")

    requested = {role: Sdf.Path(path) for role, path in planned_paths(args, cameras).items()}
    resolved, conflicts, conflict_errors = resolve_camera_paths(stage, requested, args.on_conflict)
    errors.extend(conflict_errors)
    requested_frame_root = base_path.AppendChild(args.frame_root_name)
    frame_root = requested_frame_root
    if not args.skip_environment_frames:
        frame_root, frame_errors = resolve_frame_root(stage, requested_frame_root, args.on_conflict)
        errors.extend(frame_errors)

    if args.output.resolve().parent != stage_path.parent:
        warnings.append(
            "output is not beside the input USD; relative references may need relocation or collection"
        )
    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "meters_per_unit": meters_per_unit,
        "up_axis": up_axis,
        "base_prim_type": base_prim.GetTypeName() if base_prim.IsValid() else None,
        "hand_tcp_prim_type": hand_prim.GetTypeName() if hand_prim.IsValid() else None,
        "articulation_root": articulation,
        "articulation_evidence": articulation_evidence,
        "physics_joint_count": physics_joint_count,
        "unresolved_dependencies": unresolved,
        "camera_paths": resolved,
        "camera_conflicts": conflicts,
        "frame_root": frame_root,
    }


def print_inspection(result: dict[str, Any], args: argparse.Namespace) -> None:
    print("portable_overlay_isaac_preflight_ok", str(result["ok"]).lower())
    print("stage", args.stage.resolve())
    print("meters_per_unit", result["meters_per_unit"])
    print("up_axis", result["up_axis"])
    print("base_prim", args.base_prim, result["base_prim_type"])
    print("hand_tcp_prim", args.hand_tcp_prim, result["hand_tcp_prim_type"])
    print("articulation_root", result["articulation_root"])
    print("articulation_evidence", result["articulation_evidence"])
    print("physics_joint_count", result["physics_joint_count"])
    print("on_conflict", args.on_conflict)
    for role, path in result["camera_paths"].items():
        print("camera", role, path)
    for conflict in result["camera_conflicts"]:
        print("conflict", conflict)
    for warning in result["warnings"]:
        print("warning", warning)
    for error in result["errors"]:
        print("error", error)
    sys.stdout.flush()


def main() -> int:
    args = parse_args()
    overlay_path = args.overlay.resolve()
    stage_path = args.stage.resolve()
    output_path = args.output.resolve()
    if not overlay_path.is_file():
        raise SystemExit(f"overlay does not exist: {overlay_path}")
    if not stage_path.is_file():
        raise SystemExit(f"stage does not exist: {stage_path}")
    if stage_path == output_path:
        raise SystemExit("--output must differ from --stage")

    overlay = load_yaml(overlay_path)
    if overlay.get("schema_version") != SCHEMA:
        raise SystemExit(f"unsupported overlay schema: {overlay.get('schema_version')}")
    cameras = overlay["runtime"]["cameras"]
    if args.plan_only:
        print("portable_overlay_isaac_plan_ok true")
        print("note USD and prims were not inspected; use --validate-only for stage-aware preflight")
        for role, path in planned_paths(args, cameras).items():
            print("camera", role, path)
        return 0

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(stage_path))
    if stage is None:
        print(f"failed to open USD stage: {stage_path}", file=sys.stderr, flush=True)
        os._exit(1)
    inspection = inspect_stage(stage, stage_path, args, cameras)
    print_inspection(inspection, args)
    if not inspection["ok"]:
        os._exit(2)
    if args.validate_only:
        app.close()
        return 0

    if args.on_conflict == "replace":
        for conflict in inspection["camera_conflicts"]:
            stage.RemovePrim(conflict["path"])
        if not args.skip_environment_frames:
            frame_root = inspection["frame_root"]
            if stage.GetPrimAtPath(frame_root).IsValid():
                stage.RemovePrim(frame_root)

    fixed_root = next(
        path.GetParentPath()
        for role, path in inspection["camera_paths"].items()
        if cameras[role]["parent_semantic"] == "robot_base"
    )
    UsdGeom.Xform.Define(stage, fixed_root)
    for role, camera in cameras.items():
        define_camera(stage, inspection["camera_paths"][role], camera)

    authored_frames: list[str] = []
    if not args.skip_environment_frames:
        frame_root = inspection["frame_root"]
        UsdGeom.Xform.Define(stage, frame_root)
        environment = overlay["runtime"]["environment"]
        table_path = frame_root.AppendChild("table_task")
        table = UsdGeom.Xform.Define(stage, table_path)
        set_matrix_xform(table.GetPrim(), environment["table"]["base_T_table"]["matrix"])
        nominal_path = table_path.AppendChild("nominal_table_asset_alignment")
        nominal = UsdGeom.Xform.Define(stage, nominal_path)
        set_matrix_xform(
            nominal.GetPrim(), environment["table"]["table_T_nominal_asset"]["matrix"]
        )
        plate_path = frame_root.AppendChild("threaded_plate")
        plate = UsdGeom.Xform.Define(stage, plate_path)
        set_matrix_xform(
            plate.GetPrim(), environment["threaded_plate"]["base_T_plate"]["matrix"]
        )
        authored_frames = [str(table_path), str(nominal_path), str(plate_path)]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    stage.GetRootLayer().Export(str(output_path))
    print("portable_overlay_isaac_apply_ok true")
    print("output", output_path)
    for role, path in inspection["camera_paths"].items():
        print("authored_camera", role, path)
    for path in authored_frames:
        print("authored_environment_frame", path)
    sys.stdout.flush()
    app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
