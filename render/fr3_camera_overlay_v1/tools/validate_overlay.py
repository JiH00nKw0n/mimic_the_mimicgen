#!/usr/bin/env python3
"""Validate a portable FR3 camera-overlay bundle without Isaac Sim."""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path
from typing import Any

import numpy as np
import yaml


ABSOLUTE_PATH_PATTERNS = (
    re.compile(r"(?<![A-Za-z0-9_])/(?:home|Users|opt|mnt|tmp|var)/"),
    re.compile(r"[A-Za-z]:[\\/]"),
)
PROJECT_PATH_PREFIXES = ("data/", "config/", "docs/", "assets/", "models/")
CAMERA_ROLES = ("third_person_0", "third_person_1", "third_person_2", "wrist")


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected YAML mapping: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_transform(record: Any, label: str, errors: list[str]) -> None:
    value = record.get("matrix") if isinstance(record, dict) else record
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4):
        errors.append(f"{label}: expected 4x4 matrix, got {matrix.shape}")
        return
    if not np.all(np.isfinite(matrix)):
        errors.append(f"{label}: matrix contains non-finite values")
        return
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
        errors.append(f"{label}: invalid homogeneous bottom row")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-5):
        errors.append(f"{label}: rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=2e-5):
        errors.append(f"{label}: rotation determinant is not +1")


def scan_portability(bundle: Path, errors: list[str]) -> None:
    for path in sorted(bundle.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".yaml", ".yml", ".md", ".json", ".py"}:
            continue
        text = path.read_text(encoding="utf-8")
        relative = path.relative_to(bundle)
        for pattern in ABSOLUTE_PATH_PATTERNS:
            if pattern.search(text):
                errors.append(f"{relative}: contains a machine-specific absolute path")
                break
        if relative.name in {"overlay.yaml", "manifest.yaml", "README.md"}:
            for prefix in PROJECT_PATH_PREFIXES:
                if prefix in text:
                    errors.append(f"{relative}: contains project-root path prefix {prefix!r}")


def validate(bundle: Path) -> list[str]:
    errors: list[str] = []
    manifest_path = bundle / "manifest.yaml"
    overlay_path = bundle / "overlay.yaml"
    if not manifest_path.is_file():
        errors.append("manifest.yaml is missing")
    if not overlay_path.is_file():
        errors.append("overlay.yaml is missing")
    if errors:
        return errors

    manifest = load_yaml(manifest_path)
    overlay = load_yaml(overlay_path)
    if manifest.get("schema_version") != "stage2.portable_camera_overlay_manifest.v1":
        errors.append("manifest.yaml: unexpected schema_version")
    if overlay.get("schema_version") != "stage2.fr3_camera_overlay.v1":
        errors.append("overlay.yaml: unexpected schema_version")
    if manifest.get("bundle_id") != overlay.get("bundle_id"):
        errors.append("bundle_id differs between manifest.yaml and overlay.yaml")
    if manifest.get("bundle_revision") != overlay.get("package_revision"):
        errors.append("bundle_revision differs from overlay package_revision")
    runtime_compatibility = manifest.get("runtime_compatibility", {})
    if runtime_compatibility.get("python") != ">=3.10":
        errors.append("manifest runtime compatibility does not require Python >=3.10")
    isaac_compatibility = runtime_compatibility.get("isaac_sim", {})
    if "5.1" not in isaac_compatibility.get("tested_versions", []):
        errors.append("manifest does not declare the tested Isaac Sim 5.1 runtime")
    smoke = isaac_compatibility.get("reference_scene_smoke", {})
    smoke_report_path = bundle / smoke.get("report", "missing-smoke-report")
    if smoke.get("status") != "pass" or not smoke_report_path.is_file():
        errors.append("manifest does not include a passing Isaac 5.1 smoke report")
    else:
        smoke_report = load_yaml(smoke_report_path)
        if smoke_report.get("status") != "pass":
            errors.append("Isaac 5.1 smoke report is not passing")
        image_records = list(smoke_report.get("images", {}).values())
        wrist_second = smoke_report.get("wrist_second_q")
        if wrist_second:
            image_records.append(wrist_second)
        if len(image_records) != 5 or not all(
            record.get("nonblank") is True for record in image_records
        ):
            errors.append("Isaac 5.1 smoke report does not contain five nonblank renders")
    expected_entrypoints = {
        "check_runtime",
        "validate",
        "apply_to_isaac",
        "smoke_render",
    }
    if not expected_entrypoints.issubset(manifest.get("entrypoints", {})):
        errors.append("manifest is missing one or more runtime entry points")

    for entry in manifest.get("files", []):
        relative_text = entry.get("path")
        if not isinstance(relative_text, str):
            errors.append("manifest file entry has no string path")
            continue
        relative = Path(relative_text)
        if relative.is_absolute() or ".." in relative.parts:
            errors.append(f"manifest path is not bundle-relative: {relative_text}")
            continue
        path = bundle / relative
        if not path.is_file():
            errors.append(f"manifest file is missing: {relative_text}")
            continue
        expected_hash = entry.get("sha256")
        if expected_hash and sha256(path) != expected_hash:
            errors.append(f"manifest checksum mismatch: {relative_text}")

    runtime = overlay.get("runtime", {})
    cameras = runtime.get("cameras", {})
    missing_roles = [role for role in CAMERA_ROLES if role not in cameras]
    if missing_roles:
        errors.append(f"overlay cameras missing roles: {missing_roles}")
    for role, camera in cameras.items():
        check_transform(camera.get("parent_T_camera_optical"), f"cameras.{role}.parent_T_camera_optical", errors)
        check_transform(camera.get("parent_T_camera_usd"), f"cameras.{role}.parent_T_camera_usd", errors)
        intrinsics = camera.get("intrinsics", {})
        for name in ("width", "height", "fx", "fy", "ppx", "ppy"):
            if name not in intrinsics:
                errors.append(f"cameras.{role}.intrinsics.{name} is missing")

    environment = runtime.get("environment", {})
    check_transform(environment.get("table", {}).get("base_T_table"), "environment.table.base_T_table", errors)
    check_transform(
        environment.get("threaded_plate", {}).get("base_T_plate"),
        "environment.threaded_plate.base_T_plate",
        errors,
    )
    check_transform(
        environment.get("table", {}).get("table_T_nominal_asset"),
        "environment.table.table_T_nominal_asset",
        errors,
    )

    wrist = cameras.get("wrist", {})
    if wrist.get("calibration_id") != "fr3_wrist_camera_tape_mount_depth_refined_v2":
        errors.append("wrist camera does not name the canonical depth-refined calibration")
    provenance = overlay.get("provenance", {}).get("wrist_extrinsic", {})
    if provenance.get("calibration_id") != wrist.get("calibration_id"):
        errors.append("wrist runtime and provenance calibration_id differ")

    scenes = overlay.get("validation_scenes", {})
    expected_scenes = {"fixed_camera_board_scene", "wrist_handeye_board_scene"}
    if not expected_scenes.issubset(scenes):
        errors.append("validation scenes do not distinguish fixed-camera and wrist boards")
    for name, scene in scenes.items():
        if scene.get("validation_only") is not True:
            errors.append(f"validation_scenes.{name} is not marked validation_only")
        if "base_T_board" in scene:
            check_transform(scene["base_T_board"], f"validation_scenes.{name}.base_T_board", errors)
        for artifact in scene.get("artifacts", {}).values():
            if not isinstance(artifact, str):
                continue
            path = bundle / artifact
            if not path.is_file():
                errors.append(f"validation artifact is missing: {artifact}")

    scan_portability(bundle, errors)
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", nargs="?", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    errors = validate(bundle)
    if errors:
        print("portable_overlay_validation_ok false")
        for error in errors:
            print("error", error)
        return 1
    manifest = load_yaml(bundle / "manifest.yaml")
    overlay = load_yaml(bundle / "overlay.yaml")
    print("portable_overlay_validation_ok true")
    print("bundle", bundle)
    print("bundle_id", manifest["bundle_id"])
    print("package_revision", overlay["package_revision"])
    print("camera_count", len(overlay["runtime"]["cameras"]))
    print("external_assets_embedded false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
