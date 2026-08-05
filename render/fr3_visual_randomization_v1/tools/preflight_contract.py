#!/usr/bin/env python3
"""CPU-only structural preflight for the visual randomization handoff."""

from __future__ import annotations

import re
from pathlib import Path

import yaml


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    profile_path = root / "config/visual_randomization_profiles.yaml"
    camera_path = root / "config/camera_nominal_measured_ranges.yaml"
    profile = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    camera = yaml.safe_load(camera_path.read_text(encoding="utf-8"))
    failures = []

    mixture = profile["mixture"]
    if abs(sum(float(value) for value in mixture.values()) - 1.0) > 1e-9:
        failures.append(f"mixture does not sum to one: {mixture}")
    if mixture != {"nominal_lab": 0.5, "lab_variation": 0.4, "stress_tail": 0.1}:
        failures.append(f"unexpected mixture: {mixture}")
    capture = profile["capture"]
    if capture["width"] * 9 != capture["height"] * 16:
        failures.append(f"capture is not 16:9: {capture}")
    if capture["cameras"] != ["third_person_0", "third_person_1", "wrist"]:
        failures.append(f"unexpected active cameras: {capture['cameras']}")
    if profile["gates"]["max_foreign_pixels_per_camera_env"] != 0:
        failures.append("foreign-pixel gate is not fail-closed at zero")
    if float(profile["gates"]["min_rgb_std"]) < 10.0:
        failures.append("RGB corruption threshold is weaker than 10")

    if camera.get("schema_version") != "stage2.fr3_camera_nominal_measured_ranges.v2":
        failures.append(f"unexpected camera schema: {camera.get('schema_version')}")

    external_refs = []
    usd_ref_pattern = re.compile(r"@([^@]+)@")
    for usd in sorted((root / "assets").rglob("*.usd*")):
        try:
            text = usd.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for ref in usd_ref_pattern.findall(text):
            if ref.startswith(("http://", "https://", "omniverse://")):
                external_refs.append((usd.relative_to(root).as_posix(), ref))
            else:
                resolved = (usd.parent / ref).resolve()
                if not resolved.exists():
                    failures.append(
                        f"unresolved relative USD reference: {usd.relative_to(root)} -> {ref}"
                    )

    required = [
        "assets/fr3/facelift/fr3_facelift_visual.usda",
        "assets/table/table_scene_rgb_lab.usda",
        "resources/textures/Carpet_Gray_BaseColor.png",
        "resources/hdri/studio_small_04_1k.hdr",
        "resources/hdri/studio_small_07_1k.hdr",
        "resources/hdri/empty_workshop_1k.hdr",
        "source/data_collection_rgb_cfg.py",
        "source/events.py",
    ]
    for relative in required:
        if not (root / relative).is_file():
            failures.append(f"missing required file: {relative}")

    if failures:
        print("FAIL")
        print("\n".join(failures))
        raise SystemExit(1)
    print("PASS: profile/camera/asset structure")
    if external_refs:
        print("External USD dependencies:")
        for source, ref in external_refs:
            print(f"- {source} -> {ref}")


if __name__ == "__main__":
    main()
