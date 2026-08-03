#!/usr/bin/env python3
"""Dependency-free integrity and contract checks for this handoff."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OFFICIAL_URL = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/"
    "Isaac/Robots/FrankaRobotics/FrankaFR3/fr3.usd"
)
WRAPPER_SHA256 = "6d5a3026a150b2c18a7cdbdefb31135319f005e563b9ce42e01090eabfac13ad"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fail(message: str) -> None:
    print(f"FAIL: {message}")
    raise SystemExit(1)


def main() -> int:
    checksums = ROOT / "SHA256SUMS"
    if not checksums.exists():
        fail("SHA256SUMS is missing")
    for line in checksums.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        path = ROOT / relative
        if not path.is_file():
            fail(f"missing file: {relative}")
        if sha256(path) != expected:
            fail(f"checksum mismatch: {relative}")

    wrapper = ROOT / "assets/fr3_research3.usda"
    if sha256(wrapper) != WRAPPER_SHA256:
        fail("canonical FR3 wrapper hash mismatch")
    wrapper_text = wrapper.read_text(encoding="utf-8")
    if OFFICIAL_URL not in wrapper_text:
        fail("wrapper does not reference the exact NVIDIA Isaac 5.1 FR3 URL")
    if "massfix" in wrapper_text.lower() or "FrankaEmika" in wrapper_text:
        fail("wrapper unexpectedly references a noncanonical asset")

    contract = (ROOT / "contracts/control_contract.yaml").read_text(encoding="utf-8")
    required_contract_tokens = (
        "fr3_cube_stage1_model4500_legacyosc_v1",
        "scale_xyz_axisangle: [0.02, 0.02, 0.02, 0.02, 0.02, 0.2]",
        "policy_hz: 10.0",
        "physics_hz: 120.0",
        "arm_command: joint_effort",
        "arm_actuator_stiffness: 0.0",
        "arm_actuator_damping: 0.0",
    )
    for token in required_contract_tokens:
        if token not in contract:
            fail(f"control contract token missing: {token}")

    subprocess.run(
        [sys.executable, str(ROOT / "tools/controller_adapter.py"), "--self-test"],
        check=True,
    )
    print("PASS: package checksums")
    print("PASS: official NVIDIA Isaac 5.1 FR3 wrapper identity")
    print("PASS: frozen Cube/MimicGen control contract")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

