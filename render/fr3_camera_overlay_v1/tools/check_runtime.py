#!/usr/bin/env python3
"""Check Python modules required by the portable camera-overlay entry points."""

from __future__ import annotations

import argparse
import importlib.util
import platform
import sys


REQUIREMENTS = {
    "validate": (("yaml", "PyYAML"), ("numpy", "NumPy")),
    "apply": (("yaml", "PyYAML"), ("isaacsim", "Isaac Sim Python")),
    "render": (
        ("yaml", "PyYAML"),
        ("numpy", "NumPy"),
        ("PIL", "Pillow"),
        ("isaacsim", "Isaac Sim Python"),
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=tuple(REQUIREMENTS), default="validate")
    args = parser.parse_args()

    missing = []
    print("portable_overlay_runtime_check")
    print("mode", args.mode)
    print("python", sys.executable)
    print("python_version", platform.python_version())
    if sys.version_info < (3, 10):
        missing.append("Python >= 3.10")
    for module, label in REQUIREMENTS[args.mode]:
        available = importlib.util.find_spec(module) is not None
        print("module", module, "available", str(available).lower(), "label", label)
        if not available:
            missing.append(label)

    if missing:
        print("portable_overlay_runtime_ok false")
        print("missing", missing)
        if args.mode == "validate":
            print("suggestion use a project environment or install requirements-validator.txt")
        else:
            print("suggestion run this entry point with the team's Isaac Sim Python launcher")
        return 1
    print("portable_overlay_runtime_ok true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
