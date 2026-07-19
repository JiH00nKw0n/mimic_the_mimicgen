"""Server entry point: register E-series variants, then run official MimicGen.

Usage (inside the robosuite_mimicgen venv):
    python -m genaudit.generation.run_mimicgen --config <mg_config.json> [...]

Everything after the module name is passed through verbatim to
`mimicgen.scripts.generate_dataset`, which runs unmodified — the only
difference from calling it directly is that our env variants exist first.
"""
from __future__ import annotations

import runpy
import sys

from genaudit.envs.robosuite_variants import register_custom_variants


def main() -> None:
    created = register_custom_variants()
    print(f"[genaudit] registered variants: {sorted(created)}")
    sys.argv[0] = "generate_dataset.py"
    runpy.run_module("mimicgen.scripts.generate_dataset", run_name="__main__")


if __name__ == "__main__":
    main()
