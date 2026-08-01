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

from genaudit.envs.robosuite_variants import (
    register_custom_variants,
    register_new_variants,
)


def main() -> None:
    # Our own flags are stripped before delegation (generate_dataset's
    # argparse rejects unknown arguments).
    argv = list(sys.argv[1:])
    physics_path = None
    if "--physics" in argv:
        index = argv.index("--physics")
        physics_path = argv[index + 1]
        del argv[index:index + 2]

    created = register_custom_variants()
    created.update(register_new_variants())  # motivation_new N0/N1/N2 ladder
    print(f"[genaudit] registered variants: {sorted(created)}")
    if physics_path is not None:
        from genaudit.envs.physics_variants import register_from_file

        physics_class = register_from_file(physics_path)
        print(f"[genaudit] registered physics variant: {physics_class}")
    sys.argv = ["generate_dataset.py"] + argv
    runpy.run_module("mimicgen.scripts.generate_dataset", run_name="__main__")


if __name__ == "__main__":
    main()
