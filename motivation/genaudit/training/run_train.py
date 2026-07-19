"""Server entry point: register E-series variants, then run robomimic train.

Usage (inside the training venv):
    python -m genaudit.training.run_train --config <bc_config.json> [...]

Training configs built from E-series pools carry env_name like Threading_D2E
in env_meta; robomimic's train.py builds the env at startup (in-training
rollouts), so the variants must be registered in the training process too.
Arguments pass through verbatim to robomimic.scripts.train.
"""
from __future__ import annotations

import runpy
import sys

from genaudit.envs.robosuite_variants import register_custom_variants


def main() -> None:
    created = register_custom_variants()
    print(f"[genaudit] registered variants: {sorted(created)}")
    sys.argv[0] = "train.py"
    runpy.run_module("robomimic.scripts.train", run_name="__main__")


if __name__ == "__main__":
    main()
