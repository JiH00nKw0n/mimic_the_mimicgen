"""Server entry point: register mimicgen envs (+E-variants), then run
robomimic's rollout evaluator.

Usage:
    python -m genaudit.evaluation.run_eval --agent <ckpt.pth> --n_rollouts 50 ...

Checkpoints/datasets carry mimicgen env names (Square_D2, Threading_D2E, ...)
in env_meta; robomimic's run_trained_agent.py never imports mimicgen, so the
robosuite registry lacks them without this wrapper. Arguments pass through
verbatim.
"""
from __future__ import annotations

import runpy
import sys

from genaudit.envs.robosuite_variants import register_custom_variants


def main() -> None:
    created = register_custom_variants()  # imports mimicgen -> registers all
    print(f"[genaudit] mimicgen envs registered (+{len(created)} E-variants)")
    sys.argv[0] = "run_trained_agent.py"
    runpy.run_module("robomimic.scripts.run_trained_agent", run_name="__main__")


if __name__ == "__main__":
    main()
