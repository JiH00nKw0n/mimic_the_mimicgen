"""Patch robomimic BC-RNN training configs for the E2 arms (PLAN.md §2.6).

The base config comes from mimicgen's own `generate_core_training_configs.py`
(paper-reproduction hyperparameters); we only vary dataset path, filter key,
seed, and bookkeeping — so baseline/uniform/ancestry arms train under
bit-identical recipes.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path


def patch_training_config(
    base: dict,
    *,
    dataset_path: str,
    filter_key: str,
    seed: int,
    experiment_name: str,
    output_dir: str,
    rollout_overrides: dict | None = None,
) -> dict:
    config = copy.deepcopy(base)
    config["experiment"]["name"] = experiment_name
    train = config["train"]
    train["data"] = dataset_path
    train["hdf5_filter_key"] = filter_key
    train["seed"] = seed
    train["output_dir"] = output_dir
    if rollout_overrides:
        unknown = set(rollout_overrides) - {"enabled", "n", "horizon", "rate", "terminate_on_success"}
        if unknown:
            raise ValueError(f"unknown rollout override keys: {sorted(unknown)}")
        config["experiment"]["rollout"].update(rollout_overrides)
    return config


def save_config(config: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=4))
    return path


def train_command(config_path: str | Path, robomimic_train_script: str) -> list[str]:
    """The exact launch command, for scripts/ and for logging into results."""
    return ["python", str(robomimic_train_script), "--config", str(config_path)]
