"""Generate robomimic BC-RNN training configs for every E2 arm run.

Base recipe = mimicgen's paper low-dim config for the task; we only vary
train.data (the merged arm hdf5), train.hdf5_filter_key (which arm+seed),
seed, experiment name, and output dir. In-training rollouts are DISABLED
(experiment.rollout.enabled=false): the box has 4 CPU cores, and CPU rollouts
would serialize the GPU trainings — evaluation is done afterward on the shared
frozen-reset set (the primary paired endpoint, PLAN §2.6). Prints one launch
line per run (via scripts/c_train_launch.sh).

Server usage:
  PYTHONPATH=<repo>/motivation python c_make_train_configs.py \
      --arms-root ~/mimicgen_jihoonkwon/experiments/motivation_ic/e2_arms \
      --base-config /tmp/train_cfgs/bc_rnn_low_dim_ds_<task>_D2_seed_101.json \
      --out-dir ~/mimicgen_jihoonkwon/experiments/motivation_ic/e2_train_cfgs \
      --results-dir ~/mimicgen_jihoonkwon/experiments/motivation_ic/e2_results \
      --task square --experiment-config <repo>/.../e2_square.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from genaudit.config import load_experiment_spec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms-root", required=True)
    parser.add_argument("--base-config", required=True, help="mimicgen paper low-dim config for this task")
    parser.add_argument("--experiment-config", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--results-dir", required=True)
    args = parser.parse_args()

    experiment = load_experiment_spec(args.experiment_config)
    base = json.loads(Path(args.base_config).read_text())
    arm_dir = Path(args.arms_root).expanduser() / f"{experiment.task}_{experiment.variant}"
    train_hdf5 = arm_dir / "train.hdf5"
    manifest = json.loads((arm_dir / "arms_manifest.json").read_text())
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    results_dir = str(Path(args.results_dir).expanduser())

    launch_lines = []
    for key in sorted(manifest["arms"]):
        seed = int(key.split("seed")[-1])
        config = json.loads(json.dumps(base))  # deep copy
        config["experiment"]["name"] = f"e2_{experiment.task}_{key}"
        config["experiment"]["rollout"]["enabled"] = False  # CPU-bound; eval via frozen resets
        config["experiment"]["save"]["enabled"] = True
        config["train"]["data"] = str(train_hdf5)
        config["train"]["hdf5_filter_key"] = key
        config["train"]["seed"] = seed
        config["train"]["output_dir"] = results_dir
        config["train"]["num_data_workers"] = 1  # 4 cores shared across concurrent runs
        path = out_dir / f"e2_{experiment.task}_{key}.json"
        path.write_text(json.dumps(config, indent=4))
        launch_lines.append(str(path))

    manifest_path = out_dir / f"launch_{experiment.task}.txt"
    manifest_path.write_text("\n".join(launch_lines) + "\n")
    print(f"wrote {len(launch_lines)} configs for {experiment.task} -> {manifest_path}")


if __name__ == "__main__":
    main()
