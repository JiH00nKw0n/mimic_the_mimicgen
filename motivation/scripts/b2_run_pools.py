"""B2 driver: generate the E2 attempt pools (PLAN.md §2.4).

For each e2 experiment config: pool.num_attempts TOTAL split across
pool.seeds, one generation run per (task, seed), N-way parallel. Output goes
under --out-root/<task>_<variant>/seed<seed>/ regardless of the yaml's
relative paths (those matter at extraction time). Resumable per run; writes
B2_DRIVER_DONE.json at the end.

Server usage:
  PYTHONPATH=<repo>/motivation nohup <venv python> b2_run_pools.py \
      --e2-config-dir <repo>/motivation/configs/experiments \
      --task-config-dir <repo>/motivation/configs/tasks \
      --templates <...>/exps/templates/robosuite --sources <...>/datasets/source \
      --out-root ~/mimicgen_jihoonkwon/experiments/motivation_ic/e2_pools \
      --jobs 4 > /tmp/b2_driver.log 2>&1 &
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from genaudit.config import ExperimentSpec, load_experiment_spec, load_task_spec
from genaudit.envs.robosuite_variants import variant_class_name
from genaudit.generation.mimicgen_backend import (
    build_generation_config,
    load_template,
    save_config,
)

E2_CONFIG_FILES = ("e2_square.yaml", "e2_threading.yaml", "e2_coffee.yaml", "e2_stack.yaml")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e2-config-dir", required=True)
    parser.add_argument("--task-config-dir", required=True)
    parser.add_argument("--templates", required=True)
    parser.add_argument("--sources", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()

    out_root = Path(args.out_root).expanduser()
    logs = out_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    specs = []
    for name in E2_CONFIG_FILES:
        experiment = load_experiment_spec(Path(args.e2_config_dir) / name)
        if not isinstance(experiment, ExperimentSpec):
            raise SystemExit(f"{name}: expected an e2 config")
        for seed in experiment.pool_seeds:
            specs.append((experiment, seed))

    def run_one(spec) -> tuple[str, str]:
        experiment, seed = spec
        tag = f"{experiment.task}_{experiment.variant}/seed{seed}"
        try:
            run_dir = out_root / tag
            if list(run_dir.rglob("important_stats.json")):
                print(f"SKIP (done) {tag}", flush=True)
                return tag, "skipped"
            load_task_spec(Path(args.task_config_dir) / f"{experiment.task}.yaml")
            template = load_template(f"{args.templates}/{experiment.task}.json")
            config = build_generation_config(
                template,
                task_name=variant_class_name(experiment.task, experiment.variant),
                source_dataset=f"{args.sources}/{experiment.task}.hdf5",
                output_folder=str(run_dir),
                num_attempts=experiment.attempts_per_pool_seed,
                seed=seed,
            )
            config_path = save_config(config, run_dir / "mg_config.json")
            command = [
                sys.executable, "-m", "genaudit.generation.run_mimicgen",
                "--config", str(config_path),
            ]
            log_path = logs / (tag.replace("/", "_") + ".log")
            with log_path.open("w") as log_file:
                code = subprocess.run(command, stdout=log_file, stderr=subprocess.STDOUT).returncode
            print(f"{'OK  ' if code == 0 else 'FAIL'} {tag}", flush=True)
            return tag, "ok" if code == 0 else f"exit {code}"
        except Exception as error:  # noqa: BLE001 - driver must keep going
            print(f"FAIL {tag} (spec error: {type(error).__name__}: {error})", flush=True)
            return tag, f"spec error: {error}"

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        results = dict(pool.map(run_one, specs))

    failed = {tag: status for tag, status in results.items() if status not in ("ok", "skipped")}
    (out_root / "B2_DRIVER_DONE.json").write_text(
        json.dumps({"total": len(results), "failed": failed}, indent=2)
    )
    print(f"b2 driver done: {len(results) - len(failed)}/{len(results)} ok")


if __name__ == "__main__":
    main()
