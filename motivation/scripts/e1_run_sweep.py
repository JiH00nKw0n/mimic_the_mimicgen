"""E1 driver: generate every (task, variant) pool of the fixed-attempt DGR
sweep (configs/experiments/e1_sweep.yaml) with N parallel workers.

Resumable like the B1 driver (skips pools whose important_stats.json exists);
writes E1_DRIVER_DONE.json when all pools have been attempted. Custom
E-variants must have passed the reachability gate before this runs (PLAN.md
§1.5) — pass --allow-unfrozen to override for smoke purposes only.

Server usage:
  PYTHONPATH=<repo>/motivation nohup <venv python> e1_run_sweep.py \
      --e1-config <repo>/motivation/configs/experiments/e1_sweep.yaml \
      --task-config-dir <repo>/motivation/configs/tasks \
      --templates <...>/exps/templates/robosuite --sources <...>/datasets/source \
      --out-root ~/mimicgen_jihoonkwon/experiments/motivation_ic/e1 \
      --jobs 3 > /tmp/e1_driver.log 2>&1 &
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from genaudit.config import E1SweepSpec, load_experiment_spec, load_task_spec
from genaudit.envs.robosuite_variants import variant_class_name
from genaudit.generation.mimicgen_backend import (
    build_generation_config,
    load_template,
    save_config,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e1-config", required=True)
    parser.add_argument("--task-config-dir", required=True)
    parser.add_argument("--templates", required=True)
    parser.add_argument("--sources", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument(
        "--probe-report",
        default=None,
        help="b1 probe_report.json; E-variants require bounds_frozen_ok there",
    )
    parser.add_argument("--allow-unfrozen", action="store_true")
    args = parser.parse_args()

    sweep = load_experiment_spec(args.e1_config)
    if not isinstance(sweep, E1SweepSpec):
        raise ValueError(f"{args.e1_config} is not an e1 sweep config")
    out_root = Path(args.out_root).expanduser()
    logs = out_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    frozen_ok: dict[str, bool] = {}
    if args.probe_report:
        report = json.loads(Path(args.probe_report).expanduser().read_text())
        frozen_ok = {
            name: bool(entry.get("bounds_frozen_ok")) for name, entry in report.items()
        }

    specs = []
    for task, variants in sweep.tasks.items():
        for variant in variants:
            if variant.endswith("E") and not args.allow_unfrozen:
                # a variant is gate-cleared if it was probed itself OR if the
                # task's widest E-variant (its superset) is frozen — probing
                # the superset covers every subset region (e.g. coffee D1E
                # under the frozen D2E)
                cleared = frozen_ok.get(f"{task}_{variant}") or frozen_ok.get(f"{task}_D2E")
                if not cleared:
                    raise SystemExit(
                        f"{task}_{variant}: reachability gate not passed (probe report "
                        f"{'missing entry' if args.probe_report else 'not given'}); "
                        "pass --probe-report with bounds_frozen_ok=true or --allow-unfrozen"
                    )
            specs.append((task, variant))

    def run_one(spec) -> tuple[str, str]:
        task, variant = spec
        tag = f"{task}_{variant}"
        pool_dir = out_root / tag
        if list(pool_dir.rglob("important_stats.json")):
            print(f"SKIP (done) {tag}", flush=True)
            return tag, "skipped"
        load_task_spec(Path(args.task_config_dir) / f"{task}.yaml")  # loud validation
        template = load_template(f"{args.templates}/{task}.json")
        config = build_generation_config(
            template,
            task_name=variant_class_name(task, variant),
            source_dataset=f"{args.sources}/{task}.hdf5",
            output_folder=str(pool_dir),
            num_attempts=sweep.num_attempts,
            seed=sweep.seed,
        )
        config_path = save_config(config, pool_dir / "mg_config.json")
        command = [
            sys.executable, "-m", "genaudit.generation.run_mimicgen",
            "--config", str(config_path),
        ]
        log_path = logs / f"{tag}.log"
        with log_path.open("w") as log_file:
            code = subprocess.run(command, stdout=log_file, stderr=subprocess.STDOUT).returncode
        print(f"{'OK  ' if code == 0 else 'FAIL'} {tag}", flush=True)
        return tag, "ok" if code == 0 else f"exit {code}"

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        results = dict(pool.map(run_one, specs))

    failed = {tag: status for tag, status in results.items() if status not in ("ok", "skipped")}
    (out_root / "E1_DRIVER_DONE.json").write_text(
        json.dumps({"total": len(results), "failed": failed}, indent=2)
    )
    print(f"e1 driver done: {len(results) - len(failed)}/{len(results)} ok")


if __name__ == "__main__":
    main()
