"""B1 driver: run every reachability-probe generation with N parallel workers.

Resumable: a run whose important_stats.json already exists is skipped, so the
driver can be re-launched after an interruption. Writes DRIVER_DONE.json when
every run has been attempted (the completion signal the laptop polls for).

Server usage:
  PYTHONPATH=<repo>/motivation nohup <venv python> b1_run_all_probes.py \
      --templates <...>/exps/templates/robosuite --sources <...>/datasets/source \
      --out-root ~/mimicgen_jihoonkwon/experiments/motivation_ic/b1_probe \
      --jobs 3 > /tmp/b1_driver.log 2>&1 &
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from genaudit.envs.probe import plan_probe_runs

PAIRS = (
    ("threading", "D2E"),
    ("coffee", "D2E"),
    ("stack", "D2E"),
    ("stack_three", "D2E"),
    ("mug_cleanup", "D2E"),
)


def run_tag(task: str, variant: str, obj: str | None, position: str) -> str:
    return f"{task}_{variant}/{position if obj is None else f'{obj}_{position}'}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--templates", required=True)
    parser.add_argument("--sources", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--attempts", type=int, default=50)
    parser.add_argument("--jobs", type=int, default=3)
    args = parser.parse_args()

    probe_script = Path(__file__).resolve().parent / "b1_reachability_probe.py"
    out_root = Path(args.out_root).expanduser()
    logs = out_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    specs = [
        (task, variant, run["object"], run["position"])
        for task, variant in PAIRS
        for run in plan_probe_runs(task, variant)
    ]

    def run_one(spec) -> tuple[str, str]:
        task, variant, obj, position = spec
        tag = run_tag(task, variant, obj, position)
        run_directory = out_root / tag
        if list(run_directory.rglob("important_stats.json")):
            print(f"SKIP (done) {tag}", flush=True)
            return tag, "skipped"
        command = [
            sys.executable, str(probe_script), "run",
            "--task", task, "--variant", variant, "--position", position,
            "--template", f"{args.templates}/{task}.json",
            "--source", f"{args.sources}/{task}.hdf5",
            "--out-root", str(out_root), "--attempts", str(args.attempts),
        ]
        if obj:
            command += ["--object", obj]
        log_path = logs / (tag.replace("/", "_") + ".log")
        with log_path.open("w") as log_file:
            code = subprocess.run(command, stdout=log_file, stderr=subprocess.STDOUT).returncode
        status = "ok" if code == 0 else f"exit {code}"
        print(f"{'OK  ' if code == 0 else 'FAIL'} {tag} ({status})", flush=True)
        return tag, status

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        results = dict(pool.map(run_one, specs))

    failed = {tag: status for tag, status in results.items() if status not in ("ok", "skipped")}
    summary = {"total": len(results), "failed": failed}
    (out_root / "DRIVER_DONE.json").write_text(json.dumps(summary, indent=2))
    print(f"driver done: {len(results) - len(failed)}/{len(results)} ok; wrote DRIVER_DONE.json")


if __name__ == "__main__":
    main()
