"""Launch all E2 training runs with a fixed GPU concurrency (aidas).

Reads launch_*.txt lists (one config path per line) produced by
c_make_train_configs.py, runs them N-at-a-time under the proven launcher
conditions (venv-only CUDA libs, stdin /dev/null, no in-training rollouts).
Resumable: a run whose result dir already has models/ is skipped. Writes
C_TRAIN_DONE.json when all runs are attempted.

Server usage:
  PYTHONPATH=<repo>/motivation nohup <venv python> c_train_all.py \
      --launch-lists <out-dir>/launch_square.txt <out-dir>/launch_threading.txt ... \
      --results-dir ~/mimicgen_jihoonkwon/experiments/motivation_ic/e2_results \
      --concurrency 3 > /tmp/c_train.log 2>&1 &
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HOME = Path.home()
NV = HOME / "mimicgen_jihoonkwon/robosuite_mimicgen/venv/lib/python3.10/site-packages/nvidia"
VENV_PY = HOME / "mimicgen_jihoonkwon/robosuite_mimicgen/venv/bin/python"
ROBOMIMIC = HOME / "mimicgen_jihoonkwon/robosuite_mimicgen/robomimic"


def _num_epochs(config_path: str) -> int:
    return int(json.loads(Path(config_path).read_text())["train"]["num_epochs"])


def run_one(config_path: str, results_dir: Path) -> tuple[str, str]:
    config = json.loads(Path(config_path).read_text())
    name = config["experiment"]["name"]
    final_epoch = int(config["train"]["num_epochs"])
    # a run is complete only if its FINAL checkpoint exists — a mere models/
    # dir can be a partial (interrupted) run, which must re-run, not skip
    if list(results_dir.rglob(f"{name}/**/model_epoch_{final_epoch}.pth")):
        print(f"SKIP (done) {name}", flush=True)
        return name, "skipped"
    # a partial run left a dir that would make robomimic prompt to overwrite;
    # remove it so the fresh run starts cleanly under stdin=/dev/null
    for stale in results_dir.rglob(f"{name}"):
        if stale.is_dir():
            import shutil
            shutil.rmtree(stale, ignore_errors=True)
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = f"{NV}/cu13/lib:{NV}/cudnn/lib"
    log = results_dir / f"log_{name}.txt"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as sink, open(os.devnull) as devnull:
        code = subprocess.run(
            [str(VENV_PY), "robomimic/scripts/train.py", "--config", config_path],
            cwd=str(ROBOMIMIC), env=env, stdout=sink, stderr=subprocess.STDOUT, stdin=devnull,
        ).returncode
    print(f"{'OK  ' if code == 0 else 'FAIL'} {name}", flush=True)
    return name, "ok" if code == 0 else f"exit {code}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch-lists", nargs="+", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--concurrency", type=int, default=3)
    args = parser.parse_args()

    results_dir = Path(args.results_dir).expanduser()
    configs = []
    for lst in args.launch_lists:
        configs.extend(line.strip() for line in Path(lst).read_text().splitlines() if line.strip())
    print(f"{len(configs)} runs, concurrency {args.concurrency}", flush=True)

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results = dict(pool.map(lambda c: run_one(c, results_dir), configs))

    failed = {n: s for n, s in results.items() if s not in ("ok", "skipped")}
    (results_dir / "C_TRAIN_DONE.json").write_text(
        json.dumps({"total": len(results), "failed": failed}, indent=2)
    )
    print(f"c_train done: {len(results) - len(failed)}/{len(results)} ok", flush=True)


if __name__ == "__main__":
    main()
