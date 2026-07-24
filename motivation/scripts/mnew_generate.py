"""motivation_new full generation, load-balanced.

Per task: N0/N1 = 500 attempts, N2 = 6250 (serves both the E1 top rung and E2
arm subsampling). Each pool is split into 500-attempt CHUNKS (distinct seeds)
so no single pool monopolises a core (a 6250 pool would be a ~15h long-pole);
all chunks run 8-wide, then each pool's chunks are merged into one standard
demo.hdf5 / demo_failed.hdf5 so downstream extraction is unchanged.

Resumable: chunks with important_stats.json are skipped; merged pools with a
MERGED marker are skipped.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from genaudit.config import load_task_spec
from genaudit.envs.robosuite_variants import variant_class_name
from genaudit.generation.mimicgen_backend import (
    build_generation_config, load_template, save_config,
)

# coffee_preparation dropped: 5-stage / horizon-800 makes it ~39s/attempt
# (5x the others) — it alone was ~40% of the whole generation budget.
TASKS = ["square", "threading", "coffee", "three_piece_assembly", "stack",
         "stack_three", "mug_cleanup", "hammer_cleanup"]
ATTEMPTS = {"N0": 500, "N1": 500, "N2": 6250}
CHUNK = 500
# slow tasks first (LPT) — contact-rich coffee/mug first
SLOW_RANK = {"coffee": 0, "mug_cleanup": 1, "square": 2}


def chunk_seeds(total: int) -> list[tuple[int, int]]:
    """(seed, n) chunks covering `total` attempts in CHUNK-sized pieces."""
    out = []
    seed = 1
    remaining = total
    while remaining > 0:
        n = min(CHUNK, remaining)
        out.append((seed, n))
        remaining -= n
        seed += 1
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task-config-dir", required=True)
    p.add_argument("--templates", required=True)
    p.add_argument("--sources", required=True)
    p.add_argument("--out-root", required=True)
    p.add_argument("--jobs", type=int, default=8)
    args = p.parse_args()
    out = Path(args.out_root).expanduser()
    (out / "logs").mkdir(parents=True, exist_ok=True)

    jobs = []  # (task, variant, seed, n)
    for task in TASKS:
        for variant in ("N2", "N1", "N0"):
            for seed, n in chunk_seeds(ATTEMPTS[variant]):
                jobs.append((task, variant, seed, n))
    # longest variant first, slow tasks first — keeps all cores busy to the end
    jobs.sort(key=lambda j: (-ATTEMPTS[j[1]], SLOW_RANK.get(j[0], 9), j[0], j[2]))

    def run_chunk(job):
        task, variant, seed, n = job
        tag = f"{task}_{variant}"
        chunk_dir = out / tag / f"chunk_seed{seed}"
        if list(chunk_dir.rglob("important_stats.json")):
            return (job, "skip")
        try:
            load_task_spec(Path(args.task_config_dir) / f"{task}.yaml")
            template = load_template(f"{args.templates}/{task}.json")
            cfg = build_generation_config(
                template, task_name=variant_class_name(task, variant),
                source_dataset=f"{args.sources}/{task}.hdf5",
                output_folder=str(chunk_dir), num_attempts=n, seed=seed)
            cfg_path = save_config(cfg, chunk_dir / "mg_config.json")
            log = out / "logs" / f"{tag}_s{seed}.log"
            with open(log, "w") as lf:
                code = subprocess.run(
                    [sys.executable, "-m", "genaudit.generation.run_mimicgen",
                     "--config", str(cfg_path)],
                    stdout=lf, stderr=subprocess.STDOUT).returncode
            return (job, "ok" if code == 0 else f"exit{code}")
        except Exception as error:  # noqa: BLE001
            return (job, f"err {type(error).__name__}: {error}")

    print(f"[gen] {len(jobs)} chunks across {args.jobs} workers", flush=True)
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(run_chunk, jobs))
    for job, status in results:
        if status not in ("ok", "skip"):
            print(f"[gen] FAIL {job}: {status}", flush=True)
    failed = [(j, s) for j, s in results if s not in ("ok", "skip")]
    (out / "GEN_CHUNKS_DONE.json").write_text(
        json.dumps({"total": len(results), "failed": len(failed)}, indent=2))
    print(f"[gen] chunks done: {len(results) - len(failed)}/{len(results)} ok", flush=True)


if __name__ == "__main__":
    main()
