"""B0: audit the Phase-0 keep_failed pools (PLAN.md §4 B0).

For every (task, variant) pool found under --pools-root:
  1. extract AttemptRecords (needs the ANNOTATED source hdf5),
  2. d_raw vs d_pos trend-preservation comparison (PLAN.md §1.3),
  3. per-bin DGR + Wilson lower bound -> planned E2 pool size (PLAN.md §2.4),
  4. worst-source retained count -> arm-C feasibility check.
Writes one JSON report per pool plus a combined b0_report.json.

Run on the server inside the robosuite_mimicgen venv (only numpy/h5py used):
  PYTHONPATH=<repo>/motivation python <repo>/motivation/scripts/b0_audit_phase0_pools.py \
      --pools-root ~/mimicgen_jihoonkwon/experiments/motivation_ic \
      --sources-root ~/mimicgen_jihoonkwon/datasets/source \
      --out ~/mimicgen_jihoonkwon/experiments/motivation_ic/b0
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from genaudit.analysis.ancestry import ancestry_stats
from genaudit.analysis.dgr import definition_comparison, dgr, plan_pool_size, wilson_lower_bound
from genaudit.config import load_task_spec
from genaudit.envs.bounds import get_variant
from genaudit.factors.initial_condition import build_task_geometry
from genaudit.records.extract import extract_attempt_records, load_source_initial_states
from genaudit.records.schema import write_jsonl

TASK_CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs" / "tasks"


def find_pool_files(pool_dir: Path) -> tuple[Path | None, Path | None]:
    demo = sorted(pool_dir.rglob("demo.hdf5"))
    failed = sorted(pool_dir.rglob("demo_failed.hdf5"))
    return (demo[0] if demo else None, failed[0] if failed else None)


def audit_pool(task: str, variant: str, pool_dir: Path, source_hdf5: Path, out_dir: Path) -> dict:
    task_spec = load_task_spec(TASK_CONFIG_DIR / f"{task}.yaml")
    geometry = build_task_geometry(
        task, get_variant(task, task_spec.widest_variant), task_spec.symmetry_orders
    )
    source_xy, source_yaw = load_source_initial_states(source_hdf5, geometry.movable_objects)
    demo, failed = find_pool_files(pool_dir)
    if demo is None and failed is None:
        raise FileNotFoundError(f"{pool_dir}: no demo.hdf5 / demo_failed.hdf5 found")
    records = extract_attempt_records(
        task=task,
        variant=variant,
        geometry=geometry,
        source_xy=source_xy,
        source_yaw=source_yaw,
        demo_hdf5=demo,
        failed_hdf5=failed,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(records, out_dir / f"{task}_{variant}_attempts.jsonl")

    report: dict = {
        "task": task,
        "variant": variant,
        "pool_dir": str(pool_dir),
        "n_attempts": len(records),
        "dgr": dgr(records),
    }
    try:
        report["trend_by_definition"] = {
            key: asdict(stats) for key, stats in definition_comparison(records).items()
        }
        report["pool_plan_d_pos"] = asdict(plan_pool_size(records, "d_pos"))
        report["pool_plan_d_raw"] = asdict(plan_pool_size(records, "d_raw"))
    except ValueError as error:
        report["trend_unavailable_reason"] = str(error)

    num_sources = len(source_xy)
    ancestry = ancestry_stats(records, num_sources)
    report["ancestry"] = asdict(ancestry)
    worst = min(range(num_sources), key=lambda i: ancestry.per_source_success_rate[i])
    worst_lb = wilson_lower_bound(
        ancestry.retained_counts[worst], max(ancestry.attempted_counts[worst], 1)
    )
    # arm C needs 50 retained per source: N >= 50 * n_src / p_worst (PLAN §2.4)
    report["arm_c_worst_source"] = {
        "source": worst,
        "success_rate_wilson_lb": worst_lb,
        "required_pool_for_50_per_source": int(50 * num_sources / worst_lb)
        if worst_lb > 0
        else None,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pools-root", required=True)
    parser.add_argument("--sources-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--pool",
        action="append",
        default=None,
        help="task:variant:subdir triple, e.g. square:D2:square_D2 (repeatable); "
        "default: autodiscover <task>_<variant> dirs under --pools-root",
    )
    args = parser.parse_args()

    pools_root = Path(args.pools_root).expanduser()
    sources_root = Path(args.sources_root).expanduser()
    out_dir = Path(args.out).expanduser()

    if args.pool:
        triples = [tuple(entry.split(":")) for entry in args.pool]
    else:
        triples = []
        for child in sorted(pools_root.iterdir()):
            if child.is_dir() and "_" in child.name:
                task, _, variant = child.name.rpartition("_")
                if (TASK_CONFIG_DIR / f"{task}.yaml").exists():
                    triples.append((task, variant, child.name))

    reports = []
    for task, variant, subdir in triples:
        source_hdf5 = sources_root / f"{task}.hdf5"
        if not source_hdf5.exists():
            print(f"SKIP {task}_{variant}: source {source_hdf5} missing (annotate first)")
            continue
        print(f"== auditing {task} {variant} ({pools_root / subdir}) ==")
        try:
            report = audit_pool(task, variant, pools_root / subdir, source_hdf5, out_dir)
        except Exception as error:  # keep going; report the failure loudly at the end
            report = {"task": task, "variant": variant, "error": f"{type(error).__name__}: {error}"}
        reports.append(report)
        print(json.dumps({k: v for k, v in report.items() if k not in ("trend_by_definition",)}, indent=1)[:600])

    combined = out_dir / "b0_report.json"
    combined.parent.mkdir(parents=True, exist_ok=True)
    combined.write_text(json.dumps(reports, indent=2))
    print(f"wrote {combined}")


if __name__ == "__main__":
    main()
