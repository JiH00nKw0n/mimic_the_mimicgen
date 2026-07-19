"""B1 reachability probe — gate before freezing E-variant bounds (PLAN.md §1.5).

Modes:
  plan     print the probe run list for a (task, variant) as JSON lines
  run      execute ONE probe run (pin object at a position, generate N attempts)
  analyze  aggregate all runs under --runs-root into probe_report.json + verdicts

Acceptance signal is the FIRST-subtask completion rate ("did the gripper reach
and grasp at all"), not task success — task success would conflate
reachability with transform difficulty. A position passes if its reach rate is
within --reach-drop of the interior reference and above --reach-floor.

Server usage (robosuite_mimicgen venv):
  PYTHONPATH=<repo>/motivation python b1_reachability_probe.py run \
      --task threading --variant D2E --object needle --position corner_00 \
      --template <templates>/threading.json --source <sources>/threading.hdf5 \
      --out-root <root>/b1_probe --attempts 50
"""
from __future__ import annotations

import argparse
import json
import runpy
import sys
from pathlib import Path

from genaudit.envs.probe import (
    INTERIOR,
    plan_probe_runs,
    probe_class_name,
    register_probe_variant,
)
from genaudit.generation.mimicgen_backend import (
    build_generation_config,
    load_template,
    save_config,
)


def run_dir(out_root: str, task: str, variant: str, object_name: str | None, position: str) -> Path:
    tag = INTERIOR if position == INTERIOR else f"{object_name}_{position}"
    return Path(out_root).expanduser() / f"{task}_{variant}" / tag


def cmd_plan(args) -> None:
    for spec in plan_probe_runs(args.task, args.variant):
        print(json.dumps({"task": args.task, "variant": args.variant, **spec}))


def cmd_run(args) -> None:
    from genaudit.envs.robosuite_variants import register_custom_variants

    register_custom_variants()
    if args.position == INTERIOR:
        task_name = probe_class_name(args.task, args.variant, None, INTERIOR)
    else:
        if not args.object:
            raise ValueError("--object is required for non-interior positions")
        register_probe_variant(args.task, args.variant, args.object, args.position)
        task_name = probe_class_name(args.task, args.variant, args.object, args.position)

    template = load_template(args.template)
    first_signal = template["task"]["task_spec"]["subtask_1"]["subtask_term_signal"]
    out = run_dir(args.out_root, args.task, args.variant, args.object, args.position)
    out.mkdir(parents=True, exist_ok=True)
    (out / "probe_manifest.json").write_text(
        json.dumps(
            {
                "task": args.task,
                "variant": args.variant,
                "object": args.object,
                "position": args.position,
                "first_subtask_signal": first_signal,
                "attempts": args.attempts,
                "seed": args.seed,
            },
            indent=2,
        )
    )
    config = build_generation_config(
        template,
        task_name=task_name,
        source_dataset=args.source,
        output_folder=str(out),
        num_attempts=args.attempts,
        seed=args.seed,
        experiment_name=task_name,
    )
    config_path = save_config(config, out / "mg_config.json")
    sys.argv = ["generate_dataset.py", "--config", str(config_path)]
    runpy.run_module("mimicgen.scripts.generate_dataset", run_name="__main__")
    # mimicgen catches generation errors and exits 0 ("run failed with error");
    # a run without its stats file did NOT complete — fail loudly.
    if not list(out.rglob("important_stats.json")):
        raise SystemExit(
            f"{task_name}: generation did not complete (no important_stats.json "
            f"under {out}) — see the mimicgen log in that folder"
        )


def _reach_rate(position_dir: Path, signal: str) -> tuple[float, int]:
    """Fraction of episodes (success + failed) whose first-subtask signal ever
    fired. Returns (rate, episodes_counted)."""
    import h5py

    fired = 0
    total = 0
    for name in ("demo.hdf5", "demo_failed.hdf5"):
        for path in sorted(position_dir.rglob(name)):
            with h5py.File(path, "r") as handle:
                if "data" not in handle:
                    continue
                for demo in handle["data"]:
                    signals = handle[f"data/{demo}/datagen_info/subtask_term_signals"]
                    if signal not in signals:
                        raise KeyError(
                            f"{path}:{demo}: signal {signal!r} missing ({sorted(signals)})"
                        )
                    total += 1
                    if signals[signal][()].max() > 0:
                        fired += 1
    if total == 0:
        raise FileNotFoundError(f"{position_dir}: no episodes found in demo hdf5s")
    return fired / total, total


def cmd_analyze(args) -> None:
    runs_root = Path(args.runs_root).expanduser()
    report: dict = {}
    for task_dir in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        if not list(task_dir.glob("*/probe_manifest.json")):
            continue  # not a task dir (e.g. logs/)
        positions = {}
        for position_dir in sorted(p for p in task_dir.iterdir() if p.is_dir()):
            manifest_path = position_dir / "probe_manifest.json"
            if not manifest_path.exists():
                continue
            manifest = json.loads(manifest_path.read_text())
            stats_files = sorted(position_dir.rglob("important_stats.json"))
            stats = json.loads(stats_files[0].read_text()) if stats_files else {}
            try:
                reach, episodes = _reach_rate(position_dir, manifest["first_subtask_signal"])
            except (FileNotFoundError, KeyError) as error:
                positions[position_dir.name] = {
                    "object": manifest["object"],
                    "position": manifest["position"],
                    "error": str(error),
                    "passed": False,
                }
                continue
            positions[position_dir.name] = {
                "object": manifest["object"],
                "position": manifest["position"],
                "reach_rate": reach,
                "episodes": episodes,
                "task_success_rate": stats.get("success_rate"),
                "num_problematic": stats.get("num_problematic"),
            }
        if INTERIOR not in positions or "error" in positions[INTERIOR]:
            report[task_dir.name] = {
                "error": "interior reference run missing or incomplete",
                "positions": positions,
                "bounds_frozen_ok": False,
            }
            print(f"== {task_dir.name}: INTERIOR MISSING/INCOMPLETE — rerun the probe")
            continue
        interior_reach = positions[INTERIOR]["reach_rate"]
        for tag, entry in positions.items():
            if tag == INTERIOR:
                entry["passed"] = True
                continue
            if "passed" in entry:  # incomplete run, already marked failed
                continue
            entry["passed"] = (
                entry["reach_rate"] >= interior_reach - args.reach_drop
                and entry["reach_rate"] >= args.reach_floor
            )
        failed = [tag for tag, entry in positions.items() if not entry["passed"]]
        report[task_dir.name] = {
            "interior_reach_rate": interior_reach,
            "positions": positions,
            "failed_positions": failed,
            "bounds_frozen_ok": not failed,
        }
        print(f"== {task_dir.name}: interior reach {interior_reach:.2f}, "
              f"{'ALL PASS' if not failed else 'FAILED: ' + ', '.join(failed)}")
        for tag, entry in sorted(positions.items()):
            if tag == INTERIOR:
                continue
            if "error" in entry:
                print(f"   {tag:<28} INCOMPLETE: {entry['error'][:70]}")
            else:
                print(f"   {tag:<28} reach {entry['reach_rate']:.2f} "
                      f"({entry['episodes']} eps) success {entry['task_success_rate']}")
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan")
    plan.add_argument("--task", required=True)
    plan.add_argument("--variant", required=True)

    run = sub.add_parser("run")
    run.add_argument("--task", required=True)
    run.add_argument("--variant", required=True)
    run.add_argument("--object", default=None)
    run.add_argument("--position", required=True)
    run.add_argument("--template", required=True)
    run.add_argument("--source", required=True)
    run.add_argument("--out-root", required=True)
    run.add_argument("--attempts", type=int, default=50)
    run.add_argument("--seed", type=int, default=7)

    analyze = sub.add_parser("analyze")
    analyze.add_argument("--runs-root", required=True)
    analyze.add_argument("--out", required=True)
    analyze.add_argument("--reach-drop", type=float, default=0.20)
    analyze.add_argument("--reach-floor", type=float, default=0.30)

    args = parser.parse_args()
    {"plan": cmd_plan, "run": cmd_run, "analyze": cmd_analyze}[args.command](args)


if __name__ == "__main__":
    main()
