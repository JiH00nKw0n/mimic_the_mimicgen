"""genaudit CLI — config-driven pipeline steps.

    python -m genaudit gen-config   --task-config ... --experiment-config ... --variant D2E
    python -m genaudit extract      --task-config ... --experiment-config ...
    python -m genaudit curate       --task-config ... --experiment-config ...
    python -m genaudit filter-keys  --experiment-config ...
    python -m genaudit analyze      --experiment-config ...

Each step reads/writes the paths named in the experiment config, so a step is
re-runnable and auditable in isolation. E1 sweep configs (experiment: e1)
drive gen-config; per-pool extract/analyze use an e2-shaped config (or the
python API) whose paths point at that pool.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from genaudit.config import (
    ArmSpec,
    E1SweepSpec,
    ExperimentSpec,
    TaskSpec,
    load_experiment_spec,
    load_task_spec,
)

# Fixed per-arm seed offsets: every arm draws from its own independent RNG
# stream (seeded by dataset_seed + offset), so changing K, adding an arm, or
# reordering arms never re-rolls the other arms' membership.
ARM_SEED_OFFSETS = {
    "baseline": 0,
    "transform_uniform": 1,
    "ancestry_balanced": 2,
    "raking": 3,
}


def _arm_rng(dataset_seed: int, arm_name: str) -> np.random.Generator:
    if arm_name not in ARM_SEED_OFFSETS:
        raise ValueError(
            f"arm {arm_name!r} has no seed offset; known: {sorted(ARM_SEED_OFFSETS)}"
        )
    return np.random.default_rng([dataset_seed, ARM_SEED_OFFSETS[arm_name]])


def _geometry(task_spec: TaskSpec):
    from genaudit.envs.bounds import get_variant
    from genaudit.factors.initial_condition import build_task_geometry

    return build_task_geometry(
        task_spec.task,
        get_variant(task_spec.task, task_spec.widest_variant),
        task_spec.symmetry_orders,
    )


def _report_problematic_attempts(demo_hdf5: str | None) -> None:
    """mimicgen retries exception-terminated attempts without recording them
    (num_problematic); surface the count so DGR denominators are honest."""
    if not demo_hdf5:
        return
    for candidate in (
        Path(demo_hdf5).parent / "important_stats.json",
        Path(demo_hdf5).parent.parent / "important_stats.json",
    ):
        if candidate.exists():
            stats = json.loads(candidate.read_text())
            problematic = stats.get("num_problematic", "?")
            print(
                f"note: {problematic} problematic (exception-retried) attempts "
                f"excluded from the pool by mimicgen ({candidate})"
            )
            return
    print("note: important_stats.json not found — num_problematic unreported")


def cmd_extract(task_spec: TaskSpec, experiment: ExperimentSpec) -> None:
    from genaudit.records.extract import extract_attempt_records, load_source_initial_states
    from genaudit.records.schema import write_jsonl

    geometry = _geometry(task_spec)
    source_xy, source_yaw = load_source_initial_states(
        task_spec.source_dataset, geometry.movable_objects
    )
    records = extract_attempt_records(
        task=experiment.task,
        variant=experiment.variant,
        geometry=geometry,
        source_xy=source_xy,
        source_yaw=source_yaw,
        demo_hdf5=experiment.paths.get("demo_hdf5"),
        failed_hdf5=experiment.paths.get("failed_hdf5"),
    )
    count = write_jsonl(records, experiment.paths["records"])
    successes = sum(record.success for record in records)
    print(f"extracted {count} attempts ({successes} retained) -> {experiment.paths['records']}")
    _report_problematic_attempts(experiment.paths.get("demo_hdf5"))


def _sample_arm(
    arm: ArmSpec,
    dataset_seed: int,
    retained_bins: np.ndarray,
    retained_sources: np.ndarray,
    k_used: int,
    experiment: ExperimentSpec,
):
    from genaudit.curation.samplers import (
        sample_ancestry_balanced,
        sample_baseline,
        sample_stratified_uniform,
    )

    rng = _arm_rng(dataset_seed, arm.name)
    if arm.name == "baseline":
        return sample_baseline(len(retained_bins), arm.size, rng)
    if arm.name == "transform_uniform":
        if arm.size % k_used != 0:
            raise ValueError(
                f"transform_uniform size {arm.size} not divisible by K={k_used}"
            )
        return sample_stratified_uniform(
            retained_bins,
            k_used,
            arm.size,
            rng,
            arm=arm.name,
            tv_threshold=experiment.tv_threshold,
            min_bin_fraction=experiment.min_bin_fraction,
        )
    if arm.name == "ancestry_balanced":
        num_sources = arm.num_strata  # from size / quota_per_stratum (config truth)
        observed_max = int(retained_sources.max())
        if observed_max >= num_sources:
            raise ValueError(
                f"ancestry_balanced: retained source id {observed_max} >= configured "
                f"source count {num_sources} — fix quota_per_stratum "
                f"(size/quota must equal the number of source demos)"
            )
        # source-exclusion water-filling fallback (PLAN §2.4): a thin source is
        # dropped rather than failing, since 50/source is infeasible at N=6250
        # for the hardest sources (predicted in B0).
        return sample_ancestry_balanced(retained_sources, num_sources, arm.size, rng)
    raise ValueError(f"unknown arm {arm.name!r}")


def cmd_curate(task_spec: TaskSpec, experiment: ExperimentSpec) -> None:
    from genaudit.curation.binning import assign_bins, compute_quantile_edges, save_edges
    from genaudit.curation.samplers import InsufficientPoolError
    from genaudit.records.schema import distance_value, read_jsonl

    records = list(read_jsonl(experiment.paths["records"]))
    distances = np.array(
        [distance_value(record, experiment.primary_distance) for record in records]
    )
    retained = [index for index, record in enumerate(records) if record.success]
    retained_sources = np.array([records[index].source_demo_id for index in retained])

    def frozen_bins(k: int):
        edges = compute_quantile_edges(
            distances,
            k,
            metadata={
                "task": experiment.task,
                "variant": experiment.variant,
                "distance_key": experiment.primary_distance,
            },
        )
        return edges, assign_bins(distances[retained], edges)

    manifest: dict = {"edges": experiment.paths["edges"], "arms": {}}
    out_dir = Path(experiment.paths["out_dir"])

    k_used = experiment.binning_k
    edges, retained_bins = frozen_bins(k_used)
    try:
        results = [
            (arm, dataset_seed, _sample_arm(arm, dataset_seed, retained_bins, retained_sources, k_used, experiment))
            for dataset_seed in experiment.dataset_seeds
            for arm in experiment.arms
        ]
    except InsufficientPoolError as error:
        if experiment.fallback_k is None:
            raise
        print(f"K={k_used} infeasible ({error}); applying pre-registered fallback K={experiment.fallback_k}")
        k_used = experiment.fallback_k
        edges, retained_bins = frozen_bins(k_used)
        results = [
            (arm, dataset_seed, _sample_arm(arm, dataset_seed, retained_bins, retained_sources, k_used, experiment))
            for dataset_seed in experiment.dataset_seeds
            for arm in experiment.arms
        ]

    save_edges(edges, experiment.paths["edges"])
    manifest["k_used"] = k_used
    for arm, dataset_seed, result in results:
        selected_records = [records[retained[index]] for index in result.selected]
        key = f"{arm.name}_seed{dataset_seed}"
        manifest["arms"][key] = {
            "size": result.size,
            "per_stratum_counts": list(result.per_stratum_counts),
            "certification": asdict(result.certification) if result.certification else None,
            # demo_names are what robomimic filter keys need (retained demos
            # all live in demo.hdf5); attempt_ids kept for auditability.
            "demo_names": [record.attempt_id.split("@")[0] for record in selected_records],
            "attempt_ids": [record.attempt_id for record in selected_records],
        }
        print(f"{key}: {result.per_stratum_counts}")
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "arms_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"wrote {manifest_path}")
    print("next: python -m genaudit filter-keys --experiment-config <same config>")


def cmd_filter_keys(experiment: ExperimentSpec) -> None:
    """Write one robomimic mask per (arm, dataset_seed) into demo.hdf5."""
    from genaudit.curation.filter_keys import write_filter_key

    manifest_path = Path(experiment.paths["out_dir"]) / "arms_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"{manifest_path} — run curate first")
    manifest = json.loads(manifest_path.read_text())
    demo_hdf5 = experiment.paths["demo_hdf5"]
    for key, arm in manifest["arms"].items():
        write_filter_key(demo_hdf5, key, arm["demo_names"])
        print(f"mask/{key}: {len(arm['demo_names'])} demos -> {demo_hdf5}")


def _write_generation_config(
    task_spec: TaskSpec, variant: str, num_attempts: int, seed: int, out_dir: str
) -> Path:
    from genaudit.envs.robosuite_variants import variant_class_name
    from genaudit.generation.mimicgen_backend import (
        build_generation_config,
        load_template,
        save_config,
    )

    template = load_template(task_spec.generation_template)
    config = build_generation_config(
        template,
        task_name=variant_class_name(task_spec.task, variant),
        source_dataset=task_spec.source_dataset,
        output_folder=str(Path(out_dir) / f"{variant}_seed{seed}"),
        num_attempts=num_attempts,
        seed=seed,
    )
    path = save_config(config, Path(out_dir) / f"mg_{variant}_seed{seed}.json")
    print(f"wrote {path}")
    print(f"run on server: python -m genaudit.generation.run_mimicgen --config {path}")
    return path


def cmd_gen_config(
    task_spec: TaskSpec, experiment: ExperimentSpec | E1SweepSpec, variant: str
) -> None:
    if isinstance(experiment, E1SweepSpec):
        if task_spec.task not in experiment.tasks:
            raise ValueError(
                f"task {task_spec.task!r} not in the E1 sweep ({sorted(experiment.tasks)})"
            )
        if variant not in experiment.tasks[task_spec.task]:
            raise ValueError(
                f"variant {variant!r} not planned for {task_spec.task} "
                f"(planned: {experiment.tasks[task_spec.task]})"
            )
        _write_generation_config(
            task_spec,
            variant,
            experiment.num_attempts,
            experiment.seed,
            experiment.out_dir(task_spec.task, variant),
        )
        return
    # E2: pool.num_attempts is the TOTAL, split evenly across pool seeds and
    # merged after generation (mimicgen scripts/merge_hdf5.py).
    for seed in experiment.pool_seeds:
        _write_generation_config(
            task_spec,
            variant,
            experiment.attempts_per_pool_seed,
            seed,
            experiment.paths["out_dir"],
        )


def cmd_analyze(experiment: ExperimentSpec) -> None:
    from genaudit.analysis.ancestry import ancestry_stats
    from genaudit.analysis.dgr import definition_comparison, dgr, dgr_vs_distance
    from genaudit.records.schema import read_jsonl

    records = list(read_jsonl(experiment.paths["records"]))
    num_sources = max(record.source_demo_id for record in records) + 1
    report: dict = {"n_attempts": len(records), "dgr": dgr(records)}
    # High-DGR control pools (near-100% success) make trend statistics
    # degenerate; report DGR/ancestry regardless and record the reason.
    try:
        report["dgr_curve"] = asdict(
            dgr_vs_distance(records, experiment.primary_distance, experiment.binning_k)
        )
        report["trend_by_definition"] = {
            key: asdict(stats)
            for key, stats in definition_comparison(records, k=experiment.binning_k).items()
        }
    except ValueError as error:
        report["dgr_curve"] = None
        report["trend_by_definition"] = None
        report["trend_unavailable_reason"] = str(error)
    report["ancestry"] = asdict(ancestry_stats(records, num_sources))
    out = Path(experiment.paths["out_dir"]) / "analysis.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"DGR={report['dgr']:.3f}  ancestry skew={report['ancestry']['skew_pp']:.1f}pp")
    print(f"wrote {out}")


def _require_e2(experiment: ExperimentSpec | E1SweepSpec, command: str) -> ExperimentSpec:
    if not isinstance(experiment, ExperimentSpec):
        raise ValueError(f"{command} requires an e2 experiment config, got e1 sweep")
    return experiment


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="genaudit")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("extract", "curate", "gen-config", "filter-keys", "analyze"):
        command = sub.add_parser(name)
        if name in ("extract", "curate", "gen-config"):
            command.add_argument("--task-config", required=True)
        command.add_argument("--experiment-config", required=True)
        if name == "gen-config":
            command.add_argument("--variant", required=True)
    args = parser.parse_args(argv)

    experiment = load_experiment_spec(args.experiment_config)
    if args.command == "extract":
        cmd_extract(load_task_spec(args.task_config), _require_e2(experiment, "extract"))
    elif args.command == "curate":
        cmd_curate(load_task_spec(args.task_config), _require_e2(experiment, "curate"))
    elif args.command == "gen-config":
        cmd_gen_config(load_task_spec(args.task_config), experiment, args.variant)
    elif args.command == "filter-keys":
        cmd_filter_keys(_require_e2(experiment, "filter-keys"))
    elif args.command == "analyze":
        cmd_analyze(_require_e2(experiment, "analyze"))


if __name__ == "__main__":
    main()
