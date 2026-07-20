"""C-stage preparation: pool seeds -> extract -> bin -> arms -> filter keys.

For one E2 task config:
  1. extract AttemptRecords from every (seed) pool, each seed prefixed so ids
     are unique, into one records.jsonl;
  2. build ONE merged training hdf5 holding every RETAINED demo across seeds
     (group name = "{prefix}{orig}", matching the retained records' ids);
  3. freeze K quantile edges on the pooled attempted distance distribution;
  4. sample the arms (baseline / transform_uniform / ancestry_balanced) with
     independent per-arm RNG streams; certify uniformity;
  5. write one robomimic filter key per (arm, dataset seed) into the merged
     hdf5, and an arms_manifest.json.

Everything after step 2 is auditable from the written artifacts.

Server usage:
  PYTHONPATH=<repo>/motivation python c_prepare_arms.py \
      --experiment-config <repo>/motivation/configs/experiments/e2_square.yaml \
      --task-config <repo>/motivation/configs/tasks/square.yaml \
      --pools-root ~/mimicgen_jihoonkwon/experiments/motivation_ic/e2_pools \
      --sources ~/mimicgen_jihoonkwon/experiments/motivation_ic/b0_sources \
      --out-root ~/mimicgen_jihoonkwon/experiments/motivation_ic/e2_arms
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from genaudit.cli import _arm_rng, _sample_arm
from genaudit.config import load_experiment_spec, load_task_spec
from genaudit.curation.binning import assign_bins, compute_quantile_edges, save_edges
from genaudit.curation.filter_keys import write_filter_key
from genaudit.envs.bounds import get_variant
from genaudit.factors.initial_condition import build_task_geometry
from genaudit.records.extract import extract_attempt_records, load_source_initial_states
from genaudit.records.schema import distance_value, read_jsonl, write_jsonl


def _require_h5py():
    import h5py
    return h5py


def find_pool_files(seed_dir: Path) -> tuple[Path | None, Path | None]:
    demo = sorted(seed_dir.rglob("demo.hdf5"))
    failed = sorted(seed_dir.rglob("demo_failed.hdf5"))
    return (demo[0] if demo else None, failed[0] if failed else None)


def build_merged_training_hdf5(
    seed_demo_paths: dict[str, Path], out_path: Path
) -> dict[str, str]:
    """Copy every retained demo group into one file, named demo_0..N-1.

    robomimic sorts demos by int(name[5:]), so group names must be exactly
    'demo_<int>'. The original '{prefix}{orig}' provenance (which seed +
    original index each demo came from) is kept as a group attr and returned
    as a mapping so arm selections can be translated to the new names.
    """
    h5py = _require_h5py()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    provenance_to_name: dict[str, str] = {}
    with h5py.File(out_path, "w") as dst:
        data = dst.create_group("data")
        total_samples = 0
        running = 0
        for prefix, demo_path in seed_demo_paths.items():
            with h5py.File(demo_path, "r") as src:
                src_data = src["data"]
                for name in src_data:
                    new_name = f"demo_{running}"
                    src.copy(src_data[name], data, name=new_name)
                    provenance = f"{prefix}{name}"
                    data[new_name].attrs["provenance"] = provenance
                    provenance_to_name[provenance] = new_name
                    total_samples += int(src_data[name]["actions"].shape[0])
                    running += 1
                if "env_args" in src_data.attrs and "env_args" not in data.attrs:
                    data.attrs["env_args"] = src_data.attrs["env_args"]
        data.attrs["total"] = running
        data.attrs["num_samples"] = total_samples
    return provenance_to_name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-config", required=True)
    parser.add_argument("--task-config", required=True)
    parser.add_argument("--pools-root", required=True)
    parser.add_argument("--sources", required=True)
    parser.add_argument("--out-root", required=True)
    args = parser.parse_args()

    experiment = load_experiment_spec(args.experiment_config)
    task_spec = load_task_spec(args.task_config)
    geometry = build_task_geometry(
        task_spec.task, get_variant(task_spec.task, task_spec.widest_variant),
        task_spec.symmetry_orders,
    )
    source_xy, source_yaw = load_source_initial_states(
        f"{args.sources}/{task_spec.task}.hdf5", geometry.movable_objects
    )

    pools_root = Path(args.pools_root).expanduser()
    out_dir = Path(args.out_root).expanduser() / f"{experiment.task}_{experiment.variant}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1 + 2: extract records per seed, gather retained demo files for merge
    all_records = []
    seed_demo_paths: dict[str, Path] = {}
    for seed in experiment.pool_seeds:
        seed_dir = pools_root / f"{experiment.task}_{experiment.variant}" / f"seed{seed}"
        demo, failed = find_pool_files(seed_dir)
        if demo is None:
            raise FileNotFoundError(f"{seed_dir}: no demo.hdf5")
        prefix = f"s{seed}_"
        seed_demo_paths[prefix] = demo
        all_records.extend(
            extract_attempt_records(
                task=experiment.task, variant=experiment.variant, geometry=geometry,
                source_xy=source_xy, source_yaw=source_yaw,
                demo_hdf5=demo, failed_hdf5=failed, attempt_id_prefix=prefix,
            )
        )
    records_path = out_dir / "attempts.jsonl"
    write_jsonl(all_records, records_path)
    merged_hdf5 = out_dir / "train.hdf5"
    # provenance ("s{seed}_{orig}") -> merged group name ("demo_{i}")
    provenance_to_name = build_merged_training_hdf5(seed_demo_paths, merged_hdf5)
    n_retained = sum(r.success for r in all_records)
    print(f"{experiment.task}: {len(all_records)} attempts, {n_retained} retained "
          f"(merged {len(provenance_to_name)} demos -> {merged_hdf5})")

    # 3: freeze edges on the pooled attempted distance distribution
    records = list(read_jsonl(records_path))
    distances = np.array([distance_value(r, experiment.primary_distance) for r in records])
    retained_idx = [i for i, r in enumerate(records) if r.success]
    retained_sources = np.array([records[i].source_demo_id for i in retained_idx])

    manifest = {"task": experiment.task, "variant": experiment.variant,
                "n_attempts": len(records), "n_retained": n_retained,
                "train_hdf5": str(merged_hdf5), "arms": {}}

    def frozen_bins(k):
        edges = compute_quantile_edges(
            distances, k, metadata={"task": experiment.task, "distance_key": experiment.primary_distance})
        return edges, assign_bins(distances[retained_idx], edges)

    k_used = experiment.binning_k
    edges, retained_bins = frozen_bins(k_used)
    from genaudit.curation.samplers import InsufficientPoolError
    h5py = _require_h5py()

    # 4 + 5: sample arms, write filter keys
    def run_arms(k, edges, retained_bins):
        save_edges(edges, out_dir / "bin_edges.json")
        manifest["k_used"] = k
        with h5py.File(merged_hdf5, "r") as f:
            group_names = set(f["data"].keys())
        for dataset_seed in experiment.dataset_seeds:
            for arm in experiment.arms:
                result = _sample_arm(arm, dataset_seed, retained_bins, retained_sources, k, experiment)
                # attempt_id ("s1_demo_5@demo.hdf5") -> provenance -> merged name
                provenances = [records[retained_idx[i]].attempt_id.split("@")[0] for i in result.selected]
                demo_names = [provenance_to_name[p] for p in provenances]
                missing = [n for n in demo_names if n not in group_names]
                if missing:
                    raise KeyError(f"{arm.name}: {len(missing)} names absent from merged hdf5 (e.g. {missing[:3]})")
                key = f"{arm.name}_seed{dataset_seed}"
                write_filter_key(merged_hdf5, key, demo_names)
                manifest["arms"][key] = {
                    "size": result.size,
                    "per_stratum_counts": list(result.per_stratum_counts),
                    "certification": asdict(result.certification) if result.certification else None,
                    "info": result.info,  # ancestry arm: excluded sources, n_eff
                    "demo_names": demo_names,
                }
                print(f"  {key}: {result.per_stratum_counts}")

    try:
        run_arms(k_used, edges, retained_bins)
    except InsufficientPoolError as error:
        if experiment.fallback_k is None:
            raise
        print(f"  K={k_used} infeasible ({error}); falling back to K={experiment.fallback_k}")
        k_used = experiment.fallback_k
        edges, retained_bins = frozen_bins(k_used)
        manifest["arms"] = {}
        run_arms(k_used, edges, retained_bins)

    (out_dir / "arms_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {out_dir / 'arms_manifest.json'}")


if __name__ == "__main__":
    main()
