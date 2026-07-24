"""motivation_new E2 arm preparation for all 8 tasks.

Mirrors scripts.c_prepare_arms but pointed at the single merged N2 pool
(gen/<task>_N2/demo.hdf5) with N2 geometry (bounds_new). For each task builds
train.hdf5 (retained demos), freezes K quantile bins on the attempted d_pos,
samples baseline / transform_uniform / ancestry_balanced for 2 dataset seeds,
writes robomimic filter keys + arms_manifest.json.
"""
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

REPO = "/home/ubuntu/mimicgen_jihoonkwon/mimic_the_mimicgen/motivation"
GEN = Path("/home/ubuntu/mimicgen_jihoonkwon/experiments/motivation_new/gen")
SRC = "/home/ubuntu/mimicgen_jihoonkwon/robosuite_mimicgen/mimicgen/datasets/source"
CFG = f"{REPO}/configs/tasks"
OUT = Path("/home/ubuntu/mimicgen_jihoonkwon/experiments/motivation_new/e2_arms")
TEMPLATE = f"{REPO}/configs/experiments/e2_square.yaml"  # for arm/binning params
SEEDS = [101, 102]
TASKS = ["square", "threading", "coffee", "three_piece_assembly", "stack",
         "stack_three", "mug_cleanup", "hammer_cleanup"]

sys.path.insert(0, REPO)
sys.path.insert(0, f"{REPO}/scripts")
import h5py  # noqa: E402
from c_prepare_arms import build_merged_training_hdf5  # noqa: E402
from genaudit.cli import _sample_arm  # noqa: E402
from genaudit.config import load_experiment_spec, load_task_spec  # noqa: E402
from genaudit.curation.binning import (  # noqa: E402
    assign_bins, compute_quantile_edges, save_edges,
)
from genaudit.curation.filter_keys import write_filter_key  # noqa: E402
from genaudit.curation.samplers import InsufficientPoolError  # noqa: E402
from genaudit.envs.bounds_new import NEW_BOUNDS  # noqa: E402
from genaudit.factors.initial_condition import build_task_geometry  # noqa: E402
from genaudit.records.extract import (  # noqa: E402
    extract_attempt_records, load_source_initial_states,
)
from genaudit.records.schema import distance_value, read_jsonl, write_jsonl  # noqa: E402


def prepare(task, base_exp):
    experiment = replace(base_exp, task=task, variant="N2", dataset_seeds=SEEDS)
    spec = load_task_spec(f"{CFG}/{task}.yaml")
    geometry = build_task_geometry(task, NEW_BOUNDS[task]["N2"], spec.symmetry_orders)
    src_xy, src_yaw = load_source_initial_states(f"{SRC}/{task}.hdf5", geometry.movable_objects)
    pool = GEN / f"{task}_N2"
    out_dir = OUT / f"{task}_N2"
    out_dir.mkdir(parents=True, exist_ok=True)

    records = extract_attempt_records(
        task=task, variant="N2", geometry=geometry, source_xy=src_xy, source_yaw=src_yaw,
        demo_hdf5=pool / "demo.hdf5", failed_hdf5=pool / "demo_failed.hdf5", attempt_id_prefix="")
    write_jsonl(records, out_dir / "attempts.jsonl")
    merged_hdf5 = out_dir / "train.hdf5"
    provenance_to_name = build_merged_training_hdf5({"": pool / "demo.hdf5"}, merged_hdf5)

    records = list(read_jsonl(out_dir / "attempts.jsonl"))
    distances = np.array([distance_value(r, experiment.primary_distance) for r in records])
    retained_idx = [i for i, r in enumerate(records) if r.success]
    retained_sources = np.array([records[i].source_demo_id for i in retained_idx])
    manifest = {"task": task, "variant": "N2", "n_attempts": len(records),
                "n_retained": len(retained_idx), "train_hdf5": str(merged_hdf5), "arms": {}}

    def frozen_bins(k):
        edges = compute_quantile_edges(
            distances, k, metadata={"task": task, "distance_key": experiment.primary_distance})
        return edges, assign_bins(distances[retained_idx], edges)

    def run_arms(k, edges, retained_bins):
        save_edges(edges, out_dir / "bin_edges.json")
        manifest["k_used"] = k
        with h5py.File(merged_hdf5, "r") as f:
            group_names = set(f["data"].keys())
        for dataset_seed in experiment.dataset_seeds:
            for arm in experiment.arms:
                try:
                    result = _sample_arm(arm, dataset_seed, retained_bins, retained_sources, k, experiment)
                except InsufficientPoolError as e:
                    print(f"  {task} {arm.name}_seed{dataset_seed}: SKIP ({e})", flush=True)
                    continue
                provs = [records[retained_idx[i]].attempt_id.split("@")[0] for i in result.selected]
                demo_names = [provenance_to_name[p] for p in provs]
                write_filter_key(merged_hdf5, f"{arm.name}_seed{dataset_seed}", demo_names)
                manifest["arms"][f"{arm.name}_seed{dataset_seed}"] = {
                    "size": result.size, "per_stratum_counts": list(result.per_stratum_counts),
                    "info": result.info, "demo_names": demo_names}
                print(f"  {task} {arm.name}_seed{dataset_seed}: {result.per_stratum_counts}", flush=True)

    k = experiment.binning_k
    edges, rb = frozen_bins(k)
    try:
        run_arms(k, edges, rb)
    except InsufficientPoolError as e:
        if experiment.fallback_k is None:
            raise
        print(f"  {task}: K={k} infeasible ({e}) -> K={experiment.fallback_k}", flush=True)
        k = experiment.fallback_k
        manifest["arms"] = {}
        edges, rb = frozen_bins(k)
        run_arms(k, edges, rb)
    (out_dir / "arms_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"{task}: {len(retained_idx)} retained -> arms written", flush=True)


def main():
    base_exp = load_experiment_spec(TEMPLATE)
    tasks = sys.argv[1:] if len(sys.argv) > 1 else TASKS
    for task in tasks:
        try:
            prepare(task, base_exp)
        except Exception as e:  # noqa: BLE001
            print(f"{task}: ERR {type(e).__name__}: {e}", flush=True)
    (OUT / "ARMS_DONE").touch()
    print("ARMS DONE", flush=True)


if __name__ == "__main__":
    main()
