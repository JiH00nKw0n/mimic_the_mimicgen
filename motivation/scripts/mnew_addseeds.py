"""motivation_new: add dataset seeds to existing arms WITHOUT rebuilding train.hdf5.

Seeds 101/102 are already trained + evaluated. To grow the paired sample we add
independent draws for new seeds (default 103-106): reconstruct provenance->group
from the existing train.hdf5 (build_merged stamped attrs['provenance'] on each
group), reuse the frozen bins recorded in arms_manifest (k_used) so binning is
identical to the original seeds, sample each arm under the new dataset seed, and
append the filter keys + manifest entries. Nothing about 101/102 is touched.

Usage: MNEW_SEEDS=103,104,105,106 mnew_addseeds.py <task> [<task> ...]
"""
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

REPO = "/home/ubuntu/mimicgen_jihoonkwon/mimic_the_mimicgen/motivation"
CFG = f"{REPO}/configs/tasks"
OUT = Path("/home/ubuntu/mimicgen_jihoonkwon/experiments/motivation_new/e2_arms")
TEMPLATE = f"{REPO}/configs/experiments/e2_square.yaml"
TASKS = ["square", "threading", "coffee", "three_piece_assembly", "stack",
         "stack_three", "mug_cleanup", "hammer_cleanup"]
SEEDS = [int(x) for x in os.environ.get("MNEW_SEEDS", "103,104,105,106").split(",")]

sys.path.insert(0, REPO)
sys.path.insert(0, f"{REPO}/scripts")
import h5py  # noqa: E402
from genaudit.cli import _sample_arm  # noqa: E402
from genaudit.config import load_experiment_spec, load_task_spec  # noqa: E402
from genaudit.curation.binning import assign_bins, compute_quantile_edges  # noqa: E402
from genaudit.curation.filter_keys import write_filter_key  # noqa: E402
from genaudit.curation.samplers import InsufficientPoolError  # noqa: E402
from genaudit.records.schema import distance_value, read_jsonl  # noqa: E402


def add(task, base_exp):
    experiment = replace(base_exp, task=task, variant="N2", dataset_seeds=SEEDS)
    out_dir = OUT / f"{task}_N2"
    merged = out_dir / "train.hdf5"
    manifest = json.loads((out_dir / "arms_manifest.json").read_text())
    k = int(manifest.get("k_used", experiment.binning_k))  # match original binning

    with h5py.File(merged, "r") as f:
        prov2name = {f["data"][g].attrs["provenance"]: g for g in f["data"].keys()}

    records = list(read_jsonl(out_dir / "attempts.jsonl"))
    distances = np.array([distance_value(r, experiment.primary_distance) for r in records])
    retained_idx = [i for i, r in enumerate(records) if r.success]
    retained_sources = np.array([records[i].source_demo_id for i in retained_idx])
    edges = compute_quantile_edges(
        distances, k, metadata={"task": task, "distance_key": experiment.primary_distance})
    retained_bins = assign_bins(distances[retained_idx], edges)

    added = 0
    for ds in SEEDS:
        for arm in experiment.arms:
            key = f"{arm.name}_seed{ds}"
            if key in manifest["arms"]:
                print(f"  {task} {key}: already present, skip", flush=True)
                continue
            try:
                result = _sample_arm(arm, ds, retained_bins, retained_sources, k, experiment)
            except InsufficientPoolError as e:
                print(f"  {task} {key}: SKIP ({e})", flush=True)
                continue
            provs = [records[retained_idx[i]].attempt_id.split("@")[0] for i in result.selected]
            demo_names = [prov2name[p] for p in provs]
            write_filter_key(merged, key, demo_names)
            manifest["arms"][key] = {
                "size": result.size, "per_stratum_counts": list(result.per_stratum_counts),
                "info": result.info, "demo_names": demo_names}
            added += 1
            print(f"  {task} {key}: {result.per_stratum_counts}", flush=True)
    (out_dir / "arms_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"{task}: +{added} arm-seed keys (seeds {SEEDS})", flush=True)


def main():
    base_exp = load_experiment_spec(TEMPLATE)
    for task in (sys.argv[1:] or TASKS):
        try:
            add(task, base_exp)
        except Exception as e:  # noqa: BLE001
            import traceback
            print(f"{task}: ERR {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()


if __name__ == "__main__":
    main()
