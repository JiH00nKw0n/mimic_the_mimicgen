"""Transform-distance (d_pos) profile of each arm's TRAINING set.

Tests the mechanism claim: ancestry_balanced (equal per source) pulls the
selected demos toward SMALLER transform distance than baseline, because it
up-weights sources whose survivors are all near (thin, near-only pools) and
down-weights dominant sources (whose survivors reach farther). Reconstructs
each arm's selected demos -> their d_pos from attempts.jsonl + train.hdf5
provenance + arms_manifest, pooled over the given seeds.

Reports per task: mean d_pos and % of selected demos in the NEAREST frozen
quantile bin (bin 0), for baseline / transform_uniform / ancestry_balanced,
plus the attempted and retained pools for reference.

Usage: MNEW_SEEDS_ALL=101,102 mnew_armdist.py <task> ...
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = "/home/ubuntu/mimicgen_jihoonkwon/mimic_the_mimicgen/motivation"
A = Path("/home/ubuntu/mimicgen_jihoonkwon/experiments/motivation_new/e2_arms")
SEEDS = [int(x) for x in os.environ.get("MNEW_SEEDS_ALL", "101,102").split(",")]
ARMS = ["baseline", "transform_uniform", "ancestry_balanced"]

sys.path.insert(0, REPO)
import h5py  # noqa: E402
from genaudit.curation.binning import assign_bins, compute_quantile_edges  # noqa: E402
from genaudit.records.schema import read_jsonl  # noqa: E402


def analyze(task):
    out_dir = A / f"{task}_N2"
    records = list(read_jsonl(out_dir / "attempts.jsonl"))
    d_att = np.array([r.d_pos for r in records])
    manifest = json.loads((out_dir / "arms_manifest.json").read_text())
    k = int(manifest.get("k_used", 5))
    edges = compute_quantile_edges(d_att, k, metadata={"task": task, "distance_key": "d_pos"})
    near_hi = edges.interior_edges[0]  # upper edge of nearest bin

    # provenance ("demo_X") -> d_pos for retained (success) records
    prov2dpos = {r.attempt_id.split("@")[0]: r.d_pos for r in records if r.success}
    d_ret = np.array(list(prov2dpos.values()))
    with h5py.File(out_dir / "train.hdf5", "r") as f:
        name2prov = {g: f["data"][g].attrs["provenance"] for g in f["data"].keys()}

    def arm_dpos(arm):
        vals = []
        for s in SEEDS:
            key = f"{arm}_seed{s}"
            if key not in manifest["arms"]:
                continue
            for g in manifest["arms"][key]["demo_names"]:
                prov = name2prov[g]
                if prov in prov2dpos:
                    vals.append(prov2dpos[prov])
        return np.array(vals)

    def line(label, d):
        if len(d) == 0:
            return f"    {label:22s} (none)"
        bins = assign_bins(d, edges)
        frac = np.array([np.mean(bins == i) for i in range(k)])
        tv = 0.5 * np.sum(np.abs(frac - 1.0 / k))
        hist = " ".join(f"{100*f:4.1f}" for f in frac)
        return (f"    {label:22s} mean={d.mean():.3f}  bins%[{hist}]  "
                f"TV(uniform)={tv:.3f}  (n={len(d)})")

    print(f"\n{task}  (nearest-bin upper edge d_pos={near_hi:.3f}; k={k})")
    print(line("attempted(all)", d_att))
    print(line("retained(pool)", d_ret))
    for arm in ARMS:
        print(line(arm, arm_dpos(arm)))


def main():
    for task in sys.argv[1:]:
        try:
            analyze(task)
        except Exception as e:  # noqa: BLE001
            import traceback
            print(f"{task}: ERR {type(e).__name__}: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
