"""Feasibility gate for the multi-task reinforcement of the controlled experiments.
For every task with an E2 pool: per-source DGR + retained counts, quintile bin
occupancy per source, and candidate source-pairs for each EXP family:
  quality (C2-style): hi-DGR pair vs mid-DGR pair, both with per-bin union >= quota
  sufficiency (D-style): two DGR-matched pairs with disjoint sources
  density/coverage (A/B-style): near/mid/far pool sizes for the hi pair
Prints JSON per task.
"""
import json, sys
from pathlib import Path
from itertools import combinations
import numpy as np

sys.path.insert(0, "/home/ubuntu/mimicgen_jihoonkwon/mimic_the_mimicgen/motivation")
from genaudit.records.schema import read_jsonl

NEW = Path("/home/ubuntu/mimicgen_jihoonkwon/experiments/motivation_new/e2_arms")
TASKS = ["square", "coffee", "three_piece_assembly", "stack", "hammer_cleanup",
         "stack_three", "threading", "mug_cleanup"]

out = {}
for task in TASKS:
    f = NEW / f"{task}_N2" / "attempts.jsonl"
    if not f.exists():
        out[task] = {"error": "no pool"}
        continue
    recs = list(read_jsonl(f))
    src_a = np.array([r.source_demo_id for r in recs])
    su = np.array([r.success for r in recs])
    ret = [r for r in recs if r.success]
    src = np.array([r.source_demo_id for r in ret])
    d = np.array([r.d_pos for r in ret])
    edges = np.quantile(d, [.2, .4, .6, .8])
    binid = np.searchsorted(edges, d, side="right")
    nsrc = int(src_a.max()) + 1
    dgr = {s: round(float(su[src_a == s].mean()), 3) for s in range(nsrc)}
    per_bin = {s: [int(((src == s) & (binid == b)).sum()) for b in range(5)] for s in range(nsrc)}

    def pair_quota(pair):
        return min(sum(per_bin[s][b] for s in pair) for b in range(5))

    ranked = sorted(range(nsrc), key=lambda s: -dgr[s])
    # quality: hi pair = best 2 feasible; mid pair = next distinct 2 with clearly lower DGR
    pairs = []
    for pair in combinations(ranked, 2):
        q = pair_quota(pair)
        if q >= 40:
            pairs.append({"pair": list(pair), "mean_dgr": round((dgr[pair[0]] + dgr[pair[1]]) / 2, 3), "quota": q})
    pairs.sort(key=lambda p: -p["mean_dgr"])
    hi = pairs[0] if pairs else None
    mid = next((p for p in pairs if hi and not set(p["pair"]) & set(hi["pair"])
                and p["mean_dgr"] <= hi["mean_dgr"] - 0.15), None)
    # sufficiency: two disjoint pairs with |dgr gap| <= 0.04, maximize quota
    dpairs = None
    best = -1
    for p1, p2 in combinations(pairs, 2):
        if set(p1["pair"]) & set(p2["pair"]):
            continue
        if abs(p1["mean_dgr"] - p2["mean_dgr"]) > 0.04:
            continue
        q = min(p1["quota"], p2["quota"])
        if q > best:
            best = q
            dpairs = [p1, p2]
    # density/coverage pools for hi pair
    ab = None
    if hi:
        hp = hi["pair"]
        near = sum(per_bin[s][b] for s in hp for b in (0, 1))
        midp = sum(per_bin[s][2] for s in hp)
        far = sum(per_bin[s][b] for s in hp for b in (3, 4))
        ab = {"near": near, "mid": midp, "far": far}
    out[task] = {"n_retained": len(ret), "dgr": dgr,
                 "edges": [round(float(e), 4) for e in edges],
                 "hi": hi, "mid": mid, "d_pairs": dpairs, "ab_pools": ab}
print(json.dumps(out, indent=1))
