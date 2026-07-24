"""Baseline vs transform_uniform SR as a function of d_eval (per-task quartiles).

Tests the 'specialization' hypothesis: baseline's training set is survivor-
skewed toward near-source demos, so IF each arm is best where its training data
concentrates, baseline should win the nearest quartile (Q1). Reuses deval.json
+ eval jsonls (no retraining). Pooled over available seeds.

Usage: MNEW_SEEDS_ALL=101,102[,...] mnew_quantile.py <task> ...
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

A = Path("/home/ubuntu/mimicgen_jihoonkwon/experiments/motivation_new/e2_arms")
SEEDS = [int(x) for x in os.environ.get("MNEW_SEEDS_ALL", "101,102").split(",")]
NQ = 4


def load_arm(task, arm):
    out = {}
    for s in SEEDS:
        f = A / f"{task}_N2" / "eval" / f"e2_{task}_{arm}_seed{s}.jsonl"
        if not f.exists():
            continue
        recs = [json.loads(x) for x in open(f) if x.strip()]
        if len(recs) < 200:
            continue
        for r in recs:
            out[(s, int(r["reset_index"]))] = bool(r["success"])
    return out


def mcnemar_p(base, treat, keys):
    b = c = 0
    for k in keys:
        if k in base and k in treat:
            if treat[k] and not base[k]:
                b += 1
            elif base[k] and not treat[k]:
                c += 1
    n = b + c
    if n == 0:
        return b, c, 1.0
    try:
        from scipy.stats import binomtest
        p = binomtest(min(b, c), n, 0.5).pvalue
    except Exception:  # noqa: BLE001
        from math import comb
        k0 = min(b, c)
        p = min(1.0, 2 * sum(comb(n, i) for i in range(k0 + 1)) / 2 ** n)
    return b, c, round(float(p), 3)


def analyze(task):
    dfile = A / f"{task}_N2" / "deval.json"
    if not dfile.exists():
        print(f"{task}: no deval.json"); return
    dev = {int(k): float(v) for k, v in json.load(open(dfile))["d_eval"].items()}
    resets = sorted(dev)
    d = np.array([dev[r] for r in resets])
    edges = np.quantile(d, [i / NQ for i in range(1, NQ)])
    qof = {r: int(np.searchsorted(edges, dev[r], side="right")) for r in resets}
    qsets = {q: [r for r in resets if qof[r] == q] for q in range(NQ)}

    base = load_arm(task, "baseline")
    trans = load_arm(task, "transform_uniform")
    anc = load_arm(task, "ancestry_balanced")
    has_anc = bool(anc)

    def sr(cells, subset):
        v = [cells[(s, r)] for s in SEEDS for r in subset if (s, r) in cells]
        return (np.mean(v), len(v)) if v else (float("nan"), 0)

    print(f"\n{task}  (seeds={SEEDS}; d_eval quartile edges="
          f"{[round(float(e),3) for e in edges]})")
    hdr = "  quartile   d_eval band        base   trans  Δt(p)          "
    if has_anc:
        hdr += "anc    Δa(p)"
    print(hdr)
    lo = 0.0
    for q in range(NQ):
        subset = qsets[q]
        band_hi = edges[q] if q < NQ - 1 else max(d)
        keys = [(s, r) for s in SEEDS for r in subset]
        bs, _ = sr(base, subset)
        ts, _ = sr(trans, subset)
        tb, tc, tp = mcnemar_p(base, trans, keys)
        tag = "Q%d(near)" % (q + 1) if q == 0 else ("Q%d(far)" % (q + 1) if q == NQ - 1 else "Q%d" % (q + 1))
        line = (f"  {tag:9s}  [{lo:.3f},{band_hi:.3f}]  n{len(subset):<3d}  "
                f"{bs:.3f}  {ts:.3f}  {ts-bs:+.3f}(b{tb}/c{tc},p{tp})")
        if has_anc:
            as_, _ = sr(anc, subset)
            ab, ac, ap = mcnemar_p(base, anc, keys)
            line += f"   {as_:.3f}  {as_-bs:+.3f}(b{ab}/c{ac},p{ap})"
        print(line)
        lo = band_hi


def main():
    for task in sys.argv[1:]:
        analyze(task)


if __name__ == "__main__":
    main()
