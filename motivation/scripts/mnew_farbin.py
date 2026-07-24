"""motivation_new: far-bin stratified SR + per-episode paired McNemar.

Uses {task}_N2/deval.json (d_eval per frozen reset) to split the 200 eval
states into near/mid/far terciles, then for each arm computes SR per tercile
(pooled over dataset seeds) and the paired McNemar test between baseline and
each treatment arm. The paired unit is (reset_index, seed): both arms are
rolled from the identical frozen reset under the identical dataset seed, so
per-episode pairing is exact.

Reports, per task:
  - per-arm SR overall / near / mid / far
  - baseline vs transform_uniform: b (transform wins), c (baseline wins),
    n_discordant, two-sided exact p  — overall and per tercile
  - same for baseline vs ancestry_balanced

Usage: mnew_farbin.py <task> [<task> ...]
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

A = Path("/home/ubuntu/mimicgen_jihoonkwon/experiments/motivation_new/e2_arms")
ARMS = ["baseline", "transform_uniform", "ancestry_balanced"]
SEEDS = [int(x) for x in os.environ.get("MNEW_SEEDS_ALL", "101,102").split(",")]


def _exact_two_sided_p(b, c):
    """McNemar exact two-sided p: min(b,c) tail of Binomial(b+c, 0.5)."""
    n = b + c
    if n == 0:
        return 1.0
    try:
        from scipy.stats import binomtest
        return float(binomtest(min(b, c), n, 0.5, alternative="two-sided").pvalue)
    except Exception:  # noqa: BLE001
        from math import comb
        k = min(b, c)
        tail = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
        return float(min(1.0, 2 * tail))


def _load_arm(task, arm):
    """{(seed, reset_index): success bool} across available seeds."""
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


def _sr(cells, keys):
    vals = [cells[k] for k in keys if k in cells]
    return float(np.mean(vals)) if vals else None, len(vals)


def _paired(base, treat, keys):
    b = c = 0
    for k in keys:
        if k not in base or k not in treat:
            continue
        bb, tt = base[k], treat[k]
        if tt and not bb:
            b += 1
        elif bb and not tt:
            c += 1
    return {"b_treat_wins": b, "c_base_wins": c, "n_disc": b + c,
            "p": round(_exact_two_sided_p(b, c), 4)}


def analyze(task):
    dfile = A / f"{task}_N2" / "deval.json"
    if not dfile.exists():
        print(f"[far] {task}: no deval.json", flush=True)
        return None
    dev = json.load(open(dfile))["d_eval"]  # {str(reset): d}
    dev = {int(k): float(v) for k, v in dev.items()}
    resets = sorted(dev)
    d = np.array([dev[r] for r in resets])
    lo, hi = np.quantile(d, [1 / 3, 2 / 3])
    bin_of = {}
    for r in resets:
        bin_of[r] = "near" if dev[r] <= lo else ("far" if dev[r] > hi else "mid")
    bins = {name: [r for r in resets if bin_of[r] == name]
            for name in ("near", "mid", "far")}

    arm_cells = {arm: _load_arm(task, arm) for arm in ARMS}
    arm_cells = {a: c for a, c in arm_cells.items() if c}

    def keys_for(reset_subset):
        return [(s, r) for s in SEEDS for r in reset_subset]

    summary = {"task": task, "n_resets": len(resets),
               "d_eval_edges": {"lo_t33": round(float(lo), 4), "hi_t67": round(float(hi), 4)},
               "bin_counts": {k: len(v) for k, v in bins.items()},
               "arms": {}, "paired": {}}
    for arm, cells in arm_cells.items():
        entry = {}
        for scope, subset in [("overall", resets)] + list(bins.items()):
            sr, n = _sr(cells, keys_for(subset))
            entry[scope] = {"sr": round(sr, 4) if sr is not None else None, "n": n}
        summary["arms"][arm] = entry

    base = arm_cells.get("baseline")
    if base:
        for treat_name in ("transform_uniform", "ancestry_balanced"):
            treat = arm_cells.get(treat_name)
            if not treat:
                continue
            blk = {}
            for scope, subset in [("overall", resets)] + list(bins.items()):
                blk[scope] = _paired(base, treat, keys_for(subset))
            summary["paired"][f"baseline_vs_{treat_name}"] = blk

    (A / f"{task}_N2" / "eval" / "farbin_summary.json").write_text(
        json.dumps(summary, indent=2))

    # compact print
    def row(arm):
        e = summary["arms"].get(arm)
        if not e:
            return f"    {arm:18s} (missing)"
        return (f"    {arm:18s} all={e['overall']['sr']}  "
                f"near={e['near']['sr']}  mid={e['mid']['sr']}  far={e['far']['sr']}")
    print(f"[far] {task}  (d_eval terciles: <= {lo:.3f} | > {hi:.3f}; "
          f"bins n={summary['bin_counts']})", flush=True)
    for arm in ARMS:
        print(row(arm), flush=True)
    for name, blk in summary["paired"].items():
        parts = []
        for scope in ("overall", "near", "mid", "far"):
            m = blk[scope]
            parts.append(f"{scope}:b{m['b_treat_wins']}/c{m['c_base_wins']}(p{m['p']})")
        print(f"    {name}: " + "  ".join(parts), flush=True)
    return summary


def main():
    for task in sys.argv[1:]:
        analyze(task)


if __name__ == "__main__":
    main()
