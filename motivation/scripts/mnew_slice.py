"""Exhaustive transform_uniform-vs-baseline slicing (6 seed).

The question: sliced by EVERY meaningful cut, is there any region where
transform_uniform differs from baseline? Paired unit = (task, reset, seed).
For each cut reports discordant counts b (transform wins) / c (baseline wins)
and the two-sided exact McNemar p. Writes slice_results.json and prints a
summary. No simulation — reuses deval.json + eval jsonls (success + steps).

Cuts:
  1. per-task d_eval DECILES (10 bins)
  2. per-task per NEAREST-SOURCE (which of 10 source demos the state is closest to)
  3. per-task per-SEED sign consistency (is the direction stable across 6 seeds?)
  4. continuous: Mann-Whitney of d_eval between transform-win vs baseline-win
     discordant pairs (is winning distance-dependent at all?)
  5. efficiency: among co-successes, Wilcoxon on steps (does transform reach
     success faster/slower even when both succeed?)
  6. POOLED across all 8 tasks — overall (max power ~9600 pairs) + by percentile
     decile of within-task d_eval
  7. POOLED across mid-SR tasks (headroom) — overall + deciles
  8. global min-p across every per-task decile cell with Bonferroni correction
"""
import json
import os
import sys
from math import comb
from pathlib import Path

import numpy as np

A = Path("/home/ubuntu/mimicgen_jihoonkwon/experiments/motivation_new/e2_arms")
SEEDS = [101, 102, 103, 104, 105, 106]
TASKS = ["square", "coffee", "three_piece_assembly", "stack", "hammer_cleanup",
         "stack_three", "threading", "mug_cleanup"]
MIDSR = ["square", "coffee", "stack_three", "threading"]  # SR ~0.5-0.68, most headroom
BASE, TREAT = "baseline", "transform_uniform"


def exact_p(b, c):
    n = b + c
    if n == 0:
        return 1.0
    try:
        from scipy.stats import binomtest
        return float(binomtest(min(b, c), n, 0.5, alternative="two-sided").pvalue)
    except Exception:  # noqa: BLE001
        k = min(b, c)
        return float(min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n))


def load(task, arm):
    out = {}
    for s in SEEDS:
        f = A / f"{task}_N2" / "eval" / f"e2_{task}_{arm}_seed{s}.jsonl"
        if not f.exists():
            continue
        recs = [json.loads(x) for x in open(f) if x.strip()]
        if len(recs) < 200:
            continue
        for r in recs:
            out[(s, int(r["reset_index"]))] = (bool(r["success"]), int(r.get("steps", -1)))
    return out


def paired(base, treat, keys):
    b = c = 0
    for k in keys:
        if k in base and k in treat:
            bs, ts = base[k][0], treat[k][0]
            if ts and not bs:
                b += 1
            elif bs and not ts:
                c += 1
    return b, c


def blk(base, treat, keys):
    b, c = paired(base, treat, keys)
    nb = sum(1 for k in keys if k in base and base[k][0])
    nt = sum(1 for k in keys if k in treat and treat[k][0])
    n = sum(1 for k in keys if k in base and k in treat)
    return {"n": n, "base_sr": round(nb / n, 4) if n else None,
            "treat_sr": round(nt / n, 4) if n else None,
            "b_treat": b, "c_base": c, "p": round(exact_p(b, c), 4)}


def main():
    results = {"per_task": {}, "pooled": {}, "notes": []}
    # accumulate pooled structures
    pool_all = {"base": {}, "treat": {}, "dpct": {}}   # key=(task,seed,reset)
    pool_mid = {"base": {}, "treat": {}, "dpct": {}}

    all_decile_cells = []  # (label, p, b, c) for Bonferroni

    for task in TASKS:
        df = A / f"{task}_N2" / "deval.json"
        if not df.exists():
            continue
        dj = json.load(open(df))
        dev = {int(k): float(v) for k, v in dj["d_eval"].items()}
        near = {int(k): int(v) for k, v in dj.get("nearest_source", {}).items()}
        base = load(task, BASE)
        treat = load(task, TREAT)
        if not base or not treat:
            continue
        resets = sorted(dev)
        entry = {"overall": blk(base, treat, [(s, r) for s in SEEDS for r in resets])}

        # 1. deciles
        d = np.array([dev[r] for r in resets])
        edges = np.quantile(d, [i / 10 for i in range(1, 10)])
        decs = []
        for q in range(10):
            sub = [r for r in resets if np.searchsorted(edges, dev[r], side="right") == q]
            bl = blk(base, treat, [(s, r) for s in SEEDS for r in sub])
            decs.append(bl)
            all_decile_cells.append((f"{task}:d{q}", bl["p"], bl["b_treat"], bl["c_base"]))
        entry["deciles"] = decs

        # 2. per nearest source
        srcs = {}
        for sid in sorted(set(near.values())) if near else []:
            sub = [r for r in resets if near.get(r) == sid]
            if sub:
                srcs[str(sid)] = blk(base, treat, [(s, r) for s in SEEDS for r in sub])
        entry["by_nearest_source"] = srcs

        # 3. per-seed sign
        seed_signs = []
        for s in SEEDS:
            nb = np.mean([base[(s, r)][0] for r in resets if (s, r) in base])
            nt = np.mean([treat[(s, r)][0] for r in resets if (s, r) in treat])
            seed_signs.append(round(float(nt - nb), 4))
        entry["per_seed_delta"] = seed_signs
        entry["seed_sign_consistency"] = f"{sum(1 for x in seed_signs if x > 0)}+/{sum(1 for x in seed_signs if x < 0)}-"

        # 4. continuous: d_eval of discordant pairs
        tw, bw = [], []
        for s in SEEDS:
            for r in resets:
                if (s, r) in base and (s, r) in treat:
                    bs, ts = base[(s, r)][0], treat[(s, r)][0]
                    if ts and not bs:
                        tw.append(dev[r])
                    elif bs and not ts:
                        bw.append(dev[r])
        mw = None
        if tw and bw:
            try:
                from scipy.stats import mannwhitneyu
                mw = round(float(mannwhitneyu(tw, bw, alternative="two-sided").pvalue), 4)
            except Exception:  # noqa: BLE001
                mw = None
        entry["discordant_deval"] = {"transform_win_median": round(float(np.median(tw)), 4) if tw else None,
                                     "baseline_win_median": round(float(np.median(bw)), 4) if bw else None,
                                     "mannwhitney_p": mw, "n_tw": len(tw), "n_bw": len(bw)}

        # 5. efficiency: steps among co-successes
        dsteps = []
        for s in SEEDS:
            for r in resets:
                if (s, r) in base and (s, r) in treat and base[(s, r)][0] and treat[(s, r)][0]:
                    dsteps.append(treat[(s, r)][1] - base[(s, r)][1])
        eff = {"n_cosuccess": len(dsteps), "mean_step_diff": round(float(np.mean(dsteps)), 2) if dsteps else None}
        if dsteps and any(x != 0 for x in dsteps):
            try:
                from scipy.stats import wilcoxon
                eff["wilcoxon_p"] = round(float(wilcoxon(dsteps).pvalue), 4)
            except Exception:  # noqa: BLE001
                eff["wilcoxon_p"] = None
        entry["efficiency_steps"] = eff

        results["per_task"][task] = entry

        # accumulate pooled
        pct = {r: float(np.mean(d <= dev[r])) for r in resets}  # within-task percentile
        for s in SEEDS:
            for r in resets:
                k = (task, s, r)
                if (s, r) in base:
                    pool_all["base"][k] = base[(s, r)]
                if (s, r) in treat:
                    pool_all["treat"][k] = treat[(s, r)]
                pool_all["dpct"][k] = pct[r]
                if task in MIDSR:
                    if (s, r) in base:
                        pool_mid["base"][k] = base[(s, r)]
                    if (s, r) in treat:
                        pool_mid["treat"][k] = treat[(s, r)]
                    pool_mid["dpct"][k] = pct[r]

    # 6/7 pooled
    def pooled_block(P, label):
        keys = list(P["dpct"])
        out = {"overall": blk(P["base"], P["treat"], keys)}
        decs = []
        for q in range(10):
            sub = [k for k in keys if min(int(P["dpct"][k] * 10), 9) == q]
            decs.append(blk(P["base"], P["treat"], sub))
        out["deciles_by_pct"] = decs
        return out
    results["pooled"]["all_tasks"] = pooled_block(pool_all, "all")
    results["pooled"]["mid_sr_tasks"] = pooled_block(pool_mid, "mid")

    # 8. Bonferroni over all decile cells
    m = len(all_decile_cells)
    mn = min(all_decile_cells, key=lambda x: x[1]) if all_decile_cells else None
    results["multiple_comparison"] = {
        "n_decile_cells": m,
        "min_p_cell": {"label": mn[0], "p": mn[1], "b": mn[2], "c": mn[3]} if mn else None,
        "bonferroni_min_p": round(min(1.0, mn[1] * m), 4) if mn else None,
        "n_cells_p_below_.05": sum(1 for x in all_decile_cells if x[1] < 0.05),
        "expected_false_positives_at_.05": round(0.05 * m, 1),
    }

    out = A / "slice_results.json"
    out.write_text(json.dumps(results, indent=2))

    # ---- printed summary ----
    print("==== POOLED (max power) transform_uniform vs baseline ====")
    for name in ("all_tasks", "mid_sr_tasks"):
        o = results["pooled"][name]["overall"]
        print(f"  {name:14s} n={o['n']:5d}  base={o['base_sr']} treat={o['treat_sr']}  "
              f"b={o['b_treat']}/c={o['c_base']}  p={o['p']}")
        dl = results["pooled"][name]["deciles_by_pct"]
        sig = [(i, x['p']) for i, x in enumerate(dl) if x['p'] < 0.05]
        print(f"     decile p<.05: {sig if sig else 'none'}")
    print("\n==== per-task overall + strongest single decile ====")
    for task, e in results["per_task"].items():
        o = e["overall"]
        pmin = min(e["deciles"], key=lambda x: x["p"])
        print(f"  {task:22s} overall p={o['p']} ({o['b_treat']}/{o['c_base']})  "
              f"seeds={e['seed_sign_consistency']}  min-decile p={pmin['p']} "
              f"(b{pmin['b_treat']}/c{pmin['c_base']})  discord-MWp={e['discordant_deval']['mannwhitney_p']}  "
              f"stepdiff={e['efficiency_steps'].get('mean_step_diff')}(p={e['efficiency_steps'].get('wilcoxon_p')})")
    mc = results["multiple_comparison"]
    print("\n==== multiple comparison over all decile cells ====")
    print(f"  cells={mc['n_decile_cells']}  min-p cell={mc['min_p_cell']}  "
          f"Bonferroni min-p={mc['bonferroni_min_p']}")
    print(f"  cells with p<.05: {mc['n_cells_p_below_.05']}  (expected by chance: {mc['expected_false_positives_at_.05']})")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
