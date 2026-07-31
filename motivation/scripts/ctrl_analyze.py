"""Controlled-experiment analysis: paired McNemar + TOST + distance slices.
Inputs: ctrl_eval/{stack,threading}/e2_*_seed*.jsonl, deval_matrix_*.json.
Pairs = (seed, reset_index) over the contrast's shared seeds.
"""
import json, math, os
from pathlib import Path

SP = Path(os.environ.get("CTRL_EVAL_DIR", Path(__file__).parent / "ctrl_eval"))
EDGES = {"stack": [0.2054, 0.2528, 0.2928, 0.3395],
         "threading": [0.2013, 0.2503, 0.2929, 0.3449]}
DGR = {"stack": {0: .183, 1: .641, 2: .289, 3: .739, 4: .948, 5: .557, 6: .591, 7: .939, 8: .525, 9: .137},
       "threading": {0: .531, 1: .604, 2: .629, 3: .044, 4: .301, 5: .255, 6: .44, 7: .338, 8: .608, 9: .303}}
SOURCES = {("stack", "A_nearheavy"): [4, 7], ("stack", "A_farheavy"): [4, 7],
           ("stack", "B_far"): [4, 7], ("stack", "B_nearpad"): [4, 7],
           ("stack", "C2_hi"): [4, 7], ("stack", "C2_mid"): [3, 1],
           ("threading", "C_hi"): [2, 8], ("threading", "C_mid"): [0, 6],
           ("threading", "C_lo"): [4, 5], ("threading", "D_1"): [2, 0], ("threading", "D_2"): [8, 1]}


def load(task, arm, seed):
    f = SP / task / f"e2_{task}_{arm}_seed{seed}.jsonl"
    if not f.exists():
        return None
    succ = [None] * 200
    for line in open(f):
        r = json.loads(line)
        succ[r["reset_index"]] = bool(r["success"])
    assert None not in succ
    return succ


def deval(task):
    m = json.load(open(SP / f"deval_matrix_{task}.json"))["matrix"]
    return {int(k): v for k, v in m.items()}


def dmin(task, sources):
    dv = deval(task)
    return {i: min(dv[i][s] for s in sources) for i in range(200)}


def mcnemar(pairs):
    """pairs: list of (a, b) bools. Returns dict with b, c, diff, exact p, 90% CI."""
    n = len(pairs)
    b = sum(1 for x, y in pairs if x and not y)
    c = sum(1 for x, y in pairs if not x and y)
    m = b + c
    if m == 0:
        return {"n": n, "b": b, "c": c, "diff": 0.0, "p": 1.0, "ci90": (0.0, 0.0)}
    k = min(b, c)
    p = min(1.0, 2 * sum(math.comb(m, i) for i in range(k + 1)) * 0.5 ** m)
    diff = (b - c) / n
    se = math.sqrt(max(m - (b - c) ** 2 / n, 0)) / n
    z = 1.6449
    return {"n": n, "b": b, "c": c, "diff": diff, "p": p,
            "ci90": (diff - z * se, diff + z * se)}


def contrast(task, a1, a2, seeds, label, slices=None):
    """slices: dict name -> set of reset indices (None = aggregate only)."""
    s1 = {s: load(task, a1, s) for s in seeds}
    s2 = {s: load(task, a2, s) for s in seeds}
    seeds = [s for s in seeds if s1[s] and s2[s]]
    per1 = [sum(s1[s]) / 200 for s in seeds]
    per2 = [sum(s2[s]) / 200 for s in seeds]
    print(f"\n### {label}: {a1} vs {a2}  ({task}, {len(seeds)} seeds)")
    print(f"  {a1}: mean {sum(per1)/len(per1):.3f}  seeds {[round(x,3) for x in per1]}")
    print(f"  {a2}: mean {sum(per2)/len(per2):.3f}  seeds {[round(x,3) for x in per2]}")
    all_slices = {"ALL": set(range(200))}
    if slices:
        all_slices.update(slices)
    for nm, idxset in all_slices.items():
        pairs = [(s1[s][i], s2[s][i]) for s in seeds for i in sorted(idxset)]
        r = mcnemar(pairs)
        sr1 = sum(x for x, _ in pairs) / r["n"]
        sr2 = sum(y for _, y in pairs) / r["n"]
        print(f"  [{nm:>10}] n={r['n']:5d}  {a1} {sr1:.3f} vs {a2} {sr2:.3f}  "
              f"diff {r['diff']*100:+.1f}pp  b/c {r['b']}/{r['c']}  p={r['p']:.4g}  "
              f"CI90 [{r['ci90'][0]*100:+.1f},{r['ci90'][1]*100:+.1f}]pp")


def dist_slices(task, sources):
    d = dmin(task, sources)
    e = EDGES[task]
    return {
        "near(b01)": {i for i in d if d[i] < e[1]},
        "mid(b2)": {i for i in d if e[1] <= d[i] < e[2]},
        "far(b3)": {i for i in d if e[2] <= d[i] < e[3]},
        "far(b4)": {i for i in d if d[i] >= e[3]},
    }


def own_profile(task, arm, seeds):
    src = SOURCES[(task, arm)]
    d = dmin(task, src)
    e = EDGES[task]
    bins = {"near": [i for i in d if d[i] < e[1]], "mid": [i for i in d if e[1] <= d[i] < e[2]],
            "far": [i for i in d if d[i] >= e[2]]}
    runs = [load(task, arm, s) for s in seeds]
    runs = [r for r in runs if r]
    row = []
    for nm, idxs in bins.items():
        tot = sum(r[i] for r in runs for i in idxs)
        n = len(runs) * len(idxs)
        row.append(f"{nm} {tot/n:.3f}(n{len(idxs)})")
    mdgr = sum(DGR[task][s] for s in src) / len(src)
    print(f"  {arm:<10} src{src} meanDGR {mdgr:.2f}: " + "  ".join(row))


S6 = [301, 302, 303, 304, 305, 306]
S3 = [301, 302, 303]

print("=" * 80)
print("EXP-A (stack): density crossover — same sources {4,7}, 500 demos, near:far 3:1 vs 1:3")
sl = dist_slices("stack", [4, 7])
contrast("stack", "A_nearheavy", "A_farheavy", S6, "EXP-A", sl)

print("\n" + "=" * 80)
print("EXP-B (stack): coverage — B_far(150/100/150) vs B_nearpad(300/100/0) same sources, 400")
contrast("stack", "B_far", "B_nearpad", S3, "EXP-B", sl)

print("\n" + "=" * 80)
print("EXP-C2 (stack): source DGR causal, distance-matched 70/bin — {4,7} vs {3,1}")
contrast("stack", "C2_hi", "C2_mid", S3, "EXP-C2")
print("\n  own-distance profiles (SR by distance to own sources):")
for arm in ["C2_hi", "C2_mid"]:
    own_profile("stack", arm, S3)

print("\n" + "=" * 80)
print("EXP-C (threading): DGR gradient, distance-matched 50/bin — hi{2,8}/mid{0,6}/lo{4,5}")
contrast("threading", "C_hi", "C_mid", S3, "EXP-C hi-mid")
contrast("threading", "C_hi", "C_lo", S3, "EXP-C hi-lo")
contrast("threading", "C_mid", "C_lo", S3, "EXP-C mid-lo")
print("\n  own-distance profiles:")
for arm in ["C_hi", "C_mid", "C_lo"]:
    own_profile("threading", arm, S3)

print("\n" + "=" * 80)
print("EXP-D (threading): DGR sufficiency — D_1{2,0} meanDGR .58 vs D_2{8,1} meanDGR .61")
contrast("threading", "D_1", "D_2", S3, "EXP-D")
contrast("threading", "C_hi", "D_1", S3, "extra: C_hi vs D_1")
contrast("threading", "C_hi", "D_2", S3, "extra: C_hi vs D_2")
print("\n  own-distance profiles:")
for arm in ["D_1", "D_2"]:
    own_profile("threading", arm, S3)
