"""Per-object reslice of existing baseline evals: does the policy fail where the
survivor filter depleted training data (hard-object-moved-far scenes)?

Inputs per task (env OBJ_DIR): deval_objmatrix.json (200 resets x 10 sources x
objects, normalized per-object displacement) and eval jsonls for baseline seeds.

Two tests per task:
  A) marginal: baseline SR by quartile of hard_d = min over sources of the
     hard object's displacement (confounded with total distance — shown for
     orientation only).
  B) controlled: within terciles of total distance (mean over objects, nearest
     source), split scenes by hard-share = d_hard / (d_hard + d_easy) at the
     nearest source; compare SR hard-heavy vs easy-heavy. This isolates the
     reallocation axis the filter distorts. Run-level: per-seed delta t-test.
"""
import json, math, os
from pathlib import Path

OBJ = Path(os.environ.get("OBJ_DIR", "motivation/data/objslice"))
SEEDS = [101, 102, 103, 104, 105, 106]
HARD = {"coffee": "coffee_machine", "threading": "tripod",
        "hammer_cleanup": "drawer", "stack_three": "cubeB", "mug_cleanup": "drawer"}


def t_p(d):
    n = len(d)
    m = sum(d) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1)) if n > 1 else 0
    if sd == 0:
        return m, 0.0, 1.0
    t = m / (sd / math.sqrt(n))
    # crude two-sided p via normal approx is fine at n=6? use t table for df=5: |t|>2.57 -> p<.05
    return m, t, None


def load_eval(task, seed):
    f = OBJ / task / f"e2_{task}_baseline_seed{seed}.jsonl"
    if not f.exists():
        return None
    s = [None] * 200
    for line in open(f):
        r = json.loads(line)
        s[r["reset_index"]] = bool(r["success"])
    return s


for task, hard in HARD.items():
    mp = OBJ / task / "deval_objmatrix.json"
    if not mp.exists():
        print(f"{task}: objmatrix 없음 — 스킵")
        continue
    M = json.loads(mp.read_text())
    objs = M["objects"]
    hi_idx = None
    for k, o in enumerate(objs):
        if hard in o or o in hard:
            hi_idx = k
    if hi_idx is None:
        print(f"{task}: hard object '{hard}' not in {objs} — 스킵")
        continue
    mat = {int(k): v for k, v in M["matrix"].items()}
    movN = len(objs)

    hard_d, total_d, share = {}, {}, {}
    for i, rows in mat.items():
        totals = [sum(r) / movN for r in rows]
        j = min(range(len(rows)), key=lambda j: totals[j])
        total_d[i] = totals[j]
        hard_d[i] = min(r[hi_idx] for r in rows)
        s = sum(rows[j])
        share[i] = rows[j][hi_idx] / s if s > 0 else 0.5

    runs = {s: load_eval(task, s) for s in SEEDS}
    seeds = [s for s in SEEDS if runs[s]]
    if len(seeds) < 3:
        print(f"{task}: baseline eval 부족 ({len(seeds)} seeds) — 스킵")
        continue

    print(f"\n== {task} (hard={objs[hi_idx]}, {len(seeds)} seeds) ==")
    # A) marginal quartiles of hard_d
    order = sorted(mat, key=lambda i: hard_d[i])
    q = {i: min(r * 4 // 200, 3) for r, i in enumerate(order)}
    line = []
    for b in range(4):
        idx = [i for i in mat if q[i] == b]
        sr = sum(runs[s][i] for s in seeds for i in idx) / (len(seeds) * len(idx))
        line.append(f"Q{b + 1} {sr:.3f}")
    print("  A(한계, hard_d 4분위): " + "  ".join(line))

    # B) controlled: within total-d terciles, hard-heavy vs easy-heavy
    order_t = sorted(mat, key=lambda i: total_d[i])
    terc = {i: min(r * 3 // 200, 2) for r, i in enumerate(order_t)}
    all_deltas = {s: [] for s in seeds}
    for b in range(3):
        idx = [i for i in mat if terc[i] == b]
        med = sorted(share[i] for i in idx)[len(idx) // 2]
        hh = [i for i in idx if share[i] >= med]
        eh = [i for i in idx if share[i] < med]
        ds = []
        for s in seeds:
            a = sum(runs[s][i] for i in hh) / len(hh)
            c = sum(runs[s][i] for i in eh) / len(eh)
            ds.append(a - c)
            all_deltas[s].append((a - c, len(hh) + len(eh)))
        m, t, _ = t_p(ds)
        band = ["near", "mid", "far"][b]
        print(f"  B({band}, n{len(idx)}): hard-heavy − easy-heavy = {m * 100:+.1f}pp  "
              f"seeds[{','.join(f'{x * 100:+.0f}' for x in ds)}]  t={t:.2f}")
    # pooled across bands (weighted by n)
    ds = []
    for s in seeds:
        num = sum(d * n for d, n in all_deltas[s])
        den = sum(n for _, n in all_deltas[s])
        ds.append(num / den)
    m, t, _ = t_p(ds)
    sig = "유의(df5 |t|>2.57)" if abs(t) > 2.57 else "무유의"
    print(f"  B(전체 풀링): {m * 100:+.1f}pp  seeds[{','.join(f'{x * 100:+.0f}' for x in ds)}]  t={t:.2f}  {sig}")

    # depletion-zone check: top-20% hard_d scenes
    cut = sorted(hard_d.values())[int(200 * 0.8)]
    dz = [i for i in mat if hard_d[i] >= cut]
    rest = [i for i in mat if hard_d[i] < cut]
    ds = []
    for s in seeds:
        a = sum(runs[s][i] for i in dz) / len(dz)
        c = sum(runs[s][i] for i in rest) / len(rest)
        ds.append(a - c)
    m, t, _ = t_p(ds)
    print(f"  고갈영역(top-20% hard_d, n{len(dz)}): Δ vs 나머지 = {m * 100:+.1f}pp  t={t:.2f}  (거리 교란 포함 주의)")
