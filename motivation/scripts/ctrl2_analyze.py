"""ctrl2 multi-task analysis: run-level paired stats for every EXP family on
every task that has the contrast. Pair unit = seed (per-seed SR delta), the
inference standard set by the red-team pass; pooled per-episode McNemar is
printed as descriptive support only.
Env: CTRL_EVAL_DIR = dir with <task>/e2_<task>_<arm>_seed<s>.jsonl
"""
import json, math, os
from pathlib import Path

SP = Path(os.environ.get("CTRL_EVAL_DIR", Path(__file__).parent.parent / "data" / "ctrl2_eval"))
S3 = [301, 302, 303]
CONTRASTS = [
    # (exp, task, arm1, arm2)
    ("A(밀도)", "stack", "A_nearheavy", "A_farheavy"),
    ("A(밀도)", "threading", "A_nearheavy", "A_farheavy"),
    ("A(밀도)", "square", "A_nearheavy", "A_farheavy"),
    ("A(밀도)", "coffee", "A_nearheavy", "A_farheavy"),
    ("B(커버리지)", "stack", "B_far", "B_nearpad"),
    ("B(커버리지)", "threading", "B_far", "B_nearpad"),
    ("B(커버리지)", "square", "B_far", "B_nearpad"),
    ("B(커버리지)", "coffee", "B_far", "B_nearpad"),
    ("C2(품질)", "stack", "C2_hi", "C2_mid"),
    ("C2(품질)", "square", "C2_hi", "C2_mid"),
    ("C2(품질)", "coffee", "C2_hi", "C2_mid"),
    ("C2(품질)", "three_piece_assembly", "C2_hi", "C2_mid"),
    ("C2(품질)", "stack_three", "C2_hi", "C2_mid"),
    ("D(DGR충분성)", "threading", "D_1", "D_2"),
    ("D(DGR충분성)", "stack", "D_1", "D_2"),
    ("D(DGR충분성)", "square", "D_1", "D_2"),
    ("D(DGR충분성)", "coffee", "D_1", "D_2"),
]


def load(task, arm, seed):
    f = SP / task / f"e2_{task}_{arm}_seed{seed}.jsonl"
    if not f.exists():
        return None
    succ = [None] * 200
    for line in open(f):
        r = json.loads(line)
        succ[r["reset_index"]] = bool(r["success"])
    return succ if None not in succ else None


def t_cdf_2sided(t, df):
    """two-sided p for t stat via incomplete beta (no scipy)."""
    x = df / (df + t * t)
    a, b = df / 2.0, 0.5
    # continued fraction for regularized incomplete beta I_x(a,b)
    def betacf(a, b, x):
        qab, qap, qam = a + b, a + 1.0, a - 1.0
        c, d = 1.0, 1.0 - qab * x / qap
        if abs(d) < 1e-30:
            d = 1e-30
        d = 1.0 / d
        h = d
        for m in range(1, 200):
            m2 = 2 * m
            aa = m * (b - m) * x / ((qam + m2) * (a + m2))
            d = 1.0 + aa * d
            if abs(d) < 1e-30:
                d = 1e-30
            c = 1.0 + aa / c
            if abs(c) < 1e-30:
                c = 1e-30
            d = 1.0 / d
            h *= d * c
            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
            d = 1.0 + aa * d
            if abs(d) < 1e-30:
                d = 1e-30
            c = 1.0 + aa / c
            if abs(c) < 1e-30:
                c = 1e-30
            d = 1.0 / d
            de = d * c
            h *= de
            if abs(de - 1.0) < 3e-9:
                break
        return h

    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    if x <= 0:
        ib = 0.0
    elif x >= 1:
        ib = 1.0
    else:
        front = math.exp(math.log(x) * a + math.log(1 - x) * b + lbeta) / a
        if x < (a + 1) / (a + b + 2):
            ib = front * betacf(a, b, x)
        else:
            ib = 1 - math.exp(math.log(1 - x) * b + math.log(x) * a + lbeta) / b * betacf(b, a, 1 - x)
    return ib  # = P(|T|>t) two-sided for the t <-> beta identity


def mcnemar(pairs):
    b = sum(1 for x, y in pairs if x and not y)
    c = sum(1 for x, y in pairs if not x and y)
    m = b + c
    if m == 0:
        return 1.0, b, c
    k = min(b, c)
    p = min(1.0, 2 * sum(math.comb(m, i) for i in range(k + 1)) * 0.5 ** m)
    return p, b, c


print(f"{'EXP':<14}{'task':<22}{'n':>2} {'arm1':>6} {'arm2':>6} {'Δ(pp)':>7} "
      f"{'per-seed deltas':<28}{'t':>6} {'p(run)':>8} {'p(ep)':>9}")
print("-" * 118)
for exp, task, a1, a2 in CONTRASTS:
    s1 = {s: load(task, a1, s) for s in S3}
    s2 = {s: load(task, a2, s) for s in S3}
    seeds = [s for s in S3 if s1[s] and s2[s]]
    if len(seeds) < 2:
        print(f"{exp:<14}{task:<22} -- 데이터 부족 (seeds {len(seeds)})")
        continue
    d = [sum(s1[s]) / 200 - sum(s2[s]) / 200 for s in seeds]
    n = len(d)
    m = sum(d) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1)) if n > 1 else 0.0
    t = m / (sd / math.sqrt(n)) if sd > 0 else float("inf")
    p_run = t_cdf_2sided(abs(t), n - 1) if sd > 0 else 0.0
    pairs = [(s1[s][i], s2[s][i]) for s in seeds for i in range(200)]
    p_ep, b, c = mcnemar(pairs)
    m1 = sum(sum(s1[s]) for s in seeds) / (200 * n)
    m2 = sum(sum(s2[s]) for s in seeds) / (200 * n)
    ds = ",".join(f"{x*100:+.1f}" for x in d)
    print(f"{exp:<14}{task:<22}{n:>2} {m1:>6.3f} {m2:>6.3f} {m*100:>+7.1f} "
          f"[{ds:<26}]{t:>6.2f} {p_run:>8.4f} {p_ep:>9.2g}")
