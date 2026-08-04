"""demo_bars with D0 vs widest rung side by side: per task, per source, the
share of RETAINED data at the narrowest (D0) and the widest variant, with the
uniform 10% line. Shows how widening amplifies ancestry skew.
Data: Phase-0 stats_all.json (square/threading/coffee/three_piece) +
E1 analysis_full attempts jsonls (the rest + widest E-variants).
Output: /home/ubuntu/demo_bars_9_d0.png
"""
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IC = Path("/home/ubuntu/mimicgen_jihoonkwon/experiments/motivation_ic")
AF = IC / "e1" / "analysis_full"
stats = json.load(open(IC / "analysis_all" / "stats_all.json"))

# (task label, D0 source, widest source) — source is either ("stats", task, var) or ("jsonl", filename)
TASKS = [
    ("Square", ("stats", "square", "D0"), ("jsonl", "square_D2")),
    ("Threading", ("stats", "threading", "D0"), ("jsonl", "threading_D2E")),
    ("Coffee", ("stats", "coffee", "D0"), ("jsonl", "coffee_D2E")),
    ("Three Piece", ("stats", "three_piece_assembly", "D0"), ("stats", "three_piece_assembly", "D2")),
    ("Stack", ("jsonl", "stack_D0"), ("jsonl", "stack_D2E")),
    ("Stack Three", ("jsonl", "stack_three_D0"), ("jsonl", "stack_three_D2E")),
    ("Mug Cleanup", ("jsonl", "mug_cleanup_D0"), ("jsonl", "mug_cleanup_D2E")),
    ("Hammer Cleanup", ("jsonl", "hammer_cleanup_D0"), ("jsonl", "hammer_cleanup_D1")),
    ("Coffee Prep", ("jsonl", "coffee_preparation_D0"), ("jsonl", "coffee_preparation_D1")),
]


def counts(spec):
    kind = spec[0]
    if kind == "stats":
        a = stats[spec[1]][spec[2]]["ancestry"]
        att = {int(s): int(n) for s, n in zip(a["srcs"], a["attempted"])}
        ret = {int(s): int(n) for s, n in zip(a["srcs"], a["retained"])}
        return att, ret
    att, ret = Counter(), Counter()
    for line in open(AF / f"{spec[1]}_attempts.jsonl"):
        r = json.loads(line)
        # source id from attempt provenance: use src index field
        s = r.get("source_demo_id")
        if s is None:
            s = int(r["attempt_id"].split("@")[0].split("_")[1]) % 10  # fallback (unused)
        att[int(s)] += 1
        if r["success"]:
            ret[int(s)] += 1
    return dict(att), dict(ret)


fig, axes = plt.subplots(3, 3, figsize=(15, 9.5))
for ax, (name, d0_spec, dw_spec) in zip(axes.flat, TASKS):
    a0, r0 = counts(d0_spec)
    aw, rw = counts(dw_spec)
    srcs = sorted(set(a0) | set(aw))
    tot_r0 = max(sum(r0.values()), 1)
    tot_rw = max(sum(rw.values()), 1)
    share0 = {s: r0.get(s, 0) / tot_r0 for s in srcs}
    sharew = {s: rw.get(s, 0) / tot_rw for s in srcs}
    order = sorted(srcs, key=lambda s: -sharew[s])
    x = range(len(order))
    w = 0.38
    ax.bar([i - w / 2 for i in x], [share0[s] * 100 for s in order], width=w,
           color="#9ecf9e", label="D0 (narrow)")
    ax.bar([i + w / 2 for i in x], [sharew[s] * 100 for s in order], width=w,
           color="#1a7a2e", label="widest variant")
    ax.axhline(10, color="#888", lw=1, ls="--")
    dgr0 = sum(r0.values()) / max(sum(a0.values()), 1) * 100
    dgrw = sum(rw.values()) / max(sum(aw.values()), 1) * 100
    ax.set_title(f"{name}  (gen. success {dgr0:.0f}% \u2192 {dgrw:.0f}%)", fontsize=11)
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"s{s}" for s in order], fontsize=8)
    ax.set_ylim(0, max(max(sharew.values()), max(share0.values())) * 115)
    ax.tick_params(axis="y", labelsize=8)

axes.flat[0].legend(fontsize=9, loc="upper right")
fig.suptitle("Share of retained data per source demo: narrow D0 vs widest variant (dashed = uniform 10%)", fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig("/home/ubuntu/demo_bars_9_d0.png", dpi=140)
print("saved /home/ubuntu/demo_bars_9_d0.png")
