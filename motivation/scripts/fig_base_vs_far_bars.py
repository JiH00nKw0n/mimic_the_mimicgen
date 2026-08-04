"""무작위 뽑기 학습(baseline)만: 보통 장면(40) vs 먼 장면(80)의 로봇 성공률
바 차트 — 버려진 구석의 절벽을 보여준다. 점은 시드 6개 개별값.
사용: python3 fig_base_vs_far_bars.py <diag_eval_dir> <out.png>
"""
import json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

D = Path(sys.argv[1])
OUT = sys.argv[2]
SEEDS = [101, 102, 103, 104, 105, 106]
TASKS = [
    ("stack", "블록 쌓기"),
    ("stack_three", "세 블록 쌓기"),
    ("threading", "바늘 꿰기"),
    ("coffee", "커피"),
    ("mug_cleanup", "머그 정리"),
    ("three_piece_assembly", "세 조각 조립"),
]
strata = json.loads((D / "strata.json").read_text())


def sr(task, stratum, seed):
    f = D / f"{task}_N2/eval/diag_e2_{task}_baseline_seed{seed}.jsonl"
    if not f.exists():
        return None
    succ = {}
    for line in open(f):
        r = json.loads(line)
        succ[r["reset_index"]] = r["success"]
    idx = [int(k.split("_")[1]) for k, v in strata[task].items() if v["stratum"] == stratum]
    return sum(succ[i] for i in idx) / len(idx)


BLUE = "#5b8db8"
ORANGE = "#e76f51"
fig, ax = plt.subplots(figsize=(11, 5.5))
w = 0.36
for i, (task, name) in enumerate(TASKS):
    a = [x for x in (sr(task, "base", s) for s in SEEDS) if x is not None]
    b = [x for x in (sr(task, "tail", s) for s in SEEDS) if x is not None]
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    ax.bar(i - w / 2, ma, width=w, color=BLUE, zorder=2)
    ax.bar(i + w / 2, mb, width=w, color=ORANGE, zorder=2)
    ax.scatter([i - w / 2] * len(a), a, s=14, color="#2d4a63", zorder=3, alpha=0.8)
    ax.scatter([i + w / 2] * len(b), b, s=14, color="#8a3b26", zorder=3, alpha=0.8)
    ax.text(i, max(max(a), max(b)) + 0.03, f"{(mb - ma) * 100:+.1f}%p", ha="center", fontsize=11)

ax.set_xticks(range(len(TASKS)))
ax.set_xticklabels([n for _, n in TASKS], fontsize=12)
ax.set_ylabel("로봇 성공률 (무작위 뽑기 학습)", fontsize=12)
ax.set_ylim(0, 1.0)
ax.grid(axis="y", color="#eee", zorder=0)
ax.set_axisbelow(True)
handles = [plt.Rectangle((0, 0), 1, 1, color=BLUE), plt.Rectangle((0, 0), 1, 1, color=ORANGE)]
ax.legend(handles, ["보통 장면 (40개)", "먼 장면 (80개)"], fontsize=11, frameon=False, loc="upper right")
ax.set_title("같은 로봇의 보통 장면 vs 먼 장면 성공률 — 버려진 구석의 절벽 (점은 시드 6개 개별값)", fontsize=13)
fig.tight_layout()
fig.savefig(OUT, dpi=140)
print(f"saved {OUT}")
