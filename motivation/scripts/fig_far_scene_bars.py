"""먼 장면(80개) 층에서만: 무작위 뽑기 vs 거리 균등 뽑기 학습의 로봇 성공률
바 차트. 시드 6개 개별값을 점으로 겹쳐 그린다.
사용: python3 fig_far_scene_bars.py <diag_eval_dir> <out.png>
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
    ("coffee", "커피"),
    ("threading", "바늘 꿰기"),
    ("three_piece_assembly", "세 조각 조립"),
    ("stack_three", "세 블록 쌓기"),
    ("mug_cleanup", "머그 정리"),
    ("stack", "블록 쌓기"),
]
strata = json.loads((D / "strata.json").read_text())


def far_sr(task, arm, seed):
    f = D / f"{task}_N2/eval/diag_e2_{task}_{arm}_seed{seed}.jsonl"
    if not f.exists():
        return None
    succ = {}
    for line in open(f):
        r = json.loads(line)
        succ[r["reset_index"]] = r["success"]
    idx = [int(k.split("_")[1]) for k, v in strata[task].items() if v["stratum"] == "tail"]
    return sum(succ[i] for i in idx) / len(idx)


GRAY = "#9aa2ab"
GREEN = "#2a9d8f"
fig, ax = plt.subplots(figsize=(11, 5.5))
w = 0.36
for i, (task, name) in enumerate(TASKS):
    a = [far_sr(task, "baseline", s) for s in SEEDS]
    b = [far_sr(task, "transform_uniform", s) for s in SEEDS]
    a = [x for x in a if x is not None]
    b = [x for x in b if x is not None]
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    ax.bar(i - w / 2, ma, width=w, color=GRAY, zorder=2)
    ax.bar(i + w / 2, mb, width=w, color=GREEN, zorder=2)
    ax.scatter([i - w / 2] * len(a), a, s=14, color="#4a4f55", zorder=3, alpha=0.8)
    ax.scatter([i + w / 2] * len(b), b, s=14, color="#14554c", zorder=3, alpha=0.8)
    d = (mb - ma) * 100
    star = " *" if task == "coffee" else ""
    ax.text(i, max(max(a), max(b)) + 0.03, f"{d:+.1f}%p{star}", ha="center", fontsize=11,
            fontweight="bold" if task == "coffee" else "normal")

ax.set_xticks(range(len(TASKS)))
ax.set_xticklabels([n for _, n in TASKS], fontsize=12)
ax.set_ylabel("먼 장면에서의 로봇 성공률", fontsize=12)
ax.set_ylim(0, 1.0)
ax.grid(axis="y", color="#eee", zorder=0)
ax.set_axisbelow(True)
handles = [plt.Rectangle((0, 0), 1, 1, color=GRAY), plt.Rectangle((0, 0), 1, 1, color=GREEN)]
ax.legend(handles, ["무작위 뽑기로 학습", "거리 균등 뽑기로 학습"], fontsize=11, frameon=False, loc="upper left")
ax.set_title("먼 장면(모든 원본에서 먼 평가 장면 80개)에서의 로봇 성공률 — 점은 시드 6개 개별값, * p<0.1", fontsize=13)
fig.tight_layout()
fig.savefig(OUT, dpi=140)
print(f"saved {OUT}")
