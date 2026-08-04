"""전체 200장면 기준: 무작위 뽑기 vs 원본별 균등 뽑기 학습의 로봇 성공률
바 차트 (7작업 — 머그는 생존 풀이 얇아 원본별 균등 뽑기 불가). 점 = 시드 6개.
사용: python3 fig_ancestry_bars.py <eval_root(noise dir)> <out.png>
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
    ("hammer_cleanup", "망치 정리"),
    ("stack", "블록 쌓기"),
    ("square", "너트 끼우기"),
    ("stack_three", "세 블록 쌓기"),
    ("coffee", "커피"),
    ("threading", "바늘 꿰기"),
    ("three_piece_assembly", "세 조각 조립"),
]


def sr(task, arm, seed):
    f = D / f"{task}_N2/eval/e2_{task}_{arm}_seed{seed}.jsonl"
    if not f.exists():
        return None
    vals = [json.loads(x)["success"] for x in open(f)]
    return sum(vals) / len(vals)


GRAY = "#9aa2ab"
PURPLE = "#7d6aa6"
fig, ax = plt.subplots(figsize=(11.5, 5.5))
w = 0.36
for i, (task, name) in enumerate(TASKS):
    a = [x for x in (sr(task, "baseline", s) for s in SEEDS) if x is not None]
    b = [x for x in (sr(task, "ancestry_balanced", s) for s in SEEDS) if x is not None]
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    ax.bar(i - w / 2, ma, width=w, color=GRAY, zorder=2)
    ax.bar(i + w / 2, mb, width=w, color=PURPLE, zorder=2)
    ax.scatter([i - w / 2] * len(a), a, s=14, color="#4a4f55", zorder=3, alpha=0.8)
    ax.scatter([i + w / 2] * len(b), b, s=14, color="#463a63", zorder=3, alpha=0.8)
    d = (mb - ma) * 100
    star = " **" if task == "stack" else (" *" if task == "threading" else "")
    ax.text(i, max(max(a), max(b)) + 0.03, f"{d:+.1f}%p{star}", ha="center", fontsize=10.5,
            fontweight="bold" if task == "stack" else "normal")

ax.set_xticks(range(len(TASKS)))
ax.set_xticklabels([n for _, n in TASKS], fontsize=12)
ax.set_ylabel("로봇 성공률 (평가 장면 200개 전체)", fontsize=12)
ax.set_ylim(0, 1.02)
ax.grid(axis="y", color="#eee", zorder=0)
ax.set_axisbelow(True)
handles = [plt.Rectangle((0, 0), 1, 1, color=GRAY), plt.Rectangle((0, 0), 1, 1, color=PURPLE)]
ax.legend(handles, ["무작위 뽑기로 학습", "원본별 균등 뽑기로 학습"], fontsize=11, frameon=False, loc="upper right")
ax.set_title("무작위 뽑기 vs 원본별 균등 뽑기 — 이득은 없고 블록 쌓기에서 유의한 손해 (점은 시드 6개 개별값)", fontsize=13)
fig.tight_layout()
fig.savefig(OUT, dpi=140)
print(f"saved {OUT}")
