"""원본 품질 실험: 생성 성공률 상위 원본 2개 vs 중위 원본 2개로 학습한 로봇
성공률 바 차트 (거리 분포 매칭, 시드 301-303). 점 = 시드 3개 개별값.
사용: python3 fig_quality_bars.py <data_root(motivation/data)> <out.png>
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
SEEDS = [301, 302, 303]
# (task, 표기, eval 디렉토리, run-level 유의 표기)
TASKS = [
    ("stack", "블록 쌓기", "ctrl_eval", "**"),
    ("stack_three", "세 블록 쌓기", "ctrl2_eval", "**"),
    ("square", "너트 끼우기", "ctrl2_eval", "*"),
    ("coffee", "커피", "ctrl2_eval", "**"),
    ("three_piece_assembly", "세 조각 조립", "ctrl2_eval", "*"),
]


def sr(root, task, arm, seed):
    f = D / root / task / f"e2_{task}_{arm}_seed{seed}.jsonl"
    if not f.exists():
        return None
    vals = [json.loads(x)["success"] for x in open(f)]
    return sum(vals) / len(vals)


GREEN = "#2a9d8f"
GRAY = "#9aa2ab"
fig, ax = plt.subplots(figsize=(10.5, 5.5))
w = 0.36
for i, (task, name, root, star) in enumerate(TASKS):
    a = [x for x in (sr(root, task, "C2_hi", s) for s in SEEDS) if x is not None]
    b = [x for x in (sr(root, task, "C2_mid", s) for s in SEEDS) if x is not None]
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    ax.bar(i - w / 2, ma, width=w, color=GREEN, zorder=2)
    ax.bar(i + w / 2, mb, width=w, color=GRAY, zorder=2)
    ax.scatter([i - w / 2] * len(a), a, s=16, color="#14554c", zorder=3, alpha=0.85)
    ax.scatter([i + w / 2] * len(b), b, s=16, color="#4a4f55", zorder=3, alpha=0.85)
    ax.text(i, max(max(a), max(b)) + 0.03, f"+{(ma - mb) * 100:.1f}%p {star}", ha="center",
            fontsize=11, fontweight="bold")

ax.set_xticks(range(len(TASKS)))
ax.set_xticklabels([n for _, n, _, _ in TASKS], fontsize=12)
ax.set_ylabel("로봇 성공률 (평가 장면 200개 전체)", fontsize=12)
ax.set_ylim(0, 1.05)
ax.grid(axis="y", color="#eee", zorder=0)
ax.set_axisbelow(True)
handles = [plt.Rectangle((0, 0), 1, 1, color=GREEN), plt.Rectangle((0, 0), 1, 1, color=GRAY)]
ax.legend(handles, ["생성 성공률 상위 원본 2개로 학습", "중위 원본 2개로 학습"], fontsize=11, frameon=False, loc="upper right")
ax.set_title("원본 품질 실험 — 거리 분포를 똑같이 맞춰도 원본 쌍 교체만으로 갈린다 (점은 시드 3개 개별값)", fontsize=13)
fig.tight_layout()
fig.savefig(OUT, dpi=140)
print(f"saved {OUT}")
