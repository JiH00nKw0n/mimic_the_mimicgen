"""demo_bars_9.png과 동일한 형식의 D0 버전. 입력: d0_ancestry.json
(태스크별 원본별 시도/생존 수). 출력: demo_bars_9_D0.png (로컬, 한글 폰트)."""
import json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

SRC = Path(sys.argv[1])
OUT = Path(sys.argv[2])
data = json.loads(SRC.read_text())

TITLES = [
    ("square", "Square (사각 너트 끼우기)"),
    ("threading", "Threading (바늘 꿰기)"),
    ("coffee", "Coffee (커피 팟 넣기)"),
    ("three_piece_assembly", "Three Piece Assembly (3조각 조립)"),
    ("stack", "Stack (블록 2개 쌓기)"),
    ("stack_three", "Stack Three (블록 3개 쌓기)"),
    ("mug_cleanup", "Mug Cleanup (머그 정리)"),
    ("hammer_cleanup", "Hammer Cleanup (망치 정리)"),
    ("coffee_preparation", "Coffee Preparation (커피 준비·5단계)"),
]
GRAY = "#c9c9c9"
GREEN = "#2a9d8f"

fig, axes = plt.subplots(3, 3, figsize=(17, 12.5))
for ax, (task, title) in zip(axes.flat, TITLES):
    att = {int(k): v for k, v in data[task]["att"].items()}
    ret = {int(k): v for k, v in data[task]["ret"].items()}
    srcs = sorted(set(att) | set(ret))
    ta, tr = max(sum(att.values()), 1), max(sum(ret.values()), 1)
    att_pct = {s: att.get(s, 0) / ta * 100 for s in srcs}
    ret_pct = {s: ret.get(s, 0) / tr * 100 for s in srcs}
    order = sorted(srcs, key=lambda s: -ret_pct[s])
    x = range(len(order))
    w = 0.4
    ax.bar([i - w / 2 for i in x], [att_pct[s] for s in order], width=w, color=GRAY, zorder=2)
    ax.bar([i + w / 2 for i in x], [ret_pct[s] for s in order], width=w, color=GREEN, zorder=2)
    ax.axhline(10, color="#999", lw=1.2, ls="--", zorder=1)
    ymax = max(max(att_pct.values()), max(ret_pct.values()))
    ax.text(len(order) - 0.4, 10 + ymax * 0.02, "고르면 10%", color="#999", fontsize=10, ha="right")
    ax.set_title(f"{title}  (D0)", fontsize=13)
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"s{s}" for s in order], fontsize=9)
    ax.set_ylabel("비중 (%)", fontsize=10)
    ax.tick_params(axis="y", labelsize=9)
    ax.grid(axis="y", color="#eee", zorder=0)
    ax.set_axisbelow(True)

fig.suptitle("원본 데모 10개의 기여도 — 좁은 D0 영역: 시도(회색)도 살아남은 데이터(초록)도 대체로 고르다", fontsize=15)
handles = [plt.Rectangle((0, 0), 1, 1, color=GRAY), plt.Rectangle((0, 0), 1, 1, color=GREEN)]
fig.legend(handles, ["시도 중 비중 (원본을 고르게 뽑음)", "살아남은 데이터 중 비중"],
           loc="lower center", ncol=2, fontsize=12, frameon=False)
fig.tight_layout(rect=(0, 0.03, 1, 0.95))
fig.savefig(OUT, dpi=140)
print(f"saved {OUT}")
