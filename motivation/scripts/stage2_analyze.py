"""Stage2 physics ablation analysis: DGR tables, per-source matrix, curves.

Consumes experiments/stage2_ablation/records/<task>_<arm>_attempts.jsonl and
emits analysis/{stage2_report.md, *.png}. Run on the server after extract:

  PYTHONPATH=$M $V scripts/stage2_analyze.py
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

M = Path("/home/ubuntu/mimicgen_jihoonkwon/mimic_the_mimicgen/motivation")
OUT = Path("/home/ubuntu/mimicgen_jihoonkwon/experiments/stage2_ablation")
REF_CASESTUDY = Path(
    "/home/ubuntu/mimicgen_jihoonkwon/experiments/motivation_new/e2_arms/"
    "casestudy_per_source.json")
TASKS = {"stack": (4, 7), "square": (0, 4)}
ARMS = ["P0_base", "P1_nominal", "P2_posterior", "P3_robust", "P5_omni", "P4_hisrc"]
MATRIX_ARMS = ["P0_base", "P1_nominal", "P2_posterior", "P3_robust", "P5_omni"]
PARAM_KEYS = ["s2_table_mu", "s2_cube_mu", "s2_finger_mu", "s2_force_scale"]

sys.path.insert(0, str(M))
from genaudit.analysis.dgr import dgr_vs_distance, trend_stats  # noqa: E402
from genaudit.records.schema import read_jsonl  # noqa: E402


def wilson_ci(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    if trials == 0:
        return (0.0, 0.0)
    p = successes / trials
    denom = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denom
    margin = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def spearman(a: list[float], b: list[float]) -> float:
    # pairwise-complete: a source absent from one arm must not poison the rank
    pairs = [(x, y) for x, y in zip(a, b)
             if not (math.isnan(x) or math.isnan(y))]
    if len(pairs) < 3:
        return float("nan")
    a = [p[0] for p in pairs]
    b = [p[1] for p in pairs]

    def ranks(values):
        order = sorted(range(len(values)), key=lambda i: values[i])
        rank = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                rank[order[k]] = avg
            i = j + 1
        return rank
    ra, rb = ranks(a), ranks(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va = math.sqrt(sum((x - ma) ** 2 for x in ra))
    vb = math.sqrt(sum((y - mb) ** 2 for y in rb))
    return cov / (va * vb) if va and vb else float("nan")


def load_all() -> dict:
    data: dict = {}
    for task in TASKS:
        for arm in ARMS:
            path = OUT / "records" / f"{task}_{arm}_attempts.jsonl"
            if path.exists():
                data[(task, arm)] = list(read_jsonl(path))
    return data


def source_id(record) -> int:
    return int(record.extras.get("orig_source_id", record.source_demo_id))


def per_source_dgr(records) -> dict[int, tuple[int, int]]:
    counts: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    for record in records:
        sid = source_id(record)
        counts[sid][0] += int(record.success)
        counts[sid][1] += 1
    return {sid: (s, n) for sid, (s, n) in counts.items()}


def main() -> None:
    analysis_dir = OUT / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    data = load_all()
    if not data:
        sys.exit("no records found — run stage2_ablation.py --stage extract first")

    reference = {}
    if REF_CASESTUDY.exists():
        casestudy = json.loads(REF_CASESTUDY.read_text())
        for task in TASKS:
            if task in casestudy:
                reference[task] = {int(r["src"]): float(r["DGR"])
                                   for r in casestudy[task]["rows"]}

    lines = ["# Stage2 Physics DGR Ablation — 결과 테이블", ""]

    # ------------------------------------------------ arm-level DGR table
    lines += ["## 1. Arm-level DGR (Wilson 95% CI)", "",
              "| task | arm | attempts | successes | DGR | CI95 |",
              "|---|---|---:|---:|---:|---|"]
    arm_dgr: dict = {}
    for (task, arm), records in sorted(data.items()):
        n = len(records)
        s = sum(r.success for r in records)
        if n == 0:
            lines.append(f"| {task} | {arm} | 0 | 0 | — | — |")
            arm_dgr[(task, arm)] = 0.0
            continue
        lo, hi = wilson_ci(s, n)
        arm_dgr[(task, arm)] = s / n
        lines.append(f"| {task} | {arm} | {n} | {s} | {s / n * 100:.1f}% "
                     f"| [{lo * 100:.1f}, {hi * 100:.1f}] |")
    lines.append("")

    # ------------------------------------------------ per-source matrix
    lines += ["## 2. Per-source DGR matrix (rows=source, cols=physics arm)", ""]
    matrices: dict = {}
    for task in TASKS:
        header = "| src | " + " | ".join(MATRIX_ARMS) + " | N2_ref |"
        lines += [f"### {task}", "", header,
                  "|" + "---|" * (len(MATRIX_ARMS) + 2)]
        matrix: dict[int, dict[str, float]] = defaultdict(dict)
        for arm in MATRIX_ARMS:
            records = data.get((task, arm), [])
            for sid, (s, n) in per_source_dgr(records).items():
                matrix[sid][arm] = s / n if n else float("nan")
        matrices[task] = matrix
        for sid in range(10):
            row = [f"{matrix[sid].get(arm, float('nan')) * 100:.0f}%"
                   if arm in matrix[sid] else "—" for arm in MATRIX_ARMS]
            ref = reference.get(task, {}).get(sid)
            lines.append(f"| s{sid} | " + " | ".join(row) + " | "
                         + (f"{ref * 100:.0f}%" if ref is not None else "—") + " |")
        lines.append("")

        base = [matrix[sid].get("P0_base", float("nan")) for sid in range(10)]
        lines.append("Spearman rank stability vs P0_base: " + ", ".join(
            f"{arm}={spearman(base, [matrix[s].get(arm, float('nan')) for s in range(10)]):.2f}"
            for arm in MATRIX_ARMS[1:]) + "\n")

    # ------------------------------------------------ P4 vs P3
    lines += ["## 3. hi-DGR source selection (P4, P3-physics)", "",
              "| task | P3_robust all-10 | P3 hi-2 subset | P4_hisrc | hi ids |",
              "|---|---:|---:|---:|---|"]
    for task, hi in TASKS.items():
        p3 = data.get((task, "P3_robust"), [])
        p4 = data.get((task, "P4_hisrc"), [])
        p3_hi = [r for r in p3 if source_id(r) in hi]
        def rate(rs):
            return (f"{sum(r.success for r in rs) / len(rs) * 100:.1f}%"
                    if rs else "—")
        lines.append(f"| {task} | {rate(p3)} | {rate(p3_hi)} | {rate(p4)} "
                     f"| {list(hi)} |")
    lines.append("")

    # ------------------------------------------------ DGR vs physics params
    lines += ["## 4. DGR vs realized physics (robust arms pooled, quintile bins)", ""]
    for task in TASKS:
        pooled = (data.get((task, "P3_robust"), [])
                  + data.get((task, "P5_omni"), [])
                  + data.get((task, "P4_hisrc"), []))
        pooled = [r for r in pooled if "s2_cube_mu" in r.extras]
        if not pooled:
            continue
        lines.append(f"### {task} (n={len(pooled)})")
        for key in PARAM_KEYS:
            try:
                curve = dgr_vs_distance(pooled, distance_key=key, k=5)
                trend = trend_stats(pooled, distance_key=key)
                bins = ", ".join(
                    f"{c:.2f}:{d * 100:.0f}%(n={n})" for c, d, n in
                    zip(curve.bin_centers, curve.per_bin_dgr, curve.per_bin_attempts))
                lines.append(f"- `{key}`: {bins} | spearman={trend.spearman_rho:.2f}")
            except Exception as error:  # noqa: BLE001
                lines.append(f"- `{key}`: ERR {type(error).__name__}: {error}")
        lines.append("")

    # ------------------------------------------------ realized-sample stats
    lines += ["## 5. Realized physics per arm (mean±sd of recorded s2_*)", ""]
    for (task, arm), records in sorted(data.items()):
        values: dict[str, list[float]] = defaultdict(list)
        for record in records:
            for key, value in record.extras.items():
                if key.startswith("s2_"):
                    values[key].append(float(value))
        if not values:
            continue
        stats = []
        for key in sorted(values):
            vs = values[key]
            mean = sum(vs) / len(vs)
            sd = math.sqrt(sum((v - mean) ** 2 for v in vs) / len(vs))
            stats.append(f"{key.removeprefix('s2_')}={mean:.3f}±{sd:.3f}")
        lines.append(f"- {task}/{arm}: " + ", ".join(stats))
    lines.append("")

    # ------------------------------------------------ known-inert axes
    lines += [
        "## 6. Integration notes (매핑 한계 — 해석 시 주의)", "",
        "- restitution·table dynamic friction·gripper speed_scale·cube "
        "linear/angular damping: 샘플·기록만 되고 sim에는 미적용 (stage2.py 헤더 참조).",
        "- P5 `joint_fric_scale`/`joint_armature_scale`: Panda 관절 XML에 "
        "frictionloss/armature attr이 없어 sim nominal 0 → x배율은 항상 0 (비활성 축).",
        "- `s2_force_ratio` 상향(>1)은 position servo 수요 한계(kp*error) 위에서 "
        "부분적으로 비활성 — 하향(<1)은 유효.",
        "- Square는 proxy 매핑(new condition): finger_cube->finger-nut, "
        "table_cube->table-nut, cube_cube->nut-peg.",
        "- Panda+robosuite 기하(0.04m 큐브)에 FR3+50.7mm 실측값의 '값'만 이식 — "
        "절대 충실도가 아니라 조건 간 상대 비교가 유효한 설계.",
        "",
    ]

    report = analysis_dir / "stage2_report.md"
    report.write_text("\n".join(lines))
    print(f"wrote {report}")

    # ------------------------------------------------ figures
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        figure, axes = plt.subplots(1, 2, figsize=(11, 4))
        for axis, task in zip(axes, TASKS):
            arms = [a for a in ARMS if (task, a) in data]
            rates = [arm_dgr[(task, a)] * 100 for a in arms]
            colors = ["#888" if a == "P0_base" else "#2a6" if a == "P4_hisrc"
                      else "#36c" for a in arms]
            axis.bar(range(len(arms)), rates, color=colors)
            axis.set_xticks(range(len(arms)))
            axis.set_xticklabels(arms, rotation=30, ha="right", fontsize=8)
            axis.set_title(f"{task}: DGR by physics arm")
            axis.set_ylabel("DGR %")
        figure.tight_layout()
        figure.savefig(analysis_dir / "dgr_by_arm.png", dpi=150)

        figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for axis, task in zip(axes, TASKS):
            matrix = matrices[task]
            grid = np.array([[matrix[s].get(a, np.nan) for a in MATRIX_ARMS]
                             for s in range(10)])
            image = axis.imshow(grid, vmin=0, vmax=1, cmap="viridis", aspect="auto")
            axis.set_xticks(range(len(MATRIX_ARMS)))
            axis.set_xticklabels(MATRIX_ARMS, rotation=30, ha="right", fontsize=8)
            axis.set_yticks(range(10))
            axis.set_yticklabels([f"s{i}" for i in range(10)])
            axis.set_title(f"{task}: per-source DGR")
            for i in range(10):
                for j in range(len(MATRIX_ARMS)):
                    if not np.isnan(grid[i, j]):
                        axis.text(j, i, f"{grid[i, j] * 100:.0f}",
                                  ha="center", va="center", fontsize=7,
                                  color="w" if grid[i, j] < 0.6 else "k")
            figure.colorbar(image, ax=axis, fraction=0.04)
        figure.tight_layout()
        figure.savefig(analysis_dir / "per_source_matrix.png", dpi=150)
        print("wrote figures")
    except Exception as error:  # noqa: BLE001
        print(f"figures skipped: {type(error).__name__}: {error}")


if __name__ == "__main__":
    main()
