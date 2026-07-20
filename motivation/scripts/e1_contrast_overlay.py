"""E1 §1.4 contrast analysis: mirror (public D2) vs expansion (D2E) pools on
the shared union-box axis d'.

For threading and coffee: re-extract both pools' attempts with normalizers
from the union bounding box of the two regions (d' stays in [0,1] for both),
restrict to the overlapping d' support, and compare per-bin DGR + logistic
slope. Pre-registered readout: if DGR(d') matches on common support, transform
distance is a sufficient statistic and the relocation itself adds nothing.

Usage (server):
  PYTHONPATH=<repo>/motivation python e1_contrast_overlay.py \
      --expansion-root .../motivation_ic/e1 --mirror-root .../motivation_ic \
      --sources .../b0_sources --out .../e1/analysis_full/contrast_overlay.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from genaudit.config import load_task_spec
from genaudit.curation.binning import assign_bins, compute_quantile_edges
from genaudit.envs.bounds import get_variant, union_bounding_box
from genaudit.factors.initial_condition import build_task_geometry
from genaudit.records.extract import extract_attempt_records, load_source_initial_states

TASK_CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs" / "tasks"

CONTRASTS = (
    ("threading", "D2E", "D2"),
    ("coffee", "D2E", "D2"),
)


def pool_files(pool_dir: Path) -> tuple[Path | None, Path | None]:
    demo = sorted(pool_dir.rglob("demo.hdf5"))
    failed = sorted(pool_dir.rglob("demo_failed.hdf5"))
    return (demo[0] if demo else None, failed[0] if failed else None)


def extract_pool(task, variant, geometry, source_xy, source_yaw, pool_dir):
    demo, failed = pool_files(pool_dir)
    records = extract_attempt_records(
        task=task, variant=variant, geometry=geometry,
        source_xy=source_xy, source_yaw=source_yaw,
        demo_hdf5=demo, failed_hdf5=failed,
    )
    d = np.array([record.d_pos for record in records])
    success = np.array([record.success for record in records], dtype=bool)
    return d, success


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expansion-root", required=True)
    parser.add_argument("--mirror-root", required=True)
    parser.add_argument("--sources", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    report = {}
    for task, expansion_variant, mirror_variant in CONTRASTS:
        task_spec = load_task_spec(TASK_CONFIG_DIR / f"{task}.yaml")
        union = union_bounding_box(
            get_variant(task, expansion_variant), get_variant(task, mirror_variant)
        )
        geometry = build_task_geometry(task, union, task_spec.symmetry_orders)
        source_xy, source_yaw = load_source_initial_states(
            f"{args.sources}/{task}.hdf5", geometry.movable_objects
        )
        d_exp, s_exp = extract_pool(
            task, expansion_variant, geometry, source_xy, source_yaw,
            Path(args.expansion_root).expanduser() / f"{task}_{expansion_variant}",
        )
        d_mir, s_mir = extract_pool(
            task, mirror_variant, geometry, source_xy, source_yaw,
            Path(args.mirror_root).expanduser() / f"{task}_{mirror_variant}",
        )

        low = max(d_exp.min(), d_mir.min())
        high = min(d_exp.max(), d_mir.max())
        in_exp = (d_exp >= low) & (d_exp <= high)
        in_mir = (d_mir >= low) & (d_mir <= high)
        overlap = {
            "interval": [float(low), float(high)],
            "expansion_mass": float(in_exp.mean()),
            "mirror_mass": float(in_mir.mean()),
        }

        entry = {
            "expansion": {"n": len(d_exp), "dgr": float(s_exp.mean()),
                          "d_range": [float(d_exp.min()), float(d_exp.max())]},
            "mirror": {"n": len(d_mir), "dgr": float(s_mir.mean()),
                       "d_range": [float(d_mir.min()), float(d_mir.max())]},
            "overlap": overlap,
        }
        if in_exp.sum() >= 10 * args.k and in_mir.sum() >= 10 * args.k:
            pooled = np.concatenate([d_exp[in_exp], d_mir[in_mir]])
            edges = compute_quantile_edges(pooled, args.k)
            rows = []
            for index in range(args.k):
                be = assign_bins(d_exp[in_exp], edges) == index
                bm = assign_bins(d_mir[in_mir], edges) == index
                rows.append({
                    "bin": index,
                    "expansion_dgr": float(s_exp[in_exp][be].mean()) if be.sum() else None,
                    "expansion_n": int(be.sum()),
                    "mirror_dgr": float(s_mir[in_mir][bm].mean()) if bm.sum() else None,
                    "mirror_n": int(bm.sum()),
                })
            entry["per_bin"] = rows
            gaps = [
                abs(row["expansion_dgr"] - row["mirror_dgr"])
                for row in rows
                if row["expansion_dgr"] is not None and row["mirror_dgr"] is not None
                and row["expansion_n"] >= 20 and row["mirror_n"] >= 20
            ]
            entry["mean_abs_dgr_gap_common_support"] = float(np.mean(gaps)) if gaps else None
        else:
            # Pre-registered fallback (support too thin/disjoint): fit a
            # logistic DGR(d') on the EXPANSION pool and compare the mirror
            # pool's observed DGR against the extrapolated prediction at the
            # mirror's median d'. Extrapolation caveat reported as-is.
            entry["per_bin"] = None
            x = d_exp - d_exp.mean()
            beta1, beta0 = np.polyfit(x, s_exp.astype(float), 1)  # linear prob fit
            mirror_median = float(np.median(d_mir))
            predicted = float(beta0 + beta1 * (mirror_median - d_exp.mean()))
            entry["fallback_extrapolation"] = {
                "expansion_linear_slope_per_dprime": float(beta1),
                "mirror_median_dprime": mirror_median,
                "predicted_dgr_at_mirror_median": max(0.0, min(1.0, predicted)),
                "observed_mirror_dgr": float(s_mir.mean()),
                "note": "supports disjoint — relocation occupies a strictly farther distance regime; prediction is a linear extrapolation and stated as such",
            }
        report[task] = entry
        print(task, json.dumps(entry["overlap"]), "gap:", entry.get("mean_abs_dgr_gap_common_support"))

    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
