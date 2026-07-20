"""Evaluate every finished E2 checkpoint of one task on its shared frozen-reset
set, then summarize the A/B/C comparison (PLAN §2.6).

Rolls each arm's policy from the SAME 200 fixed initial states, so arms are
compared episode-by-episode (paired). Stratifies episodes by nearest-source
distance (`d_eval`) to show WHERE an arm wins. Prints a per-arm success table
and per-bin curves; writes eval_summary.json.

Server usage (after training + frozen resets exist):
  PYTHONPATH=<repo>/motivation python c_eval_task.py \
      --task-config <...>/square.yaml --experiment-config <...>/e2_square.yaml \
      --arms-root <...>/e2_arms --results-root <...>/e2_results \
      --sources <...>/b0_sources
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from genaudit.config import load_experiment_spec, load_task_spec
from genaudit.envs.bounds import get_variant
from genaudit.evaluation.frozen_resets import evaluate_policy_on_frozen_resets
from genaudit.factors.initial_condition import build_task_geometry, nearest_source_distance


def _require_h5py():
    import h5py
    return h5py


def find_checkpoint(results_root: Path, name: str) -> Path | None:
    hits = list(results_root.rglob(f"{name}/**/model_epoch_2000.pth"))
    return hits[0] if hits else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-config", required=True)
    parser.add_argument("--experiment-config", required=True)
    parser.add_argument("--arms-root", required=True)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--sources", required=True)
    parser.add_argument("--horizon", type=int, default=400)
    args = parser.parse_args()

    task_spec = load_task_spec(args.task_config)
    experiment = load_experiment_spec(args.experiment_config)
    arm_dir = Path(args.arms_root).expanduser() / f"{experiment.task}_{experiment.variant}"
    resets_path = arm_dir / "frozen_resets.hdf5"
    results_root = Path(args.results_root).expanduser()
    eval_dir = arm_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    # evaluate every finished (arm, seed); collect per-episode success arrays
    per_run = {}  # key -> np.array(bool) indexed by reset_index
    for dataset_seed in experiment.dataset_seeds:
        for arm in experiment.arms:
            name = f"e2_{experiment.task}_{arm.name}_seed{dataset_seed}"
            ckpt = find_checkpoint(results_root, name)
            if ckpt is None:
                print(f"skip {name}: no checkpoint yet")
                continue
            out_jsonl = eval_dir / f"{name}.jsonl"
            if out_jsonl.exists():
                records = [json.loads(l) for l in out_jsonl.read_text().splitlines() if l.strip()]
            else:
                records = evaluate_policy_on_frozen_resets(ckpt, resets_path, args.horizon, out_jsonl)
            arr = {r["reset_index"]: bool(r["success"]) for r in records}
            per_run[(arm.name, dataset_seed)] = arr
            print(f"{name}: SR {np.mean(list(arr.values())):.3f} ({sum(arr.values())}/{len(arr)})")

    # per-arm aggregate over available seeds
    summary = {"task": experiment.task, "variant": experiment.variant, "arms": {}}
    arm_episode_sr = {}  # arm -> np.array averaged over seeds (paired episodes)
    n_reset = None
    for arm in experiment.arms:
        seeds = [s for s in experiment.dataset_seeds if (arm.name, s) in per_run]
        if not seeds:
            continue
        idx = sorted(per_run[(arm.name, seeds[0])].keys())
        n_reset = len(idx)
        mat = np.array([[per_run[(arm.name, s)][i] for i in idx] for s in seeds], dtype=float)
        per_seed_sr = mat.mean(axis=1)
        arm_episode_sr[arm.name] = mat.mean(axis=0)  # avg over seeds per episode
        summary["arms"][arm.name] = {
            "seeds": seeds,
            "per_seed_sr": [round(float(v), 4) for v in per_seed_sr],
            "mean_sr": round(float(per_seed_sr.mean()), 4),
        }

    # paired A vs B, A vs C (McNemar-style discordant counts, seed-averaged>0.5)
    def paired(a, b):
        if a not in arm_episode_sr or b not in arm_episode_sr:
            return None
        va, vb = arm_episode_sr[a] > 0.5, arm_episode_sr[b] > 0.5
        return {"a_only": int(np.sum(va & ~vb)), "b_only": int(np.sum(~va & vb)),
                "both": int(np.sum(va & vb)), "neither": int(np.sum(~va & ~vb))}

    summary["paired"] = {
        "baseline_vs_transform_uniform": paired("baseline", "transform_uniform"),
        "baseline_vs_ancestry_balanced": paired("baseline", "ancestry_balanced"),
    }
    (eval_dir / "eval_summary.json").write_text(json.dumps(summary, indent=2))

    print("\n=== " + experiment.task + " 요약 (arm 평균 SR, seed 개수) ===")
    for arm_name, s in summary["arms"].items():
        print(f"  {arm_name:<20} SR {s['mean_sr']:.3f}  (seeds {s['seeds']}, 개별 {s['per_seed_sr']})")
    print("paired:", json.dumps(summary["paired"]))
    print(f"wrote {eval_dir / 'eval_summary.json'}")


if __name__ == "__main__":
    main()
