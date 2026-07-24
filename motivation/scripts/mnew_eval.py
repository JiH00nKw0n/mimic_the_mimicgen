"""motivation_new policy evaluation: roll every trained (task, arm, seed)
checkpoint on its task's frozen resets, then aggregate per-arm SR + paired.

Usage:
  mnew_eval.py <task> [<task> ...]            # evaluate (policy-parallel)
  mnew_eval.py --aggregate <task> [...]       # write per-task eval_summary.json
"""
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np

REPO = "/home/ubuntu/mimicgen_jihoonkwon/mimic_the_mimicgen/motivation"
A = Path("/home/ubuntu/mimicgen_jihoonkwon/experiments/motivation_new/e2_arms")
R = Path("/home/ubuntu/mimicgen_jihoonkwon/experiments/motivation_new/e2_results")
ARMS = ["baseline", "transform_uniform", "ancestry_balanced"]
SEEDS = [int(x) for x in os.environ.get("MNEW_SEEDS_ALL", "101,102").split(",")]
HORIZON = 400
WORKERS = 6


def eval_policy(args):
    task, arm, seed = args
    sys.path.insert(0, REPO)
    from genaudit.evaluation.frozen_resets import evaluate_policy_on_frozen_resets
    out = A / f"{task}_N2" / "eval" / f"e2_{task}_{arm}_seed{seed}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and sum(1 for _ in open(out) if _.strip()) >= 200:
        return (task, arm, seed, "cached")
    ckpts = list(R.rglob(f"e2_{task}_{arm}_seed{seed}/**/model_epoch_2000.pth"))
    if not ckpts:
        return (task, arm, seed, "no_ckpt")
    resets = A / f"{task}_N2" / "frozen_resets.hdf5"
    if not resets.exists():
        return (task, arm, seed, "no_resets")
    try:
        recs = evaluate_policy_on_frozen_resets(str(ckpts[0]), str(resets), HORIZON, str(out))
        return (task, arm, seed, round(sum(r["success"] for r in recs) / len(recs), 3))
    except Exception as e:  # noqa: BLE001
        return (task, arm, seed, f"ERR {type(e).__name__}: {e}")


def aggregate(tasks):
    for task in tasks:
        d = A / f"{task}_N2" / "eval"
        summary = {"task": task, "variant": "N2", "arms": {}}
        per_arm_ep = {}
        for arm in ARMS:
            per_seed = []
            ep_by_seed = []
            for s in SEEDS:
                f = d / f"e2_{task}_{arm}_seed{s}.jsonl"
                if not f.exists():
                    continue
                recs = [json.loads(x) for x in open(f) if x.strip()]
                if len(recs) < 200:
                    continue
                sr = {r["reset_index"]: bool(r["success"]) for r in recs}
                per_seed.append(np.mean(list(sr.values())))
                ep_by_seed.append(sr)
            if per_seed:
                summary["arms"][arm] = {"mean_sr": round(float(np.mean(per_seed)), 4),
                                        "per_seed_sr": [round(float(v), 4) for v in per_seed],
                                        "n_seeds": len(per_seed)}
                if ep_by_seed:
                    idx = sorted(ep_by_seed[0])
                    per_arm_ep[arm] = np.mean([[e[i] for i in idx] for e in ep_by_seed], axis=0)

        def paired(a, b):
            if a not in per_arm_ep or b not in per_arm_ep:
                return None
            va, vb = per_arm_ep[a] > 0.5, per_arm_ep[b] > 0.5
            return {"a_only": int(np.sum(va & ~vb)), "b_only": int(np.sum(~va & vb)),
                    "both": int(np.sum(va & vb)), "neither": int(np.sum(~va & ~vb))}
        summary["paired"] = {
            "baseline_vs_transform_uniform": paired("baseline", "transform_uniform"),
            "baseline_vs_ancestry_balanced": paired("baseline", "ancestry_balanced")}
        (d / "eval_summary.json").write_text(json.dumps(summary, indent=2))
        arms_str = "  ".join(f"{a}={summary['arms'][a]['mean_sr']}" for a in summary["arms"])
        print(f"[agg] {task}: {arms_str}", flush=True)


def main():
    args = sys.argv[1:]
    if args and args[0] == "--aggregate":
        aggregate(args[1:])
        return
    tasks = args
    jobs = [(t, arm, s) for t in tasks for arm in ARMS for s in SEEDS]
    with mp.get_context("spawn").Pool(WORKERS, maxtasksperchild=1) as pool:
        for r in pool.imap_unordered(eval_policy, jobs):
            print(f"[eval] {r}", flush=True)
    aggregate(tasks)
    print("EVAL DONE", flush=True)


if __name__ == "__main__":
    main()
