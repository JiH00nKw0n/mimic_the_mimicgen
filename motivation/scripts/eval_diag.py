"""Evaluate baseline (+transform_uniform) checkpoints on the diagnostic
oversampled reset set. Writes eval/diag_e2_<task>_<arm>_seed<s>.jsonl.
Usage: eval_diag.py <task> [<task> ...]   (env MNEW_ARMS, MNEW_SEEDS override)
"""
import os, sys
from multiprocessing import Pool

REPO = "/home/ubuntu/mimicgen_jihoonkwon/mimic_the_mimicgen/motivation"
R = "/home/ubuntu/mimicgen_jihoonkwon/experiments/motivation_ic/e2_results"
A = "/home/ubuntu/mimicgen_jihoonkwon/experiments/motivation_new/e2_arms"
ARMS = os.environ.get("MNEW_ARMS", "baseline,transform_uniform").split(",")
SEEDS = [int(x) for x in os.environ.get("MNEW_SEEDS", "101,102,103,104,105,106").split(",")]
HORIZON = 400
WORKERS = 6


def eval_one(args):
    task, arm, seed = args
    sys.path.insert(0, REPO)
    from pathlib import Path
    from genaudit.evaluation.frozen_resets import evaluate_policy_on_frozen_resets
    out = Path(A) / f"{task}_N2" / "eval" / f"diag_e2_{task}_{arm}_seed{seed}.jsonl"
    if out.exists() and sum(1 for _ in open(out) if _.strip()) >= 190:
        return (task, arm, seed, "cached")
    ckpts = list(Path("/home/ubuntu/mimicgen_jihoonkwon/experiments/motivation_new/e2_results").rglob(
        f"e2_{task}_{arm}_seed{seed}/**/model_epoch_2000.pth"))
    if not ckpts:
        ckpts = list(Path(R).rglob(f"e2_{task}_{arm}_seed{seed}/**/model_epoch_2000.pth"))
    if not ckpts:
        return (task, arm, seed, "no_ckpt")
    resets = Path(A) / f"{task}_N2" / "frozen_resets_diag.hdf5"
    if not resets.exists():
        return (task, arm, seed, "no_resets")
    try:
        recs = evaluate_policy_on_frozen_resets(str(ckpts[0]), str(resets), HORIZON, str(out))
        return (task, arm, seed, round(sum(r["success"] for r in recs) / len(recs), 3))
    except Exception as e:  # noqa: BLE001
        return (task, arm, seed, f"ERR {type(e).__name__}: {e}")


if __name__ == "__main__":
    tasks = sys.argv[1:]
    jobs = [(t, a, s) for t in tasks for a in ARMS for s in SEEDS]
    with Pool(WORKERS) as p:
        for r in p.imap_unordered(eval_one, jobs):
            print("[diag-eval]", r, flush=True)
    print("DIAG EVAL DONE", flush=True)
