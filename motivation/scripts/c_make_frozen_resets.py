"""Create the shared frozen-reset evaluation set for each E2 task (PLAN §2.6).

One fixed set of `num_resets` initial states per task, sampled from that task's
D2/D2E reset distribution and stored, so every arm and seed is evaluated on the
SAME episodes (paired / McNemar analysis). Built once, before evaluation.

Server usage (light CPU; safe to run alongside training):
  PYTHONPATH=<repo>/motivation python c_make_frozen_resets.py \
      --arms-root ~/mimicgen_jihoonkwon/experiments/motivation_ic/e2_arms \
      --tasks square:D2 threading:D2E coffee:D2E stack:D2E \
      --num-resets 200 --seed 7
"""
from __future__ import annotations

import argparse
from pathlib import Path

from genaudit.evaluation.frozen_resets import create_frozen_resets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms-root", required=True)
    parser.add_argument("--tasks", nargs="+", required=True, help="task:variant pairs")
    parser.add_argument("--num-resets", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    arms_root = Path(args.arms_root).expanduser()
    for entry in args.tasks:
        task, variant = entry.split(":")
        train_hdf5 = arms_root / f"{task}_{variant}" / "train.hdf5"
        if not train_hdf5.exists():
            print(f"SKIP {entry}: {train_hdf5} missing")
            continue
        out = arms_root / f"{task}_{variant}" / "frozen_resets.hdf5"
        if out.exists():
            print(f"SKIP {entry}: {out} exists")
            continue
        create_frozen_resets(train_hdf5, args.num_resets, args.seed, out)
        print(f"wrote {out} ({args.num_resets} resets)")


if __name__ == "__main__":
    main()
