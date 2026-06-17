#!/usr/bin/env python3
"""
STEP 3 - Generate the synthetic dataset with MimicGen.

What happens here
-----------------
This is the actual data-multiplication step. Isaac Lab's `generate_dataset.py`
takes the annotated demos and, for each new randomized scene, transforms each
subtask segment to the new object positions, stitches the segments together,
and REPLAYS the result in simulation. Only attempts that actually succeed at the
task are kept. From 10 human demos it produces ~1000 synthetic ones.

Two modes:
  --mode small : 10 trials, no --headless. A quick sanity check that the whole
                 pipeline runs and produces at least one success.
  --mode full  : 1000 trials, --headless (no GUI) for throughput. This is the
                 real run; expect ~30 min on this server (CPU-bound, 4 vCPUs).

We keep num_envs at 10 (the tutorial's default for modest hardware). Higher
values help on big multi-core machines, but this server has only 4 vCPUs so the
benefit is limited.

Data flow: annotated_dataset.hdf5 (host) -> container -> generate ->
           generated_dataset[_small].hdf5 copied back to host.
"""

from __future__ import annotations

import argparse
import sys

from _common import (
    CONTAINER_DATA,
    DATASETS_DIR,
    cp_from_container,
    cp_to_container,
    ensure_container_dirs,
    in_container,
    require_container,
)

ANNOTATED_HOST = DATASETS_DIR / "annotated_dataset.hdf5"
ANNOTATED_CONTAINER = f"{CONTAINER_DATA}/annotated_dataset.hdf5"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a synthetic MimicGen dataset.")
    parser.add_argument(
        "--mode", choices=["small", "full"], default="small",
        help="small = 10 trials (sanity check); full = 1000 trials (real run, headless).",
    )
    parser.add_argument("--num-envs", type=int, default=10, help="Parallel environments (default 10).")
    args = parser.parse_args()

    require_container()
    if not ANNOTATED_HOST.exists():
        sys.exit(f"[ERROR] {ANNOTATED_HOST} not found. Run scripts/02_annotate.py first.")

    # Mode-specific settings.
    if args.mode == "small":
        num_trials = 10
        headless_flag = ""               # keep rendering on for the quick check
        out_name = "generated_dataset_small.hdf5"
    else:
        num_trials = 1000
        headless_flag = "--headless"     # no GUI -> faster for the big run
        out_name = "generated_dataset.hdf5"

    out_container = f"{CONTAINER_DATA}/{out_name}"
    out_host = DATASETS_DIR / out_name

    ensure_container_dirs()

    # 1) Make sure the annotated dataset is present inside the container.
    print("[1/3] Copying annotated dataset into the container ...")
    cp_to_container(ANNOTATED_HOST, ANNOTATED_CONTAINER)

    # 2) Run MimicGen generation inside the container.
    print(f"[2/3] Generating {num_trials} demos (mode={args.mode}, num_envs={args.num_envs}) ...")
    in_container(
        "./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/generate_dataset.py "
        f"--device cpu {headless_flag} --num_envs {args.num_envs} "
        f"--generation_num_trials {num_trials} "
        f"--input_file {ANNOTATED_CONTAINER} --output_file {out_container}"
    )

    # 3) Copy the generated dataset back to the host.
    print("[3/3] Copying generated dataset back to the host ...")
    cp_from_container(out_container, out_host)

    print(f"\nOK: {out_host}")
    if args.mode == "small":
        print("Sanity run done. For the real run: python3 scripts/03_generate.py --mode full")
    else:
        print("Next: record a video -> python3 scripts/04_record_video.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
