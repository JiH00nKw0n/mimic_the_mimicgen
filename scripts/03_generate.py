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

Two modes (both run headless, since this is a display-less server; you watch
the robot later via recorded video or WebRTC, not live here):
  --mode small : 10 trials. A quick sanity check that the whole pipeline runs
                 and produces at least one success.
  --mode full  : 1000 trials. The real run; expect ~30 min on this server
                 (CPU-bound, 4 vCPUs).

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
    get_profile,
    in_container,
    require_container,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a synthetic MimicGen dataset.")
    parser.add_argument("--profile", default="franka", help="franka (default) or gr1t2.")
    parser.add_argument(
        "--mode", choices=["small", "full"], default="small",
        help="small = 10 trials (sanity check); full = 1000 trials (real run).",
    )
    parser.add_argument(
        "--num-envs", type=int, default=None,
        help="Parallel environments (default: per-profile - franka 10, gr1t2 20).",
    )
    args = parser.parse_args()
    profile = get_profile(args.profile)
    num_envs = args.num_envs if args.num_envs is not None else profile.num_envs

    require_container()
    annotated_host = DATASETS_DIR / profile.annotated_file
    annotated_container = f"{CONTAINER_DATA}/{profile.annotated_file}"
    if not annotated_host.exists():
        prev = "01_download_dataset.py" if profile.pre_annotated else "02_annotate.py"
        sys.exit(f"[ERROR] {annotated_host} not found. Run scripts/{prev} --profile {profile.name} first.")

    # Mode-specific settings. Both run headless on this display-less server.
    headless_flag = "--headless"
    if args.mode == "small":
        num_trials = 10
        out_name = profile.generated_small_file
    else:
        num_trials = 1000
        out_name = profile.generated_file

    out_container = f"{CONTAINER_DATA}/{out_name}"
    out_host = DATASETS_DIR / out_name
    pinocchio = "--enable_pinocchio " if profile.enable_pinocchio else ""

    ensure_container_dirs()

    # 1) Make sure the annotated dataset is present inside the container.
    print("[1/3] Copying annotated dataset into the container ...")
    cp_to_container(annotated_host, annotated_container)

    # 2) Run MimicGen generation inside the container.
    print(f"[2/3] Generating {num_trials} demos (profile={profile.name}, mode={args.mode}, num_envs={num_envs}) ...")
    in_container(
        "./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/generate_dataset.py "
        f"--device cpu {headless_flag} {pinocchio}--num_envs {num_envs} "
        f"--generation_num_trials {num_trials} "
        f"--input_file {annotated_container} --output_file {out_container}"
    )

    # 3) Copy the generated dataset back to the host.
    print("[3/3] Copying generated dataset back to the host ...")
    cp_from_container(out_container, out_host)

    print(f"\nOK: {out_host}")
    if args.mode == "small":
        print(f"Sanity run done. Real run: python3 scripts/03_generate.py --profile {profile.name} --mode full")
    else:
        print(f"Next: record a video -> python3 scripts/04_record_video.py --profile {profile.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
