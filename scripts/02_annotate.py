#!/usr/bin/env python3
"""
STEP 2 - Annotate the source demos with subtask boundaries.

What "annotation" means here
----------------------------
MimicGen does not copy a whole demo blindly. It splits each demo into
object-centric "subtasks" (for cube stacking: grasp cube 1, place it, grasp
cube 2, place it, ...). To do that it needs to know WHERE each subtask starts
and ends in every source demo.

This step runs Isaac Lab's `annotate_demos.py` with `--auto`, which detects
those boundaries automatically and writes a new annotated dataset. We use the
`...-Mimic-v0` task variant: the same environment but carrying the extra Mimic
metadata the annotator/generator need.

Profiles:
  franka : the download is raw, so we annotate it here.
  gr1t2  : the download is ALREADY annotated, so this step just exits - go
           straight to step 3.

Data flow (Isaac Lab runs inside the container):
    host source -> docker cp -> container -> annotate -> docker cp -> host

Run it like:
    python3 scripts/02_annotate.py                  # franka (default)
    python3 scripts/02_annotate.py --profile gr1t2  # no-op (already annotated)
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
    parser = argparse.ArgumentParser(description="Auto-annotate subtask boundaries.")
    parser.add_argument("--profile", default="franka", help="franka (default) or gr1t2.")
    args = parser.parse_args()
    profile = get_profile(args.profile)

    # gr1t2 ships pre-annotated, so there is nothing to do here.
    if profile.pre_annotated:
        print(f"[skip] profile '{profile.name}' is already annotated.")
        print(f"Next: python3 scripts/03_generate.py --profile {profile.name} --mode small")
        return 0

    require_container()
    src_host = DATASETS_DIR / profile.source_file
    annotated_host = DATASETS_DIR / profile.annotated_file
    if not src_host.exists():
        sys.exit(f"[ERROR] {src_host} not found. Run scripts/01_download_dataset.py first.")

    src_container = f"{CONTAINER_DATA}/{profile.source_file}"
    annotated_container = f"{CONTAINER_DATA}/{profile.annotated_file}"
    pinocchio = "--enable_pinocchio " if profile.enable_pinocchio else ""

    ensure_container_dirs()

    # 1) Copy the source demos into the container.
    print("[1/3] Copying source dataset into the container ...")
    cp_to_container(src_host, src_container)

    # 2) Auto-annotate inside the container (CPU is enough; keeps GPU free).
    print("[2/3] Annotating subtasks (auto) inside the container ...")
    in_container(
        "./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/annotate_demos.py "
        f"--device cpu {pinocchio}--task {profile.mimic_task} --auto "
        f"--input_file {src_container} --output_file {annotated_container}"
    )

    # 3) Copy the annotated dataset back out to the host.
    print("[3/3] Copying annotated dataset back to the host ...")
    cp_from_container(annotated_container, annotated_host)

    print(f"\nOK: {annotated_host}")
    print(f"Next: python3 scripts/03_generate.py --profile {profile.name} --mode small")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
