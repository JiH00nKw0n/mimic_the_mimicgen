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
those boundaries automatically using the task's built-in heuristics and writes
a new annotated dataset. (Some tasks need manual annotation, but Franka cube
stacking supports the automatic path.)

Note we use the `...-Mimic-v0` task variant here: it is the same cube-stacking
environment but carries the extra Mimic metadata the annotator/generator need.

Data flow (because Isaac Lab runs inside the container):
    host datasets/source_dataset.hdf5
      --docker cp-->  container /workspace/isaaclab/datasets/source_dataset.hdf5
      --annotate-->   container .../annotated_dataset.hdf5
      --docker cp-->  host datasets/annotated_dataset.hdf5

Run it like:
    python3 scripts/02_annotate.py
"""

from __future__ import annotations

import sys

from _common import (
    CONTAINER_DATA,
    DATASETS_DIR,
    TASK_MIMIC,
    cp_from_container,
    cp_to_container,
    ensure_container_dirs,
    in_container,
    require_container,
)

SRC_HOST = DATASETS_DIR / "source_dataset.hdf5"
ANNOTATED_HOST = DATASETS_DIR / "annotated_dataset.hdf5"

# Paths inside the container's scratch datasets folder.
SRC_CONTAINER = f"{CONTAINER_DATA}/source_dataset.hdf5"
ANNOTATED_CONTAINER = f"{CONTAINER_DATA}/annotated_dataset.hdf5"


def main() -> int:
    require_container()
    if not SRC_HOST.exists():
        sys.exit(f"[ERROR] {SRC_HOST} not found. Run scripts/01_download_dataset.py first.")

    ensure_container_dirs()

    # 1) Copy the source demos into the container.
    print("[1/3] Copying source dataset into the container ...")
    cp_to_container(SRC_HOST, SRC_CONTAINER)

    # 2) Run the auto-annotator inside the container.
    #    --device cpu : annotation is light; CPU avoids competing for the GPU.
    #    --auto       : detect subtask boundaries automatically.
    print("[2/3] Annotating subtasks (auto) inside the container ...")
    in_container(
        "./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/annotate_demos.py "
        f"--device cpu --task {TASK_MIMIC} --auto "
        f"--input_file {SRC_CONTAINER} --output_file {ANNOTATED_CONTAINER}"
    )

    # 3) Copy the annotated dataset back out to the host.
    print("[3/3] Copying annotated dataset back to the host ...")
    cp_from_container(ANNOTATED_CONTAINER, ANNOTATED_HOST)

    print(f"\nOK: {ANNOTATED_HOST}")
    print("Next: python3 scripts/03_generate.py --mode small")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
