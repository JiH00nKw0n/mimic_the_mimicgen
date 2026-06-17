#!/usr/bin/env python3
"""
STEP 4 - Record an MP4 video of the generated demos (the main "view" path).

This is the host-side driver. The actual rendering happens inside the container
via `_record_video_inproc.py` (see that file for the details). Here we just:

  1. make sure imageio (the MP4 writer) is available inside the container,
  2. copy the dataset and the in-container replay script into the container,
  3. run the replay+record script with cameras enabled (headless offscreen),
  4. copy the resulting MP4s back to the host under outputs/videos/.

After this, run setup/sync_from_remote.sh on your laptop to pull the videos
down, then just open the .mp4 files.

By default it records from the full generated dataset; pass --dataset to point
at a different HDF5 (e.g. the small sanity dataset).

Run it like:
    python3 scripts/04_record_video.py                 # 5 episodes from the full set
    python3 scripts/04_record_video.py --num-episodes 3
    python3 scripts/04_record_video.py --dataset datasets/generated_dataset_small.hdf5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _common import (
    CONTAINER_DATA,
    CONTAINER_OUT,
    DATASETS_DIR,
    ISAACLAB_PATH,
    OUTPUTS_DIR,
    cp_from_container,
    cp_to_container,
    ensure_container_dirs,
    in_container,
    require_container,
)

# The in-container replay/record script (lives next to this file on the host).
INPROC_HOST = Path(__file__).resolve().parent / "_record_video_inproc.py"
INPROC_CONTAINER = f"{ISAACLAB_PATH}/_record_video_inproc.py"


def main() -> int:
    parser = argparse.ArgumentParser(description="Record MP4 videos of replayed demos.")
    parser.add_argument(
        "--dataset", default=str(DATASETS_DIR / "generated_dataset.hdf5"),
        help="Host path to the HDF5 dataset to replay (default: the full generated set).",
    )
    parser.add_argument("--num-episodes", type=int, default=5, help="How many episodes to record.")
    args = parser.parse_args()

    require_container()
    dataset_host = Path(args.dataset)
    if not dataset_host.exists():
        sys.exit(f"[ERROR] {dataset_host} not found. Generate it first (scripts/03_generate.py).")

    ensure_container_dirs()

    # 1) imageio writes the MP4. It is not in the base image, so install it once
    #    inside the container (cached afterwards). imageio-ffmpeg bundles ffmpeg.
    print("[1/4] Ensuring imageio is available in the container ...")
    in_container("./isaaclab.sh -p -m pip install --quiet imageio imageio-ffmpeg")

    # 2) Copy the dataset and the in-container record script into the container.
    print("[2/4] Copying dataset + replay script into the container ...")
    dataset_container = f"{CONTAINER_DATA}/{dataset_host.name}"
    cp_to_container(dataset_host, dataset_container)
    cp_to_container(INPROC_HOST, INPROC_CONTAINER)

    # 3) Run the headless replay+record inside the container.
    print(f"[3/4] Rendering {args.num_episodes} episode(s) to MP4 (headless, cameras on) ...")
    in_container(
        f"./isaaclab.sh -p {INPROC_CONTAINER} "
        f"--dataset_file {dataset_container} --num_episodes {args.num_episodes} "
        f"--video_dir {CONTAINER_OUT}/videos"
    )

    # 4) Copy the videos back to the host.
    print("[4/4] Copying MP4s back to the host ...")
    (OUTPUTS_DIR / "videos").mkdir(parents=True, exist_ok=True)
    cp_from_container(f"{CONTAINER_OUT}/videos/.", OUTPUTS_DIR / "videos")

    print(f"\nOK: videos under {OUTPUTS_DIR / 'videos'}")
    print("Now pull them to your laptop:  bash setup/sync_from_remote.sh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
