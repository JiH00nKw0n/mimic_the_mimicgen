#!/usr/bin/env python3
"""
STEP 1 - Download the source human demonstrations.

What this is
------------
NVIDIA provides a small HDF5 file containing 10 human teleoperation demos of a
Franka arm stacking three cubes. This is the "seed" data that MimicGen will
later multiply into ~1000 synthetic demos. We just download it onto the host;
no GPU or container is needed for this step.

The file is hosted on NVIDIA's public Omniverse S3 bucket (no login required).

Output:
    datasets/source_dataset.hdf5   (on the host)

Run it like:
    python3 scripts/01_download_dataset.py
"""

from __future__ import annotations

import sys
import urllib.request

from _common import DATASETS_DIR

# Public S3 URL for the 10-demo Franka cube-stacking dataset (from the tutorial).
DATASET_URL = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/"
    "Assets/Isaac/5.1/Isaac/IsaacLab/Mimic/franka_stack_datasets/dataset.hdf5"
)
OUT_PATH = DATASETS_DIR / "source_dataset.hdf5"


def _progress(block_num: int, block_size: int, total_size: int) -> None:
    """Print a simple percent-complete line while downloading."""
    if total_size <= 0:
        return
    downloaded = block_num * block_size
    pct = min(100.0, downloaded * 100.0 / total_size)
    mb = downloaded / 1e6
    total_mb = total_size / 1e6
    print(f"\r  downloading... {pct:5.1f}%  ({mb:6.1f} / {total_mb:6.1f} MB)", end="", flush=True)


def main() -> int:
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)

    if OUT_PATH.exists():
        print(f"[skip] {OUT_PATH} already exists ({OUT_PATH.stat().st_size/1e6:.1f} MB).")
        return 0

    print(f"Downloading source demos from:\n  {DATASET_URL}\n-> {OUT_PATH}")
    try:
        urllib.request.urlretrieve(DATASET_URL, OUT_PATH, _progress)
        print()  # newline after the progress line
    except Exception as e:  # noqa: BLE001 - we want a friendly message on any failure
        sys.exit(f"\n[ERROR] download failed: {e}")

    # HDF5 files start with the 8-byte signature b"\x89HDF\r\n\x1a\n". A quick
    # check here catches the common failure where S3 returns an HTML error page
    # instead of the file.
    with open(OUT_PATH, "rb") as f:
        magic = f.read(8)
    if magic != b"\x89HDF\r\n\x1a\n":
        OUT_PATH.unlink(missing_ok=True)
        sys.exit("[ERROR] downloaded file is not a valid HDF5 file (got unexpected header).")

    print(f"OK: {OUT_PATH}  ({OUT_PATH.stat().st_size/1e6:.1f} MB)")
    print("Next: python3 scripts/02_annotate.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
