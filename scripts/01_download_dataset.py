#!/usr/bin/env python3
"""
STEP 1 - Download the source demonstrations.

What this is
------------
NVIDIA provides the "seed" demos on its public Omniverse S3 bucket (no login).
  - franka : 10 RAW human teleop demos of a Franka stacking cubes.
  - gr1t2  : an ALREADY-annotated set of bimanual GR-1 pick-and-place demos.
MimicGen will later multiply these into ~1000 synthetic demos. We just download
the file onto the host; no GPU or container is needed for this step.

Output (depends on --profile):
    datasets/source_dataset.hdf5           (franka)
    datasets/gr1t2_annotated_dataset.hdf5  (gr1t2)

Run it like:
    python3 scripts/01_download_dataset.py                 # franka (default)
    python3 scripts/01_download_dataset.py --profile gr1t2
"""

from __future__ import annotations

import argparse
import sys
import urllib.request

from _common import DATASETS_DIR, get_profile


def _progress(block_num: int, block_size: int, total_size: int) -> None:
    """Print a simple percent-complete line while downloading.

    Only animates on an interactive terminal; when the output is piped (e.g. over
    SSH) the carriage-return bar would spam the log, so we stay quiet there.
    """
    if total_size <= 0 or not sys.stdout.isatty():
        return
    downloaded = block_num * block_size
    pct = min(100.0, downloaded * 100.0 / total_size)
    mb = downloaded / 1e6
    total_mb = total_size / 1e6
    print(f"\r  downloading... {pct:5.1f}%  ({mb:6.1f} / {total_mb:6.1f} MB)", end="", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download the seed demonstrations.")
    parser.add_argument("--profile", default="franka", help="franka (default) or gr1t2.")
    args = parser.parse_args()
    profile = get_profile(args.profile)

    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    dataset_url = profile.source_url
    out_path = DATASETS_DIR / profile.source_file

    if out_path.exists():
        print(f"[skip] {out_path} already exists ({out_path.stat().st_size/1e6:.1f} MB).")
        return 0

    print(f"Downloading source demos from:\n  {dataset_url}\n-> {out_path}")
    try:
        urllib.request.urlretrieve(dataset_url, out_path, _progress)
        print()  # newline after the progress line
    except Exception as e:  # noqa: BLE001 - we want a friendly message on any failure
        sys.exit(f"\n[ERROR] download failed: {e}")

    # HDF5 files start with the 8-byte signature b"\x89HDF\r\n\x1a\n". A quick
    # check here catches the common failure where S3 returns an HTML error page
    # instead of the file.
    with open(out_path, "rb") as f:
        magic = f.read(8)
    if magic != b"\x89HDF\r\n\x1a\n":
        out_path.unlink(missing_ok=True)
        sys.exit("[ERROR] downloaded file is not a valid HDF5 file (got unexpected header).")

    print(f"OK: {out_path}  ({out_path.stat().st_size/1e6:.1f} MB)")
    if profile.pre_annotated:
        print(f"This dataset is already annotated; skip step 2. "
              f"Next: python3 scripts/03_generate.py --profile {profile.name} --mode small")
    else:
        print(f"Next: python3 scripts/02_annotate.py --profile {profile.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
