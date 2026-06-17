#!/usr/bin/env python3
"""
STEP 4b - Inspect a generated dataset locally (no simulator needed).

This is the lightweight, always-works way to "see" what MimicGen produced. It
runs on your LAPTOP after you've synced the datasets down
(setup/sync_from_remote.sh). It opens the HDF5 and reports:

  - how many demos it contains,
  - how long each demo is (number of timesteps),
  - the structure (observation/action keys),

and saves a couple of plots to outputs/:
  - a histogram of demo lengths,
  - the action trace of the first demo.

It only needs h5py + matplotlib (both pure-Python installs on your Mac):
    pip install h5py matplotlib numpy

Run it like:
    python3 scripts/04b_inspect_dataset.py
    python3 scripts/04b_inspect_dataset.py --dataset datasets/generated_dataset.hdf5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _common import DATASETS_DIR, OUTPUTS_DIR


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect a MimicGen HDF5 dataset locally.")
    parser.add_argument(
        "--dataset", default=str(DATASETS_DIR / "generated_dataset.hdf5"),
        help="Path to the HDF5 dataset to inspect.",
    )
    args = parser.parse_args()

    # Import here so a missing dependency produces a friendly hint, not a crash
    # at import time.
    try:
        import h5py
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")  # headless backend: write PNGs without a display
        import matplotlib.pyplot as plt
    except ImportError as e:
        sys.exit(f"[ERROR] missing dependency: {e}.\n        pip install h5py matplotlib numpy")

    path = Path(args.dataset)
    if not path.exists():
        sys.exit(f"[ERROR] {path} not found. Sync it first: bash setup/sync_from_remote.sh")

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "r") as f:
        # Isaac Lab datasets store demos under the top-level "data" group as
        # "demo_0", "demo_1", ... Each demo has an "actions" dataset and an
        # "obs/" subgroup, plus a "num_samples" attribute (its length).
        if "data" not in f:
            sys.exit(f"[ERROR] unexpected file layout: no top-level 'data' group in {path}.")
        data = f["data"]
        demo_keys = sorted(data.keys(), key=lambda k: int(k.split("_")[-1]) if k.split("_")[-1].isdigit() else 0)
        n_demos = len(demo_keys)
        env_name = data.attrs.get("env_name", "<unknown>")
        print(f"file        : {path}")
        print(f"env_name    : {env_name}")
        print(f"num demos   : {n_demos}")

        if n_demos == 0:
            sys.exit("[ERROR] dataset contains 0 demos.")

        # Per-demo length.
        lengths = []
        for k in demo_keys:
            g = data[k]
            n = int(g.attrs["num_samples"]) if "num_samples" in g.attrs else int(g["actions"].shape[0])
            lengths.append(n)
        lengths = np.array(lengths)
        print(f"demo length : min={lengths.min()}  mean={lengths.mean():.1f}  max={lengths.max()}")

        # Show the structure of the first demo so you know what is inside.
        first = data[demo_keys[0]]
        print(f"\nfirst demo '{demo_keys[0]}' contents:")
        print(f"  actions shape: {first['actions'].shape}")
        if "obs" in first:
            for key in first["obs"].keys():
                print(f"  obs/{key:<24} shape: {first['obs'][key].shape}")

        # Plot 1: histogram of demo lengths.
        plt.figure(figsize=(6, 4))
        plt.hist(lengths, bins=min(30, n_demos))
        plt.xlabel("demo length (timesteps)")
        plt.ylabel("count")
        plt.title(f"Demo lengths ({n_demos} demos)\n{path.name}")
        plt.tight_layout()
        hist_path = OUTPUTS_DIR / f"{path.stem}_lengths.png"
        plt.savefig(hist_path, dpi=120)
        print(f"\nsaved: {hist_path}")

        # Plot 2: action trace of the first demo (one line per action dimension).
        actions = np.asarray(first["actions"])
        plt.figure(figsize=(8, 4))
        for d in range(actions.shape[1]):
            plt.plot(actions[:, d], label=f"a{d}")
        plt.xlabel("timestep")
        plt.ylabel("action value")
        plt.title(f"Actions of {demo_keys[0]}  ({path.name})")
        plt.legend(fontsize=7, ncol=4)
        plt.tight_layout()
        act_path = OUTPUTS_DIR / f"{path.stem}_{demo_keys[0]}_actions.png"
        plt.savefig(act_path, dpi=120)
        print(f"saved: {act_path}")

    print("\nDone. Open the PNGs in outputs/ to view.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
