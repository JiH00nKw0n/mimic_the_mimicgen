#!/usr/bin/env python3
"""Fix the demos' initial_state to be the actual trajectory start (states[0]).

In the lab teleop dataset, each demo's `initial_state` is NOT the scattered start of
the episode — it is (roughly) the stacked end configuration. Replay / annotation reset
to `initial_state` and then apply the recorded actions, so they were starting from an
already-stacked scene → garbage. The recorded `states` are correct (scattered -> stacked);
this copies states[0] into initial_state so reset_to() starts from the real beginning.

Pure h5py/numpy (no Isaac Sim).

    python fix_initial_state.py --src teleop_dataset_success.hdf5 --dst teleop_dataset_fixed.hdf5
"""

from __future__ import annotations

import argparse
import os

import h5py


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    args = ap.parse_args()
    if os.path.exists(args.dst):
        os.remove(args.dst)

    with h5py.File(args.src, "r", locking=False) as fs, h5py.File(args.dst, "w") as fd:
        fd.create_group("data")
        for key, val in fs["data"].attrs.items():
            fd["data"].attrs[key] = val

        demos = sorted(fs["data"].keys(), key=lambda d: int(d.split("_")[1]))
        for d in demos:
            fd.copy(fs[f"data/{d}"], f"data/{d}")
            g = fd[f"data/{d}"]
            changed = []
            for atype in ("articulation", "rigid_object"):
                if atype not in g["initial_state"] or atype not in g["states"]:
                    continue
                for asset in g["initial_state"][atype]:
                    for k in g["initial_state"][atype][asset]:
                        g["initial_state"][atype][asset][k][...] = g["states"][atype][asset][k][0:1]
                        changed.append(f"{asset}/{k}")
            # report cube z before/after so the fix is visible
            zs = [float(g["initial_state"]["rigid_object"][f"cube_{i}"]["root_pose"][0, 2]) for i in (1, 2, 3)]
            print(f"  {d}: initial_state := states[0]   cube z now = {[round(z, 3) for z in zs]}")

    print(f"\nwrote {args.dst}: {len(demos)} demos, {round(os.path.getsize(args.dst) / 1e6, 2)} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
