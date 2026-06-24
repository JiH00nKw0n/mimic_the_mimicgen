#!/usr/bin/env python3
"""Canonicalize any-order 3-cube stacking demos to the official cube-identity order.

Our seed demos stack the 3 cubes in ANY colour/identity order, but Isaac Lab Mimic's
subtask schema and success term are order-specific (cube_1 bottom < cube_2 mid <
cube_3 top). So for each demo we detect the actual bottom->middle->top order from the
final recorded cube poses and PERMUTE the cube identities so the demo looks canonical.

Only cube-identity channels are permuted; the actions (end-effector deltas) are left
untouched, so when the canonicalized demo is replayed, reset_to places the physical
cubes at the permuted positions and the same gripper trajectory builds a canonical
blue->red->green tower -> the official grasp_1/stack_1/grasp_2 signals and the
cubes_stacked success fire in order. See METHODOLOGY.md section 6.

Pure h5py/numpy (no Isaac Sim). Run with any python that has h5py.

    python canonicalize.py --src teleop_dataset_success.hdf5 --dst teleop_dataset_canon.hdf5
"""

from __future__ import annotations

import argparse
import os

import h5py
import numpy as np

CUBES = ["cube_1", "cube_2", "cube_3"]


def final_order(src_demo, k: int) -> list[int]:
    """Return old cube indices (0,1,2) sorted bottom->top by mean final-frame z."""
    zs = []
    for c in CUBES:
        z = src_demo[f"states/rigid_object/{c}/root_pose"][-k:, 2].mean()
        zs.append(float(z))
    return list(np.argsort(zs))  # ascending z: [bottom_old, mid_old, top_old]


def permute_demo(dst_demo, order: list[int]) -> bool:
    """In-place permute cube-identity channels so new cube_(j+1) := old cube_(order[j]+1)."""
    if order == [0, 1, 2]:
        return False

    # 1) group-keyed cube state (this is what the annotation replay actually uses)
    for base in ("initial_state/rigid_object", "states/rigid_object"):
        grp = dst_demo[base]
        old = {c: {ds: grp[f"{c}/{ds}"][()] for ds in grp[c].keys()} for c in CUBES}
        for j, c in enumerate(CUBES):
            src_c = CUBES[order[j]]
            for ds in grp[c].keys():
                grp[f"{c}/{ds}"][...] = old[src_c][ds]

    # 2) concatenated obs blocks (3 cubes side by side). Not used by annotate/generate
    #    (those re-record obs), but keep the dataset self-consistent.
    for name, stride in (("cube_positions", 3), ("cube_orientations", 4), ("object", 13)):
        path = f"obs/{name}"
        if path not in dst_demo:
            continue
        arr = dst_demo[path][()]
        if arr.shape[1] != 3 * stride:
            print(f"    [skip] obs/{name} width {arr.shape[1]} != 3x{stride}; not permuting")
            continue
        new = arr.copy()
        for j in range(3):
            new[:, j * stride:(j + 1) * stride] = arr[:, order[j] * stride:(order[j] + 1) * stride]
        dst_demo[path][...] = new
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="Canonicalize cube identities by final z-order.")
    ap.add_argument("--src", required=True, help="input HDF5 (any-order success demos)")
    ap.add_argument("--dst", required=True, help="output HDF5 (canonicalized)")
    ap.add_argument("--k", type=int, default=3, help="average final z over the last K frames")
    args = ap.parse_args()

    if os.path.exists(args.dst):
        os.remove(args.dst)

    with h5py.File(args.src, "r", locking=False) as fs, h5py.File(args.dst, "w") as fd:
        fd.create_group("data")
        for key, val in fs["data"].attrs.items():       # carry env_args, total, etc.
            fd["data"].attrs[key] = val

        demos = sorted(fs["data"].keys(), key=lambda d: int(d.split("_")[1]))
        swapped = 0
        for d in demos:
            fd.copy(fs[f"data/{d}"], f"data/{d}")        # deep copy, then permute in place
            order = final_order(fs[f"data/{d}"], args.k)
            did = permute_demo(fd[f"data/{d}"], order)
            swapped += int(did)
            tag = "->".join(f"c{o + 1}" for o in order)
            print(f"  {d:>9}: bottom->top = {tag}   {'SWAPPED' if did else 'kept'}")

    print(f"\nwrote {args.dst}: {len(demos)} demos, {swapped} permuted, "
          f"{round(os.path.getsize(args.dst) / 1e6, 2)} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
