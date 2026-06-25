#!/usr/bin/env python3
"""Dump the final cube arrangement of each demo so we can see HOW a SkillGen demo failed
(did the stack partly form, or did it end up scattered?). Reads cube positions from the
recorded obs and prints, per demo, the final z-order + pairwise xy/z gaps vs the clean-stack
criterion (z order cube_1<cube_2<cube_3, gaps ~0.0468, xy<0.04).

    python dump_final_cubes.py <dataset.hdf5> [max_demos]
"""

import sys

import h5py
import numpy as np

CUBES = ["cube_1", "cube_2", "cube_3"]


def main():
    path = sys.argv[1]
    maxd = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    with h5py.File(path, "r") as f:
        demos = list(f["data"].keys())[:maxd]
        print(f"{path}: {len(list(f['data'].keys()))} demos, showing {len(demos)}")
        for d in demos:
            g = f["data"][d]
            obs = g["obs"]
            cp = np.asarray(obs["cube_positions"][:])[-1].reshape(-1, 3)  # (3,3): cube_1,2,3
            pos = {CUBES[i]: cp[i][:3] for i in range(3)}
            z = {c: pos[c][2] for c in CUBES}
            order = sorted(CUBES, key=lambda c: z[c])
            g12 = float(np.linalg.norm(pos["cube_2"][:2] - pos["cube_1"][:2]))
            g23 = float(np.linalg.norm(pos["cube_3"][:2] - pos["cube_2"][:2]))
            dz12 = float(z["cube_2"] - z["cube_1"])
            dz23 = float(z["cube_3"] - z["cube_2"])
            print(f"  {d}: zorder={order} z1={z['cube_1']:.3f} z2={z['cube_2']:.3f} z3={z['cube_3']:.3f} "
                  f"xy12={g12:.3f} xy23={g23:.3f} dz12={dz12:.3f} dz23={dz23:.3f}")


if __name__ == "__main__":
    main()
