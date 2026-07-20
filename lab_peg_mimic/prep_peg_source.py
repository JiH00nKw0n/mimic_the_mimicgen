#!/usr/bin/env python3
"""Fix initial_state (:= states[0]) on each 'ok' peg demo and merge the 12 into one
Isaac Lab Mimic source dataset (data/demo_0..N-1). Pure h5py — no Isaac Sim.

Mirrors lab_stack_mimic/fix_initial_state.py (copy states[atype][asset][k][0:1] into
initial_state) but reads the colleague's split _epNNNN.hdf5 files (each = data/demo_0)
and writes a single merged, replay-ready source. Originals are never modified.

    python prep_peg_source.py --src_dir <ok_dir> --dst datasets/peg_source.hdf5
"""
import argparse, glob, os
import h5py, numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--src_dir", default="/Users/junekwon/Downloads/peg_in_hole/datasets/ok")
ap.add_argument("--dst", default="datasets/peg_source.hdf5")
args = ap.parse_args()

files = sorted(glob.glob(os.path.join(args.src_dir, "*.hdf5")))
assert files, f"no hdf5 in {args.src_dir}"
os.makedirs(os.path.dirname(os.path.abspath(args.dst)), exist_ok=True)
if os.path.exists(args.dst):
    os.remove(args.dst)

with h5py.File(args.dst, "w") as fd:
    grp = fd.create_group("data")
    env_args = None
    total = 0
    for i, fp in enumerate(files):
        with h5py.File(fp, "r", locking=False) as fs:
            if env_args is None and "env_args" in fs["data"].attrs:
                env_args = fs["data"].attrs["env_args"]
            src_key = list(fs["data"].keys())[0]            # each file has one demo
            dst_key = f"demo_{i}"
            fd.copy(fs[f"data/{src_key}"], f"data/{dst_key}")
            g = fd[f"data/{dst_key}"]
            # --- the fix: initial_state := states[0] for every asset/field ---
            changed = []
            for atype in ("articulation", "rigid_object"):
                if atype not in g["initial_state"] or atype not in g["states"]:
                    continue
                for asset in g["initial_state"][atype]:
                    for k in g["initial_state"][atype][asset]:
                        g["initial_state"][atype][asset][k][...] = g["states"][atype][asset][k][0:1]
                        changed.append(f"{atype}/{asset}/{k}")
            n = int(g.attrs.get("num_samples", g["actions"].shape[0]))
            total += n
            pegp = g["initial_state"]["rigid_object"]["peg"]["root_pose"][0]
            print(f"  demo_{i:<2} <- {os.path.basename(fp):32} N={n:<4} peg_init xy=({pegp[0]:+.3f},{pegp[1]:+.3f}) "
                  f"succ={bool(g.attrs.get('success'))} fixed[{len(changed)} fields]")
    grp.attrs["total"] = total
    if env_args is not None:
        grp.attrs["env_args"] = env_args

# verify
with h5py.File(args.dst, "r") as f:
    demos = sorted(f["data"].keys(), key=lambda d: int(d.split("_")[1]))
    ok = 0
    for d in demos:
        g = f["data"][d]
        ip = g["initial_state"]["rigid_object"]["peg"]["root_pose"][0]
        sp = g["states"]["rigid_object"]["peg"]["root_pose"][0]
        ij = g["initial_state"]["articulation"]["robot"]["joint_position"][0]
        sj = g["states"]["articulation"]["robot"]["joint_position"][0]
        match = np.allclose(ip, sp) and np.allclose(ij, sj)
        ok += int(match)
    print(f"\nwrote {args.dst}: {len(demos)} demos, {round(os.path.getsize(args.dst)/1e6,2)} MB")
    print(f"initial_state == states[0] now: {ok}/{len(demos)}  (must be {len(demos)}/{len(demos)})")
    print(f"data attrs: total={f['data'].attrs.get('total')}, env_args set={'env_args' in f['data'].attrs}")
