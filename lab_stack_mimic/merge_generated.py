#!/usr/bin/env python3
"""Merge several generated HDF5 datasets (and their provenance JSONs) into one.

Concatenates demos with contiguous renumbering (demo_0..demo_{N-1}), sums the
`data` attrs `total`, and merges the provenance files: concatenates per_demo,
sums n_success / n_attempts, and adds the (eef, subtask, src_ind) counters.

    python merge_generated.py --out merged.hdf5 \
        --inputs a.hdf5 b.hdf5 [--prov a.provenance.json b.provenance.json]

If --prov is omitted it defaults to "<input without .hdf5>.provenance.json" for each
input (skips any that are missing). Writes <out without .hdf5>.provenance.json too.
"""

from __future__ import annotations

import argparse
import collections
import json
import os

import h5py


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--prov", nargs="*", default=None, help="provenance JSONs (default: derived from inputs)")
    args = ap.parse_args()

    provs = args.prov if args.prov is not None else [f"{os.path.splitext(p)[0]}.provenance.json" for p in args.inputs]
    if os.path.exists(args.out):
        os.remove(args.out)

    total = 0
    n_written = 0
    env_attrs = {}
    with h5py.File(args.out, "w") as fd:
        g = fd.create_group("data")
        for ipath in args.inputs:
            with h5py.File(ipath, "r", locking=False) as fs:
                for k, v in fs["data"].attrs.items():
                    env_attrs[k] = v  # last wins for env_args; total summed below
                total += int(fs["data"].attrs.get("total", 0))
                demos = sorted(fs["data"].keys(), key=lambda d: int(d.split("_")[1]))
                for d in demos:
                    fd.copy(fs[f"data/{d}"], f"data/demo_{n_written}")
                    n_written += 1
            print(f"  {os.path.basename(ipath)}: +{len(demos)} demos -> running total {n_written}")
        for k, v in env_attrs.items():
            if k != "total":
                g.attrs[k] = v
        g.attrs["total"] = total
    print(f"\nwrote {args.out}: {n_written} demos, {round(os.path.getsize(args.out)/1e6, 1)} MB")

    # merge provenance
    merged = {
        "n_success": 0, "n_attempts": 0, "input_file": "",
        "counts_success": collections.Counter(), "counts_all": collections.Counter(),
        "per_demo": [],
    }
    found = 0
    for ppath in provs:
        if not os.path.exists(ppath):
            print(f"  [prov] missing {ppath} — skipped")
            continue
        found += 1
        with open(ppath) as fh:
            pj = json.load(fh)
        merged["n_success"] += pj.get("n_success", 0)
        merged["n_attempts"] += pj.get("n_attempts", 0)
        merged["input_file"] = pj.get("input_file", merged["input_file"])
        merged["per_demo"].extend(pj.get("per_demo", []))
        for key in ("counts_success", "counts_all"):
            for c in pj.get(key, []):
                merged[key][(c["eef"], c["subtask"], c["src_ind"])] += c["count"]

    if found:
        def rows(counter):
            return [{"eef": e, "subtask": s, "src_ind": i, "count": c} for (e, s, i), c in sorted(counter.items())]
        out_prov = {
            "n_success": merged["n_success"], "n_attempts": merged["n_attempts"],
            "input_file": merged["input_file"],
            "counts_success": rows(merged["counts_success"]),
            "counts_all": rows(merged["counts_all"]),
            "per_demo": merged["per_demo"],
        }
        opath = f"{os.path.splitext(args.out)[0]}.provenance.json"
        with open(opath, "w") as fh:
            json.dump(out_prov, fh, indent=2)
        dgr = 100 * out_prov["n_success"] / out_prov["n_attempts"] if out_prov["n_attempts"] else 0
        print(f"wrote {opath}: n_success={out_prov['n_success']} n_attempts={out_prov['n_attempts']} "
              f"DGR={dgr:.1f}%  per_demo={len(out_prov['per_demo'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
