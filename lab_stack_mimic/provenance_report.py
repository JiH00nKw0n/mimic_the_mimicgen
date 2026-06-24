#!/usr/bin/env python3
"""Turn provenance_hooks.py output into the per-seed / per-subtask contribution table.

  python provenance_report.py --provenance /tmp/provenance.json \
      --input fwd_annotated.hdf5 [--csv out.csv]

Reports, over the KEPT (successful) generated demos:
  - per subtask: how the source seed demos were used (count + % of kept demos),
  - per seed demo: total contributions and the per-subtask breakdown.

src_ind -> demo name uses the SAME h5py key order the generator loaded
(`DataGenInfoPool` iterates `get_episode_names()` = `f["data"].keys()`).
"""

from __future__ import annotations

import argparse
import collections
import json

import h5py


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provenance", required=True, help="provenance.json from provenance_hooks.py")
    ap.add_argument("--input", required=True, help="annotated source hdf5 (for src_ind -> demo name)")
    ap.add_argument("--csv", default=None, help="optional CSV of the per-source x per-subtask matrix")
    args = ap.parse_args()

    with open(args.provenance) as fh:
        prov = json.load(fh)
    with h5py.File(args.input) as f:
        names = list(f["data"].keys())  # same order the pool used

    def nm(i):
        return names[i] if 0 <= i < len(names) else f"src_{i}"

    n_succ = prov["n_success"]
    n_att = prov["n_attempts"]
    dgr = (100.0 * n_succ / n_att) if n_att else 0.0
    print(f"=== source-usage provenance ===")
    print(f"input: {args.input}")
    print(f"kept (successful) demos: {n_succ}   attempts: {n_att}   DGR: {dgr:.1f}%\n")

    # counts over kept demos: (eef, subtask, src_ind) -> count
    counts = prov["counts_success"]
    subtasks = sorted({(c["eef"], c["subtask"]) for c in counts})
    src_inds = sorted({c["src_ind"] for c in counts})

    # matrix[src_ind][(eef,subtask)] = count
    mat = collections.defaultdict(lambda: collections.defaultdict(int))
    per_subtask_total = collections.Counter()
    for c in counts:
        mat[c["src_ind"]][(c["eef"], c["subtask"])] += c["count"]
        per_subtask_total[(c["eef"], c["subtask"])] += c["count"]

    # --- per subtask: distribution over seed demos
    print("--- per subtask: source-demo usage (count, % of kept demos) ---")
    for key in subtasks:
        eef, st = key
        tot = per_subtask_total[key] or 1
        print(f"\nsubtask {st} ({eef}):  (kept demos using each seed)")
        rows = sorted(((mat[si][key], si) for si in src_inds if mat[si][key] > 0), reverse=True)
        for cnt, si in rows:
            print(f"    {nm(si):<12} {cnt:4d}  ({100*cnt/tot:5.1f}%)")

    # --- per seed demo: total + per-subtask breakdown
    print("\n--- per seed demo: total contribution across all subtasks ---")
    hdr = "seed".ljust(12) + "".join(f"st{st}".rjust(8) for (_e, st) in subtasks) + "   total"
    print(hdr)
    for si in sorted(src_inds, key=lambda i: -sum(mat[i].values())):
        cells = "".join(f"{mat[si][key]:8d}" for key in subtasks)
        tot = sum(mat[si].values())
        print(f"{nm(si):<12}{cells}   {tot:5d}")
    # seeds that were NEVER used for any subtask in a kept demo
    used = {si for si in src_inds if sum(mat[si].values()) > 0}
    unused = [nm(i) for i in range(len(names)) if i not in used]
    if unused:
        print(f"\nseeds never used in a kept demo ({len(unused)}): {', '.join(unused)}")

    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["seed"] + [f"subtask_{st}_{e}" for (e, st) in subtasks] + ["total"])
            for si in sorted(src_inds, key=lambda i: -sum(mat[i].values())):
                w.writerow([nm(si)] + [mat[si][key] for key in subtasks] + [sum(mat[si].values())])
        print(f"\n[csv] wrote {args.csv}")


if __name__ == "__main__":
    main()
