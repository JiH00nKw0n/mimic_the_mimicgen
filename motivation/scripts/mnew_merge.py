"""Merge motivation_new generation chunks into one standard pool each.

For each <task>_<variant> pool, concatenate all chunk_seed*/demo.hdf5 into
pool/demo.hdf5 (and demo_failed.hdf5), renaming demo groups globally, so the
existing extraction / arm-prep scripts see a normal single pool. 8-way parallel.
"""
import json
import multiprocessing as mp
from pathlib import Path

import h5py

GEN = Path("/home/ubuntu/mimicgen_jihoonkwon/experiments/motivation_new/gen")
TASKS = ["square", "threading", "coffee", "three_piece_assembly", "stack",
         "stack_three", "mug_cleanup", "hammer_cleanup"]
VARIANTS = ["N0", "N1", "N2"]


def _demo_key(n):
    return int(n.split("_")[1])


def _merge_files(chunk_files, out_path):
    total = 0
    gi = 0
    with h5py.File(out_path, "w") as dst:
        dg = dst.create_group("data")
        first = True
        for cf in chunk_files:
            with h5py.File(cf, "r") as src:
                if "data" not in src:
                    continue
                if first:
                    for k, v in src["data"].attrs.items():
                        dg.attrs[k] = v
                    first = False
                for name in sorted(src["data"].keys(), key=_demo_key):
                    src.copy(f"data/{name}", dg, name=f"demo_{gi}")
                    total += int(src["data"][name].attrs.get("num_samples", 0))
                    gi += 1
        dg.attrs["total"] = total
    return gi


def merge_pool(pool):
    pdir = GEN / pool
    if (pdir / "MERGED").exists():
        return (pool, "cached")
    demos = sorted(pdir.glob("chunk_seed*/*/demo.hdf5"))
    fails = sorted(pdir.glob("chunk_seed*/*/demo_failed.hdf5"))
    if not demos and not fails:
        return (pool, "NO_CHUNKS")
    try:
        n_ok = _merge_files(demos, pdir / "demo.hdf5") if demos else 0
        n_fail = _merge_files(fails, pdir / "demo_failed.hdf5") if fails else 0
        ns = nf = 0
        for f in pdir.glob("chunk_seed*/*/important_stats.json"):
            d = json.load(open(f))
            ns += d["num_success"]; nf += d["num_failures"]
        (pdir / "important_stats.json").write_text(json.dumps({
            "num_success": ns, "num_failures": nf, "num_attempts": ns + nf,
            "success_rate": 100 * ns / (ns + nf) if (ns + nf) else 0}))
        (pdir / "MERGED").touch()
        return (pool, f"merged demos={n_ok} fails={n_fail} (succ={ns})")
    except Exception as e:  # noqa: BLE001
        return (pool, f"ERR {type(e).__name__}: {e}")


if __name__ == "__main__":
    pools = [f"{t}_{v}" for t in TASKS for v in VARIANTS]
    with mp.get_context("spawn").Pool(8) as p:
        for r in p.imap_unordered(merge_pool, pools):
            print(r, flush=True)
    (GEN / "MERGE_DONE").touch()
    print("MERGE DONE", flush=True)
