"""Build two new experiment arm families from the EXISTING D2 pool (no new gen).

near_only  (coverage-vs-density): the 500 nearest-to-source retained demos
           (sampled from the nearest 40% by d_pos) — removes far coverage.
hidgr_only / lodgr_only (source-quality causal): equal-size draws from ONLY the
           high-DGR source demos vs ONLY the low-DGR source demos — isolates
           whether low-transfer ("fragile") source grasps make worse training data.

Reconstructs provenance->group from train.hdf5, samples with seeded RNG, writes
robomimic filter keys + updates arms_manifest. Seeds 201-203 (no collision with
the 101-106 main arms). Idempotent per key.
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = "/home/ubuntu/mimicgen_jihoonkwon/mimic_the_mimicgen/motivation"
A = Path("/home/ubuntu/mimicgen_jihoonkwon/experiments/motivation_new/e2_arms")
SEEDS = [201, 202, 203]
NEAR_TASKS = ["square", "coffee", "three_piece_assembly", "threading", "stack"]
HL_TASKS = ["stack", "threading"]
NEAR_SIZE = 500
NEAR_FRAC = 0.40
HL_SIZE = 350
ARM_OFFSET = {"near_only": 1, "hidgr_only": 2, "lodgr_only": 3}

sys.path.insert(0, REPO)
import h5py  # noqa: E402
from genaudit.curation.filter_keys import write_filter_key  # noqa: E402
from genaudit.records.schema import read_jsonl  # noqa: E402


def build(task):
    out_dir = A / f"{task}_N2"
    merged = out_dir / "train.hdf5"
    manifest = json.loads((out_dir / "arms_manifest.json").read_text())
    with h5py.File(merged, "r") as f:
        prov2name = {f["data"][g].attrs["provenance"]: g for g in f["data"].keys()}
    records = list(read_jsonl(out_dir / "attempts.jsonl"))
    ret = [r for r in records if r.success]
    ret_prov = [r.attempt_id.split("@")[0] for r in ret]
    ret_names = np.array([prov2name[p] for p in ret_prov])
    ret_src = np.array([r.source_demo_id for r in ret])
    ret_d = np.array([r.d_pos for r in ret])
    nsrc = int(np.array([r.source_demo_id for r in records]).max()) + 1
    allsrc = np.array([r.source_demo_id for r in records])
    allsucc = np.array([r.success for r in records])
    dgr = {s: float(allsucc[allsrc == s].mean()) if (allsrc == s).any() else 0.0 for s in range(nsrc)}
    retcount = {s: int((ret_src == s).sum()) for s in range(nsrc)}

    def write_arm(arm, pool_idx, size, info):
        pool = np.asarray(pool_idx)
        size = int(min(size, len(pool)))
        for s in SEEDS:
            rng = np.random.default_rng(s * 10 + ARM_OFFSET[arm])
            sel = rng.choice(pool, size=size, replace=False)
            names = [str(x) for x in ret_names[sel]]
            key = f"{arm}_seed{s}"
            write_filter_key(merged, key, names)
            manifest["arms"][key] = {"size": size, "demo_names": names, **info}
        print(f"  {task} {arm}: size={size} pool={len(pool)} {info}", flush=True)

    if task in NEAR_TASKS:
        thr = float(np.quantile(ret_d, NEAR_FRAC))
        pool = np.flatnonzero(ret_d <= thr)
        write_arm("near_only", pool, NEAR_SIZE,
                  {"criterion": f"d_pos<={round(thr,3)} (nearest {int(NEAR_FRAC*100)}%)"})

    if task in HL_TASKS:
        by_dgr_desc = sorted(range(nsrc), key=lambda s: -dgr[s])
        hi, tot = [], 0
        for s in by_dgr_desc:
            hi.append(s); tot += retcount[s]
            if len(hi) >= 2 and tot >= HL_SIZE:
                break
        alive_asc = sorted([s for s in range(nsrc) if dgr[s] >= 0.08], key=lambda s: dgr[s])
        lo, tot = [], 0
        for s in alive_asc:
            lo.append(s); tot += retcount[s]
            if len(lo) >= 2 and tot >= HL_SIZE:
                break
        size = min(HL_SIZE, sum(retcount[s] for s in hi), sum(retcount[s] for s in lo))
        write_arm("hidgr_only", np.flatnonzero(np.isin(ret_src, hi)), size,
                  {"sources": hi, "src_dgr": [round(dgr[s], 3) for s in hi]})
        write_arm("lodgr_only", np.flatnonzero(np.isin(ret_src, lo)), size,
                  {"sources": lo, "src_dgr": [round(dgr[s], 3) for s in lo]})

    (out_dir / "arms_manifest.json").write_text(json.dumps(manifest, indent=2))


def main():
    tasks = sys.argv[1:] or sorted(set(NEAR_TASKS + HL_TASKS))
    for task in tasks:
        try:
            build(task)
        except Exception as e:  # noqa: BLE001
            import traceback
            print(f"{task}: ERR {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
    print("NEWARMS DONE", flush=True)


if __name__ == "__main__":
    main()
