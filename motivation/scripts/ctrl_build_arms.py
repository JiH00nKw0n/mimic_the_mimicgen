"""Build controlled-experiment arms into motivation_controlled/arms/<task>/train.hdf5
(copied from motivation_new so the new keys never touch the original).

Common d_pos bins = task-retained quintiles. near={bin0,1} mid={bin2} far={bin3,4}.
EXP-A (stack {4,7}, 6 seeds): A_nearheavy 375near+125far, A_farheavy 125near+375far (500).
EXP-B (stack {4,7}, 3 seeds): B_far near150+mid100+far150, B_nearpad near300+mid100 (400).
EXP-C  (threading, 3 seeds): C_hi{2,8}/C_mid{0,6}/C_lo{4,5}, 50 per bin x5 = 250 (distance-matched).
EXP-C2 (stack, 3 seeds): C2_hi{4,7}/C2_mid{3,1}, 70 per bin x5 = 350 (distance-matched).
EXP-D  (threading, 3 seeds): D_1{2,0}/D_2{8,1}, 50 per bin x5 = 250 (distance-matched).
"""
import json, shutil, sys
from pathlib import Path
import numpy as np

REPO = "/home/ubuntu/mimicgen_jihoonkwon/mimic_the_mimicgen/motivation"
NEW = Path("/home/ubuntu/mimicgen_jihoonkwon/experiments/motivation_new/e2_arms")
CTRL = Path("/home/ubuntu/mimicgen_jihoonkwon/experiments/motivation_controlled/arms")
sys.path.insert(0, REPO)
import h5py  # noqa: E402
from genaudit.curation.filter_keys import write_filter_key  # noqa: E402
from genaudit.records.schema import read_jsonl  # noqa: E402

OFF = {"A_nearheavy": 1, "A_farheavy": 2, "B_far": 3, "B_nearpad": 4,
       "C_hi": 5, "C_mid": 6, "C_lo": 7, "C2_hi": 8, "C2_mid": 9, "D_1": 10, "D_2": 11}


def prep(task):
    dst = CTRL / task
    dst.mkdir(parents=True, exist_ok=True)
    merged = dst / "train.hdf5"
    if not merged.exists():
        shutil.copy(NEW / f"{task}_N2" / "train.hdf5", merged)
    with h5py.File(merged, "r") as f:
        prov2name = {f["data"][g].attrs["provenance"]: g for g in f["data"].keys()}
    recs = list(read_jsonl(NEW / f"{task}_N2" / "attempts.jsonl"))
    ret = [r for r in recs if r.success]
    prov = np.array([r.attempt_id.split("@")[0] for r in ret])
    names = np.array([prov2name[p] for p in prov])
    src = np.array([r.source_demo_id for r in ret])
    d = np.array([r.d_pos for r in ret])
    edges = np.quantile(d, [.2, .4, .6, .8])
    binid = np.searchsorted(edges, d, side="right")  # 0..4
    manifest = {"task": task, "arms": {}}
    return merged, names, src, d, binid, manifest


def idx(src, binid, sources, bins):
    return np.flatnonzero(np.isin(src, sources) & np.isin(binid, bins))


def build_stack():
    merged, names, src, d, binid, manifest = prep("stack")
    S = [4, 7]
    near = idx(src, binid, S, [0, 1]); mid = idx(src, binid, S, [2]); far = idx(src, binid, S, [3, 4])

    def emit(arm, seeds, plan):  # plan = list of (pool, n)
        for s in seeds:
            rng = np.random.default_rng(s * 100 + OFF[arm])
            sel = np.concatenate([rng.choice(pool, n, replace=False) for pool, n in plan])
            nm = [str(x) for x in names[sel]]
            write_filter_key(merged, f"{arm}_seed{s}", nm)
            manifest["arms"][f"{arm}_seed{s}"] = {"size": len(nm), "demo_names": nm}
        print(f"  stack {arm}: n={sum(n for _,n in plan)} x{len(seeds)}seed  pools near{len(near)}/mid{len(mid)}/far{len(far)}")

    A_SEEDS = [301, 302, 303, 304, 305, 306]; S3 = [301, 302, 303]
    emit("A_nearheavy", A_SEEDS, [(near, 375), (far, 125)])
    emit("A_farheavy", A_SEEDS, [(near, 125), (far, 375)])
    emit("B_far", S3, [(near, 150), (mid, 100), (far, 150)])
    emit("B_nearpad", S3, [(near, 300), (mid, 100)])
    # C2: distance-matched 70/bin from {4,7} vs {3,1}
    for arm, S2 in [("C2_hi", [4, 7]), ("C2_mid", [3, 1])]:
        for s in S3:
            rng = np.random.default_rng(s * 100 + OFF[arm])
            sel = np.concatenate([rng.choice(idx(src, binid, S2, [b]), 70, replace=False) for b in range(5)])
            nm = [str(x) for x in names[sel]]
            write_filter_key(merged, f"{arm}_seed{s}", nm)
            manifest["arms"][f"{arm}_seed{s}"] = {"size": len(nm), "demo_names": nm}
        print(f"  stack {arm} {S2}: 70/bin x5 =350 x3seed")
    (CTRL / "stack" / "arms_manifest.json").write_text(json.dumps(manifest, indent=2))


def build_threading():
    merged, names, src, d, binid, manifest = prep("threading")
    S3 = [301, 302, 303]
    specs = [("C_hi", [2, 8], 50), ("C_mid", [0, 6], 50), ("C_lo", [4, 5], 50),
             ("D_1", [2, 0], 50), ("D_2", [8, 1], 50)]
    for arm, S2, per in specs:
        for s in S3:
            rng = np.random.default_rng(s * 100 + OFF[arm])
            sel = np.concatenate([rng.choice(idx(src, binid, S2, [b]), per, replace=False) for b in range(5)])
            nm = [str(x) for x in names[sel]]
            write_filter_key(merged, f"{arm}_seed{s}", nm)
            manifest["arms"][f"{arm}_seed{s}"] = {"size": len(nm), "demo_names": nm}
        print(f"  threading {arm} {S2}: {per}/bin x5 ={per*5} x3seed")
    (CTRL / "threading" / "arms_manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    t = sys.argv[1] if len(sys.argv) > 1 else "all"
    if t in ("stack", "all"): build_stack()
    if t in ("threading", "all"): build_threading()
    print("BUILD DONE", flush=True)
