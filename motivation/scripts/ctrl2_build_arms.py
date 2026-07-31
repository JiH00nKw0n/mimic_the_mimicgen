"""ctrl2: multi-task reinforcement of the controlled experiments (EXP-A/B/C2/D).
Gate: /home/ubuntu/ctrl2_gate.py output, 2026-07-31. Source pairs per task chosen
so hi-vs-mid DGR gap >= 0.15 (C2), pair mean-DGR gap <= 0.03 (D), and every
distance bin holds the quota. Writes filter keys + extends arms_manifest.json in
motivation_controlled/arms/<task>_N2 (idempotent: existing keys are skipped).
Recipes (3 seeds 301-303): A 300 = near225+far75 vs near75+far225;
B 250 = near100+mid50+far100 vs near200+mid50 (far zero); C2/D 50/bin x5 = 250.
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
       "C2_hi": 8, "C2_mid": 9, "D_1": 10, "D_2": 11}
S3 = [301, 302, 303]
SPECS = {
    "square": {"AB": [0, 4], "C2": ([0, 4], [6, 3]), "D": ([4, 6], [0, 8])},
    "coffee": {"AB": [0, 3], "C2": ([0, 3], [1, 4]), "D": ([0, 6], [3, 5])},
    "three_piece_assembly": {"C2": ([3, 7], [9, 2])},
    "stack_three": {"C2": ([7, 9], [6, 1])},
    "threading": {"AB": [2, 8]},
    "stack": {"D": ([4, 1], [7, 6])},
}


def build(task, spec):
    dst = CTRL / f"{task}_N2"
    dst.mkdir(parents=True, exist_ok=True)
    merged = dst / "train.hdf5"
    if not merged.exists():
        shutil.copy(NEW / f"{task}_N2" / "train.hdf5", merged)
    fr = dst / "frozen_resets.hdf5"
    if not fr.exists():
        shutil.copy(NEW / f"{task}_N2" / "frozen_resets.hdf5", fr)
    mpath = dst / "arms_manifest.json"
    manifest = json.loads(mpath.read_text()) if mpath.exists() else {"task": task, "arms": {}}
    with h5py.File(merged, "r") as f:
        prov2name = {f["data"][g].attrs["provenance"]: g for g in f["data"].keys()}
    recs = [r for r in read_jsonl(NEW / f"{task}_N2" / "attempts.jsonl") if r.success]
    prov = np.array([r.attempt_id.split("@")[0] for r in recs])
    names = np.array([prov2name[p] for p in prov])
    src = np.array([r.source_demo_id for r in recs])
    d = np.array([r.d_pos for r in recs])
    edges = np.quantile(d, [.2, .4, .6, .8])
    binid = np.searchsorted(edges, d, side="right")

    def idx(sources, bins):
        return np.flatnonzero(np.isin(src, sources) & np.isin(binid, bins))

    def emit(arm, plan):
        made = 0
        for s in S3:
            key = f"{arm}_seed{s}"
            if key in manifest["arms"]:
                continue
            rng = np.random.default_rng(s * 100 + OFF[arm])
            sel = np.concatenate([rng.choice(pool, n, replace=False) for pool, n in plan])
            nm = [str(x) for x in names[sel]]
            write_filter_key(merged, key, nm)
            manifest["arms"][key] = {"size": len(nm), "demo_names": nm}
            made += 1
        print(f"  {task} {arm}: n={sum(n for _, n in plan)} new_seeds={made}", flush=True)

    if "AB" in spec:
        S = spec["AB"]
        near, mid, far = idx(S, [0, 1]), idx(S, [2]), idx(S, [3, 4])
        emit("A_nearheavy", [(near, 225), (far, 75)])
        emit("A_farheavy", [(near, 75), (far, 225)])
        emit("B_far", [(near, 100), (mid, 50), (far, 100)])
        emit("B_nearpad", [(near, 200), (mid, 50)])
    if "C2" in spec:
        for arm, pair in zip(("C2_hi", "C2_mid"), spec["C2"]):
            emit(arm, [(idx(pair, [b]), 50) for b in range(5)])
    if "D" in spec:
        for arm, pair in zip(("D_1", "D_2"), spec["D"]):
            emit(arm, [(idx(pair, [b]), 50) for b in range(5)])
    mpath.write_text(json.dumps(manifest, indent=2))


for t, s in SPECS.items():
    build(t, s)
print("CTRL2 BUILD DONE", flush=True)
