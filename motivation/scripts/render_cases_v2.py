"""Case-study renderer v2. Two axes per task, up to 10 clips per case:
REGION axis: attempts generated in the N0 (narrowest) region pool, vs attempts
  from the N2 pool whose initial placement falls OUTSIDE the N1 region
  (ring = N2 minus N1; N1 box estimated empirically from the N1 pool's frame-0
  object poses, all attempts incl. failed). Cases: {n0, ring} x {OK, FAIL} x 10.
REGION DGR per band is recorded in the manifest.
SOURCE axis (v1 extended): hi/lo-DGR source x {OK-near, OK-far, FAIL-far} x 10,
  samples spread across d_pos within the band.
Output: /home/ubuntu/case_renders_v2/<task>/*.mp4 + manifest.json
Run niced so ctrl2 training keeps priority.
"""
import json, sys, traceback
from pathlib import Path
import numpy as np

REPO = "/home/ubuntu/mimicgen_jihoonkwon/mimic_the_mimicgen/motivation"
GEN = Path("/home/ubuntu/mimicgen_jihoonkwon/experiments/motivation_new/gen")
ARM = Path("/home/ubuntu/mimicgen_jihoonkwon/experiments/motivation_new/e2_arms")
OUT = Path("/home/ubuntu/case_renders_v2"); OUT.mkdir(exist_ok=True)
TASKS = sys.argv[1:] or ["stack", "square", "coffee", "threading", "three_piece_assembly",
                         "stack_three", "hammer_cleanup", "mug_cleanup"]
K = 10  # clips per case
RES = 160
sys.path.insert(0, REPO)
import h5py, imageio
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
from genaudit.evaluation.frozen_resets import _register_variants

_register_variants()
ObsUtils.initialize_obs_utils_with_obs_specs({"obs": {"low_dim": ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"], "rgb": []}})


def frame0_xy(grp):
    op = grp["datagen_info"]["object_poses"]
    return {k: op[k][0][:2, 3] for k in op.keys()}


def pool_files(d):
    out = {}
    for name in ["demo.hdf5", "demo_failed.hdf5"]:
        p = d / name
        if p.exists():
            out[name] = h5py.File(p, "r")
    return out


def n1_box(task):
    """Per-object empirical xy bounds over ALL N1 attempts (success+fail)."""
    files = pool_files(GEN / f"{task}_N1")
    lo, hi = {}, {}
    for f in files.values():
        for g in f["data"].keys():
            for obj, xy in frame0_xy(f["data"][g]).items():
                if obj not in lo:
                    lo[obj] = xy.copy(); hi[obj] = xy.copy()
                lo[obj] = np.minimum(lo[obj], xy); hi[obj] = np.maximum(hi[obj], xy)
    for f in files.values():
        f.close()
    return lo, hi


def spread(cands, key, k):
    """Pick up to k candidates spread evenly along key(c)."""
    if len(cands) <= k:
        return list(cands)
    cs = sorted(cands, key=key)
    idx = np.linspace(0, len(cs) - 1, k).round().astype(int)
    return [cs[i] for i in sorted(set(idx))]


def render_clip(env, grp, path):
    states = grp["states"][()]
    model = grp.attrs["model_file"]
    frames = []
    for t in range(0, len(states), 4):
        env.reset_to({"model": model, "states": states[t]})
        frames.append(env.render(mode="rgb_array", height=RES, width=RES, camera_name="agentview"))
    imageio.mimsave(str(path), frames, fps=12, macro_block_size=1)
    return len(states)


for task in TASKS:
    try:
        tdir = OUT / task; tdir.mkdir(exist_ok=True)
        man = {"task": task, "clips": [], "bands": {}}
        env_meta = FileUtils.get_env_metadata_from_dataset(str(GEN / f"{task}_N2" / "demo.hdf5"))
        env = EnvUtils.create_env_from_metadata(env_meta=env_meta, render=False, render_offscreen=True)

        # ---------- REGION axis ----------
        # N0 band: straight from the N0 pool
        n0f = pool_files(GEN / f"{task}_N0")
        n_ok = len(n0f["demo.hdf5"]["data"]) if "demo.hdf5" in n0f else 0
        n_fail = len(n0f["demo_failed.hdf5"]["data"]) if "demo_failed.hdf5" in n0f else 0
        man["bands"]["n0"] = {"dgr": round(n_ok / max(n_ok + n_fail, 1), 3), "n": n_ok + n_fail}
        for succ, fname in [(True, "demo.hdf5"), (False, "demo_failed.hdf5")]:
            if fname not in n0f:
                continue
            groups = sorted(n0f[fname]["data"].keys())
            for i, g in enumerate(spread(groups, lambda x: hash(x) % 9973, K)):
                tag = f"region_n0_{'OK' if succ else 'FAIL'}_{i:02d}"
                n = render_clip(env, n0f[fname]["data"][g], tdir / f"{tag}.mp4")
                man["clips"].append({"axis": "region", "band": "n0", "success": succ, "file": f"{tag}.mp4", "ep_len": n})
                print(f"  {task} {tag}", flush=True)
        for f in n0f.values():
            f.close()

        # ring band: N2 attempts with any object outside the empirical N1 box
        lo, hi = n1_box(task)
        recs = [json.loads(x) for x in open(ARM / f"{task}_N2" / "attempts.jsonl")]
        n2f = pool_files(GEN / f"{task}_N2")
        ring = []
        n_ring_ok = n_ring = 0
        for r in recs:
            g, fname = r["attempt_id"].split("@")
            if fname not in n2f:
                continue
            xys = frame0_xy(n2f[fname]["data"][g])
            outside = any((xy[0] < lo[o][0] - 3e-3 or xy[0] > hi[o][0] + 3e-3 or
                           xy[1] < lo[o][1] - 3e-3 or xy[1] > hi[o][1] + 3e-3)
                          for o, xy in xys.items() if o in lo)
            if outside:
                n_ring += 1; n_ring_ok += int(r["success"])
                ring.append(r)
        man["bands"]["ring"] = {"dgr": round(n_ring_ok / max(n_ring, 1), 3), "n": n_ring,
                                "n2_total": len(recs),
                                "n2_dgr": round(sum(r["success"] for r in recs) / len(recs), 3)}
        for succ in [True, False]:
            cs = [r for r in ring if r["success"] == succ]
            for i, r in enumerate(spread(cs, lambda x: x["d_pos"], K)):
                g, fname = r["attempt_id"].split("@")
                tag = f"region_ring_{'OK' if succ else 'FAIL'}_{i:02d}_d{r['d_pos']:.2f}"
                n = render_clip(env, n2f[fname]["data"][g], tdir / f"{tag}.mp4")
                man["clips"].append({"axis": "region", "band": "ring", "success": succ, "d_pos": round(r["d_pos"], 3),
                                     "src": r["source_demo_id"], "file": f"{tag}.mp4", "ep_len": n})
                print(f"  {task} {tag}", flush=True)

        # ---------- SOURCE axis (v1 x10) ----------
        src = np.array([r["source_demo_id"] for r in recs]); su = np.array([r["success"] for r in recs])
        d = np.array([r["d_pos"] for r in recs])
        nsrc = int(src.max()) + 1
        dgr = {s: float(su[src == s].mean()) for s in range(nsrc)}
        nok = {s: int((su & (src == s)).sum()) for s in range(nsrc)}
        renderable = [s for s in range(nsrc) if nok[s] >= 3]
        hi_s = max(renderable, key=lambda s: dgr[s]); lo_s = min(renderable, key=lambda s: dgr[s])
        q40, q60 = np.quantile(d[su], .4), np.quantile(d[su], .6)
        man["hi_src"] = {"id": hi_s, "dgr": round(dgr[hi_s], 2)}
        man["lo_src"] = {"id": lo_s, "dgr": round(dgr[lo_s], 2)}
        for role, s in [("hi", hi_s), ("lo", lo_s)]:
            cases = [("OK", "near", [r for r in recs if r["source_demo_id"] == s and r["success"] and r["d_pos"] <= q40]),
                     ("OK", "far", [r for r in recs if r["source_demo_id"] == s and r["success"] and r["d_pos"] >= q60]),
                     ("FAIL", "far", [r for r in recs if r["source_demo_id"] == s and not r["success"] and r["d_pos"] >= q60])]
            for oc, band, cs in cases:
                for i, r in enumerate(spread(cs, lambda x: x["d_pos"], K)):
                    g, fname = r["attempt_id"].split("@")
                    if fname not in n2f:
                        continue
                    tag = f"src_{role}_s{s}_DGR{int(dgr[s]*100):02d}_{oc}_{band}_{i:02d}_d{r['d_pos']:.2f}"
                    n = render_clip(env, n2f[fname]["data"][g], tdir / f"{tag}.mp4")
                    man["clips"].append({"axis": "source", "role": role, "src": s, "dgr": round(dgr[s], 2),
                                         "success": oc == "OK", "band": band, "d_pos": round(r["d_pos"], 3),
                                         "file": f"{tag}.mp4", "ep_len": n})
                    print(f"  {task} {tag}", flush=True)
        for f in n2f.values():
            f.close()
        env.env.close() if hasattr(env, "env") else None
        (tdir / "manifest.json").write_text(json.dumps(man, indent=1))
        print(f"{task}: DONE {len(man['clips'])} clips  n0DGR={man['bands']['n0']['dgr']} ringDGR={man['bands']['ring']['dgr']} (n2 {man['bands']['ring']['n2_dgr']})", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"{task}: ERR {e}"); traceback.print_exc()
print("RENDER V2 ALL DONE", flush=True)
