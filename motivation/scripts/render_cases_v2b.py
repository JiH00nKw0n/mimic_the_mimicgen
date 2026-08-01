"""Fallback case renderer for tasks whose N0/N1 pools are infeasible
(three_piece_assembly, stack_three: the tiny shared N0 box cannot place all
objects, so those pools never existed). Region bands are built from the N2
pool alone, split by the REGISTRY N1 box: inner = every object within the N1
box (the pre-expansion region), ring = any object outside (newly opened area).
Source axis identical to render_cases_v2. Overwrites the task's manifest.
"""
import json, sys, traceback
from pathlib import Path
import numpy as np

REPO = "/home/ubuntu/mimicgen_jihoonkwon/mimic_the_mimicgen/motivation"
GEN = Path("/home/ubuntu/mimicgen_jihoonkwon/experiments/motivation_new/gen")
ARM = Path("/home/ubuntu/mimicgen_jihoonkwon/experiments/motivation_new/e2_arms")
OUT = Path("/home/ubuntu/case_renders_v2")
TASKS = sys.argv[1:] or ["three_piece_assembly", "stack_three"]
K = 10
RES = 160
PAD = 3e-3
sys.path.insert(0, REPO)
import h5py, imageio
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
from genaudit.evaluation.frozen_resets import _register_variants
from genaudit.envs.bounds_new import NEW_BOUNDS

_register_variants()
ObsUtils.initialize_obs_utils_with_obs_specs({"obs": {"low_dim": ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"], "rgb": []}})


def frame0_xy(grp):
    op = grp["datagen_info"]["object_poses"]
    return {k: op[k][0][:2, 3] for k in op.keys()}


def spread(cands, key, k):
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
        for old in tdir.glob("region_*.mp4"):
            old.unlink()
        boxes = {o: pb for o, pb in NEW_BOUNDS[task]["N1"].items()}
        recs = [json.loads(x) for x in open(ARM / f"{task}_N2" / "attempts.jsonl")]
        files = {}
        for name in ["demo.hdf5", "demo_failed.hdf5"]:
            p = GEN / f"{task}_N2" / name
            if p.exists():
                files[name] = h5py.File(p, "r")
        env_meta = FileUtils.get_env_metadata_from_dataset(str(GEN / f"{task}_N2" / "demo.hdf5"))
        env = EnvUtils.create_env_from_metadata(env_meta=env_meta, render=False, render_offscreen=True)

        inner, ring = [], []
        for r in recs:
            g, fname = r["attempt_id"].split("@")
            if fname not in files:
                continue
            xys = frame0_xy(files[fname]["data"][g])
            outside = False
            for o, xy in xys.items():
                pb = boxes.get(o)
                if pb is None:
                    continue
                if (xy[0] < pb.x[0] - PAD or xy[0] > pb.x[1] + PAD or
                        xy[1] < pb.y[0] - PAD or xy[1] > pb.y[1] + PAD):
                    outside = True
                    break
            (ring if outside else inner).append(r)

        man = {"task": task, "clips": [], "bands": {}}
        n2_dgr = round(sum(r["success"] for r in recs) / len(recs), 3)
        man["bands"]["n0"] = {"mode": "inner", "n": len(inner),
                              "dgr": round(sum(r["success"] for r in inner) / max(len(inner), 1), 3)}
        man["bands"]["ring"] = {"n": len(ring), "n2_total": len(recs), "n2_dgr": n2_dgr,
                                "dgr": round(sum(r["success"] for r in ring) / max(len(ring), 1), 3)}
        for band, pool in [("n0", inner), ("ring", ring)]:
            for succ in [True, False]:
                cs = [r for r in pool if r["success"] == succ]
                for i, r in enumerate(spread(cs, lambda x: x["d_pos"], K)):
                    g, fname = r["attempt_id"].split("@")
                    bname = "n0" if band == "n0" else "ring"
                    tag = f"region_{bname}_{'OK' if succ else 'FAIL'}_{i:02d}_d{r['d_pos']:.2f}"
                    n = render_clip(env, files[fname]["data"][g], tdir / f"{tag}.mp4")
                    man["clips"].append({"axis": "region", "band": bname, "success": succ,
                                         "d_pos": round(r["d_pos"], 3), "src": r["source_demo_id"],
                                         "file": f"{tag}.mp4", "ep_len": n})
                    print(f"  {task} {tag}", flush=True)

        # source axis (same as v2)
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
        have = {c["file"] for c in man["clips"]}
        for role, s in [("hi", hi_s), ("lo", lo_s)]:
            cases = [("OK", "near", [r for r in recs if r["source_demo_id"] == s and r["success"] and r["d_pos"] <= q40]),
                     ("OK", "far", [r for r in recs if r["source_demo_id"] == s and r["success"] and r["d_pos"] >= q60]),
                     ("FAIL", "far", [r for r in recs if r["source_demo_id"] == s and not r["success"] and r["d_pos"] >= q60])]
            for oc, band, cs in cases:
                for i, r in enumerate(spread(cs, lambda x: x["d_pos"], K)):
                    g, fname = r["attempt_id"].split("@")
                    if fname not in files:
                        continue
                    tag = f"src_{role}_s{s}_DGR{int(dgr[s]*100):02d}_{oc}_{band}_{i:02d}_d{r['d_pos']:.2f}"
                    if not (tdir / f"{tag}.mp4").exists():
                        n = render_clip(env, files[fname]["data"][g], tdir / f"{tag}.mp4")
                    else:
                        n = 0
                    man["clips"].append({"axis": "source", "role": role, "src": s, "dgr": round(dgr[s], 2),
                                         "success": oc == "OK", "band": band, "d_pos": round(r["d_pos"], 3),
                                         "file": f"{tag}.mp4", "ep_len": n})
                    print(f"  {task} {tag}", flush=True)
        for f in files.values():
            f.close()
        env.env.close() if hasattr(env, "env") else None
        (tdir / "manifest.json").write_text(json.dumps(man, indent=1))
        print(f"{task}: DONE {len(man['clips'])} clips  innerDGR={man['bands']['n0']['dgr']}(n{man['bands']['n0']['n']}) ringDGR={man['bands']['ring']['dgr']}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"{task}: ERR {e}"); traceback.print_exc()
print("RENDER V2B ALL DONE", flush=True)
