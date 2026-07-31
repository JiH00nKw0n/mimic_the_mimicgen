"""Case-study renderer for ALL tasks. Per task: auto-pick the highest-DGR
('frequently used') and lowest-DGR ('rarely used') source demo that still has
>=3 successes, then render synthetic demos: hi/lo x {success near, success far,
failure far} to mp4. Labels each with source/DGR/success/d_pos.
Output: /home/ubuntu/case_renders/<task>/*.mp4 + manifest.json
"""
import json, sys, traceback
from pathlib import Path
import numpy as np

REPO = "/home/ubuntu/mimicgen_jihoonkwon/mimic_the_mimicgen/motivation"
GEN = Path("/home/ubuntu/mimicgen_jihoonkwon/experiments/motivation_new/gen")
ARM = Path("/home/ubuntu/mimicgen_jihoonkwon/experiments/motivation_new/e2_arms")
OUT = Path("/home/ubuntu/case_renders"); OUT.mkdir(exist_ok=True)
TASKS = sys.argv[1:] or ["square", "coffee", "three_piece_assembly", "stack",
                         "hammer_cleanup", "stack_three", "threading", "mug_cleanup"]
sys.path.insert(0, REPO)
import h5py, imageio
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
from genaudit.evaluation.frozen_resets import _register_variants

_register_variants()
ObsUtils.initialize_obs_utils_with_obs_specs({"obs": {"low_dim": ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"], "rgb": []}})
CAM = {"default": "agentview"}
allman = {}

def pick(recs, src, succ, dtarget):
    c = [r for r in recs if r["source_demo_id"] == src and r["success"] == succ]
    return min(c, key=lambda r: abs(r["d_pos"] - dtarget)) if c else None

for task in TASKS:
    try:
        gd = GEN / f"{task}_N2"
        recs = [json.loads(x) for x in open(ARM / f"{task}_N2" / "attempts.jsonl")]
        src = np.array([r["source_demo_id"] for r in recs]); su = np.array([r["success"] for r in recs])
        nsrc = int(src.max()) + 1
        dgr = {s: float(su[src == s].mean()) if (src == s).any() else 0 for s in range(nsrc)}
        nok = {s: int((su & (src == s)).sum()) for s in range(nsrc)}
        renderable = [s for s in range(nsrc) if nok[s] >= 3]
        hi = max(renderable, key=lambda s: dgr[s]); lo = min(renderable, key=lambda s: dgr[s])
        d_all = np.array([r["d_pos"] for r in recs if r["success"]])
        dn, df = float(np.quantile(d_all, .25)), float(np.quantile(d_all, .85))

        env_meta = FileUtils.get_env_metadata_from_dataset(str(gd / "demo.hdf5"))
        env = EnvUtils.create_env_from_metadata(env_meta=env_meta, render=False, render_offscreen=True)
        files = {"demo.hdf5": h5py.File(gd / "demo.hdf5", "r")}
        fp = gd / "demo_failed.hdf5"
        if fp.exists(): files["demo_failed.hdf5"] = h5py.File(fp, "r")
        (OUT / task).mkdir(exist_ok=True)
        man = {"task": task, "hi_src": hi, "hi_dgr": round(dgr[hi], 2), "lo_src": lo, "lo_dgr": round(dgr[lo], 2), "clips": []}
        specs = [("hi", hi, True, "near", dn), ("hi", hi, True, "far", df), ("hi", hi, False, "far", df),
                 ("lo", lo, True, "near", dn), ("lo", lo, True, "far", df), ("lo", lo, False, "far", df)]
        for role, s, succ, band, dt in specs:
            r = pick(recs, s, succ, dt)
            if r is None:
                continue
            gname, fname = r["attempt_id"].split("@")
            if fname not in files:
                continue
            grp = files[fname]["data"][gname]
            states = grp["states"][()]; model = grp.attrs["model_file"]
            frames = []
            for t in range(0, len(states), 4):
                env.reset_to({"model": model, "states": states[t]})
                frames.append(env.render(mode="rgb_array", height=220, width=220, camera_name=CAM["default"]))
            tag = f"{role}_s{s}_DGR{int(dgr[s]*100):02d}_{'OK' if succ else 'FAIL'}_{band}_d{r['d_pos']:.2f}"
            imageio.mimsave(str(OUT / task / f"{tag}.mp4"), frames, fps=12, macro_block_size=1)
            man["clips"].append({"role": role, "src": s, "dgr": round(dgr[s], 2), "success": succ,
                                 "band": band, "d_pos": round(r["d_pos"], 3), "ep_len": int(len(states)), "file": f"{tag}.mp4"})
            print(f"  {task} {tag}: {len(states)}st", flush=True)
        for f in files.values():
            f.close()
        env.env.close() if hasattr(env, "env") else None
        allman[task] = man
        print(f"{task}: hi=s{hi}(DGR{dgr[hi]:.2f}) lo=s{lo}(DGR{dgr[lo]:.2f}) {len(man['clips'])} clips", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"{task}: ERR {e}"); traceback.print_exc()

(OUT / "manifest.json").write_text(json.dumps(allman, indent=2))
print("RENDER ALL DONE", flush=True)
