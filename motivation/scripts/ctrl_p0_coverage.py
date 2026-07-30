"""P0 gate: d_own coverage map. For each experiment source set, re-slice the 200
eval states by d_own (distance to the arm's TRAINING sources) and check whether
far-d_own eval states fall within the training demos' d_pos support (else the
'far' claim for that arm must be restricted to the coverable range)."""
import json, numpy as np
from pathlib import Path
NEW=Path("/home/ubuntu/mimicgen_jihoonkwon/experiments/motivation_new/e2_arms")
CTRL=Path("/home/ubuntu/mimicgen_jihoonkwon/experiments/motivation_controlled")
SETS={"stack":{"AB_C2hi[4,7]":[4,7],"C2mid[3,1]":[3,1]},
      "threading":{"Chi[2,8]":[2,8],"Cmid[0,6]":[0,6],"Clo[4,5]":[4,5],"D1[2,0]":[2,0],"D2[8,1]":[8,1]}}
res={}
for task,sets in SETS.items():
    mf=NEW/f"{task}_N2"/"deval_matrix.json"
    if not mf.exists(): print(f"{task}: matrix missing"); continue
    M=json.load(open(mf)); mat={int(k):v for k,v in M["matrix"].items()}
    recs=[json.loads(x) for x in open(NEW/f"{task}_N2"/"attempts.jsonl")]
    ret=[r for r in recs if r["success"]]
    print(f"\n==== {task} ====")
    res[task]={}
    for nm,S in sets.items():
        d_own=np.array([min(mat[i][j] for j in S) for i in sorted(mat)])           # eval->nearest training source
        tr=np.array([r["d_pos"] for r in ret if r["source_demo_id"] in S])          # training demo d_pos support
        p90=np.quantile(tr,0.90); tmax=tr.max()
        q=np.quantile(d_own,[1/3,2/3])
        far=d_own>q[1]
        covered_far=np.mean(d_own[far]<=p90)                                        # far-eval within training support?
        frac_extrap=np.mean(d_own>p90)
        res[task][nm]={"eval_d_own_terciles":[round(float(x),3) for x in q],
                       "eval_d_own_max":round(float(d_own.max()),3),
                       "train_dpos_med":round(float(np.median(tr)),3),"train_dpos_p90":round(float(p90),3),"train_dpos_max":round(float(tmax),3),
                       "n_train":int(len(tr)),
                       "far_eval_covered_frac":round(float(covered_far),3),
                       "overall_extrap_frac":round(float(frac_extrap),3)}
        print(f"  {nm}: eval d_own med~{np.median(d_own):.3f} max {d_own.max():.3f} | train d_pos med {np.median(tr):.3f} p90 {p90:.3f} max {tmax:.3f}")
        print(f"      far-eval covered(d_own<=train p90): {covered_far:.0%}  | overall beyond-support: {frac_extrap:.0%}")
CTRL.joinpath("gates","p0_coverage.json").write_text(json.dumps(res,indent=2))
# also copy matrices into gates/
import shutil
for task in SETS:
    src=NEW/f"{task}_N2"/"deval_matrix.json"
    if src.exists(): shutil.copy(src, CTRL/"gates"/f"deval_matrix_{task}.json")
print("\nwrote", CTRL/"gates"/"p0_coverage.json")
