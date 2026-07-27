"""Per-source generation case study — why does ancestry_balanced hurt (esp. stack)?

For each task, from attempts.jsonl compute per SOURCE demo (0..9):
  n_att, n_ret, DGR=n_ret/n_att, mean d_pos of retained, mean episode_length of retained.
baseline's source composition == retained share (it draws random from retained).
ancestry_balanced forces ~equal counts per source (from arms_manifest), so it
UP-weights low-retained (=low-DGR) sources. Test the 'balancing dilutes good
demos' hypothesis: is the ancestry mass-shift toward LOW-DGR / LONG-episode
(=lower quality proxy) sources? Report per-source table + the correlations
corr(shift, DGR) and corr(shift, mean_eplen).

Usage: mnew_casestudy.py <task> ...
"""
import json
import sys
from pathlib import Path

import numpy as np

A = Path("/home/ubuntu/mimicgen_jihoonkwon/experiments/motivation_new/e2_arms")


def analyze(task):
    recs = [json.loads(x) for x in open(A / f"{task}_N2" / "attempts.jsonl") if x.strip()]
    src = np.array([r["source_demo_id"] for r in recs])
    succ = np.array([r["success"] for r in recs])
    dpos = np.array([r["d_pos"] for r in recs])
    eplen = np.array([r.get("episode_length", -1) for r in recs])
    nsrc = int(src.max()) + 1

    rows = []
    for s in range(nsrc):
        att = src == s
        ret = att & succ
        n_att, n_ret = int(att.sum()), int(ret.sum())
        rows.append({
            "src": s, "n_att": n_att, "n_ret": n_ret,
            "DGR": round(n_ret / n_att, 3) if n_att else 0.0,
            "d_ret": round(float(dpos[ret].mean()), 3) if n_ret else None,
            "eplen_ret": round(float(eplen[ret][eplen[ret] > 0].mean()), 1)
            if n_ret and (eplen[ret] > 0).any() else None,
        })
    tot_ret = sum(r["n_ret"] for r in rows)
    for r in rows:
        r["base_share"] = round(r["n_ret"] / tot_ret, 3) if tot_ret else 0.0

    # ancestry per-source counts from manifest (seed101)
    manifest = json.loads((A / f"{task}_N2" / "arms_manifest.json").read_text())
    anc_key = next((k for k in manifest["arms"] if k.startswith("ancestry_balanced")), None)
    if anc_key:
        anc_counts = manifest["arms"][anc_key]["per_stratum_counts"]
        tot_anc = sum(anc_counts)
        for r in rows:
            r["anc_share"] = round(anc_counts[r["src"]] / tot_anc, 3) if tot_anc else 0.0
            r["shift"] = round(r["anc_share"] - r["base_share"], 3)  # + = ancestry upweights
    else:
        for r in rows:
            r["anc_share"] = r["shift"] = None

    # correlations across sources (only where ancestry exists)
    corr = {}
    if anc_key:
        shift = np.array([r["shift"] for r in rows], float)
        dgr = np.array([r["DGR"] for r in rows], float)
        epl = np.array([r["eplen_ret"] if r["eplen_ret"] else np.nan for r in rows], float)
        m = ~np.isnan(epl)
        if shift.std() > 0 and dgr.std() > 0:
            corr["shift_vs_DGR"] = round(float(np.corrcoef(shift, dgr)[0, 1]), 3)
        if m.sum() > 2 and shift[m].std() > 0 and epl[m].std() > 0:
            corr["shift_vs_eplen"] = round(float(np.corrcoef(shift[m], epl[m])[0, 1]), 3)
        if dgr.std() > 0 and m.sum() > 2 and dgr[m].std() > 0 and epl[m].std() > 0:
            corr["DGR_vs_eplen"] = round(float(np.corrcoef(dgr[m], epl[m])[0, 1]), 3)

    # print
    print(f"\n==== {task}  (retained total={tot_ret}, sources={nsrc}) ====")
    print("  src  n_att  n_ret   DGR   d_ret  eplen  base%  anc%   shift")
    for r in sorted(rows, key=lambda x: -x["n_ret"]):
        print(f"  s{r['src']:<3d} {r['n_att']:5d}  {r['n_ret']:5d}  {r['DGR']:.3f}  "
              f"{str(r['d_ret']):>5}  {str(r['eplen_ret']):>5}  "
              f"{r['base_share'] if r['base_share'] is not None else '—'!s:>5}  "
              f"{r['anc_share'] if r['anc_share'] is not None else '—'!s:>5}  "
              f"{r['shift'] if r['shift'] is not None else '—'!s:>6}")
    print(f"  corr: {corr}  "
          f"(shift_vs_DGR<0 → ancestry가 저DGR source로 질량 이동; "
          f"DGR_vs_eplen<0 → 저DGR가 긴(나쁜) 데모)")
    return {"task": task, "rows": rows, "corr": corr, "retained_total": tot_ret}


def main():
    out = {}
    for task in sys.argv[1:]:
        try:
            out[task] = analyze(task)
        except Exception as e:  # noqa: BLE001
            import traceback
            print(f"{task}: ERR {e}"); traceback.print_exc()
    (A / "casestudy_per_source.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {A / 'casestudy_per_source.json'}")


if __name__ == "__main__":
    main()
