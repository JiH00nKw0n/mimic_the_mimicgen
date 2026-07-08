import sys, h5py, numpy as np
from collections import defaultdict
path = sys.argv[1] if len(sys.argv) > 1 else "datasets/generated/square_sart.hdf5"
f = h5py.File(path, "r"); d = f["data"]
demos = sorted(d.keys(), key=lambda x: int(x.split("_")[1]))
groups = defaultdict(list)
for dk in demos:
    g = d[dk]; eef = g["obs"]["robot0_eef_pos"][:]
    src = int(np.array(g["src_demo_inds"]).reshape(-1)[0])
    groups[src].append(eef)
K = 90
peak, profiles = [], []
for src, arrs in groups.items():
    if len(arrs) < 2:
        continue
    m = min(a.shape[0] for a in arrs); w = min(K, m)
    stack = np.array([a[-w:] for a in arrs])            # (n, w, 3), aligned by END (insertion)
    std_k = stack.std(axis=0).mean(axis=-1)             # (w,) cross-demo std per reverse step
    profiles.append(std_k); peak.append(float(std_k.max()))
print("n_demos:", len(demos), "| source groups w/>=2:", len(peak))
print("PEAK same-source approach eef std (SART diversity, m):", round(float(np.mean(peak)), 4))
print("  per-group peak:", [round(p, 4) for p in peak])
w = min(len(p) for p in profiles)
avg = np.mean([p[-w:] for p in profiles], axis=0)       # index -1 = last (insertion end)
for k in [3, 10, 20, 30, 40, 50, 60]:
    if k <= w:
        print("  reverse-idx -%d (end=verbatim insertion): mean same-source eef std = %.4f m" % (k, avg[-k]))
