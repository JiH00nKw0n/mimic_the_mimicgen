import h5py, glob
import numpy as np
import xml.etree.ElementTree as ET

def nut_geom_size(model_xml):
    """Sum of SquareNut geom size components -> proxy for nut scale."""
    root = ET.fromstring(model_xml)
    sizes = []
    for geom in root.iter('geom'):
        nm = (geom.get('name') or '')
        if 'SquareNut' in nm or 'nut' in nm.lower():
            s = geom.get('size')
            if s:
                sizes.append(sum(float(x) for x in s.split()))
    return sum(sizes) if sizes else None

for tag, pat in [("SUCCESS", "out/run1/successes/*.hdf5"), ("FAILURE", "out/run1/failures/*.hdf5")]:
    files = sorted(glob.glob(pat))
    vals = []
    for fp in files:
        f = h5py.File(fp, "r"); d = f["data"]
        dk = list(d.keys())[0]; g = d[dk]
        mf = g.attrs.get("model_file")
        if isinstance(mf, bytes): mf = mf.decode()
        ns = nut_geom_size(mf) if mf is not None else None
        if ns is not None: vals.append(ns)
        f.close()
    print(f"=== {tag}: {len(files)} demos, {len(vals)} with nut size ===")
    if vals:
        print("  nut total-geom-size per demo:", [round(v, 4) for v in vals])
        if len(vals) > 1:
            print("  spread: min=%.4f max=%.4f std=%.5f range/mean=%.1f%%" %
                  (min(vals), max(vals), np.std(vals), 100*(max(vals)-min(vals))/np.mean(vals)))
