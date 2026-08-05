"""Join rendered camera frames with contract-format actions.

The two halves of an RGB-Action sample are produced by different tools:

  render/render_viewpoints.py  -> per-frame camera images at the SOURCE rate
                                  (20 Hz for the lab generation env)
  contract/convert_demo.py     -> contract actions at the CONTRACT rate (10 Hz)

This script pairs them. The pairing is index arithmetic, not interpolation:
the contract conversion resamples the source clock by an integer factor
(20 -> 10 Hz, factor 2), so contract step k is exactly source frame k*factor.
The factor is derived per demo from the two lengths and rejected unless it is
integral, so a rate change on either side fails loudly instead of silently
misaligning images and actions.

Output keeps the contract schema (so the RL team's validator still applies)
and adds one uint8 image dataset per camera under obs/.

Pure h5py/numpy — runs on the host:
  python3 join_rgb_contract.py --rgb rgb.hdf5 --contract contract.hdf5 \
      --output rgb_action.hdf5 [--cameras third_person_0,third_person_1,wrist]
"""
from __future__ import annotations

import argparse
import json

import h5py
import numpy as np

CONTRACT_CAMERAS = ("third_person_0", "third_person_1", "wrist")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rgb", required=True)
    ap.add_argument("--contract", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--cameras", default=",".join(CONTRACT_CAMERAS),
                    help="camera roles to keep; the visual-randomization "
                         "contract stores exactly third_person_0/1 + wrist")
    ap.add_argument("--no_compress", action="store_true")
    args = ap.parse_args()
    roles = [r.strip() for r in args.cameras.split(",") if r.strip()]

    rgb = h5py.File(args.rgb, "r", locking=False)
    con = h5py.File(args.contract, "r", locking=False)
    out = h5py.File(args.output, "w")
    data = out.create_group("data")
    for k, v in con["data"].attrs.items():
        data.attrs[k] = v
    for k, v in rgb["data"].attrs.items():
        if k in ("visual_randomization", "fr3_camera_overlay", "fr3_binding"):
            data.attrs[k] = v
    data.attrs["rgb_source"] = args.rgb
    data.attrs["cameras"] = json.dumps(roles)

    comp = {} if args.no_compress else {
        "compression": "gzip", "compression_opts": 4, "shuffle": True}
    kept, report = 0, {}
    names = [n for n in con["data"].keys() if n in rgb["data"]]
    missing = [n for n in con["data"].keys() if n not in rgb["data"]]
    for name in names:
        cg, rg = con[f"data/{name}"], rgb[f"data/{name}"]
        n_con = int(cg.attrs["num_samples"])
        n_rgb = int(rg.attrs["num_samples"])
        factor = n_rgb / n_con
        if abs(factor - round(factor)) > 1e-6 or round(factor) < 1:
            report[name] = f"REJECT non-integral rate ratio {n_rgb}/{n_con}"
            continue
        factor = int(round(factor))
        idx = np.arange(n_con) * factor
        if idx[-1] >= n_rgb:
            idx = idx[idx < n_rgb]
        con.copy(cg, data, name=name)
        ep = data[name]
        n_keep = len(idx)
        if n_keep < n_con:
            # trim every contract track so images and actions stay 1:1
            for key in list(ep.keys()):
                if isinstance(ep[key], h5py.Group):
                    for sub in list(ep[key].keys()):
                        arr = ep[f"{key}/{sub}"][()]
                        del ep[f"{key}/{sub}"]
                        ep.create_dataset(f"{key}/{sub}", data=arr[:n_keep])
                else:
                    arr = ep[key][()]
                    del ep[key]
                    ep.create_dataset(key, data=arr[:n_keep])
            ep.attrs["num_samples"] = n_keep
        obs = ep.require_group("obs")
        for role in roles:
            key = f"{role}_image"
            if key not in rg["obs"]:
                report.setdefault(name, "")
                report[name] += f" missing:{key}"
                continue
            arr = rg[f"obs/{key}"][()][idx]
            obs.create_dataset(key, data=arr, chunks=(1,) + arr.shape[1:], **comp)
        ep.attrs["rgb_stride"] = factor
        kept += 1
        report[name] = report.get(name, "") or f"ok stride={factor} T={n_keep}"

    data.attrs["total"] = int(sum(int(data[n].attrs["num_samples"]) for n in data.keys()))
    rgb.close()
    con.close()
    out.close()
    for name, msg in sorted(report.items()):
        print(f"{name}: {msg}")
    if missing:
        print(f"[join] {len(missing)} contract demos had no rendered frames: {missing[:5]}")
    print(f"[join] wrote {args.output}: {kept} episodes, cameras={roles}")


if __name__ == "__main__":
    main()
