"""Make a documentation GIF from a rendered RGB dataset.

Lays the requested camera roles side by side for one episode and writes an
animated GIF small enough to live in the repo next to the docs.

  python3 make_preview_gif.py --rgb rgb.hdf5 --demo demo_0 \
      --output render/media/rgb_demo_0.gif [--stride 4] [--scale 0.75]
"""
from __future__ import annotations

import argparse
import os

import h5py
import numpy as np
from PIL import Image

DEFAULT_ROLES = ("third_person_0", "third_person_1", "wrist")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rgb", required=True)
    ap.add_argument("--demo", default="demo_0")
    ap.add_argument("--output", required=True)
    ap.add_argument("--roles", default=",".join(DEFAULT_ROLES))
    ap.add_argument("--stride", type=int, default=4, help="keep every Nth frame")
    ap.add_argument("--max_frames", type=int, default=64)
    ap.add_argument("--scale", type=float, default=0.75)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--colors", type=int, default=128)
    args = ap.parse_args()

    roles = [r.strip() for r in args.roles.split(",") if r.strip()]
    with h5py.File(args.rgb, "r", locking=False) as fh:
        obs = fh[f"data/{args.demo}/obs"]
        keys = [f"{r}_image" for r in roles if f"{r}_image" in obs]
        if not keys:
            raise SystemExit(f"no image datasets for roles={roles}")
        n = obs[keys[0]].shape[0]
        idx = list(range(0, n, args.stride))[: args.max_frames]
        frames = []
        for t in idx:
            tiles = []
            for k in keys:
                img = Image.fromarray(obs[k][t])
                if args.scale != 1.0:
                    img = img.resize(
                        (int(img.width * args.scale), int(img.height * args.scale)),
                        Image.LANCZOS)
                tiles.append(np.asarray(img))
            frames.append(Image.fromarray(np.concatenate(tiles, axis=1)))

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    pal = [f.convert("P", palette=Image.ADAPTIVE, colors=args.colors) for f in frames]
    pal[0].save(args.output, save_all=True, append_images=pal[1:],
                duration=int(1000 * args.stride / args.fps), loop=0, optimize=True)
    size_mb = os.path.getsize(args.output) / 1e6
    print(f"[gif] {args.output}: {len(frames)} frames, "
          f"{pal[0].width}x{pal[0].height}, {size_mb:.2f} MB, roles={keys}")


if __name__ == "__main__":
    main()
