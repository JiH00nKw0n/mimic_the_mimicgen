#!/usr/bin/env python3
"""Replay each demo and report: original recorded stack order vs replayed stack order
vs replay success (order-agnostic valid tower).

For every demo: reset_to(scattered start), apply the recorded actions, then read the
final cube z-heights -> bottom->top order, and whether it is a physically valid 3-tower
(both z-gaps ~ one cube, xy aligned). Also reads the ORIGINAL order from the dataset's
last recorded state. This tells us whether replay reproduces the demo's stack and order.

    python replay_order.py --task ...Fwd... --dataset teleop_random_fixed.hdf5
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-Stack-Cube-LabFR3-Fwd-IK-Rel-Mimic-v0")
parser.add_argument("--dataset", default="/home/ubuntu/mimicgen_jihoonkwon/mimic_the_mimicgen/datasets/teleop_random_fixed.hdf5")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import torch
import numpy as np
import h5py
import gymnasium as gym

from isaaclab.utils.datasets import HDF5DatasetFileHandler
import isaaclab_mimic.envs  # noqa
import lab_register  # noqa
import isaaclab_tasks  # noqa
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
env_cfg.terminations = None
env_cfg.recorders = {}
env = gym.make(args.task, cfg=env_cfg).unwrapped


def stack_order_and_valid(zpos, xypos):
    """zpos/xypos: dict cube-> z / xy. Return (order tuple bottom->top, valid bool)."""
    order = tuple(sorted(zpos, key=lambda c: zpos[c]))
    zs = sorted(zpos.values())
    gap_ok = (0.035 < zs[1] - zs[0] < 0.065) and (0.035 < zs[2] - zs[1] < 0.065)
    bot, mid, top = order
    xy_ok = (np.linalg.norm(xypos[mid] - xypos[bot]) < 0.05) and (np.linalg.norm(xypos[top] - xypos[mid]) < 0.05)
    return order, bool(gap_ok and xy_ok)


SHORT = {"cube_1": "1", "cube_2": "2", "cube_3": "3"}
def fmt(order):
    return "".join(SHORT[c] for c in order)


# original orders from the raw dataset (last recorded state)
orig = {}
with h5py.File(args.dataset) as h:
    demos = sorted(h["data"].keys(), key=lambda s: int(s.split("_")[1]))
    for k in demos:
        st = h["data"][k]["states"]["rigid_object"]
        z = {c: float(st[c]["root_pose"][-1][2]) for c in ["cube_1", "cube_2", "cube_3"]}
        xy = {c: st[c]["root_pose"][-1][:2].astype(float) for c in ["cube_1", "cube_2", "cube_3"]}
        orig[k] = stack_order_and_valid(z, xy)

handler = HDF5DatasetFileHandler()
handler.open(args.dataset)

rows = []
print(f"\n demo   | orig(b->t) ok | replay(b->t) ok | order_match")
with torch.inference_mode():
    for k in demos:
        ep = handler.load_episode(k, env.device)
        env.reset()
        env.reset_to(ep.get_initial_state(), torch.tensor([0], device=env.device), is_relative=True)
        while True:
            act = ep.get_next_action()
            if act is None:
                break
            env.step(act.unsqueeze(0) if act.ndim == 1 else act)
        org = env.scene.env_origins[0]
        z = {c: float(env.scene[c].data.root_pos_w[0, 2] - org[2]) for c in ["cube_1", "cube_2", "cube_3"]}
        xy = {c: (env.scene[c].data.root_pos_w[0, :2] - org[:2]).cpu().numpy() for c in ["cube_1", "cube_2", "cube_3"]}
        rorder, rvalid = stack_order_and_valid(z, xy)
        oorder, ovalid = orig[k]
        match = rorder == oorder
        rows.append((k, oorder, ovalid, rorder, rvalid, match))
        print(f" {k:>7} | {fmt(oorder)}  {'Y' if ovalid else 'n'}       | {fmt(rorder)}  {'Y' if rvalid else 'n'}        | {'==' if match else 'DIFF'}")

n = len(rows)
n_rvalid = sum(r[4] for r in rows)
n_match = sum(r[5] for r in rows)
n_match_and_valid = sum(1 for r in rows if r[5] and r[4])
print(f"\n[summary] demos={n}")
print(f"[summary] replay valid tower (any order): {n_rvalid}/{n}")
print(f"[summary] replay order == original order: {n_match}/{n}")
print(f"[summary] replay reproduces SAME order AND valid: {n_match_and_valid}/{n}")
from collections import Counter
print(f"[summary] replay order distribution: {dict(Counter(fmt(r[3]) for r in rows))}")
print(f"[summary] failed-to-stack demos: {[r[0] for r in rows if not r[4]]}")

env.close()
app.close()
