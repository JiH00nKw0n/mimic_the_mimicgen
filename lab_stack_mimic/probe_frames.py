#!/usr/bin/env python3
"""Pin down the eef-pose frame. For both the cfg-default reset and a reset_to(demo),
print get_robot_eef_pose vs the physical fr3_hand body pose vs the ee_frame target,
and compare to the demo's STORED obs eef_pos. Tells us if our ee_frame/get_robot_eef_pose
matches what the demos were recorded with."""

from __future__ import annotations
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-Stack-Cube-LabFR3-Fwd-IK-Rel-Mimic-v0")
parser.add_argument("--dataset", default="/home/ubuntu/mimicgen_jihoonkwon/mimic_the_mimicgen/datasets/teleop_dataset_fixed.hdf5")
parser.add_argument("--demo", default="demo_0")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import torch, numpy as np, h5py
import gymnasium as gym
import isaaclab.utils.math as PoseUtils
from isaaclab.utils.datasets import HDF5DatasetFileHandler
import isaaclab_mimic.envs  # noqa
import lab_register  # noqa
import isaaclab_tasks  # noqa
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
env_cfg.terminations = None
env_cfg.recorders = {}
env = gym.make(args.task, cfg=env_cfg).unwrapped
eef_name = list(env.cfg.subtask_configs.keys())[0]
robot = env.scene["robot"]
hand_idx = robot.body_names.index("fr3_hand")
origin = env.scene.env_origins[0]


def dump(tag):
    p = env.get_robot_eef_pose(eef_name, env_ids=[0])[0]
    gep = (p[:3, 3]).tolist()
    hand = (robot.data.body_pos_w[0, hand_idx] - origin).tolist()
    tpw = (env.scene["ee_frame"].data.target_pos_w[0, 0] - origin).tolist()
    j = robot.data.joint_pos[0].tolist()
    print(f"[{tag}] joints={[round(v,3) for v in j]}")
    print(f"[{tag}] get_robot_eef_pose pos = {[round(v,3) for v in gep]}")
    print(f"[{tag}] fr3_hand body  (w-org) = {[round(v,3) for v in hand]}")
    print(f"[{tag}] ee_frame tgt[0](w-org) = {[round(v,3) for v in tpw]}")
    for i in (1, 2, 3):
        c = (env.scene[f"cube_{i}"].data.root_pos_w[0] - origin).tolist()
        print(f"[{tag}] cube_{i} = {[round(v,3) for v in c]}")


with torch.inference_mode():
    env.reset()
    dump("cfg-reset")

    handler = HDF5DatasetFileHandler()
    handler.open(args.dataset)
    ep = handler.load_episode(args.demo, env.device)
    env.reset_to(ep.get_initial_state(), torch.tensor([0], device=env.device), is_relative=True)
    print()
    dump("reset_to-demo")

# the demo's STORED obs (what jake recorded)
with h5py.File(args.dataset) as h:
    d = h["data"][args.demo]
    print(f"\n[stored-obs] eef_pos[0]  = {np.round(d['obs']['eef_pos'][0],3).tolist()}")
    print(f"[stored-obs] eef_quat[0] = {np.round(d['obs']['eef_quat'][0],3).tolist()}")
    jp0 = d['states']['articulation']['robot']['joint_position'][0]
    print(f"[stored-obs] states joint_position[0] = {np.round(jp0,3).tolist()}")

env.close()
app.close()
