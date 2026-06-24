#!/usr/bin/env python3
"""Diagnose why the grasp subtask signal doesn't fire on the FR3.

Replays one demo in the lab Mimic env and logs, per step, the distance from the
ee_frame TCP (what object_grasped checks) to each cube, the finger joint values,
and the env's subtask term signals. Tells us whether grasp_1 fails because the
ee_frame TCP is mis-placed (distance never < threshold) or the gripper check fails.

    python diag_grasp.py --dataset <hdf5> --demo demo_2
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-Stack-Cube-LabFR3-Fwd-IK-Rel-Mimic-v0")
parser.add_argument("--dataset", default="/home/ubuntu/jake/aidas/3cube_stack/datasets/teleop_dataset_success.hdf5")
parser.add_argument("--demo", default="demo_2")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import torch
import gymnasium as gym

from isaaclab.utils.datasets import HDF5DatasetFileHandler
import isaaclab_mimic.envs  # noqa: F401
import lab_register  # noqa: F401
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
env_cfg.terminations = None
env_cfg.recorders = {}
env = gym.make(args.task, cfg=env_cfg).unwrapped

robot = env.scene["robot"]
finger_idx = [i for i, n in enumerate(robot.joint_names) if "finger" in n]
print(f"[diag] finger joints: {[robot.joint_names[i] for i in finger_idx]}  open_val={env.cfg.gripper_open_val}")

handler = HDF5DatasetFileHandler()
handler.open(args.dataset)
ep = handler.load_episode(args.demo, env.device)

env.reset()
env.reset_to(ep.get_initial_state(), torch.tensor([0], device=env.device), is_relative=True)

# All-steps closest approach (no filter) of the gripper HARDWARE to each cube,
# to separate "ee_frame offset is wrong" (hand reaches cube but the frame is offset)
# from "replay doesn't reach the cube" (fidelity). Track three reference points:
#   end_effector frame[0], finger-tip midpoint (frames[1,2]), raw fr3_hand body.
n_frames = env.scene["ee_frame"].data.target_pos_w.shape[1]
hand_idx = robot.body_names.index("fr3_hand") if "fr3_hand" in robot.body_names else 0
min_ee = {1: 1e9, 2: 1e9, 3: 1e9}
min_fmid = {1: 1e9, 2: 1e9, 3: 1e9}
min_hand = {1: 1e9, 2: 1e9, 3: 1e9}
finger_at_closest = {1: None, 2: None, 3: None}
sig_max = {}
steps = 0
with torch.inference_mode():
    while True:
        act = ep.get_next_action()
        if act is None:
            break
        env.step(act.unsqueeze(0) if act.ndim == 1 else act)
        steps += 1
        tpw = env.scene["ee_frame"].data.target_pos_w[0]
        ee = tpw[0]
        fmid = (tpw[1] + tpw[2]) / 2 if n_frames >= 3 else ee
        hand = robot.data.body_pos_w[0, hand_idx]
        fp = robot.data.joint_pos[0, finger_idx]
        for i in (1, 2, 3):
            cpos = env.scene[f"cube_{i}"].data.root_pos_w[0]
            d_hand = float(torch.linalg.vector_norm(cpos - hand))
            min_ee[i] = min(min_ee[i], float(torch.linalg.vector_norm(cpos - ee)))
            min_fmid[i] = min(min_fmid[i], float(torch.linalg.vector_norm(cpos - fmid)))
            if d_hand < min_hand[i]:
                min_hand[i] = d_hand
                finger_at_closest[i] = [round(float(fp[0]), 4), round(float(fp[1]), 4)]
        sig = env.get_subtask_term_signals()
        for k, v in sig.items():
            sig_max[k] = max(sig_max.get(k, 0.0), float(v.reshape(-1)[0]))

print(f"\n[diag] demo={args.demo} steps={steps}  ee_frame target frames={n_frames}  hand_body_idx={hand_idx}")
print(f"[diag] min fr3_hand(body)->cube (all steps): c1={min_hand[1]:.4f} c2={min_hand[2]:.4f} c3={min_hand[3]:.4f}")
print(f"[diag] min end_effector[0]->cube (all steps): c1={min_ee[1]:.4f} c2={min_ee[2]:.4f} c3={min_ee[3]:.4f}")
print(f"[diag] min finger-midpoint->cube (all steps): c1={min_fmid[1]:.4f} c2={min_fmid[2]:.4f} c3={min_fmid[3]:.4f}")
print(f"[diag] finger joints at closest-hand step per cube: c1={finger_at_closest[1]} c2={finger_at_closest[2]} c3={finger_at_closest[3]}  (open=0.04, closed~0)")
print(f"[diag] subtask signals ever-fired: {sig_max}")
print("[diag] -> hand/fingers reach cube (~<0.05) => ee_frame offset bug; stays far (~>0.08) => replay fidelity")
env.close()
app.close()
