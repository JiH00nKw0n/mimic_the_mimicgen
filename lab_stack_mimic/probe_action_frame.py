#!/usr/bin/env python3
"""Closed-loop probe: does commanding the EE toward a WORLD-frame target actually
move it there? This isolates the IK-rel action<->frame mapping from the full
MimicGen transform/stitching, so we can tell whether the base-frame fix works at
the control level.

For each test we hold a constant world-frame target pose (current eef translated by
a known world delta), and every sim step recompute the action via the env's
`target_eef_pose_to_action(target, curr)` and step. After N steps we measure how
far the eef got toward the target. Convergence => the action mapping is correct.

    python probe_action_frame.py --task ...Fwd... --dataset <fixed.hdf5> --demo demo_2
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-Stack-Cube-LabFR3-Fwd-IK-Rel-Mimic-v0")
parser.add_argument("--dataset", default="/home/ubuntu/mimicgen_jihoonkwon/mimic_the_mimicgen/datasets/teleop_dataset_fixed.hdf5")
parser.add_argument("--demo", default="demo_2")
parser.add_argument("--steps", type=int, default=25)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import torch
import gymnasium as gym

import isaaclab.utils.math as PoseUtils
from isaaclab.utils.datasets import HDF5DatasetFileHandler
import isaaclab_mimic.envs  # noqa: F401
import lab_register  # noqa: F401
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from isaaclab_mimic.envs.franka_stack_ik_rel_mimic_env import FrankaCubeStackIKRelMimicEnv

env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
env_cfg.terminations = None
env_cfg.recorders = {}
env = gym.make(args.task, cfg=env_cfg).unwrapped

eef_name = list(env.cfg.subtask_configs.keys())[0]

# --- 0) confirm our override is the one being used
cls = type(env)
overridden = cls.target_eef_pose_to_action is not FrankaCubeStackIKRelMimicEnv.target_eef_pose_to_action
print(f"\n[probe] env class = {cls.__name__}")
print(f"[probe] target_eef_pose_to_action OVERRIDDEN by lab subclass: {overridden}")

handler = HDF5DatasetFileHandler()
handler.open(args.dataset)
ep = handler.load_episode(args.demo, env.device)
env.reset()
env.reset_to(ep.get_initial_state(), torch.tensor([0], device=env.device), is_relative=True)

robot = env.scene["robot"]
root_pos = robot.data.root_pos_w[0]
root_quat = robot.data.root_quat_w[0]
print(f"[probe] robot root_pos_w={[round(float(x),3) for x in root_pos]}  root_quat_w(wxyz)={[round(float(x),3) for x in root_quat]}")
# decode yaw from quat
import math
w, x, y, z = [float(v) for v in root_quat]
yaw = math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
print(f"[probe] robot base yaw ~= {yaw:.1f} deg")


def cur_eef():
    p = env.get_robot_eef_pose(eef_name, env_ids=[0])[0]
    pos, rot = PoseUtils.unmake_pose(p)
    return pos.clone(), rot.clone()


def reset_demo():
    env.reset_to(ep.get_initial_state(), torch.tensor([0], device=env.device), is_relative=True)


def run_to_target(target_pos, target_rot, gripper=1.0, steps=25):
    """Hold constant world target; recompute action each step; return start/end err."""
    start_pos, _ = cur_eef()
    target_pose = PoseUtils.make_pose(target_pos, target_rot)
    g = torch.tensor([float(gripper)], device=env.device)
    for _ in range(steps):
        act = env.target_eef_pose_to_action(
            {eef_name: target_pose}, {eef_name: g}, action_noise_dict=None, env_id=0
        )
        env.step(act.unsqueeze(0))
    end_pos, _ = cur_eef()
    return start_pos, end_pos


def ori_err_deg(end_rot, target_rot):
    errR = end_rot.matmul(target_rot.transpose(-1, -2))
    q = PoseUtils.quat_from_matrix(errR.unsqueeze(0))
    aa = PoseUtils.axis_angle_from_quat(q)[0]
    return math.degrees(float(torch.linalg.vector_norm(aa)))


with torch.inference_mode():
    # cfg-default reset pose (what GENERATION actually starts from — no reset_to)
    env.reset()
    print(f"\n[probe] cfg-DEFAULT reset robot joints (generation start) = "
          f"{[round(float(v),3) for v in robot.data.joint_pos[0]]}")
    print(f"[probe] my FR3 init expects ~ [0.0,-0.569,0.0,-2.810,0.0,3.037,0.741, 0.04,0.04]")

    print("\n[probe] === closed-loop reach tests (constant world target) ===")
    tests = {
        "+x_world(+5cm)": torch.tensor([0.05, 0.0, 0.0], device=env.device),
        "+y_world(+5cm)": torch.tensor([0.0, 0.05, 0.0], device=env.device),
        "+z_world(+5cm)": torch.tensor([0.0, 0.0, 0.05], device=env.device),
    }
    for name, dvec in tests.items():
        reset_demo()
        s, _r = cur_eef()
        tgt = s + dvec
        start, end = run_to_target(tgt, _r, gripper=1.0, steps=args.steps)
        moved = end - start
        dn = dvec / torch.linalg.vector_norm(dvec)
        along = float(torch.dot(moved, dn))
        err_end = float(torch.linalg.vector_norm(end - tgt))
        print(f"  {name}: moved={[round(float(v),3) for v in moved]}  "
              f"along_desired={along:+.3f}m (want +0.050)  final_err={err_end:.3f}m  "
              f"{'OK' if along > 0.02 else 'WRONG-WAY' if along < -0.005 else 'STUCK'}")

    # Reach toward cube_2 (the first grasp target) — the real generation goal.
    reset_demo()
    s, r = cur_eef()
    c2 = (env.scene["cube_2"].data.root_pos_w[0] - env.scene.env_origins[0]).clone()
    tgt = c2.clone()
    tgt[2] = tgt[2] + 0.10  # hover 10cm above cube_2
    start, end = run_to_target(tgt, r, gripper=1.0, steps=args.steps * 2)
    print(f"\n[probe] reach above cube_2: start={[round(float(v),3) for v in start]} "
          f"end={[round(float(v),3) for v in end]} target={[round(float(v),3) for v in tgt]} "
          f"final_err={float(torch.linalg.vector_norm(end - tgt)):.3f}m")

    # === ROTATION tracking tests (this is what the probe never exercised before) ===
    print("\n[probe] === closed-loop ROTATION tests (hold position, rotate orientation) ===")
    rot_tests = {
        "z+45deg(world)": (math.radians(45), [0.0, 0.0, 1.0]),
        "x+45deg(world)": (math.radians(45), [1.0, 0.0, 0.0]),
        "y+45deg(world)": (math.radians(45), [0.0, 1.0, 0.0]),
    }
    for name, (ang, ax) in rot_tests.items():
        reset_demo()
        s, r0 = cur_eef()
        angt = torch.tensor([ang], device=env.device)
        axt = torch.tensor([ax], device=env.device)
        dq = PoseUtils.quat_from_angle_axis(angt, axt)[0]
        dR = PoseUtils.matrix_from_quat(dq.unsqueeze(0))[0]
        target_rot = dR.matmul(r0)  # rotate current orientation by dR about a WORLD axis
        start_err = ori_err_deg(r0, target_rot)
        _s, _e = run_to_target(s, target_rot, gripper=1.0, steps=args.steps)
        _ep, end_rot = cur_eef()
        end_err = ori_err_deg(end_rot, target_rot)
        pos_drift = float(torch.linalg.vector_norm(_ep - s))
        print(f"  {name}: start_ori_err={start_err:.1f}deg -> final_ori_err={end_err:.1f}deg "
              f"pos_drift={pos_drift:.3f}m  {'OK' if end_err < 5 else 'DIVERGED'}")

env.close()
app.close()
