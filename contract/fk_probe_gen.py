"""FK probe for GENERATED episodes — pixel-free ground truth.

Writes the episode's obs/joint_pos into the sim frame by frame (the proven
convert_demo pattern: write + sim.forward + scene.update reads physics
buffers, no render sync involved) and reports, in the sim's own world frame:
  - TCP (fr3_hand + z 0.1034 in hand frame) vs each initial cube position;
  - FK TCP vs the recorded obs/eef_pos (decodes the obs frame convention).

Run inside the Isaac docker:
  isaaclab.sh -p contract/fk_probe_gen.py --device cpu \
      --dataset /out/gen_rl_v7_10_failed.hdf5 --demos demo_10 --every 10
"""
from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True)
parser.add_argument("--demos", required=True)
parser.add_argument("--every", type=int, default=10)
parser.add_argument("--table_usd", default="/nonexistent.usdc")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = False
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import gymnasium as gym  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
for rel in ("../render", "../lab_stack_mimic"):
    sys.path.insert(0, os.path.normpath(os.path.join(HERE, rel)))
import lab_env  # noqa: E402


def quat_rot(q, v):
    w, x, y, z = q
    u = np.array([x, y, z])
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


def main():
    cfg = lab_env.build_env_cfg(args.device, args.table_usd, cameras=None,
                                num_envs=1)
    env = gym.make(lab_env.TASK, cfg=cfg).unwrapped
    env.reset()
    scene = env.scene
    robot = scene["robot"]
    hand_index = robot.find_bodies("fr3_hand")[0][0]
    origin = scene.env_origins[0].cpu().numpy()
    dev = env.device

    with h5py.File(args.dataset, "r", locking=False) as src:
        for name in args.demos.split(","):
            g = src[f"data/{name}"]
            init = g["initial_state"]
            root = init["articulation/robot/root_pose"][0].copy()
            root[:3] += origin
            robot.write_root_pose_to_sim(
                torch.tensor(root, dtype=torch.float32,
                             device=dev).unsqueeze(0))
            robot.write_root_velocity_to_sim(torch.zeros(1, 6, device=dev))
            cubes = {}
            for c in (1, 2, 3):
                pose = init[f"rigid_object/cube_{c}/root_pose"][0].copy()
                cubes[c] = pose[:3] + origin

            joints = g["obs/joint_pos"][()]
            obs_ee = g["obs/eef_pos"][()]
            grip_act = g["actions"][()][:, 6]
            print(f"[fk] {name}: root={np.round(root, 3).tolist()} "
                  f"cubes={ {c: np.round(v, 3).tolist() for c, v in cubes.items()} }",
                  flush=True)
            for t in range(0, len(joints), args.every):
                robot.write_joint_state_to_sim(
                    torch.tensor(joints[t], dtype=torch.float32,
                                 device=dev).unsqueeze(0),
                    torch.zeros(1, joints.shape[1], device=dev))
                scene.write_data_to_sim()
                env.sim.forward()
                scene.update(dt=0.0)
                hand_p = robot.data.body_pos_w[0, hand_index].cpu().numpy()
                hand_q = robot.data.body_quat_w[0, hand_index].cpu().numpy()
                tcp = hand_p + quat_rot(hand_q, np.array([0.0, 0.0, 0.1034]))
                d = {c: float(np.linalg.norm(tcp - cubes[c])) for c in cubes}
                fk_vs_obs = float(np.linalg.norm(
                    (tcp - origin) - obs_ee[t]))
                fk_vs_obs_raw = float(np.linalg.norm(tcp - obs_ee[t]))
                print(f"[fk] {name} t{t:04d}: tcp={np.round(tcp, 3).tolist()} "
                      f"d1={d[1]*100:.1f} d2={d[2]*100:.1f} d3={d[3]*100:.1f}cm "
                      f"grip={grip_act[t]:+.0f} "
                      f"|fk-obs|={fk_vs_obs*100:.1f}/{fk_vs_obs_raw*100:.1f}cm",
                      flush=True)
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
