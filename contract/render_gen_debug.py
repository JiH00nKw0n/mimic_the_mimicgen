"""Render debug frames from GENERATED (mimic) episodes.

Generated episodes carry no states group, so we reconstruct the scene per
frame from what they do carry: robot joints from obs/joint_pos, robot root
and cube poses from initial_state (cubes barely move in the failure modes
under investigation — a static-cube view is exactly what we need to see
where the hand goes relative to them).

Run inside the Isaac docker (GPU needed for RTX camera):
  isaaclab.sh -p contract/render_gen_debug.py --device cuda:0 \
      --dataset /out/gen_rl_v7_10_failed.hdf5 --demos demo_10,demo_12 \
      --out /out/debug_frames --every 15
"""
from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True)
parser.add_argument("--demos", required=True, help="comma list")
parser.add_argument("--out", required=True)
parser.add_argument("--every", type=int, default=15)
parser.add_argument("--table_usd", default="/nonexistent.usdc")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import gymnasium as gym  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sensors import CameraCfg  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
for rel in ("../render", "../lab_stack_mimic"):
    sys.path.insert(0, os.path.normpath(os.path.join(HERE, rel)))
import lab_env  # noqa: E402


def main():
    cam_cfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/DebugCam",
        update_period=0.0, height=480, width=640, data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=16.0,
                                         clipping_range=(0.01, 20.0)))
    cfg = lab_env.build_env_cfg(args.device, args.table_usd,
                                cameras={"debug_cam": cam_cfg}, num_envs=1)
    env = gym.make(lab_env.TASK, cfg=cfg).unwrapped
    env.reset()
    scene = env.scene
    robot = scene["robot"]
    cam = scene["debug_cam"]
    origin = scene.env_origins[0].cpu().numpy()
    eye = torch.tensor([origin + np.array([-0.75, 1.05, 1.55])],
                       dtype=torch.float32, device=env.device)
    target = torch.tensor([origin + np.array([0.30, 0.14, 0.80])],
                          dtype=torch.float32, device=env.device)
    cam.set_world_poses_from_view(eye, target)
    os.makedirs(args.out, exist_ok=True)

    dev = env.device
    with h5py.File(args.dataset, "r", locking=False) as src:
        for name in args.demos.split(","):
            g = src[f"data/{name}"]
            init = g["initial_state"]
            root = init["articulation/robot/root_pose"][0].copy()
            root[:3] += origin
            robot.write_root_pose_to_sim(
                torch.tensor(root, dtype=torch.float32, device=dev).unsqueeze(0))
            robot.write_root_velocity_to_sim(
                torch.zeros(1, 6, device=dev))
            for c in (1, 2, 3):
                pose = init[f"rigid_object/cube_{c}/root_pose"][0].copy()
                pose[:3] += origin
                cube = scene[f"cube_{c}"]
                cube.write_root_pose_to_sim(
                    torch.tensor(pose, dtype=torch.float32,
                                 device=dev).unsqueeze(0))
                cube.write_root_velocity_to_sim(torch.zeros(1, 6, device=dev))

            joints = g["obs/joint_pos"][()]
            grip = np.abs(g["obs/gripper_pos"][()]).mean(1)
            cube_state = {}
            for c in (1, 2, 3):
                pose = init[f"rigid_object/cube_{c}/root_pose"][0].copy()
                pose[:3] += origin
                cube_state[c] = torch.tensor(pose, dtype=torch.float32,
                                             device=dev).unsqueeze(0)
            for t in range(0, len(joints), args.every):
                jq = torch.tensor(joints[t], dtype=torch.float32,
                                  device=dev).unsqueeze(0)
                robot.write_joint_state_to_sim(
                    jq, torch.zeros(1, joints.shape[1], device=dev))
                robot.set_joint_position_target(jq)
                for c in (1, 2, 3):
                    scene[f"cube_{c}"].write_root_pose_to_sim(cube_state[c])
                    scene[f"cube_{c}"].write_root_velocity_to_sim(
                        torch.zeros(1, 6, device=dev))
                scene.write_data_to_sim()
                # a real physics step is required to push link transforms to
                # the render hierarchy on Isaac Lab 3.0 (render-only loops
                # leave the robot frozen at its spawn pose)
                env.sim.step(render=False)
                scene.update(dt=env.physics_dt)
                for _ in range(3 if t == 0 else 2):
                    env.sim.render()
                img = cam.data.output["rgb"][0].cpu().numpy()
                imageio.imwrite(
                    os.path.join(args.out,
                                 f"{name}_t{t:04d}_f{grip[t]*1000:.0f}mm.png"),
                    img.astype(np.uint8))
            print(f"[debug] {name}: {len(range(0, len(joints), args.every))} "
                  f"frames -> {args.out}", flush=True)
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
