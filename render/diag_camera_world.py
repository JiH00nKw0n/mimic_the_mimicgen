"""Diagnose whether camera prims follow the WRITTEN robot root pose.

Symptom this exists for: the calibrated fixed cameras render the floor instead
of the workspace. Projection math says they should see the cubes IF the robot
base carries its 180 deg yaw, and matches the observed frames if the base is
unrotated — i.e. the cameras appear to sit at the SPAWN pose while the
articulation sits at the WRITTEN pose.

This prints, for one demo's first frame: the robot root pose as the sim
reports it, each camera's world pose as the sensor reports it, and where each
camera's optical axis crosses the desk plane. Compare the crossing points with
the cube positions printed alongside.

Run in the Isaac docker (needs cameras -> GPU):
  ./run_render_aidas.sh is for the renderer; for this use the same docker line
  with -p /repo/render/diag_camera_world.py
"""
from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True)
parser.add_argument("--demo", default="demo_0")
parser.add_argument("--table_usd", default="/nonexistent.usdc")
parser.add_argument("--overlay", default=os.path.join(os.path.dirname(__file__),
                                                      "fr3_camera_overlay_v1/overlay.yaml"))
parser.add_argument("--binding", default=os.path.join(os.path.dirname(__file__),
                                                      "fr3_binding.yaml"))
parser.add_argument("--spawn_rot", default="", help="override robot spawn quat, e.g. 0,0,1,0")
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

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lab_env  # noqa: E402
from overlay_cameras import (  # noqa: E402
    ALL_ROLES, build_camera_cfgs, camera_metadata, load_binding, load_overlay,
)


def rot_from_quat(q, order):
    w, x, y, z = (q[0], q[1], q[2], q[3]) if order == "wxyz" else (q[3], q[0], q[1], q[2])
    n = float(np.sqrt(w * w + x * x + y * y + z * z))
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def main():
    overlay = load_overlay(args.overlay)
    hand_T_tcp, base_adapter, _ = load_binding(args.binding, overlay)
    cam_cfgs = build_camera_cfgs(overlay, hand_T_tcp, base_adapter, 320, 180)
    cfg = lab_env.build_env_cfg(args.device, args.table_usd, cameras=cam_cfgs, num_envs=1)
    if args.spawn_rot:
        q = tuple(float(v) for v in args.spawn_rot.split(","))
        cfg.scene.robot.init_state.rot = q
        print(f"[diag] spawn rot overridden -> {q}", flush=True)
    print(f"[diag] cfg robot init rot (as authored) = {cfg.scene.robot.init_state.rot}",
          flush=True)
    env = gym.make(lab_env.TASK, cfg=cfg).unwrapped
    env.reset()
    scene = env.scene
    robot = scene["robot"]
    origin = scene.env_origins[0].cpu().numpy()

    with h5py.File(args.dataset, "r", locking=False) as src:
        g = src[f"data/{args.demo}"]
        init = g["initial_state"]
        root = init["articulation/robot/root_pose"][0].copy()
        print(f"[diag] recorded root_pose = {np.round(root, 4).tolist()}", flush=True)
        r = root.copy()
        r[:3] += origin
        robot.write_root_pose_to_sim(
            torch.tensor(r, dtype=torch.float32, device=env.device).unsqueeze(0))
        robot.write_root_velocity_to_sim(torch.zeros(1, 6, device=env.device))
        jq = init["articulation/robot/joint_position"][0]
        robot.write_joint_state_to_sim(
            torch.tensor(jq, dtype=torch.float32, device=env.device).unsqueeze(0),
            torch.zeros(1, len(jq), device=env.device))
        cubes = {}
        for c in (1, 2, 3):
            p = init[f"rigid_object/cube_{c}/root_pose"][0].copy()
            p[:3] += origin
            scene[f"cube_{c}"].write_root_pose_to_sim(
                torch.tensor(p, dtype=torch.float32, device=env.device).unsqueeze(0))
            cubes[c] = p[:3]
        scene.write_data_to_sim()
        env.sim.step(render=False)
        scene.update(env.physics_dt)

    root_pos = robot.data.root_pos_w[0].cpu().numpy()
    root_quat = robot.data.root_quat_w[0].cpu().numpy()
    print(f"[diag] sim root_pos_w={np.round(root_pos, 4).tolist()} "
          f"root_quat_w={np.round(root_quat, 4).tolist()}", flush=True)
    for order in ("wxyz", "xyzw"):
        fwd = rot_from_quat(root_quat, order) @ np.array([1.0, 0.0, 0.0])
        print(f"[diag]   base +x under {order} = {np.round(fwd, 3).tolist()}", flush=True)
    print(f"[diag] cubes(world) = { {k: np.round(v, 3).tolist() for k, v in cubes.items()} }",
          flush=True)

    desk_z = float(np.mean([v[2] for v in cubes.values()])) - 0.025
    for role in ALL_ROLES:
        cam = scene[role]
        pos = cam.data.pos_w[0].cpu().numpy()
        quat = cam.data.quat_w_world[0].cpu().numpy()
        for order in ("wxyz", "xyzw"):
            R = rot_from_quat(quat, order)
            fwd = -R[:, 2]
            if abs(fwd[2]) < 1e-6:
                hit = np.array([np.nan] * 3)
            else:
                hit = pos + ((desk_z - pos[2]) / fwd[2]) * fwd
            d = min(float(np.linalg.norm(hit[:2] - c[:2])) for c in cubes.values())
            print(f"[diag] {role:16s} quat_as={order} pos={np.round(pos, 3).tolist()} "
                  f"aim_desk={np.round(hit[:2], 3).tolist()} dist_to_nearest_cube={d:.3f} m",
                  flush=True)
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        raise
