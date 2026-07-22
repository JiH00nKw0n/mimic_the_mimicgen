#!/usr/bin/env python3
"""Diagnose camera placement for the world-pose-driven pipeline.

Sets each camera's world pose from physx FK exactly like render_viewpoints.py,
then reads the pose back from Camera.data and reports the round-trip error.
Historical note: cameras were originally parented under robot link prims, but
with PhysX+fabric the articulation spawn rotation and link motion never reach
the USD hierarchy (measured on Isaac Lab 3.0: a child camera keeps the
USD-default pose), hence the explicit world-pose driving.
"""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--overlay", default=os.path.join(os.path.dirname(__file__), "fr3_camera_overlay_v1/overlay.yaml"))
parser.add_argument("--binding", default=os.path.join(os.path.dirname(__file__), "fr3_binding.yaml"))
parser.add_argument("--table_usd", default="/home/ubuntu/jake/aidas/3cube_stack/table_scene.usdc")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True
app = AppLauncher(args).app

import numpy as np
import torch
import gymnasium as gym

import lab_env
from overlay_cameras import (
    ALL_ROLES, R_from_quat_wxyz, build_camera_cfgs, camera_link_transforms, camera_quat_order,
    load_binding, load_overlay, quat_wxyz_from_R, rot_angle_deg,
)


def main():
    ov = load_overlay(args.overlay)
    hand_T_tcp, base_adapter, _ = load_binding(args.binding, ov)
    cams = build_camera_cfgs(ov, None, None, 320, 180, standalone=True)
    link_T = camera_link_transforms(ov, hand_T_tcp, base_adapter)
    env = gym.make(lab_env.TASK, cfg=lab_env.build_env_cfg(args.device, args.table_usd, cameras=cams)).unwrapped
    robot = env.scene["robot"]

    def link_pose_T(i):
        d = robot.data
        if hasattr(d, "body_link_pos_w"):
            p, q = d.body_link_pos_w[0, i].cpu().numpy(), d.body_link_quat_w[0, i].cpu().numpy()
        else:
            p, q = d.body_pos_w[0, i].cpu().numpy(), d.body_quat_w[0, i].cpu().numpy()
        T = np.eye(4)
        T[:3, :3] = R_from_quat_wxyz(q)
        T[:3, 3] = p
        return T

    quat_order = camera_quat_order()
    print(f"[campose] camera quaternion order for this Isaac Lab: {quat_order}")

    with torch.inference_mode():
        env.reset()
        env.scene.update(env.physics_dt)
        for role in ALL_ROLES:
            ln, T_loc = link_T[role]
            W = link_pose_T(robot.body_names.index(ln)) @ T_loc
            cam = env.scene[role]
            q = quat_wxyz_from_R(W[:3, :3])
            if quat_order == "xyzw":
                q = np.array([q[1], q[2], q[3], q[0]])
            pos = torch.tensor(W[:3, 3], dtype=torch.float32, device=env.device).unsqueeze(0)
            quat = torch.tensor(q, dtype=torch.float32, device=env.device).unsqueeze(0)
            cam.set_world_poses(positions=pos, orientations=quat, convention="opengl")
        for _ in range(4):
            env.sim.render()
        env.scene.update(env.physics_dt)

        ok = True
        for role in ALL_ROLES:
            ln, T_loc = link_T[role]
            W = link_pose_T(robot.body_names.index(ln)) @ T_loc
            data = env.scene[role].data
            pw = data.pos_w[0].cpu().numpy()
            dpos = float(np.linalg.norm(pw - W[:3, 3]))
            print(f"=== {role} (on {ln})")
            print(f"  set world t  = {np.round(W[:3, 3], 4).tolist()}")
            print(f"  read pos_w   = {np.round(pw, 4).tolist()}   dpos={dpos * 1000:.2f} mm")
            if hasattr(data, "quat_w_opengl"):
                q = data.quat_w_opengl[0].cpu().numpy()
                if quat_order == "xyzw":
                    q = np.array([q[3], q[0], q[1], q[2]])
                drot = rot_angle_deg(R_from_quat_wxyz(q).T @ W[:3, :3])
                print(f"  quat_w_opengl round-trip drot = {drot:.2f} deg (readback may be unreliable; images decide)")
            ok = ok and dpos < 1e-3
        print(f"[campose] round-trip {'OK' if ok else 'FAIL'}")
    env.close()
    return 0 if ok else 2


if __name__ == "__main__":
    import traceback
    try:
        code = main()
    except BaseException:
        traceback.print_exc()
        sys.stderr.flush()
        code = 1
    finally:
        sys.stdout.flush()
        app.close()
    sys.exit(code)
