#!/usr/bin/env python3
"""One-shot empirical test of Camera.set_world_poses quaternion handling.

All four overlay cameras are placed at THE SAME target world pose (third_person_2:
in front of the robot, looking back at the workspace), but each with a different
encoding:

    third_person_0 : quat as (x,y,z,w)
    third_person_1 : quat as (w,x,y,z)
    third_person_2 : quat as (x,y,z,w) CONJUGATE (inverse rotation)
    wrist          : set_world_poses_from_view look-at (no quaternion at all)

One render -> four pngs. The png whose content matches the look-at reference
(up to camera roll) reveals the encoding the installed Isaac Lab actually uses.
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
parser.add_argument("--out_dir", default="/work/out")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True
app = AppLauncher(args).app

import numpy as np
import torch
import gymnasium as gym
import imageio.v2 as imageio

import lab_env
from overlay_cameras import (
    ALL_ROLES, R_from_quat_wxyz, build_camera_cfgs, camera_link_transforms, load_binding, load_overlay,
    quat_wxyz_from_R,
)


def main():
    ov = load_overlay(args.overlay)
    hand_T_tcp, base_adapter, _ = load_binding(args.binding, ov)
    cams = build_camera_cfgs(ov, None, None, 640, 360, standalone=True)
    link_T = camera_link_transforms(ov, hand_T_tcp, base_adapter)
    env = gym.make(lab_env.TASK, cfg=lab_env.build_env_cfg(args.device, args.table_usd, cameras=cams)).unwrapped
    robot = env.scene["robot"]

    with torch.inference_mode():
        env.reset()
        env.scene.update(env.physics_dt)

        d = robot.data
        i = robot.body_names.index("fr3_link0")
        if hasattr(d, "body_link_pos_w"):
            p, q = d.body_link_pos_w[0, i].cpu().numpy(), d.body_link_quat_w[0, i].cpu().numpy()
        else:
            p, q = d.body_pos_w[0, i].cpu().numpy(), d.body_quat_w[0, i].cpu().numpy()
        W_base = np.eye(4)
        W_base[:3, :3] = R_from_quat_wxyz(q)
        W_base[:3, 3] = p
        W = W_base @ link_T["third_person_2"][1]  # target pose for ALL variants
        pos = torch.tensor(W[:3, 3], dtype=torch.float32, device=env.device).unsqueeze(0)
        qw = quat_wxyz_from_R(W[:3, :3])  # (w,x,y,z)
        view_dir = -W[:3, 2]  # usd camera looks along -Z
        target = W[:3, 3] + view_dir * 1.2
        print(f"[diag] target world pose t={np.round(W[:3, 3], 3).tolist()} view_dir={np.round(view_dir, 3).tolist()}")

        def t(a):
            return torch.tensor(np.asarray(a, dtype=np.float32), device=env.device).unsqueeze(0)

        env.scene["third_person_0"].set_world_poses(
            positions=pos, orientations=t([qw[1], qw[2], qw[3], qw[0]]), convention="opengl")
        env.scene["third_person_1"].set_world_poses(
            positions=pos, orientations=t([qw[0], qw[1], qw[2], qw[3]]), convention="opengl")
        env.scene["third_person_2"].set_world_poses(
            positions=pos, orientations=t([-qw[1], -qw[2], -qw[3], qw[0]]), convention="opengl")
        env.scene["wrist"].set_world_poses_from_view(eyes=pos, targets=t(target))

        for _ in range(10):
            env.sim.render()
        env.scene.update(env.physics_dt)
        names = {"third_person_0": "variant_xyzw", "third_person_1": "variant_wxyz",
                 "third_person_2": "variant_xyzw_conj", "wrist": "variant_lookat"}
        for role, name in names.items():
            img = env.scene[role].data.output["rgb"][0]
            if isinstance(img, torch.Tensor):
                img = img.detach().cpu().numpy()
            path = os.path.join(args.out_dir, f"{name}.png")
            imageio.imwrite(path, np.ascontiguousarray(img[..., :3]).astype(np.uint8))
            print(f"[diag] wrote {path}")
    env.close()
    return 0


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
