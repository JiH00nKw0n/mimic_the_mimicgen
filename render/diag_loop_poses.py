#!/usr/bin/env python3
"""Reproduce the replay-loop camera failure and test the tensor-lifetime fix.

Replays the first 30 states of a demo while driving all cameras to the SAME
target pose (third_person_2's calibrated view) with different call styles:

    third_person_0 : set ONCE before the loop (fresh tensors)
    third_person_1 : set EVERY frame via PERSISTENT tensors (.copy_ into them)
    third_person_2 : set EVERY frame via FRESH tensors (the failing render loop style)
    wrist          : set EVERY frame via set_world_poses_from_view (look-at)

Dumps pngs at k=0 and k=25. If tp2 is garbage while tp1 is fine, the warp
kernel in FrameView.set_world_poses reads freed tensor memory (async launch)
and the fix is persistent buffers.
"""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--overlay", default=os.path.join(os.path.dirname(__file__), "fr3_camera_overlay_v1/overlay.yaml"))
parser.add_argument("--binding", default=os.path.join(os.path.dirname(__file__), "fr3_binding.yaml"))
parser.add_argument("--dataset", default="/data/fwd_annotated.hdf5")
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

from isaaclab.utils.datasets import HDF5DatasetFileHandler

import lab_env
from overlay_cameras import (
    R_from_quat_wxyz, build_camera_cfgs, camera_link_transforms, load_binding, load_overlay, quat_wxyz_from_R,
)


def main():
    ov = load_overlay(args.overlay)
    hand_T_tcp, base_adapter, _ = load_binding(args.binding, ov)
    cams = build_camera_cfgs(ov, None, None, 640, 360, standalone=True)
    link_T = camera_link_transforms(ov, hand_T_tcp, base_adapter)
    env = gym.make(lab_env.TASK, cfg=lab_env.build_env_cfg(args.device, args.table_usd, cameras=cams)).unwrapped
    robot = env.scene["robot"]
    cube_assets = {i: env.scene[f"cube_{i}"] for i in (1, 2, 3)}
    origin = env.scene.env_origins

    handler = HDF5DatasetFileHandler()
    handler.open(args.dataset)
    ep = handler.load_episode(sorted(handler.get_episode_names())[0], env.device)
    S = ep.data["states"]
    jp = S["articulation"]["robot"]["joint_position"]
    jv = S["articulation"]["robot"]["joint_velocity"]
    cp = {i: S["rigid_object"][f"cube_{i}"]["root_pose"] for i in (1, 2, 3)}
    cv = {i: S["rigid_object"][f"cube_{i}"]["root_velocity"] for i in (1, 2, 3)}

    def fk_base_T():
        d = robot.data
        i = robot.body_names.index("fr3_link0")
        if hasattr(d, "body_link_pos_w"):
            p, q = d.body_link_pos_w[0, i].cpu().numpy(), d.body_link_quat_w[0, i].cpu().numpy()
        else:
            p, q = d.body_pos_w[0, i].cpu().numpy(), d.body_quat_w[0, i].cpu().numpy()
        T = np.eye(4)
        T[:3, :3] = R_from_quat_wxyz(q)
        T[:3, 3] = p
        return T

    with torch.inference_mode():
        env.reset()
        env.scene.update(env.physics_dt)
        W = fk_base_T() @ link_T["third_person_2"][1]
        pos_np = W[:3, 3].astype(np.float32)
        qw = quat_wxyz_from_R(W[:3, :3])
        q_xyzw = np.array([qw[1], qw[2], qw[3], qw[0]], dtype=np.float32)
        view_dir = -W[:3, 2]
        tgt_np = (W[:3, 3] + view_dir * 1.2).astype(np.float32)
        print(f"[diag2] target t={np.round(pos_np, 3).tolist()}")

        def fresh(a):
            return torch.tensor(np.asarray(a, dtype=np.float32), device=env.device).unsqueeze(0)

        # persistent buffers for tp1
        p_pos = fresh(pos_np).clone()
        p_quat = fresh(q_xyzw).clone()

        # tp0: set once, never again
        env.scene["third_person_0"].set_world_poses(positions=fresh(pos_np), orientations=fresh(q_xyzw),
                                                    convention="opengl")

        for k in range(30):
            robot.write_joint_state_to_sim(jp[k:k + 1], jv[k:k + 1])
            for i in (1, 2, 3):
                p = cp[i][k:k + 1].clone(); p[:, :3] += origin
                cube_assets[i].write_root_pose_to_sim(p)
                cube_assets[i].write_root_velocity_to_sim(cv[i][k:k + 1])
            env.scene.write_data_to_sim()
            env.scene.update(env.physics_dt)

            p_pos.copy_(fresh(pos_np)); p_quat.copy_(fresh(q_xyzw))
            env.scene["third_person_1"].set_world_poses(positions=p_pos, orientations=p_quat, convention="opengl")
            env.scene["third_person_2"].set_world_poses(positions=fresh(pos_np), orientations=fresh(q_xyzw),
                                                        convention="opengl")
            env.scene["wrist"].set_world_poses_from_view(eyes=fresh(pos_np), targets=fresh(tgt_np))

            n = 8 if k == 0 else 1
            for _ in range(n):
                env.sim.render()

            if k in (0, 25):
                for role, tag in (("third_person_0", "once"), ("third_person_1", "persistent"),
                                  ("third_person_2", "freshloop"), ("wrist", "lookat")):
                    img = env.scene[role].data.output["rgb"][0]
                    if isinstance(img, torch.Tensor):
                        img = img.detach().cpu().numpy()
                    path = os.path.join(args.out_dir, f"loop_{tag}_k{k}.png")
                    imageio.imwrite(path, np.ascontiguousarray(img[..., :3]).astype(np.uint8))
                    print(f"[diag2] wrote {path}")
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
