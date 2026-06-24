#!/usr/bin/env python3
"""Render lab 3-cube demos to mp4, grouped, in one Isaac Sim session.

Two modes:
  --mode states  : play back the RECORDED states directly (the true demo; no physics/actions)
  --mode replay  : reset to initial_state, then apply the recorded ACTIONS open-loop
                   (use the FIXED dataset so initial_state == real scattered start)

Groups: --groups "label1=demo_1,demo_2;label2=demo_4,demo_5" renders each label to
<out_prefix>_<label>.mp4. Each frame is labelled (demo, end order, 3-tower yes/no).

    python record_video.py --mode states --dataset src.hdf5 --out_prefix /tmp/demo \
        --groups "fwd_success=demo_1,demo_2;rev_success=demo_4"
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True)
parser.add_argument("--mode", choices=["states", "replay"], required=True)
parser.add_argument("--groups", default="", help='"label=demo_a,demo_b;label2=demo_c"')
parser.add_argument("--out_prefix", default="/tmp/labvid")
parser.add_argument("--table_usd", default="/home/ubuntu/jake/aidas/3cube_stack/table_scene.usdc")
parser.add_argument("--every", type=int, default=2)
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--width", type=int, default=720)
parser.add_argument("--height", type=int, default=480)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True
app = AppLauncher(args).app

import numpy as np
import torch
import gymnasium as gym
import imageio.v2 as imageio
from PIL import Image, ImageDraw

import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as mdp
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.datasets import HDF5DatasetFileHandler
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

from success_criteria import tower_status

DEV = args.device
DESK_Z, CUBE = 0.720, 0.05
ROBOT_POS, ROBOT_ROT = (0.72, 0.138, 0.722), (0.0, 0.0, 0.0, 1.0)
BASE_XY = (0.32, 0.138)

FR3_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/FrankaRobotics/FrankaFR3/fr3.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, max_depenetration_velocity=5.0),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0)),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=ROBOT_POS, rot=ROBOT_ROT,
        joint_pos={"fr3_joint1": 0.0, "fr3_joint2": -0.569, "fr3_joint3": 0.0, "fr3_joint4": -2.810,
                   "fr3_joint5": 0.0, "fr3_joint6": 3.037, "fr3_joint7": 0.741, "fr3_finger_joint.*": 0.04}),
    actuators={
        "a1": ImplicitActuatorCfg(joint_names_expr=["fr3_joint[1-4]"], stiffness=400.0, damping=80.0),
        "a2": ImplicitActuatorCfg(joint_names_expr=["fr3_joint[5-7]"], stiffness=400.0, damping=80.0),
        "h": ImplicitActuatorCfg(joint_names_expr=["fr3_finger_joint.*"], effort_limit_sim=200.0, stiffness=2e3, damping=1e2)},
    soft_joint_pos_limit_factor=1.0,
)


def _cube_cfg(name, color, xy):
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/" + name,
        init_state=RigidObjectCfg.InitialStateCfg(pos=(xy[0], xy[1], DESK_Z + CUBE + 0.01), rot=(1, 0, 0, 0)),
        spawn=sim_utils.CuboidCfg(size=(0.05, 0.05, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, max_depenetration_velocity=5.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05), collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color)))


def build_env_cfg():
    cfg = parse_env_cfg("Isaac-Stack-Cube-Franka-IK-Rel-v0", device=DEV, num_envs=1)
    cfg.scene.table = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0, 0), rot=(1, 0, 0, 0)),
        spawn=sim_utils.UsdFileCfg(usd_path=args.table_usd))
    cfg.scene.work_surface = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/WorkSurface",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.20, BASE_XY[1], DESK_Z - 0.01)),
        spawn=sim_utils.CuboidCfg(size=(0.55, 0.6, 0.02), collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.55, 0.58), opacity=0.0)))
    cfg.scene.robot = FR3_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    cfg.scene.cube_1 = _cube_cfg("Cube_1", (1.0, 0.0, 0.0), (BASE_XY[0], BASE_XY[1] - 0.10))
    cfg.scene.cube_2 = _cube_cfg("Cube_2", (0.0, 0.0, 1.0), (BASE_XY[0], BASE_XY[1]))
    cfg.scene.cube_3 = _cube_cfg("Cube_3", (1.0, 1.0, 0.0), (BASE_XY[0], BASE_XY[1] + 0.10))
    cfg.actions.arm_action = DifferentialInverseKinematicsActionCfg(
        asset_name="robot", joint_names=["fr3_joint.*"], body_name="fr3_hand", scale=1.0,
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=(0.0, 0.0, 0.1034)))
    cfg.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot", joint_names=["fr3_finger_joint.*"],
        open_command_expr={"fr3_finger_joint.*": 0.04}, close_command_expr={"fr3_finger_joint.*": 0.0})
    if hasattr(cfg, "gripper_joint_names"):
        cfg.gripper_joint_names = ["fr3_finger_joint.*"]
    for ev in ("init_franka_arm_pose", "randomize_franka_joint_state", "randomize_cube_positions"):
        if hasattr(cfg.events, ev):
            setattr(cfg.events, ev, None)
    if hasattr(cfg.scene, "ee_frame"):
        cfg.scene.ee_frame.prim_path = "{ENV_REGEX_NS}/Robot/fr3_link0"
        for fr in cfg.scene.ee_frame.target_frames:
            fr.prim_path = (fr.prim_path.replace("panda_hand", "fr3_hand")
                            .replace("panda_rightfinger", "fr3_rightfinger").replace("panda_leftfinger", "fr3_leftfinger"))
    cfg.terminations = {}
    cfg.recorders = {}
    cfg.viewer.eye = (1.45, -0.55, 1.25)
    cfg.viewer.lookat = (0.32, 0.14, 0.76)
    cfg.viewer.origin_type = "world"
    return cfg


def label(frame, text):
    img = Image.fromarray(frame).resize((args.width, args.height))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, args.width, 26], fill=(0, 0, 0))
    d.text((8, 6), text, fill=(255, 255, 0))
    return np.asarray(img)


def parse_groups(spec, all_names):
    if not spec:
        return [("all", list(all_names))]
    out = []
    for part in spec.split(";"):
        part = part.strip()
        if not part:
            continue
        label, demos = part.split("=")
        out.append((label.strip(), [d.strip() for d in demos.split(",") if d.strip()]))
    return out


def main():
    handler = HDF5DatasetFileHandler()
    handler.open(args.dataset)
    names = list(handler.get_episode_names())
    groups = parse_groups(args.groups, names)
    print(f"[video] mode={args.mode} dataset={args.dataset}  groups={[(l, len(d)) for l, d in groups]}")

    env = gym.make("Isaac-Stack-Cube-Franka-IK-Rel-v0", cfg=build_env_cfg(), render_mode="rgb_array").unwrapped
    finger_idx = [i for i, n in enumerate(env.scene["robot"].joint_names) if "finger" in n]
    robot = env.scene["robot"]
    cube_assets = {i: env.scene[f"cube_{i}"] for i in (1, 2, 3)}
    origin = env.scene.env_origins
    env.reset()

    with torch.inference_mode():
        for glabel, demo_names in groups:
            out = f"{args.out_prefix}_{glabel}.mp4"
            writer = imageio.get_writer(out, fps=args.fps, codec="libx264", quality=7, macro_block_size=8)
            for name in demo_names:
                if name not in names:
                    print(f"  [skip] {name} not in dataset")
                    continue
                ep = handler.load_episode(name, env.device)
                frames = []
                if args.mode == "states":
                    S = ep.data["states"]
                    jp = S["articulation"]["robot"]["joint_position"]
                    jv = S["articulation"]["robot"]["joint_velocity"]
                    cp = {i: S["rigid_object"][f"cube_{i}"]["root_pose"] for i in (1, 2, 3)}
                    cv = {i: S["rigid_object"][f"cube_{i}"]["root_velocity"] for i in (1, 2, 3)}
                    T = jp.shape[0]
                    env.reset()
                    for t in range(T):
                        robot.write_joint_state_to_sim(jp[t:t + 1], jv[t:t + 1])
                        for i in (1, 2, 3):
                            p = cp[i][t:t + 1].clone(); p[:, :3] += origin
                            cube_assets[i].write_root_pose_to_sim(p)
                            cube_assets[i].write_root_velocity_to_sim(cv[i][t:t + 1])
                        env.scene.write_data_to_sim()
                        env.sim.render()
                        if t % args.every == 0 and (f := env.render()) is not None:
                            frames.append(np.asarray(f))
                    cubes = [cp[i][-1].tolist() for i in (1, 2, 3)]
                    fingers = jp[-1, finger_idx].tolist()
                else:  # replay
                    env.reset_to(ep.get_initial_state(), torch.tensor([0], device=env.device), is_relative=True)
                    T = 0
                    while True:
                        act = ep.get_next_action()
                        if act is None:
                            break
                        env.step(act.unsqueeze(0) if act.ndim == 1 else act)
                        T += 1
                        if T % args.every == 0 and (f := env.render()) is not None:
                            frames.append(np.asarray(f))
                    cubes = [cube_assets[i].data.root_pos_w[0].tolist() for i in (1, 2, 3)]
                    fingers = robot.data.joint_pos[0, finger_idx].tolist()

                st = tower_status(cubes, fingers, canonical=False)
                order = "->".join(f"c{o + 1}" for o in st["order"])
                tag = f"{name}  {args.mode.upper()}  end-order={order}  3-tower={'YES' if st['ok'] else 'no'}  steps={T}"
                print(f"  [{glabel}] {tag}  ({len(frames)} frames)")
                for f in frames:
                    writer.append_data(label(f, tag))
            writer.close()
            print(f"[video] wrote {out}")
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
