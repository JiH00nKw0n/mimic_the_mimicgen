"""Closed-loop contract execution: replay converted targets through the
legacy Cube RelCartesianOSC controller (UWLab) in the lab FR3 3-cube scene.

This is the warm-start completion step from the handoff: at each 10 Hz policy
step the action is computed ONLINE as
    target_pose_to_action(actual_ee_pose_now, demo_target_pose_k)
so the reference is always the current actual pose (contract semantics), then
executed by RelCartesianOSCAction at 120 Hz (decimation 12) with the
cube_legacy_profile gains on zero-stiffness effort-mode arm joints.

Needs UWLab importable (arpa: source /home/ubuntu/jake/env_uwlab/bin/activate).

  python contract/warmstart_replay.py --device cpu \
      --contract <contract.hdf5 from convert_demo.py> --demo demo_0 \
      --source <source .hdf5 with states> --output <executed.hdf5>
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from types import SimpleNamespace

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--contract", required=True, help="offline-converted hdf5")
parser.add_argument("--source", required=True, help="source episode hdf5 (states)")
parser.add_argument("--demo", default="demo_0")
parser.add_argument("--output", required=True)
parser.add_argument("--table_usd", default=os.environ.get(
    "LAB_TABLE_USD", "/home/ubuntu/jake/aidas/3cube_stack/table_scene.usdc"))
parser.add_argument("--settle_steps", type=int, default=5)
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
from isaaclab.utils.math import subtract_frame_transforms  # noqa: E402
from uwlab_tasks.manager_based.manipulation.omnireset.mdp.actions.actions_cfg import (  # noqa: E402
    RelCartesianOSCActionCfg,
)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
for rel in ("../render", "../lab_stack_mimic"):
    sys.path.insert(0, os.path.normpath(os.path.join(HERE, rel)))

import adapter  # noqa: E402
import cube_legacy_profile  # noqa: E402
import schema_io  # noqa: E402
from lab_env import build_env_cfg  # noqa: E402
from success_criteria import tower_status  # noqa: E402

GRIPPER_OPEN_THRESHOLD = 0.02


def make_contract_env_cfg():
    cfg = build_env_cfg(args.device, args.table_usd, cameras=None, num_envs=1)
    # contract controller: torque-mode OSC on zero-stiffness arm joints
    cfg.scene.robot.actuators["a1"].stiffness = 0.0
    cfg.scene.robot.actuators["a1"].damping = 0.0
    cfg.scene.robot.actuators["a2"].stiffness = 0.0
    cfg.scene.robot.actuators["a2"].damping = 0.0
    arm = RelCartesianOSCActionCfg(
        asset_name="robot",
        joint_names=["fr3_joint.*"],
        body_name="fr3_hand",
        scale_xyz_axisangle=(0.02, 0.02, 0.02, 0.02, 0.02, 0.2),
    )
    # apply the frozen legacy profile through the handoff's own function
    proxy = SimpleNamespace(actions=SimpleNamespace(arm=arm),
                            sim=cfg.sim, decimation=cfg.decimation)
    cube_legacy_profile.apply_to_env_cfg(proxy)
    cfg.decimation = proxy.decimation
    cfg.actions.arm_action = arm
    # gripper: keep binary joint position (sign-compatible with the contract's
    # AbsBinary term for the +-1 actions the adapter emits)
    cfg.episode_length_s = 120.0
    # disable terminations so the replay always runs to the end of the track
    if getattr(cfg, "terminations", None) is not None:
        for field in list(vars(cfg.terminations)):
            setattr(cfg.terminations, field, None)
    return cfg


def main():
    with h5py.File(args.contract, "r") as handle:
        group = handle[f"data/{args.demo}"]
        targets = group["commanded_target_pose"][()]
        grip_actions = group["actions"][()][:, 6]
    with h5py.File(args.source, "r") as handle:
        states = handle[f"data/{args.demo}/states"]
        joints0 = states["articulation/robot/joint_position"][0]
        joint_vel0 = states["articulation/robot/joint_velocity"][0]
        cubes0 = {i: states[f"rigid_object/cube_{i}/root_pose"][0]
                  for i in (1, 2, 3)}

    cfg = make_contract_env_cfg()
    env = gym.make("Isaac-Stack-Cube-Franka-IK-Rel-v0", cfg=cfg).unwrapped
    env.reset()
    scene = env.scene
    robot = scene["robot"]
    hand_index = robot.find_bodies("fr3_hand")[0][0]
    device = env.device

    # ---- initial state from the source demo
    robot.write_joint_state_to_sim(
        torch.tensor(joints0, dtype=torch.float32, device=device).unsqueeze(0),
        torch.tensor(joint_vel0, dtype=torch.float32, device=device).unsqueeze(0))
    for i in (1, 2, 3):
        cube = scene[f"cube_{i}"]
        pose = torch.tensor(cubes0[i], dtype=torch.float32,
                            device=device).unsqueeze(0)
        cube.write_root_pose_to_sim(pose)
        cube.write_root_velocity_to_sim(torch.zeros(1, 6, device=device))
    scene.write_data_to_sim()
    env.sim.forward()
    scene.update(dt=0.0)

    def actual_ee():
        pos_b, quat_b = subtract_frame_transforms(
            robot.data.root_pos_w[0], robot.data.root_quat_w[0],
            robot.data.body_pos_w[0, hand_index],
            robot.data.body_quat_w[0, hand_index])
        return ([float(v) for v in pos_b], [float(v) for v in quat_b])

    # settle: hold current pose a few policy steps
    grip0 = 1.0 if float(np.mean(joints0[7:9])) > GRIPPER_OPEN_THRESHOLD else -1.0
    hold = torch.tensor([[0, 0, 0, 0, 0, 0, grip0]], dtype=torch.float32,
                        device=device)
    for _ in range(args.settle_steps):
        env.step(hold)

    # ---- closed-loop contract execution
    executed = {"actions": [], "deltas": [], "targets": [], "actuals": [],
                "joints": [], "joint_vels": [], "grips": [], "cubes": []}
    track_err = []
    for k in range(len(targets)):
        pos_now, quat_now = actual_ee()
        target = targets[k]
        action = adapter.target_pose_to_action(
            pos_now, quat_now, target[:3].tolist(), target[3:7].tolist(),
            float(grip_actions[k]))
        env.step(torch.tensor([list(action)], dtype=torch.float32,
                              device=device))
        pos_after, quat_after = actual_ee()
        track_err.append(math.dist(pos_after, target[:3].tolist()))
        executed["actions"].append(list(action))
        executed["deltas"].append(
            [action[i] * adapter.ACTION_SCALE[i] for i in range(6)])
        executed["targets"].append(target.tolist())
        executed["actuals"].append(list(pos_after) + list(quat_after))
        joint_pos = robot.data.joint_pos[0].tolist()
        executed["joints"].append(joint_pos)
        executed["joint_vels"].append(robot.data.joint_vel[0].tolist())
        executed["grips"].append(joint_pos[7:9])
        row = []
        for i in (1, 2, 3):
            cube = scene[f"cube_{i}"]
            cp, cq = subtract_frame_transforms(
                robot.data.root_pos_w[0], robot.data.root_quat_w[0],
                cube.data.root_pos_w[0], cube.data.root_quat_w[0])
            row.append([float(v) for v in cp] + [float(v) for v in cq])
        executed["cubes"].append(row)

    # ---- success judgement on final cube world poses (any-order tower)
    cube_pos_w = [scene[f"cube_{i}"].data.root_pos_w[0].cpu().numpy().tolist()
                  for i in (1, 2, 3)]
    finger_pos = robot.data.joint_pos[0, 7:9].cpu().numpy().tolist()
    status = tower_status(cube_pos_w, finger_pos)
    success = bool(status["ok"])

    with h5py.File(args.output, "w") as out:
        schema_io.write_episode(
            out, args.demo,
            actions=executed["actions"],
            processed_delta=executed["deltas"],
            commanded_target_pose=executed["targets"],
            actual_ee_pose=executed["actuals"],
            joint_position=executed["joints"],
            joint_velocity=executed["joint_vels"],
            gripper_state=executed["grips"],
            cube_pose=executed["cubes"],
            success=success,
            source_human_demo_id=f"{os.path.basename(args.source)}::{args.demo}",
            retarget_version="contract_replay_v1",
        )
        schema_io.finalize_file(out, env_args={
            "env_name": "LabFR3Cube+RelCartesianOSC(legacy profile)",
            "type": 99,
            "env_kwargs": {"contract_id": schema_io.CONTRACT_ID,
                           "policy_hz": 10, "physics_hz": 120}})

    report = {
        "demo": args.demo,
        "steps": len(targets),
        "success": success,
        "tower_status": str(status),
        "tracking_error_m": {
            "mean": float(np.mean(track_err)),
            "p95": float(np.percentile(track_err, 95)),
            "max": float(np.max(track_err)),
        },
        "schema_violations": schema_io.validate_file(args.output),
    }
    with open(args.output + ".report.json", "w") as handle:
        json.dump(report, handle, indent=2)
    print("[warmstart]", json.dumps(report), flush=True)
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
