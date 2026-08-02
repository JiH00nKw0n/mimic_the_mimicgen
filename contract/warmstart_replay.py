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
    "LAB_TABLE_USD", "/nonexistent.usdc"))  # desk-slab fallback (lab_env)
parser.add_argument("--settle_steps", type=int, default=5)
parser.add_argument("--bundle", default=None,
                    help="fr3_cube_system_calibration_bundle_v1 dir: apply the "
                         "dynamics module's joint armature/friction/viscous "
                         "nominal values to the arm actuators")
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

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
for rel in ("../render", "../lab_stack_mimic"):
    sys.path.insert(0, os.path.normpath(os.path.join(HERE, rel)))

try:  # live UWLab checkout if present; otherwise the handoff's frozen copy
    from uwlab_tasks.manager_based.manipulation.omnireset.mdp.actions.actions_cfg import (
        RelCartesianOSCActionCfg,
    )
    OSC_SOURCE = "uwlab_tasks"
except ImportError:
    from uwlab_frozen import RelCartesianOSCActionCfg
    OSC_SOURCE = "uwlab_frozen(handoff historical_09f7e5b)"

# Isaac Lab 3.0-beta2 runtime shim (frozen file kept verbatim): during the
# FIRST env.reset() the articulation buffers are still fabric/warp proxies,
# so the frozen reset's EE read crashes. Skipping that latch is semantically
# safe here: process_actions recomputes the desired pose from the CURRENT
# actual EE every policy step, and our settle loop starts with zero actions
# (target = current pose). Null-space recapture behaviour is preserved.
_action_class = RelCartesianOSCActionCfg.class_type


def _patched_reset(self, env_ids=None):
    if env_ids is None:
        env_ids = slice(None)
    self._raw_actions[env_ids] = 0.0
    if getattr(self, "_has_nullspace", False) and getattr(
            self, "_regulate_to_reset", False):
        self._need_capture[env_ids] = True


_action_class.reset = _patched_reset

# Second shim: data.root_pos_w/root_quat_w come back as fabric/warp proxies in
# this beta image. The arm base is FIXED at the spawn pose, so substituting the
# spawn constants is exact — body link reads stay on the physx tensor path.
import isaaclab.utils.math as _math_utils  # noqa: E402
from lab_env import ROBOT_POS as _ROOT_POS, ROBOT_ROT as _ROOT_QUAT  # noqa: E402


def _patched_get_ee_pose_root_frame(self):
    ee_pos_w = self._asset.data.body_pos_w[:, self._ee_body_idx]
    ee_quat_w = self._asset.data.body_quat_w[:, self._ee_body_idx]
    n = ee_pos_w.shape[0]
    root_pos = torch.tensor(_ROOT_POS, dtype=ee_pos_w.dtype,
                            device=ee_pos_w.device).expand(n, 3)
    root_quat = torch.tensor(_ROOT_QUAT, dtype=ee_quat_w.dtype,
                             device=ee_quat_w.device).expand(n, 4)
    return _math_utils.subtract_frame_transforms(
        root_pos, root_quat, ee_pos_w, ee_quat_w)


_action_class._get_ee_pose_root_frame = _patched_get_ee_pose_root_frame


def _patched_compute_jacobian(self, joint_pos):
    # identical math to the frozen version; inputs coerced to torch (this
    # beta's physx view returns warp arrays) and the fixed-base spawn quat
    # substituted for the fabric-backed root read.
    jac_full = self._asset.root_physx_view.get_jacobians()
    if not isinstance(jac_full, torch.Tensor):
        import warp as wp
        jac_full = wp.to_torch(jac_full)
    jac_w = jac_full[:, self._jacobi_body_idx, :, self._jacobi_joint_ids]
    n = jac_w.shape[0]
    root_quat = torch.tensor([list(_ROOT_QUAT)], dtype=jac_w.dtype,
                             device=jac_w.device).repeat(n, 1)
    base_rot = _math_utils.matrix_from_quat(_math_utils.quat_inv(root_quat))
    return torch.cat([torch.bmm(base_rot, jac_w[:, :3, :]),
                      torch.bmm(base_rot, jac_w[:, 3:, :])], dim=1)


_action_class._compute_jacobian = _patched_compute_jacobian

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
    if args.bundle:
        # calibrated joint dynamics from the system bundle (identified on the
        # real FR3 under the same frozen OSC): armature + friction + viscous
        # damping — the quantities absent from the nucleus USD scene.
        import bundle_integration
        dyn = bundle_integration.load_dynamics(args.bundle)
        arm = dyn["armature_kg_m2"]["nominal"]
        fric = dyn["static_friction"]["nominal"]
        visc = dyn.get("viscous_friction", {}).get("nominal", [0.0] * 7)
        # a1 = fr3_joint1-4, a2 = fr3_joint5-7 (lab_env actuator grouping)
        cfg.scene.robot.actuators["a1"].armature = float(np.mean(arm[:4]))
        cfg.scene.robot.actuators["a2"].armature = float(np.mean(arm[4:]))
        cfg.scene.robot.actuators["a1"].friction = float(np.mean(fric[:4]))
        cfg.scene.robot.actuators["a2"].friction = float(np.mean(fric[4:]))
        cfg.scene.robot.actuators["a1"].damping = float(np.mean(visc[:4]))
        cfg.scene.robot.actuators["a2"].damping = float(np.mean(visc[4:]))
        print(f"[warmstart] bundle dynamics applied: armature={arm} "
              f"friction={fric} viscous={visc}", flush=True)
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
    terminations = getattr(cfg, "terminations", None)
    if terminations is not None:
        fields = set(getattr(type(terminations), "__annotations__", {}))
        fields.update(getattr(terminations, "__dict__", {}))
        for field in fields:
            if not field.startswith("_"):
                try:
                    setattr(terminations, field, None)
                except AttributeError:
                    pass
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

    root_pos_c = torch.tensor(list(_ROOT_POS), dtype=torch.float32,
                              device=device)
    root_quat_c = torch.tensor(list(_ROOT_QUAT), dtype=torch.float32,
                               device=device)

    def actual_ee():
        pos_b, quat_b = subtract_frame_transforms(
            root_pos_c, root_quat_c,
            robot.data.body_pos_w[0, hand_index],
            robot.data.body_quat_w[0, hand_index])
        return ([float(v) for v in pos_b], [float(v) for v in quat_b])

    # neutralize the USD's bogus fr3_link8 inertia (invalid {1,1,1} tensor /
    # negative mass in the nucleus asset — PhysX only sphere-approximates it)
    try:
        import warp as wp

        def as_torch(x):
            return x if isinstance(x, torch.Tensor) else wp.to_torch(x)

        link8 = robot.find_bodies("fr3_link8")[0][0]
        masses = as_torch(robot.root_physx_view.get_masses()).cpu().clone()
        masses[:, link8] = 1e-3
        robot.root_physx_view.set_masses(masses, torch.arange(1))
        inertias = as_torch(robot.root_physx_view.get_inertias()).cpu().clone()
        inertias[:, link8] = torch.tensor(
            [1e-5, 0, 0, 0, 1e-5, 0, 0, 0, 1e-5], dtype=inertias.dtype)
        robot.root_physx_view.set_inertias(inertias, torch.arange(1))
        print("[warmstart] fr3_link8 mass/inertia neutralized", flush=True)
    except Exception as error:  # noqa: BLE001
        print(f"[warmstart] link8 fix skipped: {type(error).__name__}: {error}",
              flush=True)

    # settle: actively hold the demo's INITIAL pose (a zero action would be a
    # follow-me servo that ratifies drift instead of resisting it)
    grip0 = 1.0 if float(np.mean(np.abs(joints0[7:9]))) > GRIPPER_OPEN_THRESHOLD else -1.0
    pos_init, quat_init = actual_ee()
    for _ in range(args.settle_steps):
        pos_now, quat_now = actual_ee()
        action0 = adapter.target_pose_to_action(
            pos_now, quat_now, pos_init, quat_init, grip0)
        env.step(torch.tensor([list(action0)], dtype=torch.float32,
                              device=device))

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
            cube_pos = torch.as_tensor(
                np.array(cube.data.root_pos_w[0].cpu()
                         if hasattr(cube.data.root_pos_w[0], "cpu")
                         else cube.data.root_pos_w[0]),
                dtype=torch.float32, device=device)
            cube_quat = torch.as_tensor(
                np.array(cube.data.root_quat_w[0].cpu()
                         if hasattr(cube.data.root_quat_w[0], "cpu")
                         else cube.data.root_quat_w[0]),
                dtype=torch.float32, device=device)
            cp, cq = subtract_frame_transforms(
                root_pos_c, root_quat_c, cube_pos, cube_quat)
            row.append([float(v) for v in cp] + [float(v) for v in cq])
        executed["cubes"].append(row)

    # ---- success judgement on final cube poses (tower_status only uses
    # relative geometry, so the base-frame row we just recorded suffices)
    final_cubes = [executed["cubes"][-1][c][:3] for c in range(3)]
    finger_pos = executed["grips"][-1]
    status = tower_status(final_cubes, finger_pos)
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
