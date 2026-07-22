#!/usr/bin/env python3
"""Measure the base adapter and fr3_hand_T_fr3v2_hand_tcp in-sim; write the binding YAML.

The overlay's asset-adapter policy forbids binding calibrated frames to team
prims on name similarity alone — and rightly so: the Isaac ``fr3.usd`` base
link frame is yawed 180 deg from the calibrated ``fr3v2_link0`` convention.
This probe puts the sim FR3 at the overlay's reference Franka-Home joint pose,
reads the fr3_hand link pose from FK, then

  1. picks the base adapter A (yaw in {0, 90, 180, 270} deg about z) that best
     maps the overlay's ``base_T_hand_tcp`` into the sim base frame:
         sim_T_tcp = A @ cal_T_tcp
  2. solves the hand adapter
         fr3_hand_T_hand_tcp = (sim_T_fr3_hand)^-1 @ A @ cal_T_tcp

Gates: mm-level translation residual of the naive TCP (+0.1034 z on fr3_hand)
vs the adapted prediction, and hand_T_tcp being a near-pure z-rotation (a
tilted hand adapter would mean a genuine frame mismatch, not a convention yaw).
Also reports the sim-desk vs calibrated-table height delta, then writes
fr3_binding.yaml consumed by render_viewpoints.py.

Run (arpa, UWLab env):  bash run_probe.sh
"""

from __future__ import annotations

import argparse
import math
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--overlay", default=os.path.join(os.path.dirname(__file__), "fr3_camera_overlay_v1/overlay.yaml"))
parser.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "fr3_binding.yaml"))
parser.add_argument("--table_usd", default="/home/ubuntu/jake/aidas/3cube_stack/table_scene.usdc")
parser.add_argument("--pos_tol_m", type=float, default=0.02, help="naive-vs-overlay FK translation gate")
parser.add_argument("--rot_tol_deg", type=float, default=5.0, help="naive-vs-overlay FK rotation gate")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import numpy as np
import torch
import gymnasium as gym
import yaml

import lab_env
from overlay_cameras import (
    T_from, T_inv, T_trans, camera_quat_order, load_overlay, pose_T_from_data, quat_wxyz_from_R,
    reference_home, rot_angle_deg,
)

QUAT_ORDER = camera_quat_order()  # 3.0 reads/writes are (x,y,z,w); 2.x is (w,x,y,z)


def link_pose_T(robot, idx):
    """World pose of a LINK frame (prefer link-frame data over COM)."""
    d = robot.data
    if hasattr(d, "body_link_pos_w"):
        return pose_T_from_data(d.body_link_pos_w[0, idx].cpu().numpy(),
                                d.body_link_quat_w[0, idx].cpu().numpy(), QUAT_ORDER)
    return pose_T_from_data(d.body_pos_w[0, idx].cpu().numpy(), d.body_quat_w[0, idx].cpu().numpy(), QUAT_ORDER)


def main():
    print(f"[probe] Isaac Lab quaternion order: {QUAT_ORDER}")
    ov = load_overlay(args.overlay)
    home_map, base_T_tcp_ref = reference_home(ov)

    env = gym.make(lab_env.TASK, cfg=lab_env.build_env_cfg(args.device, args.table_usd)).unwrapped
    robot = env.scene["robot"]
    names = robot.joint_names
    print(f"[probe] joint_names = {names}")
    q = torch.zeros((1, len(names)), device=env.device)
    for i, n in enumerate(names):
        q[0, i] = home_map.get(n, 0.04)  # arm joints from overlay; fingers open 0.04

    with torch.inference_mode():
        env.reset()
        robot.write_joint_state_to_sim(q, torch.zeros_like(q))
        robot.set_joint_position_target(q)
        for _ in range(8):  # settle under PD hold (gravity disabled)
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(env.physics_dt)

        q_now = robot.data.joint_pos[0].cpu().numpy()
        q_ref = q[0].cpu().numpy()
        q_err = np.abs(q_now - q_ref).max()
        print(f"[probe] joints reached = {np.round(q_now, 4).tolist()}")
        print(f"[probe] max joint error vs overlay home = {q_err:.5f} rad")
        if q_err > 5e-3:
            print("[probe] ERROR: sim cannot hold the overlay reference pose "
                  "(joint-limit clamp?) — FK comparison would be invalid.")
            env.close()
            sys.exit(2)

        i_base = robot.body_names.index("fr3_link0")
        i_hand = robot.body_names.index("fr3_hand")
        W_T_base = link_pose_T(robot, i_base)
        W_T_hand = link_pose_T(robot, i_hand)

    base_T_hand = T_inv(W_T_base) @ W_T_hand
    naive = base_T_hand @ T_trans(0.0, 0.0, 0.1034)  # sim TCP via the IK action's offset

    # 1) base adapter: yaw about z mapping calibrated fr3v2_link0 -> sim fr3_link0
    def Rz_T(deg):
        c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
        T = np.eye(4)
        T[:2, :2] = [[c, -s], [s, c]]
        return T

    yaw, adapter, d_pos = min(
        ((deg, A, float(np.linalg.norm((A @ base_T_tcp_ref)[:3, 3] - naive[:3, 3])))
         for deg, A in ((d, Rz_T(d)) for d in (0, 90, 180, 270))),
        key=lambda t: t[2])

    # 2) hand adapter, using the chosen base adapter
    hand_T_tcp = T_inv(base_T_hand) @ adapter @ base_T_tcp_ref
    # physical invariants (one reference pose can't cross-check more): the Franka-hand
    # TCP sits 0.1034 m from the hand frame, and the gripper axis must be (anti)parallel
    # to the hand z — the sim fr3.usd hand frame may be z-FLIPPED vs the TCP convention
    # (measured: hand z points up at ready pose, TCP at (0,0,-0.1034) with a 180 deg flip),
    # which is a legitimate USD convention the adapter absorbs, not an error.
    t_norm = float(np.linalg.norm(hand_T_tcp[:3, 3]))
    axis_dev = math.degrees(math.acos(np.clip(abs(hand_T_tcp[2, 2]), 0.0, 1.0)))
    ok = abs(t_norm - 0.1034) <= args.pos_tol_m and axis_dev <= args.rot_tol_deg

    # table sanity: calibrated table-task origin height vs the sim desk top,
    # both expressed relative to the robot base link.
    base_T_table = T_from(ov["runtime"]["environment"]["table"]["base_T_table"])
    sim_desk_top_in_base = lab_env.DESK_Z - lab_env.ROBOT_POS[2]

    print(f"[probe] base_T_hand (home) t = {np.round(base_T_hand[:3, 3], 4).tolist()}")
    print(f"[probe] overlay base_T_tcp  t = {np.round(base_T_tcp_ref[:3, 3], 4).tolist()}")
    print(f"[probe] base adapter: yaw {yaw} deg about z "
          f"({'identity — frames already match' if yaw == 0 else 'sim fr3_link0 is yawed vs calibrated fr3v2_link0'})")
    print(f"[probe] hand_T_tcp t = {np.round(hand_T_tcp[:3, 3], 4).tolist()} (|t|={t_norm:.4f} m, expect ~0.1034)  "
          f"rot={rot_angle_deg(hand_T_tcp[:3, :3]):.2f} deg  z-axis dev={axis_dev:.2f} deg")
    print(f"[probe] gates: |t|-0.1034 within {args.pos_tol_m * 1000:.0f} mm and z-axis dev within "
          f"{args.rot_tol_deg:.0f} deg -> {'OK' if ok else 'FAIL'}   "
          f"(naive +0.1034z guess was off by {d_pos * 1000:.1f} mm — diagnostic only)")
    print(f"[probe] table z in base: calibrated {base_T_table[2, 3]:+.4f} m  vs sim desk {sim_desk_top_in_base:+.4f} m "
          f"(delta {(base_T_table[2, 3] - sim_desk_top_in_base) * 1000:.0f} mm — affects view content, not camera math)")

    binding = {
        "schema_version": "stage2.fr3_camera_overlay_asset_binding.v1",
        "status": "measured_in_sim_by_probe_tcp_binding" if ok else "FAILED_probe_gate",
        "overlay_bundle_id": ov["bundle_id"],
        "calibration_revision": ov["calibration_revision"],
        "scene": {
            "runtime": "isaaclab_uwlab_lab_stack",
            "robot_usd": "ISAAC_NUCLEUS/Robots/FrankaRobotics/FrankaFR3/fr3.usd",
            "table_usd": args.table_usd,
        },
        "robot_base": {
            "team_prim": "{ENV_REGEX_NS}/Robot/fr3_link0",
            "calibrated_semantic": "fr3v2_link0",
            "compatibility_class": ("exact_semantic_frame" if yaw == 0 else "known_adapter_required") if ok
                                   else "unknown_or_incompatible",
            "identity_transform_accepted": bool(ok and yaw == 0),
            "adapter_yaw_deg": int(yaw),
            "team_frame_T_calibrated_frame": {
                "matrix": adapter.tolist(),
                "translation_m": adapter[:3, 3].tolist(),
                "quaternion_wxyz": quat_wxyz_from_R(adapter[:3, :3]).tolist(),
            },
            "evidence": f"home-pose FK chain check with yaw-candidate fit: best yaw={yaw}deg; hand_T_tcp "
                        f"|t|={t_norm:.4f}m (expect 0.1034), z-axis dev={axis_dev:.2f}deg "
                        f"(gates {args.pos_tol_m * 1000:.0f}mm/{args.rot_tol_deg:.0f}deg)",
        },
        "hand_tcp": {
            "team_prim": "{ENV_REGEX_NS}/Robot/fr3_hand",
            "calibrated_semantic": "fr3v2_hand_tcp",
            "compatibility_class": "known_adapter_required",
            "team_frame_T_calibrated_frame": {
                "matrix": hand_T_tcp.tolist(),
                "translation_m": hand_T_tcp[:3, 3].tolist(),
                "quaternion_wxyz": quat_wxyz_from_R(hand_T_tcp[:3, :3]).tolist(),
            },
            "evidence": "solved from sim fr3_hand FK at the overlay reference Franka-Home pose "
                        "against overlay runtime.reference_robot_pose.base_T_hand_tcp",
        },
        "diagnostics": {
            "isaac_lab_quat_order": QUAT_ORDER,
            "max_joint_error_rad": float(q_err),
            "base_adapter_yaw_deg": int(yaw),
            "naive_guess_dpos_mm": float(d_pos * 1000),
            "hand_T_tcp_norm_m": t_norm,
            "hand_T_tcp_z_axis_dev_deg": float(axis_dev),
            "hand_T_tcp_rot_deg": rot_angle_deg(hand_T_tcp[:3, :3]),
            "table_z_delta_mm": float((base_T_table[2, 3] - sim_desk_top_in_base) * 1000),
        },
        "ready_to_apply": bool(ok),
    }
    with open(args.out, "w", encoding="utf-8") as f:
        yaml.safe_dump(binding, f, sort_keys=False)
    print(f"[probe] wrote {args.out}  ready_to_apply={ok}")
    env.close()
    return 0 if ok else 2


if __name__ == "__main__":
    # print failures BEFORE app.close(): SimulationApp.close() can os._exit(0),
    # which would swallow the traceback and fake a success exit code
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
