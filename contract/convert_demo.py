"""Convert FR3 3-cube episodes to the Stage-1 control-contract format.

Offline (kinematic) conversion — no policy execution:
  1. states-playback each source frame into the FR3 scene (write joint +
     cube states to sim, no physics stepping) and read the fr3_hand pose in
     the robot base frame — this is the unambiguous FK pass;
  2. resample the actual-EE track to 10 Hz (contract policy rate);
  3. actions[t] = target_pose_to_action(actual_t, actual_{t+1}) — reference
     is the current actual pose, per the contract;
  4. report pose/action round-trip error, action percentiles, and the
     fraction outside the RL reference envelope;
  5. export the contract HDF5 (dataset_schema.yaml) + JSON report.

The closed-loop legacy-OSC execution (warm-start final step) is a separate
script (warmstart_replay.py) because it needs UWLab's RelCartesianOSCAction.

Run inside Isaac Lab (docker isaac-lab image on aidas, or env_uwlab on arpa):
  isaaclab.sh -p contract/convert_demo.py --device cpu \
      --dataset <src.hdf5> --output <contract.hdf5> --count 1 \
      [--reference <rl_bundle.hdf5>] [--source_hz <hz>]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--report", default=None, help="JSON report path")
parser.add_argument("--demos", default=None, help="comma list, default: first --count")
parser.add_argument("--count", type=int, default=1)
parser.add_argument("--reference", default=None,
                    help="RL bundle hdf5 for the action reference envelope")
parser.add_argument("--source_hz", type=float, default=None,
                    help="override source sample rate (default: env step rate)")
parser.add_argument("--table_usd", default="/nonexistent.usdc")
parser.add_argument("--retarget_version", default="offline_fk_v1")
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


def _rot_wxyz(quat):
    w, x, y, z = np.asarray(quat, dtype=np.float64)
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
for rel in ("../render", "../lab_stack_mimic"):
    sys.path.insert(0, os.path.normpath(os.path.join(HERE, rel)))

import schema_io  # noqa: E402
import traj_tools  # noqa: E402
from lab_env import build_env_cfg  # noqa: E402  (repo render/lab_env.py)

GRIPPER_OPEN_THRESHOLD = 0.02  # finger joint > 2 cm -> open
# hand -> TCP translation from render/fr3_binding.yaml (|z| = 0.1034 m).
# Sign follows this asset's fr3_hand frame (z toward the wrist): the TCP sits
# at -z. Only used for the sanity check against recorded obs/eef_pos (world
# TCP); the contract export itself is the fr3_hand body in robot_base.
HAND_T_TCP = (0.0, 0.0, -0.1034)


def episode_arrays(group):
    states = group["states"]
    joints = states["articulation/robot/joint_position"][()]
    joint_vel = states["articulation/robot/joint_velocity"][()]
    root = states["articulation/robot/root_pose"][()]
    cubes = np.stack(
        [states[f"rigid_object/cube_{i}/root_pose"][()] for i in (1, 2, 3)],
        axis=1)  # [T,3,7]
    return joints, joint_vel, root, cubes


def to_base(pos_w, quat_w, root_pos_w, root_quat_w):
    pos_b, quat_b = subtract_frame_transforms(
        root_pos_w, root_quat_w, pos_w, quat_w)
    return pos_b, quat_b


def main():
    env_cfg = build_env_cfg(args.device, args.table_usd, cameras=None, num_envs=1)
    env = gym.make("Isaac-Stack-Cube-Franka-IK-Rel-v0", cfg=env_cfg).unwrapped
    env.reset()
    scene = env.scene
    robot = scene["robot"]
    hand_index = robot.find_bodies("fr3_hand")[0][0]
    step_dt = float(env.step_dt)

    reference = None
    if args.reference:
        with h5py.File(args.reference, "r") as handle:
            acts = np.concatenate(
                [handle[f"data/{d}/actions"][()] for d in handle["data"].keys()])
        low = np.percentile(acts[:, :6], 0.5, axis=0)
        high = np.percentile(acts[:, :6], 99.5, axis=0)
        reference = (low.tolist(), high.tolist(),
                     {"n": int(acts.shape[0]), "p005": low.tolist(),
                      "p995": high.tolist()})

    report: dict = {"dataset": args.dataset, "retarget": args.retarget_version,
                    "demos": {}}
    out = h5py.File(args.output, "w")
    with h5py.File(args.dataset, "r") as src:
        names = (args.demos.split(",") if args.demos
                 else list(src["data"].keys())[: args.count])
        for name in names:
            group = src[f"data/{name}"]
            joints, joint_vel, root, cubes = episode_arrays(group)
            T = len(joints)
            source_hz = args.source_hz or (1.0 / step_dt)
            times = np.arange(T) / source_hz

            ee_pos, ee_quat, grip = [], [], []
            tcp_world = []
            cube_b = []
            for t in range(T):
                robot.write_joint_state_to_sim(
                    torch.tensor(joints[t], dtype=torch.float32,
                                 device=env.device).unsqueeze(0),
                    torch.tensor(joint_vel[t], dtype=torch.float32,
                                 device=env.device).unsqueeze(0))
                scene.write_data_to_sim()
                env.sim.forward()
                scene.update(dt=0.0)
                pos_w = robot.data.body_pos_w[0, hand_index]
                quat_w = robot.data.body_quat_w[0, hand_index]
                root_pos = robot.data.root_pos_w[0]
                root_quat = robot.data.root_quat_w[0]
                pos_b, quat_b = to_base(pos_w, quat_w, root_pos, root_quat)
                ee_pos.append([float(v) for v in pos_b])
                ee_quat.append([float(v) for v in quat_b])
                # sanity-check TCP in the RECORDED world: base-frame FK is
                # invariant to the sim robot's spawn pose, so reconstruct the
                # world pose under the source demo's recorded root (wxyz,
                # yaw180 in the lab recordings), then add the hand->TCP offset
                # along the hand frame. Explicit numpy math — frame-convention
                # bugs here were only visible numerically, so keep it audited.
                rot_root = _rot_wxyz(root[t, 3:7])
                rot_hand_w = rot_root @ _rot_wxyz(
                    [float(v) for v in quat_b])
                tcp = (root[t, :3]
                       + rot_root @ np.array([float(v) for v in pos_b])
                       + rot_hand_w @ np.asarray(HAND_T_TCP))
                tcp_world.append(tcp.tolist())
                grip.append(1.0 if float(joints[t, 7:9].mean())
                            > GRIPPER_OPEN_THRESHOLD else -1.0)
                # cube poses: recorded world states -> demo's own base frame
                root_pos_rec = torch.tensor(root[t, :3], dtype=torch.float32)
                root_quat_rec = torch.tensor(root[t, 3:7], dtype=torch.float32)
                row = []
                for c in range(3):
                    cp, cq = to_base(
                        torch.tensor(cubes[t, c, :3], dtype=torch.float32),
                        torch.tensor(cubes[t, c, 3:7], dtype=torch.float32),
                        root_pos_rec, root_quat_rec)
                    row.append([float(v) for v in cp] + [float(v) for v in cq])
                cube_b.append(row)

            # sanity: recorded obs/eef_pos is the WORLD-frame TCP in the lab
            # datasets — compare against our FK hand pose + hand->TCP offset.
            fk_check = None
            if "obs" in group and "eef_pos" in group["obs"]:
                rec = group["obs/eef_pos"][()]
                fk = np.asarray(tcp_world)
                n = min(len(rec), len(fk))
                errs = np.linalg.norm(rec[:n] - fk[:n], axis=1)
                fk_check = {
                    "mean_m": float(np.mean(errs)),
                    "p50_m": float(np.median(errs)),
                    "p95_m": float(np.percentile(errs, 95)),
                    "max_m": float(np.max(errs)),
                    "argmax": int(np.argmax(errs)),
                    "first_mid_last_m": [float(errs[0]), float(errs[n // 2]),
                                         float(errs[-1])],
                    "sample_mid": {"rec": rec[n // 2].tolist(),
                                   "fk_tcp": fk[n // 2].tolist()},
                }

            rt, rp, rq, rg = traj_tools.resample_pose_track(
                times.tolist(), ee_pos, ee_quat, grip)
            actions, targets, deltas = traj_tools.track_to_actions(rp, rq, rg)
            errors = traj_tools.round_trip_errors(rp, rq, actions)
            percentiles = traj_tools.action_percentiles(actions)

            # cube + joint tracks resampled on the same clock (nearest frame)
            index = np.minimum(
                np.round(np.asarray(rt) * source_hz).astype(int), T - 1)[:-1]
            success = bool(group.attrs.get(
                "success", group.attrs.get("replay_success_any_order", False)))
            schema_io.write_episode(
                out, name,
                actions=actions,
                processed_delta=deltas,
                commanded_target_pose=targets,
                actual_ee_pose=[list(rp[k]) + list(rq[k])
                                for k in range(len(actions))],
                joint_position=joints[index],
                joint_velocity=joint_vel[index],
                gripper_state=joints[index][:, 7:9],
                cube_pose=np.asarray(cube_b)[index],
                success=success,
                source_human_demo_id=f"{os.path.basename(args.dataset)}::{name}",
                retarget_version=args.retarget_version,
                extras={"source_target_pose": targets},
            )
            demo_report = {
                "T_source": T, "T_contract": len(actions),
                "source_hz": source_hz, "round_trip": errors,
                "action_percentiles": percentiles,
                "fk_vs_recorded_eef_mean_m": fk_check,
                "success_attr": success,
            }
            if reference:
                demo_report["outside_reference_envelope"] = (
                    traj_tools.envelope_fraction(actions, reference[0], reference[1]))
                demo_report["reference_envelope"] = reference[2]
            report["demos"][name] = demo_report
            print(f"[convert] {name}: T {T}->{len(actions)} @10Hz "
                  f"rt_pos={errors['max_position_error_m']:.2e} "
                  f"rt_rot={errors['max_rotation_error_rad']:.2e} "
                  f"fk_check={fk_check}", flush=True)

    schema_io.finalize_file(out, env_args={
        "env_name": "contract_offline_conversion",
        "type": 99,
        "env_kwargs": {"source": args.dataset,
                       "contract_id": schema_io.CONTRACT_ID}})
    out.close()
    issues = schema_io.validate_file(args.output)
    report["schema_violations"] = issues
    report_path = args.report or (args.output + ".report.json")
    with open(report_path, "w") as handle:
        json.dump(report, handle, indent=2)
    print(f"[convert] schema: {'PASS' if not issues else issues}", flush=True)
    print(f"[convert] wrote {args.output} and {report_path}", flush=True)
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
