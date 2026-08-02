"""Convert RL-teacher 3-cube episodes into lab_stack_mimic input episodes.

Purpose: the human-vs-RL source comparison must run both source sets through
the SAME generation pipeline (LabFR3 Isaac Lab Mimic). Human teleop demos are
already in that format; this script brings the RL bundle episodes
(OmniReset-Fr3PandaCube, 10 Hz contract actions, Isaac states) into it:

  1. FK pass (states playback in the lab scene) -> actual EE track in base
     frame — identical machinery to convert_demo.py;
  2. scene remap: keep base-frame xy, shift z by (lab cube rest height -
     RL cube rest height) so cubes rest on the lab desk; the EE track gets the
     same shift, preserving grasp geometry;
  3. resample the EE track to the LAB env rate (20 Hz) and emit IK-rel actions
     [dpos(3), axis_angle(3), grip] (scale 1.0 — the annotate-time convention);
  4. export lab-format episodes: initial_state + states + actions (+obs eef),
     ready for run_annotate/run_generate.

Run in the Isaac Lab docker on aidas:
  ./run_isaac_aidas.sh /repo/contract/rl_to_lab.py --device cpu \
      --dataset /rl_demos/fr3_three_cube_fullstack_success_50.hdf5 \
      --output /out/rl_as_lab_3cube.hdf5 [--count 50]
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
parser.add_argument("--count", type=int, default=50)
parser.add_argument("--source_hz", type=float, default=10.0)
parser.add_argument("--time_stretch", type=float, default=1.0,
                    help="optional demo slow-down before resampling (stretched "
                         "sources make generated rollouts proportionally "
                         "longer and can hit the episode cap — prefer the "
                         "LAB_SUBTASK_OFFSETS generation knob instead)")
parser.add_argument("--table_usd", default="/nonexistent.usdc")
parser.add_argument("--lab_desk_top_z", type=float, default=0.720)
parser.add_argument("--cube_half_m", type=float, default=0.0254)
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
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "../render")))

import traj_tools  # noqa: E402
from adapter import quat_conjugate, quat_multiply, quat_to_axis_angle  # noqa: E402
from lab_env import build_env_cfg, ROBOT_POS, ROBOT_ROT  # noqa: E402

LAB_HZ = 20.0  # stock lab stack env policy rate (sim.dt*decimation = 0.05 s)
GRIPPER_OPEN_THRESHOLD = 0.02


def _rot_wxyz(quat):
    w, x, y, z = np.asarray(quat, dtype=np.float64)
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def base_to_lab_world(pos_b, quat_b_wxyz):
    """Lab world pose from a base-frame pose (lab robot base = ROBOT_POS/ROT)."""
    rot = _rot_wxyz(ROBOT_ROT)
    pos = np.asarray(ROBOT_POS) + rot @ np.asarray(pos_b)
    quat = quat_multiply(ROBOT_ROT, quat_b_wxyz)
    return pos, quat


def main():
    env_cfg = build_env_cfg(args.device, args.table_usd, cameras=None, num_envs=1)
    env = gym.make("Isaac-Stack-Cube-Franka-IK-Rel-v0", cfg=env_cfg).unwrapped
    env.reset()
    scene = env.scene
    robot = scene["robot"]
    hand_index = robot.find_bodies("fr3_hand")[0][0]

    lab_rest_z_b = (args.lab_desk_top_z + args.cube_half_m) - float(ROBOT_POS[2])

    out = h5py.File(args.output, "w")
    out_data = out.create_group("data")
    report = {"dataset": args.dataset, "demos": {}}
    with h5py.File(args.dataset, "r") as src:
        names = list(src["data"].keys())[: args.count]
        for name in names:
            group = src[f"data/{name}"]
            states = group["states"]
            joints = states["articulation/robot/joint_position"][()]
            joint_vel = states["articulation/robot/joint_velocity"][()]
            root = states["articulation/robot/root_pose"][()]
            cubes = np.stack(
                [states[f"rigid_object/cube_{i}/root_pose"][()]
                 for i in (1, 2, 3)], axis=1)
            rl_actions = group["actions"][()]
            T = len(joints)
            times = np.arange(T) / args.source_hz * args.time_stretch

            # ---- FK pass: EE track in base frame (spawn-invariant)
            ee_pos, ee_quat = [], []
            for t in range(T):
                robot.write_joint_state_to_sim(
                    torch.tensor(joints[t], dtype=torch.float32,
                                 device=env.device).unsqueeze(0),
                    torch.tensor(joint_vel[t], dtype=torch.float32,
                                 device=env.device).unsqueeze(0))
                scene.write_data_to_sim()
                env.sim.forward()
                scene.update(dt=0.0)
                pos_b, quat_b = subtract_frame_transforms(
                    robot.data.root_pos_w[0], robot.data.root_quat_w[0],
                    robot.data.body_pos_w[0, hand_index],
                    robot.data.body_quat_w[0, hand_index])
                ee_pos.append([float(v) for v in pos_b])
                ee_quat.append([float(v) for v in quat_b])

            # ---- cube tracks in RL base frame + rest-height shift
            rot_rl = _rot_wxyz(root[0, 3:7]).T  # world->base
            cubes_b = np.zeros((T, 3, 7))
            for t in range(T):
                for c in range(3):
                    cubes_b[t, c, :3] = rot_rl @ (cubes[t, c, :3] - root[t, :3])
                    q = quat_multiply(quat_conjugate(root[t, 3:7]),
                                      cubes[t, c, 3:7])
                    cubes_b[t, c, 3:7] = q
            rl_rest_z_b = float(np.min(cubes_b[0, :, 2]))
            z_shift = lab_rest_z_b - rl_rest_z_b
            cubes_b[:, :, 2] += z_shift
            ee_track = np.asarray(ee_pos)
            ee_track[:, 2] += z_shift

            # ---- gripper from RL contract actions (sign), held per step
            grip = [1.0 if a > 0 else -1.0 for a in rl_actions[:, 6]]
            grip = grip + [grip[-1]] * (T - len(grip))

            # ---- resample EE track to the LAB rate, emit IK-rel actions
            rt, rp, rq, rg = traj_tools.resample_pose_track(
                times.tolist(), ee_track.tolist(), ee_quat, grip,
                target_dt=1.0 / LAB_HZ)
            lab_actions = []
            for k in range(len(rp) - 1):
                dp = [rp[k + 1][i] - rp[k][i] for i in range(3)]
                dr = quat_to_axis_angle(
                    quat_multiply(rq[k + 1], quat_conjugate(rq[k])))
                lab_actions.append(list(dp) + list(dr) + [rg[k + 1]])
            # stationary tail: RL demos end right after the last grasp, which
            # violates isaaclab_mimic's subtask-boundary sanity (last boundary
            # + max offset must fit inside the demo). Padding with hold frames
            # is datagen-neutral and keeps the task cfg identical to the human
            # comparison arm.
            PAD = 30
            lab_actions += [[0.0] * 6 + [rg[-1]]] * PAD
            rp = list(rp) + [rp[-1]] * PAD
            rq = list(rq) + [rq[-1]] * PAD
            rg = list(rg) + [rg[-1]] * PAD

            # ---- lab-world states for initial_state/states groups
            steps = len(lab_actions)
            index = np.minimum(np.round(np.asarray(rt) * args.source_hz)
                               .astype(int), T - 1)
            index = np.concatenate(
                [index, np.repeat(index[-1], steps + 1 - len(index))])[: steps + 1]
            episode = out_data.create_group(name)
            episode.attrs["num_samples"] = steps
            episode.attrs["success"] = bool(group.attrs.get("success", True))
            episode.attrs["source"] = "rl_teacher"

            def write_state_group(state_group, ts):
                robot_group = state_group.create_group("articulation/robot")
                robot_group.create_dataset(
                    "joint_position", data=joints[ts].astype(np.float32))
                robot_group.create_dataset(
                    "joint_velocity", data=np.zeros_like(joints[ts],
                                                         dtype=np.float32))
                root_lab = np.concatenate(
                    [np.asarray(ROBOT_POS), np.asarray(ROBOT_ROT)])
                robot_group.create_dataset(
                    "root_pose",
                    data=np.tile(root_lab, (len(np.atleast_1d(ts)), 1))
                    .astype(np.float32) if np.ndim(ts) else
                    root_lab.astype(np.float32)[None])
                robot_group.create_dataset(
                    "root_velocity",
                    data=np.zeros((len(np.atleast_1d(ts)) if np.ndim(ts)
                                   else 1, 6), dtype=np.float32))
                for c in (1, 2, 3):
                    cube_group = state_group.create_group(
                        f"rigid_object/cube_{c}")
                    rows = np.atleast_1d(ts)
                    poses = []
                    for t in rows:
                        pw, qw = base_to_lab_world(
                            cubes_b[t, c - 1, :3], cubes_b[t, c - 1, 3:7])
                        poses.append(list(pw) + list(qw))
                    cube_group.create_dataset(
                        "root_pose", data=np.asarray(poses, dtype=np.float32))
                    cube_group.create_dataset(
                        "root_velocity",
                        data=np.zeros((len(rows), 6), dtype=np.float32))

            write_state_group(episode.create_group("initial_state"),
                              np.asarray([0]))
            write_state_group(episode.create_group("states"), index)
            episode.create_dataset(
                "actions", data=np.asarray(lab_actions, dtype=np.float32))

            # ---- obs + OFFLINE ANNOTATION (fwd_annotated schema): MimicGen
            # generation consumes datagen_info, not source-action replays, so
            # subtask signals are synthesized from state predicates. RL
            # collection is canonical fwd (always cube_2->cube_1 then
            # cube_3->cube_2, verified on the bundle).
            ee_world = [base_to_lab_world(rp[k], rq[k]) for k in range(steps)]
            hand_pos_w = np.asarray([p for p, _ in ee_world], dtype=np.float32)
            ee_quat_w = np.asarray([q for _, q in ee_world], dtype=np.float32)
            # the lab annotation convention (verified on fwd_annotated) puts
            # eef everywhere in TCP coordinates, not the fr3_hand body: shift
            # along the hand +z (empirically validated on the grasp events)
            ee_pos_w = np.stack([
                hand_pos_w[k] + _rot_wxyz(ee_quat_w[k]) @ np.array([0, 0, 0.1034])
                for k in range(steps)]).astype(np.float32)
            cubes_w = np.zeros((steps, 3, 7), dtype=np.float32)
            for k in range(steps):
                t = index[k]
                for c in range(3):
                    pw, qw = base_to_lab_world(cubes_b[t, c, :3],
                                               cubes_b[t, c, 3:7])
                    cubes_w[k, c] = list(pw) + list(qw)
            fingers = np.abs(joints[index[:steps], 7:9]).mean(axis=1)

            def pose44(pos, quat):
                mat = np.eye(4, dtype=np.float32)
                mat[:3, :3] = _rot_wxyz(quat)
                mat[:3, 3] = pos
                return mat

            eef44 = np.stack([pose44(ee_pos_w[k], ee_quat_w[k])
                              for k in range(steps)])
            closed = fingers < 0.03          # closing/closed (open = 0.04)
            released = fingers > 0.03
            # ee_pos_w is already TCP (see above) — use it directly for
            # grasp proximity (min dist to the grasped cube ~1.6 cm)
            near = lambda c: np.linalg.norm(  # noqa: E731
                ee_pos_w - cubes_w[:, c, :3], axis=1) < 0.10
            stacked = lambda top, low: (  # noqa: E731
                (np.linalg.norm(cubes_w[:, top, :2] - cubes_w[:, low, :2],
                                axis=1) < 0.04)
                & (cubes_w[:, top, 2] - cubes_w[:, low, 2] > 0.035)
                & (cubes_w[:, top, 2] - cubes_w[:, low, 2] < 0.065)
                & released)
            grasp_1 = closed & near(1)                      # holding cube_2
            stack_1 = stacked(1, 0)                         # cube_2 on cube_1
            grasp_2 = closed & near(2) & stack_1.cumsum().astype(bool)
            obs = episode.create_group("obs")
            obs.create_dataset("eef_pos", data=ee_pos_w)
            obs.create_dataset("eef_quat", data=ee_quat_w)
            obs.create_dataset("gripper_pos",
                               data=joints[index[:steps], 7:9]
                               .astype(np.float32))
            obs.create_dataset("joint_pos",
                               data=joints[index[:steps]].astype(np.float32))
            obs.create_dataset("joint_vel",
                               data=np.zeros_like(joints[index[:steps]],
                                                  dtype=np.float32))
            obs.create_dataset(
                "cube_positions",
                data=cubes_w[:, :, :3].reshape(steps, 9).astype(np.float32))
            obs.create_dataset(
                "cube_orientations",
                data=cubes_w[:, :, 3:7].reshape(steps, 12).astype(np.float32))
            info = obs.create_group("datagen_info")
            info.create_dataset("eef_pose/franka", data=eef44)
            target44 = np.concatenate([eef44[1:], eef44[-1:]])
            info.create_dataset("target_eef_pose/franka", data=target44)
            for c in (1, 2, 3):
                info.create_dataset(
                    f"object_pose/cube_{c}",
                    data=np.stack([pose44(cubes_w[k, c - 1, :3],
                                          cubes_w[k, c - 1, 3:7])
                                   for k in range(steps)]))
            info.create_dataset("subtask_term_signals/grasp_1", data=grasp_1)
            info.create_dataset("subtask_term_signals/stack_1", data=stack_1)
            info.create_dataset("subtask_term_signals/grasp_2", data=grasp_2)
            def first_true(mask):
                hits = np.flatnonzero(mask)
                return int(hits[0]) if len(hits) else -1
            signals = {"grasp_1": first_true(grasp_1),
                       "stack_1": first_true(stack_1),
                       "grasp_2": first_true(grasp_2)}
            MIN_GAP = 7  # boundary sanity with LAB_SUBTASK_OFFSETS=0,5 (max
            #              offset 5 + margin); tail covered by the PAD frames
            gaps_ok = (signals["stack_1"] - signals["grasp_1"] >= MIN_GAP
                       and signals["grasp_2"] - signals["stack_1"] >= MIN_GAP
                       and (steps - 1) - signals["grasp_2"] >= MIN_GAP + 5)
            annotation_ok = all(v >= 0 for v in signals.values()) and (
                signals["grasp_1"] < signals["stack_1"] < signals["grasp_2"]
            ) and gaps_ok
            report["demos"][name] = {
                "T_rl": T, "T_lab": steps, "z_shift_m": z_shift,
                "signal_first_true": signals, "annotation_ok": annotation_ok,
            }
            print(f"[rl2lab] {name}: {T}@10Hz -> {steps}@{LAB_HZ:.0f}Hz "
                  f"z_shift={z_shift:+.4f}m signals={signals} "
                  f"ok={annotation_ok}", flush=True)

    out_data.attrs["total"] = sum(
        int(out_data[k].attrs["num_samples"]) for k in out_data.keys())
    out_data.attrs["env_args"] = json.dumps({
        "env_name": "Isaac-Stack-Cube-LabFR3-Fwd-IK-Rel-Mimic-v0", "type": 2,
        "env_kwargs": {}})
    out.close()
    with open(args.output + ".report.json", "w") as handle:
        json.dump(report, handle, indent=2)
    print(f"[rl2lab] wrote {args.output}", flush=True)
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
