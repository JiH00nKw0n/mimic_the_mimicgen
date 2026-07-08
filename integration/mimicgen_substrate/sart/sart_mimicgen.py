#!/usr/bin/env python
"""
SART augmentation on robosuite MimicGen Square (CORRECTED one-way converging approach).

Integrates SART (Self-Augmented Robot Trajectory) into the working robosuite MimicGen
pipeline. For each MimicGen-generated source demo, it keeps the transport + the
tight-tolerance insertion VERBATIM and only DIVERSIFIES THE APPROACH: divert to a
sampled offset pose near the convergence point, then converge ONE-WAY back to the demo's
convergence pose, then replay the recorded insertion. Executes in robosuite via mimicgen's
WaypointTrajectory + target_pose_to_action (closed-loop OSC), success-filters, and writes
schema-compatible HDF5 (reuses write_demo_to_hdf5 / merge_all_hdf5).

Corrections over the naive design (all 3 adversarial reviews flagged the same defect):
 - NO "out-and-back" (which recorded anti-convergent actions); the approach is monotone
   one-way offset -> convergence.
 - Sources = MULTIPLE MimicGen-generated demos (varied nut/peg scenes), not N resets to
   one state -> real scene diversity underneath the approach diversity.
 - Success is the filter (nut inserted); no gate against the un-achievable commanded pose.
CPU-only. No cuRobo, no GPU.
"""
import argparse
import json
import os

import h5py
import numpy as np

import mimicgen.utils.file_utils as MG_FileUtils
import mimicgen.utils.robomimic_utils as RobomimicUtils
from mimicgen.datagen.waypoint import WaypointSequence, WaypointTrajectory
from mimicgen.env_interfaces.base import make_interface
from robomimic.utils.file_utils import get_env_metadata_from_dataset


def rand_rot(max_angle, rng):
    """Small random rotation matrix via Rodrigues, |angle| <= max_angle."""
    axis = rng.standard_normal(3)
    axis /= (np.linalg.norm(axis) + 1e-9)
    ang = rng.uniform(-max_angle, max_angle)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)


def sample_offset(P_conv, peg_pos, r_max, ang, rng, position_fix, z_margin=0.005):
    """Sample an approach-basin offset around the pre-insertion pose: full lateral/upward
    spread within r_max, floored at peg-top so it never dips into the hole.
    (The earlier 0.5*height cap collapsed the offset to ~0 at descent onset -> degenerate
    approaches; the falsifiable diversity metric caught it.)"""
    center = P_conv[:3, 3].copy()
    peg_top_z = peg_pos[2] + z_margin
    p = center.copy()
    for _ in range(20):
        v = rng.standard_normal(3); v /= (np.linalg.norm(v) + 1e-9)
        rad = r_max * (rng.random() ** (1.0 / 3))
        cand = center + rad * v
        if cand[2] >= peg_top_z:
            p = cand; break
    P = np.eye(4); P[:3, 3] = p
    P[:3, :3] = P_conv[:3, :3] if position_fix else P_conv[:3, :3] @ rand_rot(ang, rng)
    return P


def find_t_conv(eef_pose, obj_poses, t2s, T, tail_len):
    """Descent-onset index in [t2s, T): eef-z decreases >=3 steps while nut is near peg (xy)."""
    z = eef_pose[:, 2, 3]
    nut_xy = obj_poses["square_nut"][:, :2, 3]
    peg_xy = obj_poses["square_peg"][:, :2, 3]
    dxy = np.linalg.norm(nut_xy - peg_xy, axis=1)
    thresh = np.percentile(dxy[t2s:T], 35)
    for t in range(max(t2s, 1), T - 3):
        if z[t + 1] < z[t] and z[t + 2] < z[t + 1] and z[t + 3] < z[t + 2] and dxy[t] < thresh:
            return t
    return max(t2s + 1, T - tail_len)


def build_traj(dgi, t2s, t_conv, P_offset, grip_conv, k_off, k_interp, k_fixed):
    """A' verbatim transport ++ divert-to-offset ++ one-way converge ++ settle ++ C verbatim insertion."""
    tp = dgi["target_pose"]; grip = dgi["gripper_action"]; T = tp.shape[0]
    t_branch = max(1, t_conv - k_interp)
    P_conv = tp[t_conv]
    traj = WaypointTrajectory()
    # A' verbatim transport [0:t_branch]
    traj.add_waypoint_sequence(WaypointSequence.from_poses(tp[0:t_branch], grip[0:t_branch], 0.0))
    # divert to the sampled offset (mimicgen interpolates from last pose)
    traj.add_waypoint_sequence_for_target_pose(pose=P_offset, gripper_action=grip_conv,
                                               num_steps=k_off, action_noise=0.0)
    # ONE-WAY converge to the demo's convergence pose (the SART core)
    traj.add_waypoint_sequence_for_target_pose(pose=P_conv, gripper_action=grip_conv,
                                               num_steps=k_interp, action_noise=0.0)
    # settle at convergence so OSC actually reaches it
    traj.add_waypoint_sequence(WaypointSequence.from_poses(
        np.repeat(P_conv[None], k_fixed, axis=0), np.repeat(grip_conv[None], k_fixed, axis=0), 0.0))
    # C verbatim insertion [t_conv:T]
    traj.add_waypoint_sequence(WaypointSequence.from_poses(tp[t_conv:T], grip[t_conv:T], 0.0))
    return traj


def load_dgi(g):
    d = g["datagen_info"]
    return {
        "target_pose": d["target_pose"][:], "eef_pose": d["eef_pose"][:],
        "gripper_action": d["gripper_action"][:],
        "object_poses": {k: d["object_poses"][k][:] for k in d["object_poses"].keys()},
        "subtask_term_signals": {k: d["subtask_term_signals"][k][:] for k in d["subtask_term_signals"].keys()},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="datasets/generated/square_D0_poc/demo.hdf5")
    ap.add_argument("--out", default="datasets/generated/square_sart")
    ap.add_argument("--n_sources", type=int, default=10)
    ap.add_argument("--n_per", type=int, default=5)
    ap.add_argument("--radius", type=float, default=0.05)
    ap.add_argument("--rot_deg", type=float, default=10.0)
    ap.add_argument("--position_fix", action="store_true")
    ap.add_argument("--k_off", type=int, default=10)
    ap.add_argument("--k_interp", type=int, default=20)
    ap.add_argument("--k_fixed", type=int, default=5)
    ap.add_argument("--tail_len", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keep_failed", action="store_true")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    env_meta = get_env_metadata_from_dataset(dataset_path=args.source)
    env = RobomimicUtils.create_env(
        env_meta=env_meta, env_name=env_meta["env_name"],
        camera_names=[], camera_height=84, camera_width=84,
        render=False, render_offscreen=False, use_image_obs=False, use_depth_obs=False,
    )
    ei = make_interface(name="MG_Square", interface_type="robosuite", env=env.base_env)

    f = h5py.File(args.source, "r"); data = f["data"]
    demo_keys = sorted(data.keys(), key=lambda x: int(x.split("_")[1]))[: args.n_sources]

    succ_dir = args.out; fail_dir = args.out + "_failed"
    os.makedirs(succ_dir, exist_ok=True); os.makedirs(fail_dir, exist_ok=True)

    n_succ = n_att = 0
    approach_paths = []  # for diversity metric
    for dk in demo_keys:
        g = data[dk]; dgi = load_dgi(g)
        T = dgi["target_pose"].shape[0]
        gr = dgi["subtask_term_signals"]["grasp"].reshape(-1)
        t2s = int(np.argmax(gr > 0.5)) if gr.max() > 0.5 else 0
        t_conv = find_t_conv(dgi["eef_pose"], dgi["object_poses"], t2s, T, args.tail_len)
        grip_conv = dgi["gripper_action"][t_conv]
        model_xml = g.attrs["model_file"]
        if isinstance(model_xml, bytes):
            model_xml = model_xml.decode()
        state0 = g["states"][0][:]
        for j in range(args.n_per):
            n_att += 1
            P_offset = sample_offset(
                dgi["target_pose"][t_conv],
                dgi["object_poses"]["square_peg"][t_conv][:3, 3],
                args.radius, np.deg2rad(args.rot_deg), rng, args.position_fix)
            env.reset()
            env.reset_to({"model": model_xml, "states": state0})
            init_state = env.get_state()
            traj = build_traj(dgi, t2s, t_conv, P_offset, grip_conv,
                              args.k_off, args.k_interp, args.k_fixed)
            res = traj.execute(env=env, env_interface=ei)
            if len(res["states"]) == 0:
                print("[%d] src=%s j=%d EMPTY-exec" % (n_att, dk, j), flush=True); continue
            s = res["success"]; ok = bool(s["task"]) if isinstance(s, dict) else bool(s)
            sidx = int(dk.split("_")[1])
            if ok or args.keep_failed:
                MG_FileUtils.write_demo_to_hdf5(
                    folder=(succ_dir if ok else fail_dir), env=env, initial_state=init_state,
                    states=res["states"], observations=res["observations"],
                    datagen_info=res["datagen_infos"], actions=res["actions"],
                    src_demo_inds=[sidx],
                    src_demo_labels=sidx * np.ones((res["actions"].shape[0], 1), dtype=int))
            if ok:
                n_succ += 1
                approach_paths.append(P_offset[:3, 3].copy())
            print("[%d] src=%s j=%d success=%s  DGR=%.1f%%" %
                  (n_att, dk, j, ok, 100.0 * n_succ / n_att), flush=True)

    out_hdf5 = args.out + ".hdf5"
    n_merged = MG_FileUtils.merge_all_hdf5(folder=succ_dir, new_hdf5_path=out_hdf5, delete_folder=False)
    div = float(np.std(np.array(approach_paths), axis=0).mean()) if len(approach_paths) > 1 else 0.0
    stats = {"num_attempts": n_att, "num_success": n_succ,
             "dgr_pct": round(100.0 * n_succ / max(n_att, 1), 1),
             "n_merged": int(n_merged), "offset_pos_std_m": round(div, 4),
             "sources": args.n_sources, "n_per": args.n_per,
             "radius": args.radius, "rot_deg": args.rot_deg, "position_fix": args.position_fix}
    json.dump(stats, open(args.out + "_stats.json", "w"), indent=2)
    print("SART_DONE " + json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
