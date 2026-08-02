"""Temporally densify a lab-mimic source file by an integer factor.

Why: RL teacher demos move ~3.5x faster per frame than human demos
(mean EE step 14.5 mm vs 4.2 mm). The data generator executes one source
waypoint per env step, so the transformed plan inherits that speed and the
computed IK-rel actions saturate the [-1, 1] clip (p95 0.39, max 1.0 vs
human p95 0.13) — the arm undershoots every segment and grasps miss.
Inserting interpolated frames divides the per-step delta by the factor
without touching the contract or the generation protocol.

Only per-frame tracks are resampled: positions/joints linearly, quaternions
and 4x4 rotations by slerp, boolean subtask signals by hold (value at the
floor source index). target_eef_pose is rebuilt as the next densified eef
pose rather than interpolated, preserving the "target = next frame" rule.
initial_state is copied unchanged; actions are rebuilt with pos/rot deltas
divided by the factor (generation does not consume them — length
consistency only).

Pure h5py/numpy — runs on the host:
  python3 densify_source.py <in.hdf5> <out.hdf5> [factor=2]
"""
from __future__ import annotations

import json
import math
import sys

import h5py
import numpy as np


def slerp_track(quats, lo, hi, frac):
    """Interpolate a (T,4) wxyz quaternion track at lo->hi with weight frac."""
    q0, q1 = quats[lo], quats[hi]
    out = np.empty((len(lo), 4))
    for i in range(len(lo)):
        a, b = q0[i].astype(np.float64), q1[i].astype(np.float64)
        dot = float(np.dot(a, b))
        if dot < 0.0:
            b, dot = -b, -dot
        if dot > 0.9995:
            q = a + frac[i] * (b - a)
        else:
            theta = math.acos(min(1.0, dot))
            q = (math.sin((1 - frac[i]) * theta) * a
                 + math.sin(frac[i] * theta) * b) / math.sin(theta)
        out[i] = q / np.linalg.norm(q)
    return out


def rot_to_quat(mats):
    """(T,3,3) -> (T,4) wxyz."""
    out = np.empty((len(mats), 4))
    for i, m in enumerate(mats.astype(np.float64)):
        tr = m[0, 0] + m[1, 1] + m[2, 2]
        if tr > 0:
            s = math.sqrt(tr + 1.0) * 2
            out[i] = [0.25 * s, (m[2, 1] - m[1, 2]) / s,
                      (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
        else:
            j = int(np.argmax([m[0, 0], m[1, 1], m[2, 2]]))
            k, l = (j + 1) % 3, (j + 2) % 3
            s = math.sqrt(1.0 + m[j, j] - m[k, k] - m[l, l]) * 2
            q = np.empty(4)
            q[0] = (m[l, k] - m[k, l]) / s
            q[1 + j] = 0.25 * s
            q[1 + k] = (m[k, j] + m[j, k]) / s
            q[1 + l] = (m[l, j] + m[j, l]) / s
            out[i] = q
    return out


def quat_to_rot(quats):
    """(T,4) wxyz -> (T,3,3)."""
    out = np.empty((len(quats), 3, 3))
    for i, (w, x, y, z) in enumerate(quats.astype(np.float64)):
        n = math.sqrt(w * w + x * x + y * y + z * z)
        w, x, y, z = w / n, x / n, y / n, z / n
        out[i] = [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    return out


def grid(T, factor):
    """Dense fractional indices over 0..T-1: length (T-1)*factor+1."""
    pos = np.arange((T - 1) * factor + 1, dtype=np.float64) / factor
    lo = np.minimum(pos.astype(int), T - 2 if T > 1 else 0)
    frac = pos - lo
    return lo, lo + 1 if T > 1 else lo, frac


def densify_pose44(track, lo, hi, frac):
    trans = track[:, :3, 3]
    dense_t = trans[lo] + frac[:, None] * (trans[hi] - trans[lo])
    quats = rot_to_quat(track[:, :3, :3])
    dense_q = slerp_track(quats, lo, hi, frac)
    out = np.tile(np.eye(4, dtype=np.float32), (len(lo), 1, 1))
    out[:, :3, :3] = quat_to_rot(dense_q).astype(np.float32)
    out[:, :3, 3] = dense_t.astype(np.float32)
    return out


def main():
    src_path, dst_path = sys.argv[1], sys.argv[2]
    factor = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    src = h5py.File(src_path, "r")
    dst = h5py.File(dst_path, "w")
    data = dst.create_group("data")
    for key, value in src["data"].attrs.items():
        data.attrs[key] = value

    total = 0
    for name in src["data"].keys():
        g = src[f"data/{name}"]
        T = int(g.attrs["num_samples"])
        lo, hi, frac = grid(T, factor)
        new_T = len(lo)
        ep = data.create_group(name)
        for key, value in g.attrs.items():
            ep.attrs[key] = value
        ep.attrs["num_samples"] = new_T
        ep.attrs["densify_factor"] = factor
        src.copy(g["initial_state"], ep, name="initial_state")

        def interp(arr):
            return (arr[lo] + frac.reshape([-1] + [1] * (arr.ndim - 1))
                    * (arr[hi] - arr[lo])).astype(arr.dtype)

        obs_in, obs = g["obs"], ep.create_group("obs")
        for key in ("eef_pos", "cube_positions", "gripper_pos",
                    "joint_pos", "joint_vel"):
            if key in obs_in:
                obs.create_dataset(key, data=interp(obs_in[key][()]))
        if "eef_quat" in obs_in:
            obs.create_dataset("eef_quat", data=slerp_track(
                obs_in["eef_quat"][()], lo, hi, frac).astype(np.float32))
        if "cube_orientations" in obs_in:
            q = obs_in["cube_orientations"][()].reshape(T, 3, 4)
            dense = np.stack([slerp_track(q[:, c], lo, hi, frac)
                              for c in range(3)], axis=1)
            obs.create_dataset("cube_orientations",
                               data=dense.reshape(new_T, 12).astype(np.float32))

        info_in = obs_in["datagen_info"]
        info = obs.create_group("datagen_info")
        eef_dense = densify_pose44(info_in["eef_pose/franka"][()], lo, hi, frac)
        info.create_dataset("eef_pose/franka", data=eef_dense)
        info.create_dataset("target_eef_pose/franka",
                            data=np.concatenate([eef_dense[1:], eef_dense[-1:]]))
        for c in (1, 2, 3):
            info.create_dataset(
                f"object_pose/cube_{c}",
                data=densify_pose44(info_in[f"object_pose/cube_{c}"][()],
                                    lo, hi, frac))
        for key in info_in["subtask_term_signals"]:
            sig = info_in[f"subtask_term_signals/{key}"][()]
            # hold the floor frame's value; frac==1 only at the final grid
            # point, where the last source frame must be preserved
            idx_sig = np.minimum(lo + (frac >= 0.999).astype(int), T - 1)
            info.create_dataset(f"subtask_term_signals/{key}",
                                data=sig[idx_sig])

        acts = g["actions"][()]
        idx = np.minimum(lo, len(acts) - 1)
        dense_a = acts[idx].copy()
        dense_a[:, :6] /= factor
        ep.create_dataset("actions", data=dense_a)

        if "states" in g:
            st_in, st = g["states"], ep.create_group("states")
            Ts = len(st_in["articulation/robot/joint_position"])
            slo, shi, sfrac = grid(Ts, factor)

            def s_interp(arr):
                return (arr[slo] + sfrac.reshape([-1] + [1] * (arr.ndim - 1))
                        * (arr[shi] - arr[slo])).astype(arr.dtype)

            def walk(gin, gout):
                for key in gin:
                    if isinstance(gin[key], h5py.Group):
                        walk(gin[key], gout.create_group(key))
                    elif key.endswith("root_pose"):
                        arr = gin[key][()]
                        dense = np.concatenate([
                            s_interp(arr[:, :3]),
                            slerp_track(arr[:, 3:7], slo, shi,
                                        sfrac).astype(np.float32)], axis=1)
                        gout.create_dataset(key, data=dense)
                    else:
                        gout.create_dataset(key, data=s_interp(gin[key][()]))
            walk(st_in, st)

        total += new_T
        step = np.linalg.norm(np.diff(eef_dense[:, :3, 3], axis=0), axis=1)
        print(f"{name}: T {T}->{new_T} eef_step mean "
              f"{step.mean() * 1000:.1f}mm p95 "
              f"{np.percentile(step, 95) * 1000:.1f}mm")

    data.attrs["total"] = total
    env_args = json.loads(data.attrs["env_args"]) if "env_args" in data.attrs \
        else {}
    env_args.setdefault("env_kwargs", {})["densify_factor"] = factor
    data.attrs["env_args"] = json.dumps(env_args)
    src.close()
    dst.close()
    print("wrote", dst_path)


if __name__ == "__main__":
    main()
