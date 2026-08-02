"""Isolation experiment: does MY datagen_info synthesis break generation?

Copy the human fwd_annotated dataset unchanged (states/actions/obs), but
REPLACE obs/datagen_info with the same synthesis rl_to_lab.py performs —
signals from state predicates, 4x4 poses from obs eef/cube tracks. Generating
from this file at a known-good baseline (~16% with the stock annotation)
cleanly splits writer bugs from RL-content differences.

Pure h5py/numpy — runs on the host: python3 identity_test.py <in> <out>
"""
from __future__ import annotations

import math
import sys

import h5py
import numpy as np


def rot_wxyz(quat):
    w, x, y, z = np.asarray(quat, dtype=np.float64)
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def pose44(pos, quat):
    mat = np.eye(4, dtype=np.float32)
    mat[:3, :3] = rot_wxyz(quat)
    mat[:3, 3] = pos
    return mat


def main():
    src_path, dst_path = sys.argv[1], sys.argv[2]
    src = h5py.File(src_path, "r")
    dst = h5py.File(dst_path, "w")
    data = dst.create_group("data")
    for key, value in src["data"].attrs.items():
        data.attrs[key] = value

    for name in src["data"].keys():
        group = src[f"data/{name}"]
        src.copy(f"data/{name}", data, name=name)
        episode = data[name]
        obs = episode["obs"]

        # quat order self-check: original datagen eef rotation vs obs eef_quat
        orig = group["obs/datagen_info/eef_pose/franka"][0][:3, :3]
        quat0 = group["obs/eef_quat"][0]
        as_wxyz = rot_wxyz(quat0)
        as_xyzw = rot_wxyz([quat0[3], quat0[0], quat0[1], quat0[2]])
        use_wxyz = (np.abs(as_wxyz - orig).max()
                    <= np.abs(as_xyzw - orig).max())
        order = "wxyz" if use_wxyz else "xyzw"

        def to_wxyz(quat):
            return quat if use_wxyz else [quat[3], quat[0], quat[1], quat[2]]

        ee_pos = obs["eef_pos"][()]
        ee_quat = obs["eef_quat"][()]
        cube_pos = obs["cube_positions"][()].reshape(-1, 3, 3)
        cube_quat = obs["cube_orientations"][()].reshape(-1, 3, 4)
        fingers = np.abs(obs["gripper_pos"][()]).mean(axis=1)
        T = len(ee_pos)

        closed = fingers < 0.03
        released = fingers > 0.03
        near = lambda c: np.linalg.norm(  # noqa: E731
            ee_pos - cube_pos[:, c], axis=1) < 0.10
        stacked = ((np.linalg.norm(cube_pos[:, 1, :2] - cube_pos[:, 0, :2],
                                   axis=1) < 0.04)
                   & (cube_pos[:, 1, 2] - cube_pos[:, 0, 2] > 0.035)
                   & (cube_pos[:, 1, 2] - cube_pos[:, 0, 2] < 0.065)
                   & released)
        grasp_1 = closed & near(1)
        stack_1 = stacked
        grasp_2 = closed & near(2) & stack_1.cumsum().astype(bool)

        del episode["obs/datagen_info"]
        info = episode["obs"].create_group("datagen_info")
        eef44 = np.stack([pose44(ee_pos[k], to_wxyz(ee_quat[k]))
                          for k in range(T)])
        info.create_dataset("eef_pose/franka", data=eef44)
        info.create_dataset("target_eef_pose/franka",
                            data=np.concatenate([eef44[1:], eef44[-1:]]))
        for c in (1, 2, 3):
            info.create_dataset(
                f"object_pose/cube_{c}",
                data=np.stack([pose44(cube_pos[k, c - 1],
                                      to_wxyz(cube_quat[k, c - 1]))
                               for k in range(T)]))
        info.create_dataset("subtask_term_signals/grasp_1", data=grasp_1)
        info.create_dataset("subtask_term_signals/stack_1", data=stack_1)
        info.create_dataset("subtask_term_signals/grasp_2", data=grasp_2)

        def first_true(mask):
            hits = np.flatnonzero(mask)
            return int(hits[0]) if len(hits) else -1
        print(f"{name}: quat_order={order} grasp_1={first_true(grasp_1)} "
              f"stack_1={first_true(stack_1)} grasp_2={first_true(grasp_2)} "
              f"T={T}")
    src.close()
    dst.close()
    print("wrote", dst_path)


if __name__ == "__main__":
    main()
