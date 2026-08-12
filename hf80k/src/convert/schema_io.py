"""Writer + validator for the contract HDF5 (fr3.cube.mimicgen.hdf5.v1).

Field list mirrors dataset_schema.yaml in this directory. Requires h5py+numpy
(present in every runtime we target); everything else is stdlib.
"""
from __future__ import annotations

import json
import math

CONTRACT_ID = "fr3_cube_stage1_model4500_legacyosc_v1"
SCHEMA_VERSION = "fr3.cube.mimicgen.hdf5.v1"
SAMPLE_HZ = 10.0


def write_episode(
    handle,
    demo_name: str,
    *,
    actions,                 # [T,7] raw unclipped
    processed_delta,         # [T,6] scaled cartesian delta (m / rad)
    commanded_target_pose,   # [T,7] x,y,z,qw,qx,qy,qz robot_base
    actual_ee_pose,          # [T,7]
    joint_position,          # [T,9]
    joint_velocity,          # [T,9]
    gripper_state,           # [T,2]
    cube_pose,               # [T,3,7] robot_base
    success: bool,
    source_human_demo_id: str,
    retarget_version: str,
    extras: dict | None = None,   # recommended datasets, e.g. source_target_pose
):
    import numpy as np

    T = len(actions)
    group = handle.require_group("data").create_group(demo_name)
    group.attrs["num_samples"] = T
    group.attrs["success"] = bool(success)
    group.attrs["source_human_demo_id"] = source_human_demo_id
    group.attrs["retarget_version"] = retarget_version
    group.create_dataset("actions", data=np.asarray(actions, dtype=np.float32))
    group.create_dataset("processed_cartesian_delta",
                         data=np.asarray(processed_delta, dtype=np.float32))
    group.create_dataset("commanded_target_pose",
                         data=np.asarray(commanded_target_pose, dtype=np.float32))
    group.create_dataset("actual_ee_pose",
                         data=np.asarray(actual_ee_pose, dtype=np.float32))
    group.create_dataset("joint_position",
                         data=np.asarray(joint_position, dtype=np.float32))
    group.create_dataset("joint_velocity",
                         data=np.asarray(joint_velocity, dtype=np.float32))
    group.create_dataset("gripper_state",
                         data=np.asarray(gripper_state, dtype=np.float32))
    group.create_dataset("cube_pose", data=np.asarray(cube_pose, dtype=np.float32))
    group.create_dataset(
        "timestamps",
        data=np.arange(T, dtype=np.float64) / SAMPLE_HZ)
    for key, value in (extras or {}).items():
        group.create_dataset(key, data=np.asarray(value, dtype=np.float32))
    return group


def finalize_file(handle, env_args: dict):
    data = handle.require_group("data")
    total = sum(int(data[k].attrs["num_samples"]) for k in data.keys())
    data.attrs["total"] = total
    data.attrs["contract_id"] = CONTRACT_ID
    data.attrs["env_args"] = json.dumps(env_args)
    data.attrs["schema_version"] = SCHEMA_VERSION


def validate_file(path: str) -> list[str]:
    """Return a list of violations (empty = passes the schema gate)."""
    import h5py
    import numpy as np

    required = {
        "actions": (2, 7), "processed_cartesian_delta": (2, 6),
        "commanded_target_pose": (2, 7), "actual_ee_pose": (2, 7),
        "joint_position": (2, 9), "joint_velocity": (2, 9),
        "gripper_state": (2, 2), "cube_pose": (3, 7), "timestamps": (1, None),
    }
    problems: list[str] = []
    with h5py.File(path, "r") as handle:
        data = handle.get("data")
        if data is None:
            return ["missing root group 'data'"]
        if data.attrs.get("contract_id", "") != CONTRACT_ID:
            problems.append("root contract_id missing/incorrect")
        if "env_args" not in data.attrs:
            problems.append("root env_args missing")
        for demo_name in data.keys():
            group = data[demo_name]
            for attr in ("num_samples", "success", "source_human_demo_id",
                         "retarget_version"):
                if attr not in group.attrs:
                    problems.append(f"{demo_name}: missing attr {attr}")
            lengths = set()
            for key, (ndim, last) in required.items():
                if key not in group:
                    problems.append(f"{demo_name}: missing dataset {key}")
                    continue
                shape = group[key].shape
                if len(shape) != ndim or (last is not None and shape[-1] != last):
                    problems.append(f"{demo_name}: {key} shape {shape}")
                if key == "cube_pose":
                    if shape[1:] != (3, 7):
                        problems.append(f"{demo_name}: cube_pose shape {shape}")
                lengths.add(shape[0])
            if len(lengths) > 1:
                problems.append(f"{demo_name}: unequal T across datasets {lengths}")
            if "timestamps" in group:
                ts = group["timestamps"][()]
                if not np.all(np.diff(ts) > 0):
                    problems.append(f"{demo_name}: timestamps not strictly increasing")
            for pose_key in ("commanded_target_pose", "actual_ee_pose"):
                if pose_key in group:
                    quat = group[pose_key][:, 3:7].astype(np.float64)
                    norms = np.linalg.norm(quat, axis=1)
                    if np.max(np.abs(norms - 1.0)) > 1e-4:
                        problems.append(
                            f"{demo_name}: {pose_key} quat norm off by "
                            f"{np.max(np.abs(norms - 1.0)):.2e}")
    return problems


if __name__ == "__main__":
    import sys

    issues = validate_file(sys.argv[1])
    for issue in issues:
        print("FAIL", issue)
    print("SCHEMA", "PASS" if not issues else f"{len(issues)} violations")
    sys.exit(0 if not issues else 1)
