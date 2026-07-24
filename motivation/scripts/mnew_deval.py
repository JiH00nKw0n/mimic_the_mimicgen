"""motivation_new: per-frozen-reset d_eval extraction.

For each task's 200 frozen resets, reset the eval env to the stored mujoco
state, read object poses through the SAME mimicgen env interface generation
used (identical world frame as the source datagen_info), and compute
d_eval = min over source demos of d_pos (nearest_source_distance, PLAN §2.6).
Writes {task}_N2/deval.json = {reset_index: d_eval}. This is arm/seed
independent, so it runs once per task.

Usage: mnew_deval.py <task> [<task> ...]
"""
import json
import sys
from pathlib import Path

REPO = "/home/ubuntu/mimicgen_jihoonkwon/mimic_the_mimicgen/motivation"
CFG = f"{REPO}/configs/tasks"
SRC = "/home/ubuntu/mimicgen_jihoonkwon/robosuite_mimicgen/mimicgen/datasets/source"
A = Path("/home/ubuntu/mimicgen_jihoonkwon/experiments/motivation_new/e2_arms")


def deval_for_task(task):
    sys.path.insert(0, REPO)
    import h5py
    import numpy as np
    import robomimic.utils.env_utils as EnvUtils
    import robomimic.utils.obs_utils as ObsUtils
    from mimicgen.env_interfaces.base import make_interface

    from genaudit.config import load_task_spec
    from genaudit.envs.bounds_new import NEW_BOUNDS
    from genaudit.evaluation.frozen_resets import _register_variants
    from genaudit.factors.initial_condition import (
        build_task_geometry, nearest_source_distance,
    )
    from genaudit.records.extract import _pose_to_xy_yaw, load_source_initial_states

    spec = load_task_spec(f"{CFG}/{task}.yaml")
    geometry = build_task_geometry(task, NEW_BOUNDS[task]["N2"], spec.symmetry_orders)
    movable = list(geometry.movable_objects)
    src_xy, src_yaw = load_source_initial_states(f"{SRC}/{task}.hdf5", movable)

    _register_variants()
    ObsUtils.initialize_obs_utils_with_obs_specs(
        {"obs": {"low_dim": ["robot0_eef_pos", "robot0_eef_quat",
                             "robot0_gripper_qpos", "object"], "rgb": []}}
    )
    resets_path = A / f"{task}_N2" / "frozen_resets.hdf5"
    with h5py.File(resets_path, "r") as h:
        env_meta = json.loads(h.attrs["env_meta"])
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta, render=False, render_offscreen=False)
    rs_env = env.env  # underlying robosuite env
    interface = make_interface(spec.env_interface, "robosuite", rs_env)

    deval = {}
    nearest = {}
    with h5py.File(resets_path, "r") as h:
        names = sorted(h.keys(), key=lambda n: int(n.split("_")[1]))
        for name in names:
            g = h[name]
            env.reset()
            env.reset_to({"model": g.attrs["model"], "states": g["states"][()]})
            poses = interface.get_object_poses()
            new_xy, new_yaw = {}, {}
            for m in movable:
                x, y, yaw = _pose_to_xy_yaw(poses[m])
                new_xy[m] = (x, y)
                new_yaw[m] = yaw
            d, idx = nearest_source_distance(geometry, new_xy, new_yaw, src_xy, src_yaw)
            ri = int(name.split("_")[1])
            deval[ri] = float(d)
            nearest[ri] = int(idx)
    out = A / f"{task}_N2" / "deval.json"
    out.write_text(json.dumps({"task": task, "movable": movable,
                               "d_eval": deval, "nearest_source": nearest}, indent=2))
    vals = np.array(sorted(deval.values()))
    q = np.quantile(vals, [0, .33, .5, .67, 1.0])
    print(f"[deval] {task}: n={len(vals)} "
          f"min={q[0]:.3f} t33={q[1]:.3f} med={q[2]:.3f} t67={q[3]:.3f} max={q[4]:.3f}",
          flush=True)


def main():
    for task in sys.argv[1:]:
        try:
            deval_for_task(task)
        except Exception as e:  # noqa: BLE001
            import traceback
            print(f"[deval] {task}: ERR {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()


if __name__ == "__main__":
    main()
