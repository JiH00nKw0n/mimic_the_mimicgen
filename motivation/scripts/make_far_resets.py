"""Build a DIAGNOSTIC frozen-reset set that oversamples the filter-depleted
corners, by rejection sampling env resets into three strata:
  tail  (80): d_eval (min over sources of mean-object distance) in the top
              decile of the uniform-reset distribution — the far tail the
              original 200-set barely covers,
  hard  (80): the tolerance-critical object's min-over-sources displacement in
              its top quintile (machine-far corner) regardless of total d,
  base  (40): plain uniform draws (calibration overlap with the original set).
Each group stores stratum + measured distances as attrs. Output:
e2_arms/<task>_N2/frozen_resets_diag.hdf5. Then evaluate arms on it with the
usual evaluate_policy_on_frozen_resets.
"""
import json, math, sys
from pathlib import Path

REPO = "/home/ubuntu/mimicgen_jihoonkwon/mimic_the_mimicgen/motivation"
CFG = f"{REPO}/configs/tasks"
SRC = "/home/ubuntu/mimicgen_jihoonkwon/robosuite_mimicgen/mimicgen/datasets/source"
A = Path("/home/ubuntu/mimicgen_jihoonkwon/experiments/motivation_new/e2_arms")
HARD = {"coffee": "coffee_machine", "threading": "tripod", "stack": "cubeA",
        "three_piece_assembly": "base", "mug_cleanup": "drawer", "stack_three": "cubeB"}
QUOTA = {"tail": 80, "hard": 80, "base": 40}


def run(task):
    sys.path.insert(0, REPO)
    import h5py
    import numpy as np
    import robomimic.utils.env_utils as EnvUtils
    import robomimic.utils.file_utils as FileUtils
    import robomimic.utils.obs_utils as ObsUtils
    from mimicgen.env_interfaces.base import make_interface
    from genaudit.config import load_task_spec
    from genaudit.envs.bounds_new import NEW_BOUNDS
    from genaudit.evaluation.frozen_resets import _register_variants
    from genaudit.factors.initial_condition import build_task_geometry
    from genaudit.records.extract import _pose_to_xy_yaw, load_source_initial_states

    spec = load_task_spec(f"{CFG}/{task}.yaml")
    geom = build_task_geometry(task, NEW_BOUNDS[task]["N2"], spec.symmetry_orders)
    mov = list(geom.movable_objects)
    L = dict(geom.normalizers_m)
    src_xy, _ = load_source_initial_states(f"{SRC}/{task}.hdf5", mov)
    hard_obj = HARD[task]
    _register_variants()
    ObsUtils.initialize_obs_utils_with_obs_specs({"obs": {"low_dim": ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"], "rgb": []}})
    train = A / f"{task}_N2" / "train.hdf5"
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=str(train))
    env = EnvUtils.create_env_from_metadata(env_meta=env_meta, render=False, render_offscreen=False)
    itf = make_interface(spec.env_interface, "robosuite", env.env)
    np.random.seed(2026)

    # phase 1: calibrate stratum thresholds from 400 uniform probes
    def measure():
        poses = itf.get_object_poses()
        cur = {}
        for m in mov:
            x, y, _ = _pose_to_xy_yaw(poses[m])
            cur[m] = (x, y)
        per_src = []
        for j in range(len(src_xy)):
            ds = [math.hypot(cur[m][0] - src_xy[j][m][0], cur[m][1] - src_xy[j][m][1]) / L[m] for m in mov]
            per_src.append(sum(ds) / len(ds))
        d_eval = min(per_src)
        hard_d = min(math.hypot(cur[hard_obj][0] - src_xy[j][hard_obj][0],
                                cur[hard_obj][1] - src_xy[j][hard_obj][1]) / L[hard_obj]
                     for j in range(len(src_xy)))
        return d_eval, hard_d

    probe = []
    for _ in range(400):
        env.reset()
        probe.append(measure())
    tail_cut = sorted(x[0] for x in probe)[int(400 * 0.9)]
    hard_cut = sorted(x[1] for x in probe)[int(400 * 0.8)]
    print(f"{task}: tail_cut(d_eval p90)={tail_cut:.3f} hard_cut(hard_d p80)={hard_cut:.3f}", flush=True)

    out = A / f"{task}_N2" / "frozen_resets_diag.hdf5"
    got = {k: 0 for k in QUOTA}
    idx = 0
    with h5py.File(out, "w") as handle:
        handle.attrs["dataset_path"] = str(train)
        handle.attrs["seed"] = 2026
        handle.attrs["env_meta"] = json.dumps(env_meta)
        handle.attrs["tail_cut"] = tail_cut
        handle.attrs["hard_cut"] = hard_cut
        tries = 0
        while any(got[k] < QUOTA[k] for k in QUOTA) and tries < 30000:
            tries += 1
            env.reset()
            d_eval, hard_d = measure()
            if d_eval >= tail_cut and got["tail"] < QUOTA["tail"]:
                stratum = "tail"
            elif hard_d >= hard_cut and got["hard"] < QUOTA["hard"]:
                stratum = "hard"
            elif got["base"] < QUOTA["base"]:
                stratum = "base"
            else:
                continue
            state = env.get_state()
            g = handle.create_group(f"reset_{idx}")
            g.create_dataset("states", data=state["states"])
            g.attrs["model"] = state["model"]
            g.attrs["stratum"] = stratum
            g.attrs["d_eval"] = d_eval
            g.attrs["hard_d"] = hard_d
            got[stratum] += 1
            idx += 1
        print(f"{task}: {idx} resets ({got}) in {tries} tries -> {out}", flush=True)


if __name__ == "__main__":
    for t in sys.argv[1:]:
        try:
            run(t)
        except Exception as e:  # noqa: BLE001
            import traceback
            print(f"{t}: ERR {e}")
            traceback.print_exc()
