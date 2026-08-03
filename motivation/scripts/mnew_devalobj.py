"""Per-object eval distance matrix: for each frozen reset and each source demo,
save EACH movable object's xy displacement separately (not the d_pos average).
Output deval_objmatrix.json = {task, objects, matrix: {reset: [[d_obj per obj] per src]}}.
Distances are normalized by the same per-object L_m used in d_pos so bands are
comparable with the generation-side analysis.
"""
import json, sys
from pathlib import Path
REPO = "/home/ubuntu/mimicgen_jihoonkwon/mimic_the_mimicgen/motivation"
CFG = f"{REPO}/configs/tasks"
SRC = "/home/ubuntu/mimicgen_jihoonkwon/robosuite_mimicgen/mimicgen/datasets/source"
A = Path("/home/ubuntu/mimicgen_jihoonkwon/experiments/motivation_new/e2_arms")


def run(task):
    sys.path.insert(0, REPO)
    import h5py, math
    import robomimic.utils.env_utils as EnvUtils
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
    _register_variants()
    ObsUtils.initialize_obs_utils_with_obs_specs({"obs": {"low_dim": ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"], "rgb": []}})
    rp = A / f"{task}_N2" / "frozen_resets.hdf5"
    with h5py.File(rp, "r") as h:
        env_meta = json.loads(h.attrs["env_meta"])
    env = EnvUtils.create_env_from_metadata(env_meta=env_meta, render=False, render_offscreen=False)
    itf = make_interface(spec.env_interface, "robosuite", env.env)
    mat = {}
    with h5py.File(rp, "r") as h:
        for name in sorted(h.keys(), key=lambda n: int(n.split("_")[1])):
            g = h[name]
            env.reset()
            env.reset_to({"model": g.attrs["model"], "states": g["states"][()]})
            poses = itf.get_object_poses()
            cur = {}
            for m in mov:
                x, y, _ = _pose_to_xy_yaw(poses[m])
                cur[m] = (x, y)
            rows = []
            for j in range(len(src_xy)):
                row = []
                for m in mov:
                    dx = cur[m][0] - src_xy[j][m][0]
                    dy = cur[m][1] - src_xy[j][m][1]
                    d = math.hypot(dx, dy)
                    if L.get(m):
                        d = d / L[m]
                    row.append(round(d, 4))
                rows.append(row)
            mat[int(name.split("_")[1])] = rows
    out = A / f"{task}_N2" / "deval_objmatrix.json"
    out.write_text(json.dumps({"task": task, "objects": mov, "norm": {m: L.get(m) for m in mov}, "matrix": mat}))
    print(f"[devalobj] {task}: {len(mat)} states, objects={mov}", flush=True)


if __name__ == "__main__":
    for t in sys.argv[1:]:
        try:
            run(t)
        except Exception as e:  # noqa: BLE001
            import traceback
            print(f"{t}: ERR {e}")
            traceback.print_exc()
