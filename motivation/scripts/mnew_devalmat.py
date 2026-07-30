"""P0 gate: full (eval x source) d_pos distance matrix per task.

Like mnew_deval but saves distance from each of the 200 frozen eval states to
EVERY source demo (not just the min), so d_own = min over an ARM's training
sources can be computed, and coverage of far eval by a source subset checked.
Saves deval_matrix.json = {task, n_src, matrix:{reset_index:[d_pos to src j]}}.
"""
import json, sys
from pathlib import Path
REPO = "/home/ubuntu/mimicgen_jihoonkwon/mimic_the_mimicgen/motivation"
CFG = f"{REPO}/configs/tasks"
SRC = "/home/ubuntu/mimicgen_jihoonkwon/robosuite_mimicgen/mimicgen/datasets/source"
A = Path("/home/ubuntu/mimicgen_jihoonkwon/experiments/motivation_new/e2_arms")

def run(task):
    sys.path.insert(0, REPO)
    import h5py, numpy as np
    import robomimic.utils.env_utils as EnvUtils
    import robomimic.utils.obs_utils as ObsUtils
    from mimicgen.env_interfaces.base import make_interface
    from genaudit.config import load_task_spec
    from genaudit.envs.bounds_new import NEW_BOUNDS
    from genaudit.evaluation.frozen_resets import _register_variants
    from genaudit.factors.initial_condition import build_task_geometry, transform_distances
    from genaudit.records.extract import _pose_to_xy_yaw, load_source_initial_states
    spec = load_task_spec(f"{CFG}/{task}.yaml")
    geom = build_task_geometry(task, NEW_BOUNDS[task]["N2"], spec.symmetry_orders)
    mov = list(geom.movable_objects)
    src_xy, src_yaw = load_source_initial_states(f"{SRC}/{task}.hdf5", mov)
    _register_variants()
    ObsUtils.initialize_obs_utils_with_obs_specs({"obs":{"low_dim":["robot0_eef_pos","robot0_eef_quat","robot0_gripper_qpos","object"],"rgb":[]}})
    rp = A/f"{task}_N2"/"frozen_resets.hdf5"
    with h5py.File(rp,"r") as h: env_meta=json.loads(h.attrs["env_meta"])
    env = EnvUtils.create_env_from_metadata(env_meta=env_meta, render=False, render_offscreen=False)
    itf = make_interface(spec.env_interface, "robosuite", env.env)
    mat={}
    with h5py.File(rp,"r") as h:
        for name in sorted(h.keys(), key=lambda n:int(n.split("_")[1])):
            g=h[name]; env.reset(); env.reset_to({"model":g.attrs["model"],"states":g["states"][()]})
            poses=itf.get_object_poses(); nx,ny={},{}
            for m in mov:
                x,y,yaw=_pose_to_xy_yaw(poses[m]); nx[m]=(x,y); ny[m]=yaw
            row=[round(transform_distances(geom,nx,src_xy[j],ny,src_yaw[j]).d_pos,4) for j in range(len(src_xy))]
            mat[int(name.split("_")[1])]=row
    out=A/f"{task}_N2"/"deval_matrix.json"
    out.write_text(json.dumps({"task":task,"n_src":len(src_xy),"matrix":mat}))
    print(f"[devalmat] {task}: {len(mat)} states x {len(src_xy)} sources -> {out}", flush=True)

if __name__=="__main__":
    for t in sys.argv[1:]:
        try: run(t)
        except Exception as e:
            import traceback; print(f"{t}: ERR {e}"); traceback.print_exc()
