"""Visual-DR render validation for the stage2 ablation (OmniReset Table 2).

Visual randomization cannot change DGR in low-dim generation, so it is a
render-time validation, not an arm: for P1_nominal and P3_robust, replay a few
successful demos with their OWN stored physics model (model_file attr) and
randomized camera/lighting/texture per clip:

  camera  agentview pos +-5 cm, rot +-2 deg, fovy +-2   (Table 2 Visual/Camera;
          fovy stands in for focal length — MuJoCo has no focal parameter)
  texture/color, lighting via robosuite TextureModder / LightingModder
          (HDRI rows approximated — MuJoCo has no HDRI environment maps)

Output: ~/stage2_render_dr/<task>_<arm>/*.mp4 + manifest.json. Run niced.

  PYTHONPATH=$M MUJOCO_GL=egl nice -n 10 $V scripts/stage2_render_dr.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = "/home/ubuntu/mimicgen_jihoonkwon/mimic_the_mimicgen/motivation"
GEN = Path("/home/ubuntu/mimicgen_jihoonkwon/experiments/stage2_ablation/gen")
OUT = Path("/home/ubuntu/stage2_render_dr")
TASKS = ["stack", "square"]
RENDER_ARMS = ["P1_nominal", "P3_robust"]
CLIPS = 6
RES = 256
SEED0 = 20260801

sys.path.insert(0, REPO)
import h5py  # noqa: E402
import imageio  # noqa: E402
import numpy as np  # noqa: E402
import robomimic.utils.env_utils as EnvUtils  # noqa: E402
import robomimic.utils.file_utils as FileUtils  # noqa: E402
import robomimic.utils.obs_utils as ObsUtils  # noqa: E402
from robosuite.utils.mjmod import CameraModder, LightingModder, TextureModder  # noqa: E402

from genaudit.envs.robosuite_variants import (  # noqa: E402
    register_custom_variants, register_new_variants,
)
from genaudit.envs.physics_variants import register_physics_variant  # noqa: E402

register_custom_variants()
register_new_variants()
ObsUtils.initialize_obs_utils_with_obs_specs({"obs": {"low_dim": [
    "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"],
    "rgb": []}})


def render_clip(env, group, path: Path, seed: int, visual_dr: bool) -> None:
    states = group["states"][()]
    model = group.attrs["model_file"]
    env.reset_to({"model": model, "states": states[0]})
    if visual_dr:
        rng = np.random.RandomState(seed)
        sim = env.base_env.sim
        camera = CameraModder(
            sim, camera_names=["agentview"], random_state=rng,
            position_perturbation_size=0.05,          # +-5 cm  (Table 2)
            rotation_perturbation_size=np.deg2rad(2),  # +-2 deg
            fovy_perturbation_size=2.0)                # focal proxy
        texture = TextureModder(sim, random_state=rng)
        lighting = LightingModder(sim, random_state=rng)
        for modder in (camera, texture, lighting):
            modder.randomize()
    frames = []
    for t in range(0, len(states), 4):
        env.reset_to({"states": states[t]})
        frames.append(env.render(mode="rgb_array", height=RES, width=RES,
                                 camera_name="agentview"))
    imageio.mimsave(str(path), frames, fps=12, macro_block_size=1)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    manifest: dict = {"clips": []}
    for task in TASKS:
        for arm in RENDER_ARMS:
            pool = GEN / f"{task}_{arm}" / "demo.hdf5"
            if not pool.exists():
                print(f"skip {task}_{arm}: no merged demo.hdf5")
                continue
            # physics classes referenced by env_meta must exist before playback
            profile = "nominal" if arm == "P1_nominal" else "robust"
            suffix = "s2n" if arm == "P1_nominal" else "s2r"
            register_physics_variant(
                task=task, profile=profile, suffix=suffix, seed=0,
                contract_dir="/home/ubuntu/stage2_contact_calibration_v2")
            clip_dir = OUT / f"{task}_{arm}"
            clip_dir.mkdir(exist_ok=True)
            env_meta = FileUtils.get_env_metadata_from_dataset(str(pool))
            env = EnvUtils.create_env_from_metadata(
                env_meta=env_meta, render=False, render_offscreen=True)
            with h5py.File(pool, "r") as handle:
                names = list(handle["data"].keys())[:CLIPS]
                for index, name in enumerate(names):
                    group = handle[f"data/{name}"]
                    visual_dr = index > 0  # clip 0 = no-DR reference
                    tag = "dr" if visual_dr else "ref"
                    path = clip_dir / f"{name}_{tag}.mp4"
                    try:
                        render_clip(env, group, path,
                                    seed=SEED0 + index, visual_dr=visual_dr)
                        manifest["clips"].append(
                            {"task": task, "arm": arm, "demo": name,
                             "visual_dr": visual_dr, "path": str(path)})
                        print(f"rendered {path}", flush=True)
                    except Exception as error:  # noqa: BLE001
                        print(f"ERR {path}: {type(error).__name__}: {error}",
                              flush=True)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("RENDER DONE", flush=True)


if __name__ == "__main__":
    main()
