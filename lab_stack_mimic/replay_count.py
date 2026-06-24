#!/usr/bin/env python3
"""Replay teleop demos in the LAB FR3 3-cube-stack sim and count how many succeed.

Why
---
The 29 seed demos were success-filtered offline from the *recorded* cube poses
(teleop/filter_success.py). That does NOT prove they reproduce a stack when their
recorded actions are replayed open-loop in the simulator (replay can drift). This
script does exactly that replay and re-judges success, so we know how many demos
are actually usable as MimicGen seeds.

What it does
------------
1. Rebuilds the exact env the teleop used: `Isaac-Stack-Cube-Franka-IK-Rel-v0`
   retargeted to the lab setup (FR3 arm at the lab desk, lab table USD,
   work-surface collision pad, 3x 50 mm cubes, IK-rel action). Same overrides as
   `aidas/3cube_stack/teleop/lab_teleop.py`.
2. For each demo: reset to the demo's recorded initial state, step the recorded
   actions one by one, then judge the final cube configuration with the
   order-agnostic "any 3-cube tower" criterion (success_criteria.py) AND the
   stricter canonical (cube_1<cube_2<cube_3) criterion.
3. Prints per-demo PASS/FAIL + the stacking order, and the totals.

Nothing is hard-coded: pass the dataset and (optionally) the table USD as args.

Run it (on the GPU server, in the UWLab env -- see run_replay.sh):
    python replay_count.py --dataset_file /path/to/teleop_dataset_success.hdf5
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

# --- CLI (paths are arguments, never hard-coded) ----------------------------
parser = argparse.ArgumentParser(description="Replay lab 3-cube demos and count successes.")
parser.add_argument("--dataset_file", required=True, help="HDF5 of teleop demos to replay.")
parser.add_argument(
    "--table_usd",
    default="/home/ubuntu/jake/aidas/3cube_stack/table_scene.usdc",
    help="Lab table USD used when the demos were recorded.",
)
parser.add_argument("--select_episodes", type=int, nargs="+", default=[], help="Replay only these demo indices.")
parser.add_argument("--settle_steps", type=int, default=0, help="Idle physics steps before judging (default 0).")
parser.add_argument("--report", default=None, help="Optional path to write a JSON summary.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True            # display-less server; we only need the numbers
args.enable_cameras = False
app = AppLauncher(args).app

# --- everything below runs after Isaac Sim has started -----------------------
import json

import torch
import gymnasium as gym

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
import isaaclab.envs.mdp as mdp
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.datasets import HDF5DatasetFileHandler
import isaaclab_tasks  # noqa: F401  (registers Isaac-Stack-Cube-Franka-* tasks)
from isaaclab_tasks.utils import parse_env_cfg

from success_criteria import tower_status

DEV = args.device
DESK_Z = 0.720
CUBE = 0.05
ROBOT_POS = (0.72, 0.138, 0.722)
ROBOT_ROT = (0.0, 0.0, 0.0, 1.0)
BASE_XY = (0.32, 0.138)

# ---- FR3 (IK teleop config: arm gravity disabled so it holds the IK target) ----
FR3_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/FrankaRobotics/FrankaFR3/fr3.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, max_depenetration_velocity=5.0),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=ROBOT_POS, rot=ROBOT_ROT,
        joint_pos={
            "fr3_joint1": 0.0, "fr3_joint2": -0.569, "fr3_joint3": 0.0, "fr3_joint4": -2.810,
            "fr3_joint5": 0.0, "fr3_joint6": 3.037, "fr3_joint7": 0.741, "fr3_finger_joint.*": 0.04,
        },
    ),
    actuators={
        "a1": ImplicitActuatorCfg(joint_names_expr=["fr3_joint[1-4]"], stiffness=400.0, damping=80.0),
        "a2": ImplicitActuatorCfg(joint_names_expr=["fr3_joint[5-7]"], stiffness=400.0, damping=80.0),
        "h": ImplicitActuatorCfg(
            joint_names_expr=["fr3_finger_joint.*"], effort_limit_sim=200.0, stiffness=2e3, damping=1e2
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)


def _cube_cfg(name, color, xy):
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/" + name,
        init_state=RigidObjectCfg.InitialStateCfg(pos=(xy[0], xy[1], DESK_Z + CUBE + 0.01), rot=(1, 0, 0, 0)),
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.05, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, max_depenetration_velocity=5.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
        ),
    )


def build_env_cfg():
    """Rebuild the lab teleop env config (mirrors aidas/3cube_stack/teleop/lab_teleop.py)."""
    cfg = parse_env_cfg("Isaac-Stack-Cube-Franka-IK-Rel-v0", device=DEV, num_envs=1)

    cfg.scene.table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0, 0), rot=(1, 0, 0, 0)),
        spawn=sim_utils.UsdFileCfg(usd_path=args.table_usd),
    )
    cfg.scene.work_surface = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/WorkSurface",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.20, BASE_XY[1], DESK_Z - 0.01)),
        spawn=sim_utils.CuboidCfg(
            size=(0.55, 0.6, 0.02),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.55, 0.58), opacity=0.0),
        ),
    )
    cfg.scene.robot = FR3_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    cfg.scene.cube_1 = _cube_cfg("Cube_1", (1.0, 0.0, 0.0), (BASE_XY[0], BASE_XY[1] - 0.10))
    cfg.scene.cube_2 = _cube_cfg("Cube_2", (0.0, 0.0, 1.0), (BASE_XY[0], BASE_XY[1]))
    cfg.scene.cube_3 = _cube_cfg("Cube_3", (1.0, 1.0, 0.0), (BASE_XY[0], BASE_XY[1] + 0.10))

    # retarget the IK-rel arm action + binary gripper to the FR3
    cfg.actions.arm_action = DifferentialInverseKinematicsActionCfg(
        asset_name="robot", joint_names=["fr3_joint.*"], body_name="fr3_hand", scale=1.0,
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=(0.0, 0.0, 0.1034)),
    )
    cfg.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot", joint_names=["fr3_finger_joint.*"],
        open_command_expr={"fr3_finger_joint.*": 0.04}, close_command_expr={"fr3_finger_joint.*": 0.0},
    )
    if hasattr(cfg, "gripper_joint_names"):
        cfg.gripper_joint_names = ["fr3_finger_joint.*"]
    for ev in ("init_franka_arm_pose", "randomize_franka_joint_state", "randomize_cube_positions"):
        if hasattr(cfg.events, ev):
            setattr(cfg.events, ev, None)
    if hasattr(cfg.scene, "ee_frame"):
        cfg.scene.ee_frame.prim_path = "{ENV_REGEX_NS}/Robot/fr3_link0"
        for fr in cfg.scene.ee_frame.target_frames:
            fr.prim_path = (
                fr.prim_path.replace("panda_hand", "fr3_hand")
                .replace("panda_rightfinger", "fr3_rightfinger")
                .replace("panda_leftfinger", "fr3_leftfinger")
            )

    # replay must run the full recorded action sequence: no early termination, no recording
    cfg.terminations = {}
    cfg.recorders = {}
    return cfg


def main() -> int:
    handler = HDF5DatasetFileHandler()
    handler.open(args.dataset_file)
    episode_names = list(handler.get_episode_names())
    total = len(episode_names)
    indices = args.select_episodes or list(range(total))
    print(f"[replay] dataset={args.dataset_file}  episodes={total}  replaying={len(indices)}")

    env = gym.make("Isaac-Stack-Cube-Franka-IK-Rel-v0", cfg=build_env_cfg(), render_mode=None).unwrapped
    finger_idx = [i for i, n in enumerate(env.scene["robot"].joint_names) if "finger" in n]
    print(f"[replay] action_space={env.action_space}  finger_joint_idx={finger_idx}")
    env.reset()

    results = []
    any_ok = canon_ok = 0
    with torch.inference_mode():
        for n in indices:
            name = episode_names[n]
            ep = handler.load_episode(name, env.device)
            env.reset_to(ep.get_initial_state(), torch.tensor([0], device=env.device), is_relative=True)
            print(f"  {name}: replaying ...", flush=True)
            steps = 0
            CAP = 3000  # safety: recorded demos are ~300-400 steps; break runaway loops
            while steps < CAP:
                act = ep.get_next_action()
                if act is None:
                    break
                env.step(act.unsqueeze(0) if act.ndim == 1 else act)
                steps += 1
                if steps % 100 == 0:
                    print(f"    {name}: {steps} steps ...", flush=True)
            else:
                print(f"  {name}: WARNING hit step cap {CAP} (get_next_action never None)", flush=True)
            for _ in range(args.settle_steps):
                env.sim.step(render=False)
                env.scene.update(env.sim.get_physics_dt())

            cubes = [env.scene[f"cube_{i}"].data.root_pos_w[0].tolist() for i in (1, 2, 3)]
            fingers = env.scene["robot"].data.joint_pos[0, finger_idx].tolist()
            st = tower_status(cubes, fingers, canonical=False)
            stc = tower_status(cubes, fingers, canonical=True)
            any_ok += int(st["ok"])
            canon_ok += int(stc["ok"])
            order_str = "->".join(f"c{o + 1}" for o in st["order"])  # bottom->top
            print(
                f"  {name:>9}: steps={steps:4d}  ANY={'PASS' if st['ok'] else 'fail'}  "
                f"CANON={'PASS' if stc['ok'] else 'fail'}  order(bottom->top)={order_str}  "
                f"gaps={st['gaps']}  released={st['released']}  xy_ok={st['xy_ok']}"
            )
            results.append({"demo": name, "steps": steps, "any_order": st, "canonical": stc["ok"]})

    n = len(indices)
    print("\n==================== REPLAY SUCCESS ====================")
    print(f"  any-order 3-tower : {any_ok}/{n}")
    print(f"  canonical c1<c2<c3: {canon_ok}/{n}")
    print("=======================================================")

    if args.report:
        with open(args.report, "w") as f:
            json.dump(
                {"dataset": args.dataset_file, "n": n, "any_order_success": any_ok,
                 "canonical_success": canon_ok, "results": results}, f, indent=2,
            )
        print(f"[replay] wrote {args.report}")

    env.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        app.close()
