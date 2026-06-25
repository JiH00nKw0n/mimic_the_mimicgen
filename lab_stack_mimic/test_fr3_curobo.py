#!/usr/bin/env python3
"""De-risk test: can cuRobo build a MotionGen for our FR3 config and plan one motion?

This is the single biggest risk in the SkillGen extension — cuRobo only ships a Franka
Panda config; we generated an FR3 cuRobo config (fr3_curobo.yml = franka.yml with
panda_->fr3_ + the FR3 URDF). This standalone script (no Isaac Sim) mirrors how
CuroboPlanner builds its planner, then asks for ONE plan, to confirm the FR3 kinematics
parse and the collision spheres attach before we build the rest of the SkillGen pipeline.

    python test_fr3_curobo.py
"""

import os
import sys
import torch

# warp 1.14 moved the torch interop to warp._src.torch and dropped the public `warp.torch`
# namespace that cuRobo 0.7.7 still expects (wp.torch.device_from_torch). Re-expose it.
import warp as wp  # noqa: E402
import warp._src.torch as _wp_torch  # noqa: E402
sys.modules.setdefault("warp.torch", _wp_torch)
wp.torch = _wp_torch

from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.state import JointState
from curobo.geom.types import WorldConfig
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.util_file import load_yaml
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig

HERE = os.path.dirname(os.path.abspath(__file__))
CFG = os.path.join(HERE, "fr3_curobo.yml")

tdt = TensorDeviceType()
print(f"[test] CUDA available: {torch.cuda.is_available()}")

robot_cfg = load_yaml(CFG)["robot_cfg"]
# use the inline (fr3-renamed) collision spheres; don't override with a franka sphere file
print(f"[test] robot base_link={robot_cfg['kinematics']['base_link']} "
      f"tool={robot_cfg['kinematics'].get('tool_frames')} "
      f"n_collision_links={len(robot_cfg['kinematics']['collision_link_names'])}")
print(f"[test] joints={robot_cfg['kinematics']['cspace']['joint_names']}")

# simple world: a flat table plane (like collision_table.yml) under the robot
world_cfg = WorldConfig.from_dict({
    "cuboid": {
        # keep the table clear of the FR3 base (base spheres sit at z~0); top at z=-0.2
        "table": {"dims": [2.0, 2.0, 0.1], "pose": [0.0, 0.0, -0.25, 1, 0, 0, 0]},
    }
})

print("[test] building MotionGenConfig.load_from_robot_config ...")
mg_cfg = MotionGenConfig.load_from_robot_config(
    robot_cfg,
    world_cfg,
    tensor_args=tdt,
    collision_checker_type=CollisionCheckerType.MESH,
    num_trajopt_seeds=12,
    num_graph_seeds=12,
    interpolation_dt=0.02,
)
mg = MotionGen(mg_cfg)
print("[test] warming up (this builds the FR3 kinematics + collision model) ...")
mg.warmup(enable_graph=True, warmup_js_trajopt=False)
print("[test] OK: MotionGen built + warmed up for FR3")

# Plan one motion: from the FR3 home pose to a small pose offset.
cspace = robot_cfg["kinematics"]["cspace"]
# cuRobo locks the fingers (lock_joints), so it plans over the ACTIVE arm joints only.
active = list(mg.kinematics.joint_names)
full_home = cspace.get("retract_config") or cspace.get("default_joint_position")
name2home = dict(zip(cspace["joint_names"], full_home))
home = [name2home[j] for j in active]
print(f"[test] active planning joints ({len(active)}): {active}")
q = torch.tensor([home], device=tdt.device, dtype=tdt.dtype)
start = JointState.from_position(q, joint_names=active)
# current ee pose via FK, then ask to move +5cm in x
fk = mg.compute_kinematics(start)
ee = fk.ee_pose
print(f"[test] FK home ee pos = {ee.position.tolist()}")
goal = Pose(ee.position + torch.tensor([[0.05, 0.0, 0.0]], device=tdt.device), ee.quaternion)
res = mg.plan_single(start, goal, MotionGenPlanConfig(max_attempts=3, enable_graph=True))
print(f"[test] plan_single success={bool(res.success.item())}  "
      f"status={getattr(res,'status',None)}  "
      f"steps={None if res.get_interpolated_plan() is None else res.get_interpolated_plan().position.shape}")
print("[test] DONE")
