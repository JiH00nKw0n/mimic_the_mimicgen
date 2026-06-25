"""Register the lab FR3 SkillGen task + teach cuRobo about the FR3.

Imported (after Isaac launches) by the SkillGen annotate/generate runners. Does two things
without editing the shared Isaac Lab / cuRobo source:

  1. Registers `Isaac-Stack-Cube-LabFR3-Skillgen-IK-Rel-v0` -> LabFR3CubeStackIKRelSkillgenEnv
     with our lab SkillGen cfg.
  2. Patches `CuroboPlannerCfg.from_task_name` to return an FR3 planner config for our task.
     cuRobo only ships a Franka Panda config; we built fr3_curobo.yml (= franka.yml with
     panda_->fr3_ + the FR3 URDF + FR3-named collision spheres). FR3 and Panda share the same
     link/joint naming and near-identical kinematics, so the renamed config is valid for the FR3
     URDF. We mirror `franka_stack_cube_config` (flat collision_table world + runtime cube sync),
     just pointed at the FR3 robot + FR3 finger joints/links.
"""

from __future__ import annotations

import os

import gymnasium as gym

# Must run before any cuRobo planner is built (cuRobo 0.7.7 needs warp.torch, which
# warp 1.14 relocated). Importing this module installs the alias.
import warp_torch_shim  # noqa: F401

from isaaclab_mimic.motion_planners.curobo.curobo_planner_cfg import CuroboPlannerCfg

HERE = os.path.dirname(os.path.abspath(__file__))
FR3_YML = os.path.join(HERE, "fr3_curobo.yml")

TASK = "Isaac-Stack-Cube-LabFR3-Skillgen-IK-Rel-v0"
if TASK not in gym.registry:
    gym.register(
        id=TASK,
        entry_point="lab_skillgen_env:LabFR3CubeStackIKRelSkillgenEnv",
        kwargs={"env_cfg_entry_point": "lab_skillgen_cfg:LabFR3CubeStackIKRelSkillgenEnvCfg"},
        disable_env_checker=True,
    )
    print(f"[skillgen_register] registered {TASK}")


def fr3_stack_cube_config() -> CuroboPlannerCfg:
    """cuRobo planner config for the lab FR3 stacking a cube (mirror of franka_stack_cube_config)."""
    cfg = CuroboPlannerCfg(
        robot_config_file=FR3_YML,            # our FR3 cuRobo robot config (inline FR3 spheres)
        robot_name="fr3",
        gripper_joint_names=["fr3_finger_joint1", "fr3_finger_joint2"],
        gripper_open_positions={"fr3_finger_joint1": 0.04, "fr3_finger_joint2": 0.04},
        gripper_closed_positions={"fr3_finger_joint1": 0.02, "fr3_finger_joint2": 0.02},
        hand_link_names=["fr3_leftfinger", "fr3_rightfinger", "fr3_hand"],
        collision_spheres_file=None,          # None -> keep the inline FR3 spheres in fr3_curobo.yml
        grasp_gripper_open_val=0.04,
        approach_distance=0.05,
        retreat_distance=0.05,
        surface_sphere_radius=0.01,
        collision_activation_distance=0.01,
        max_planning_attempts=1,
        time_dilation_factor=0.6,
        # cuRobo planning speed. The stock setup (graph search + 12 trajopt seeds + finetune) costs
        # ~20 s/trial of the ~26 s total — but our SkillGen transitions are simple free-space hops
        # above the work surface, so the heavy GRAPH planner (collision-free roadmap search) is
        # overkill: trajopt seeded from the retract config solves these directly. LAB_SKILLGEN_FAST=1
        # disables the graph planner and halves the trajopt seeds for a large planning speedup. Keep
        # finetune on (it sharpens the approach pose the replayed skill starts from). Validate DGR
        # before a long run: if some transitions need to route around cubes, plan success may dip.
        enable_graph=os.environ.get("LAB_SKILLGEN_FAST", "0") != "1",
        num_trajopt_seeds=int(os.environ.get(
            "LAB_SKILLGEN_TRAJOPT_SEEDS", "6" if os.environ.get("LAB_SKILLGEN_FAST", "0") == "1" else "12")),
        enable_finetune_trajopt=True,
        sphere_update_freq=5,
        motion_noise_scale=0.02,
        visualize_spheres=False,
        visualize_plan=False,
        debug_planner=os.environ.get("LAB_SKILLGEN_DEBUG", "0") == "1",
        static_objects=[],
        # The planner's _initialize_static_world scrapes ALL geometry under /World/envs/env_<id>
        # as collision obstacles, excluding only these substrings. Our lab desk is a full-height
        # solid mesh (z~0..0.74) and the FR3 is mounted on top, so cuRobo sees the robot's base
        # links embedded inside the desk volume -> the start state is ALWAYS in world collision
        # (INVALID_START_STATE_WORLD_COLLISION) and every plan is rejected. We exclude the desk
        # ("/Table") from cuRobo's world: all SkillGen transitions happen above the table surface
        # (the table-surface contact lives in the replayed skills, not the planned transitions),
        # and cuRobo still avoids the cubes (synced separately). Bare substrings hold across every
        # env_<id>. (Robot/target must stay excluded too.)
        # /Table is the desk mesh; /WorkSurface is the flat work surface the FR3 is MOUNTED on and
        # the cubes sit on. The base (root at z~0.72) rests on /WorkSurface, so cuRobo sees the base
        # collision spheres touching it -> INVALID_START_STATE_WORLD_COLLISION. Confirmed from the
        # planner's own world dump: obstacles were [Cube_1, Cube_2, Cube_3, WorkSurface]. Exclude
        # both surfaces; the cubes remain as obstacles and all transitions are above the surface.
        world_ignore_substrings=[
            "/Robot", "/target", "/World/defaultGroundPlane", "/curobo", "/Table", "/WorkSurface",
        ],
    )
    # cuRobo's base world. The stock franka_stack_cube_config returns its collision_table.yml
    # table cuboid (a table at the stock Franka height z~0); in our lab scene the FR3 base also
    # sits at z~0 so that cuboid lands on the base -> start state always in world collision. We
    # exclude the surface MESH from cuRobo's world (a full-height mesh embeds the base spheres ->
    # INVALID_START_STATE_WORLD_COLLISION). But with NO surface at all the trajectory optimizer
    # freely routes the wrist BELOW the surface: a smoke demo dove to base-frame z=-0.25 (world
    # z=0.47, 0.25 m under the work surface) right beside the base, which collides with the real
    # solid surface in sim -> the demo diverges and never stacks (confirmed from the recorded
    # eef_z trajectory: a U-shaped dip to 0.47 mid-demo). cuRobo plans in the FR3 BASE frame
    # (base link at the origin, mount plane at base-frame z=0), so we model the surface as a thin
    # slab whose TOP sits a few cm BELOW the mount: it clears the base + grasping-finger spheres
    # (all at z>=~0) yet forbids the deep sub-surface dive. This is base-frame native (no env_origin
    # dependency). LAB_SKILLGEN_SURFACE_TOP tunes the top height; -1.0 restores the old harmless
    # floor. dims/thickness keep warmup happy (an empty world hangs graph warmup).
    def _surface_world():
        from curobo.geom.types import Cuboid, WorldConfig

        top = float(os.environ.get("LAB_SKILLGEN_SURFACE_TOP", "-0.06"))
        thick = 1.0
        slab = Cuboid(
            name="work_surface",
            pose=[0.0, 0.0, top - thick / 2.0, 1.0, 0.0, 0.0, 0.0],
            dims=[4.0, 4.0, thick],
        )
        return WorldConfig(cuboid=[slab])

    cfg.get_world_config = _surface_world
    return cfg


_orig_from_task_name = CuroboPlannerCfg.from_task_name.__func__


@classmethod
def _patched_from_task_name(cls, task_name: str) -> CuroboPlannerCfg:
    if "labfr3" in task_name.lower():
        print(f"[skillgen_register] cuRobo using FR3 config for task '{task_name}'")
        return fr3_stack_cube_config()
    return _orig_from_task_name(cls, task_name)


CuroboPlannerCfg.from_task_name = _patched_from_task_name


# --- base-frame fix: transform the planning target into the FR3 base frame ---------------------
# The CuroboPlanner plans with the robot base at the ORIGIN of cuRobo's frame and feeds the target
# pose straight through (plan_motion does no base subtraction). That is only correct when the robot
# sits at the environment origin with identity orientation (the stock Franka). Our lab FR3 is
# mounted on the desk at root ~[0.72, 0.14, 0.72] with yaw=180, so an env-frame target is ~0.7 m
# off and 180-flipped relative to the base -> cuRobo cannot solve IK for it -> systematic IK_FAIL.
# We wrap plan_motion to express the target in the real base frame: T_base = inv(T_env_base) @ T_env.
from isaaclab_mimic.motion_planners.curobo.curobo_planner import CuroboPlanner  # noqa: E402

import torch as _torch  # noqa: E402
from isaaclab.utils.math import matrix_from_quat as _matrix_from_quat  # noqa: E402
from isaaclab.utils.math import quat_from_matrix as _quat_from_matrix  # noqa: E402


def _base_T(planner, dev, dt):
    """T_env_base: the FR3 base pose in the env frame (4x4 on dev/dt). Maps a base-frame pose
    into the env frame: T_env = T_env_base @ T_base."""
    p = (planner.robot.data.root_pos_w[planner.env_id, :3]
         - planner.env.scene.env_origins[planner.env_id, :3]).to(device=dev, dtype=dt)
    q = planner.robot.data.root_quat_w[planner.env_id].to(device=dev, dtype=dt)  # (w, x, y, z)
    T = _torch.eye(4, device=dev, dtype=dt)
    T[:3, :3] = _matrix_from_quat(q.unsqueeze(0))[0]
    T[:3, 3] = p
    return T


def _base_T_inv(planner, dev, dt):
    """inv(T_env_base): maps an env-frame pose into the FR3 base frame (4x4 on dev/dt)."""
    return _torch.inverse(_base_T(planner, dev, dt))


# (1) Target: express the planning goal in the base frame.
_orig_plan_motion = CuroboPlanner.plan_motion


def _plan_motion_base_frame(self, target_pose, step_size=None, enable_retiming=None):
    Tbw = _base_T_inv(self, target_pose.device, target_pose.dtype)
    return _orig_plan_motion(self, Tbw @ target_pose, step_size, enable_retiming)


CuroboPlanner.plan_motion = _plan_motion_base_frame


# (1b) Planned poses: express cuRobo's output trajectory back in the ENV frame. get_planned_poses()
# computes EE poses via motion_gen.compute_kinematics(), i.e. cuRobo forward kinematics with the
# robot base at the ORIGIN -> the poses are in the BASE frame (the get_*_ee_pose docstring claiming
# "world coordinates" is only true for the stock Franka at the origin). SkillGen feeds these poses
# straight to the controller as env-frame waypoint targets. With our FR3 mounted off-origin + yaw180,
# a base-frame pose executed as an env-frame target lands ~0.7 m off and below the surface: the
# recorded transition dove to world z=0.47 beside the base (a U-shaped dip between the correct
# env-frame skill segments). This is the EXACT inverse of the target transform above, so we undo it
# on the way out: T_env = T_env_base @ T_base. (Confirmed root cause: the work-surface slab could not
# stop the dip because the dip is a frame-misplaced transition, not a path through real geometry.)
_orig_get_planned_poses = CuroboPlanner.get_planned_poses


def _get_planned_poses_env_frame(self):
    poses = _orig_get_planned_poses(self)
    if not poses:
        return poses
    Teb = _base_T(self, poses[0].device, poses[0].dtype)
    return [Teb @ p for p in poses]


CuroboPlanner.get_planned_poses = _get_planned_poses_env_frame


# (1b-defensive) get_next_waypoint_ee_pose() has the SAME base-frame FK issue as get_planned_poses
# and is currently NOT on the generate path (data_generator converts plans via get_planned_poses).
# Wrap it identically so a future consumer can't silently reintroduce the dip.
_orig_get_next_wp = CuroboPlanner.get_next_waypoint_ee_pose


def _get_next_waypoint_ee_pose_env_frame(self):
    pose = _orig_get_next_wp(self)
    try:
        from curobo.types.math import Pose as _CuroboPose

        dev = pose.position.device
        dt = pose.position.dtype
        Teb = _base_T(self, dev, dt)
        p = pose.position.reshape(3).to(device=dev, dtype=dt)
        T = _torch.eye(4, device=dev, dtype=dt)
        T[:3, :3] = _matrix_from_quat(pose.quaternion.reshape(4).unsqueeze(0))[0]
        T[:3, 3] = p
        Te = Teb @ T
        return _CuroboPose(
            position=Te[:3, 3].unsqueeze(0),
            quaternion=_quat_from_matrix(Te[:3, :3].unsqueeze(0)),
        )
    except Exception as e:  # latent path; never break generation over it
        print(f"[skillgen_register] get_next_waypoint_ee_pose base->env wrap skipped: {e}")
        return pose


CuroboPlanner.get_next_waypoint_ee_pose = _get_next_waypoint_ee_pose_env_frame


# (2) Obstacles: place the cube obstacles in the SAME base frame as the target/plan. The stock sync
# (_sync_object_poses_with_isaaclab) writes cube poses at (root_pos_w - env_origin) = the ENV frame,
# which is inconsistent with everything else once the FR3 is off the origin: the goal (#1), the
# planned-pose output (#1b), the robot FK start state, the scraped static world (reference=robot),
# and the slab are ALL base frame. With the env-frame sync the cubes are mislocated by inv(T_env_base)
# (~0.79 m / yaw180) so cuRobo dodges phantom cubes and plows through the real ones (observed: the arm
# bumps the OTHER cubes; cube_1 approach stalls at wrist-dist 0.165). It ALSO breaks object ATTACH:
# attach_objects_to_robot (curobo_planner.py) fits the carried cube's spheres from the base-frame EE
# FK against the cube pose stored in world_model -> with an env-frame cube pose the carried-cube
# spheres float ~0.79 m off the gripper on every grasp->stack transition. Re-expressing every dynamic
# obstacle in the base frame fixes obstacle avoidance AND attach in one shot.
def _sync_object_poses_base_frame(self):
    object_mappings = self._get_object_mappings()
    world_model = self.motion_gen.world_coll_checker.world_model
    rigid_objects = self.env.scene.rigid_objects
    dev = self.tensor_args.device
    dt = self.tensor_args.dtype
    Tbw = _base_T_inv(self, dev, dt)
    static = getattr(self.config, "static_objects", [])
    for name, path in object_mappings.items():
        if name not in rigid_objects:
            continue
        if any(s in name.lower() for s in static):
            continue
        obj = rigid_objects[name]
        origin = self.env.scene.env_origins[self.env_id]
        p_env = (obj.data.root_pos_w[self.env_id] - origin).to(device=dev, dtype=dt)
        q_env = obj.data.root_quat_w[self.env_id].to(device=dev, dtype=dt)
        T = _torch.eye(4, device=dev, dtype=dt)
        T[:3, :3] = _matrix_from_quat(q_env.unsqueeze(0))[0]
        T[:3, 3] = p_env
        Tb = Tbw @ T
        p_b = Tb[:3, 3]
        q_b = _quat_from_matrix(Tb[:3, :3].unsqueeze(0))[0]
        pose_list = [float(p_b[0]), float(p_b[1]), float(p_b[2]),
                     float(q_b[0]), float(q_b[1]), float(q_b[2]), float(q_b[3])]
        # BOTH writes are load-bearing: update_obstacle_pose fixes collision avoidance, while the
        # world_model write feeds attach_objects_to_robot (it reads the cube pose from world_model
        # via get_obstacle). Dropping the world_model write would silently reintroduce the ~0.79 m
        # attach-sphere mislocation on grasp->stack transitions.
        self._update_object_in_world_model(world_model, name, path, pose_list)
        self.motion_gen.world_coll_checker.update_obstacle_pose(
            path, self._make_pose(position=p_b, quaternion=q_b), update_cpu_reference=True)


# Base-frame obstacle sync is now ON BY DEFAULT: it is the deepest correct fix and the only one that
# makes cuRobo's whole world internally consistent with the base-frame goal (#1) and planned-pose
# output (#1b). An earlier trial regressed plan success 45->6 / 0 demos, but that was BEFORE the
# planned-pose output fix (#1b) existed: with the output still in the base frame, base-frame cubes
# only compounded the inconsistency. Now that #1b is in, this is the right default. Set
# LAB_SKILLGEN_OBS_BASEFRAME=0 to fall back to the (broken-for-FR3) stock env-frame sync for A/B.
if os.environ.get("LAB_SKILLGEN_OBS_BASEFRAME", "1") != "0":
    CuroboPlanner._sync_object_poses_with_isaaclab = _sync_object_poses_base_frame


# --- surface slab into the REAL planning world -------------------------------------------------
# cfg.get_world_config (above) only seeds cuRobo's WARMUP world; _initialize_static_world then
# SCRAPES the USD scene (relative to the robot prim -> base frame) and calls motion_gen.update_world
# with that, OVERWRITING the warmup world. So the get_world_config slab never reaches the planner
# that actually plans transitions (confirmed: adding it there changed nothing, demos byte-identical).
# We instead wrap _initialize_static_world to APPEND the surface slab to the scraped world (already
# base frame, so the slab's base-frame z is consistent) and re-push it. Disable with
# LAB_SKILLGEN_SURFACE_TOP=-1.0 (slab drops 1 m below, harmless). LAB_SKILLGEN_DEBUG=1 logs the
# resulting world obstacles + robot base/joint state.
_DEBUG = os.environ.get("LAB_SKILLGEN_DEBUG", "0") == "1"
_orig_isw = CuroboPlanner._initialize_static_world


def _isw_with_slab(self):
    _orig_isw(self)
    top = float(os.environ.get("LAB_SKILLGEN_SURFACE_TOP", "-0.06"))
    try:
        from curobo.geom.types import Cuboid

        w = self._static_world_config
        if getattr(w, "cuboid", None) is None:
            w.cuboid = []
        thick = 1.0
        w.cuboid.append(Cuboid(
            name="lab_work_surface",
            pose=[0.0, 0.0, top - thick / 2.0, 1.0, 0.0, 0.0, 0.0],
            dims=[4.0, 4.0, thick],
        ))
        self.motion_gen.update_world(w)
    except Exception as e:
        print(f"[skillgen_register] FAILED to add surface slab: {e}")
    if _DEBUG:
        w = self._static_world_config
        mesh = [getattr(o, "name", "?") for o in (getattr(w, "mesh", None) or [])]
        cub = [getattr(o, "name", "?") for o in (getattr(w, "cuboid", None) or [])]
        print(f"[skillgen_register] cuRobo static world (slab top={top}): "
              f"{len(mesh)} mesh {mesh} | {len(cub)} cuboid {cub}")
        try:
            base = self.robot.data.root_pos_w[self.env_id].detach().cpu().tolist()
            print(f"[skillgen_register] robot base_pos_w={base}")
        except Exception as e:
            print(f"[skillgen_register] could not read robot state: {e}")


CuroboPlanner._initialize_static_world = _isw_with_slab
