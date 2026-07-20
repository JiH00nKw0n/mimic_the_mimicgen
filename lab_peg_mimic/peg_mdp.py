"""Peg-in-hole MimicGen mdp functions for the lab FR3 task.

Three functions, modelled on the official Isaac Lab stack mdp
(`isaaclab_tasks.manager_based.manipulation.stack.mdp`):

  * ``peg_grasped``      -> the ``grasp_peg`` subtask-term signal  (mirror of ``mdp.object_grasped``)
  * ``peg_inserted``     -> the ``success`` termination            (ports teleop ``insertion_state()``)
  * ``randomize_peg_xy`` -> a ``mode="reset"`` event term          (peg xy randomization)

All geometry is expressed in the ENV-LOCAL frame (world minus ``env.scene.env_origins``),
because the socket target is a CONSTANT ``HOLE_XY`` authored relative to each env origin
(the socket is a static triangle-mesh collider, NOT a rigid object we can read a pose from).
Reading absolute ``root_pos_w`` without subtracting the origin would only be correct for a
single env at the world origin (as in the teleop script); the mimic pipeline tiles many envs.

ON-BOX TODO(imports): if you would rather delegate to the shipped stack fn instead of the
self-contained copy in ``peg_grasped``, use
``from isaaclab_tasks.manager_based.manipulation.stack import mdp as stack_mdp`` and call
``stack_mdp.object_grasped``. That exact import path is what the peg teleop script uses.
"""

from __future__ import annotations

import torch

import isaaclab.utils.math as PoseUtils
from isaaclab.managers import SceneEntityCfg

# --- scene geometry (mirrors peg_hole_teleop_webxr.py; world == env-local for a single env) ---
DESK_Z = 0.720                     # desk top world/env-local z
PEG_LEN = 0.060                    # peg z-height (PEG_SIZE[2]); half-height = 0.030
HOLE_XY = (0.091, 0.104)           # socket centre, env-local (static collider, not a rigid object)
HOLE_HEIGHT = 0.0406               # socket total height; rim sits at DESK_Z + HOLE_HEIGHT = 0.7606
HOLE_INNER = 0.032                 # square socket bore
PEG_SPAWN_Z = DESK_Z + PEG_LEN / 2 + 0.001   # 0.751 — upright rest height on the desk

# insertion thresholds, ported verbatim from teleop insertion_state()
RADIAL_INSIDE = HOLE_INNER / 2     # 0.016 — peg centre over the bore (gates "depth" meaningfulness)
RADIAL_SUCCESS = 0.010
DEPTH_SUCCESS = 0.020
UPRIGHT_SUCCESS = 0.9

# FR3 binary gripper (match env.cfg.gripper_open_val / gripper_threshold set in peg_mimic_cfg)
GRIPPER_OPEN_VAL = 0.04
GRIPPER_THRESHOLD = 0.005

# peg random-spawn region (env-local xy on the desk) + socket keep-out (teleop PEG_REGION_*)
PEG_REGION_X = (0.08, 0.26)
PEG_REGION_Y = (-0.12, 0.12)
PEG_MIN_SEP = 0.075


def peg_grasped(
    env,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("peg"),
    diff_threshold: float = 0.13,
    gripper_open_val: float = GRIPPER_OPEN_VAL,
    gripper_threshold: float = GRIPPER_THRESHOLD,
) -> torch.Tensor:
    """``grasp_peg`` signal: EE within ``diff_threshold`` of the peg centre AND both fingers
    closed past ``gripper_threshold`` from the open value.

    Verbatim structure of the stack ``mdp.object_grasped`` (…/stack/mdp/observations.py),
    retargeted to the peg. ``diff_threshold`` is looser than the cube's 0.06 because the peg is
    grasped near its TOP (~18 mm above its centre) and carried there, so the EE-to-centre
    distance stays ~0.02 m through carry/insert — well under 0.13 — keeping the signal latched
    high after the grasp, which is what marks the 0->1 subtask boundary.

    ON-BOX TODO(fingers): like the stock fn this reads ``joint_pos[:, -1]`` and ``[:, -2]`` as
    the two gripper fingers (true for the FR3: fr3_finger_joint1/2 are the last two joints).
    Verify the joint order in the container; if fingers aren't last, index them via
    env.cfg.gripper_joint_names instead.
    """
    robot = env.scene[robot_cfg.name]
    ee_frame = env.scene[ee_frame_cfg.name]
    obj = env.scene[object_cfg.name]

    object_pos = obj.data.root_pos_w
    ee_pos = ee_frame.data.target_pos_w[:, 0, :]
    pose_diff = torch.linalg.vector_norm(object_pos - ee_pos, dim=1)

    grasped = torch.logical_and(
        pose_diff < diff_threshold,
        torch.abs(robot.data.joint_pos[:, -1] - gripper_open_val) > gripper_threshold,
    )
    grasped = torch.logical_and(
        grasped,
        torch.abs(robot.data.joint_pos[:, -2] - gripper_open_val) > gripper_threshold,
    )
    return grasped


def peg_inserted(
    env,
    peg_cfg: SceneEntityCfg = SceneEntityCfg("peg"),
    hole_xy: tuple = HOLE_XY,
    desk_z: float = DESK_Z,
    hole_height: float = HOLE_HEIGHT,
    peg_len: float = PEG_LEN,
    radial_inside: float = RADIAL_INSIDE,
    radial_success: float = RADIAL_SUCCESS,
    depth_success: float = DEPTH_SUCCESS,
    upright_success: float = UPRIGHT_SUCCESS,
) -> torch.Tensor:
    """``success`` termination: True where the peg is inserted in the socket.

    Ports teleop ``insertion_state()`` geometry, vectorised over envs and referencing the
    CONSTANT socket xy (the socket is a static collider):
        radial   = hypot(peg_x - HOLE_XY[0], peg_y - HOLE_XY[1])       (env-local)
        depth    = (rim_z - peg_bottom_z)  gated on radial < 0.016     (over the bore)
        upright  = peg body z-axis . world z
        SUCCESS  = radial < 0.010 AND depth > 0.020 AND upright > 0.9

    ON-BOX TODO(quat): reads peg ``.data.root_quat_w`` as WXYZ (Isaac Lab math convention, per
    the task facts) and takes the world-z component of the peg's body z-axis for uprightness.
    The peg teleop compat shim instead assumed ``.data.*`` quats are XYZW in this 3.0 container
    — if that holds here, reorder to WXYZ before ``matrix_from_quat``. THIS IS THE #1 THING TO
    VERIFY ON-BOX (see the summary's ranked risks).
    """
    peg = env.scene[peg_cfg.name]
    pos = peg.data.root_pos_w - env.scene.env_origins   # env-local (num_envs, 3)
    quat = peg.data.root_quat_w                          # WXYZ (num_envs, 4)

    rot = PoseUtils.matrix_from_quat(quat)               # (num_envs, 3, 3), body->world
    upright = rot[:, 2, 2]                                # peg body z-axis . world z

    radial = torch.hypot(pos[:, 0] - hole_xy[0], pos[:, 1] - hole_xy[1])
    rim_z = desk_z + hole_height
    peg_bottom_z = pos[:, 2] - peg_len / 2
    inside = radial < radial_inside
    depth = torch.where(inside, rim_z - peg_bottom_z, torch.zeros_like(peg_bottom_z))

    return (radial < radial_success) & (depth > depth_success) & (upright > upright_success)


def randomize_peg_xy(
    env,
    env_ids: torch.Tensor,
    x_range: tuple = PEG_REGION_X,
    y_range: tuple = PEG_REGION_Y,
    hole_xy: tuple = HOLE_XY,
    min_sep: float = PEG_MIN_SEP,
    spawn_z: float = PEG_SPAWN_Z,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("peg"),
    max_tries: int = 100,
) -> None:
    """``mode="reset"`` event: sample the peg start xy uniformly in the region, rejecting draws
    within ``min_sep`` of the socket, and write an ABSOLUTE upright world pose at desk height.

    Only affects GENERATION — annotation's ``reset_to`` overrides the peg from the recorded
    initial_state, so this term is a no-op there. Mirrors teleop ``_sample_peg_xy`` + ``_set_peg_xy``.

    ON-BOX TODO(write API): uses the standard event API ``write_root_pose_to_sim`` /
    ``write_root_velocity_to_sim`` (env_ids). The beta2 container's teleop used the
    ``write_root_*_to_sim_index`` variants because the combined ``write_root_state_to_sim`` raised
    NotImplementedError there; switch to the index variants if these raise.
    ON-BOX TODO(quat): writes upright as WXYZ ``(1,0,0,0)`` per the task facts. The teleop wrote
    XYZW ``(0,0,0,1)`` for the same upright peg — if the peg spawns UPSIDE-DOWN, flip this to
    ``(0,0,0,1)`` (same convention question as ``peg_inserted``; both are driven by whether this
    container's runtime quats are WXYZ or XYZW).
    """
    peg = env.scene[asset_cfg.name]
    device = env.device
    if not isinstance(env_ids, torch.Tensor):
        env_ids = torch.tensor(env_ids, device=device, dtype=torch.long)
    n = int(env_ids.numel())
    if n == 0:
        return

    xs = torch.empty(n, device=device)
    ys = torch.empty(n, device=device)
    todo = torch.arange(n, device=device)
    for _ in range(max_tries):
        if todo.numel() == 0:
            break
        m = int(todo.numel())
        cx = torch.empty(m, device=device).uniform_(x_range[0], x_range[1])
        cy = torch.empty(m, device=device).uniform_(y_range[0], y_range[1])
        keep = torch.hypot(cx - hole_xy[0], cy - hole_xy[1]) > min_sep
        sel = todo[keep]
        xs[sel] = cx[keep]
        ys[sel] = cy[keep]
        todo = todo[~keep]
    if todo.numel() > 0:                                 # fallback for any never-satisfied draw
        xs[todo] = float(sum(x_range) / 2.0)
        ys[todo] = float(y_range[1])

    origins = env.scene.env_origins[env_ids]
    pos = torch.stack([xs, ys, torch.full((n,), spawn_z, device=device)], dim=1) + origins
    quat = torch.zeros((n, 4), device=device)
    quat[:, 0] = 1.0                                     # WXYZ upright — see ON-BOX TODO(quat)
    root_pose = torch.cat([pos, quat], dim=1)
    peg.write_root_pose_to_sim(root_pose, env_ids=env_ids)
    peg.write_root_velocity_to_sim(torch.zeros((n, 6), device=device), env_ids=env_ids)
