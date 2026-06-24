# Generation debugging — why DGR was 0%, and the two fixes

This documents the root-cause analysis that took our lab FR3 3-cube-stack MimicGen
generation from **0% DGR → ~13–20% DGR (real stacks)**. Auto-annotation was already
working (24/29 demos annotate); the failure was purely in **generation** (transform +
noise + stitching + execution). Everything below is verified, not theorized.

## Symptom

Generation produced 0 successes over dozens of attempts. Inspecting a failed trajectory:
the robot approached the cubes' vicinity then **diverged**, the cubes were **never
touched** (Δpos = 0), the arm whipped to an extreme joint config, and the IK-rel
**actions saturated at ±1**. Annotation/replay (which replay the *recorded* actions)
worked fine — so the bug was specific to actions that generation **re-derives**.

## Root cause 1 — IK-rel action computed in the wrong frame (secondary)

`get_robot_eef_pose` returns the end-effector pose in the **world** frame (the
`ee_frame` obs, minus env origin). `target_eef_pose_to_action` then forms the action as
a raw delta `target_pos − curr_pos` (world frame). But the IK-rel controller
(`DifferentialInverseKinematicsAction`, relative mode) applies that delta in the robot
**base** frame: `_compute_frame_pose` does `subtract_frame_transforms(root_pose_w, …)`
then `apply_delta_pose(ee_pos_base, …, command)` adds the delta directly in base coords
(`task_space_actions.py` + `differential_ik.py:139–144`, `math.py::apply_delta_pose`).

The official Franka sits at the world origin with identity rotation, so world ≡ base and
this is invisible. **Our FR3 is mounted at `ROBOT_ROT=(0,0,0,1)` = yaw 180°** (it faces
−x toward the desk). With a 180° base yaw a world delta `(dx,dy,dz)` is applied as
`(−dx,−dy,dz)` → x/y are sign-flipped → the robot drives away from the target → error
grows → action saturates.

**Fix** (`lab_mimic_env.py::LabFR3CubeStackIKRelMimicEnv`): rotate the delta into the
base frame at the action boundary — `quat_apply_inverse(root_quat_w, delta)` for both the
position delta and the axis-angle rotation delta (conjugating a rotation by R rotates its
axis-angle vector by R, so the same op is correct for both). `action_to_target_eef_pose`
does the inverse (`quat_apply`). The object-frame transform pipeline (eef + object poses
both world) is untouched; only the action boundary is corrected. Registered by pointing
`lab_register.py`'s entry point at this subclass.

**Verified** (`probe_action_frame.py`): commanding the EE toward a world-frame target now
moves it there exactly — world +x/+y/+z reach tests converge to `0.000 m` error, and a
`reach above cube_2` test lands on target. (World-axis x/z rotations also track to 0.1°;
a world-**y** rotation can still diverge, but that is a genuine FR3 wrist near-singularity,
not a frame bug — the conjugation handles x/y symmetrically.)

## Root cause 2 — reset put the arm at the USD ~zero pose (dominant)

This was the real blocker. I had dropped the Franka reset events
(`init_franka_arm_pose`, `randomize_franka_joint_state`) thinking the FR3's
`ArticulationCfg.init_state.joint_pos` would set the start pose. It does not:

- `_reset_idx` runs `scene.reset()` **before** the reset events.
- `set_default_joint_pose` (the `init_franka_arm_pose` event) only updates the
  `default_joint_pos` **buffer** — it never writes to sim.
- The actual write to sim is done by `randomize_franka_joint_state`
  (`randomize_joint_by_gaussian_offset` → `set_joint_position_target`), which I had
  disabled.

Net effect: `env.reset()` left the arm at the USD default (~all-zero joints), i.e. the
arm pointing nearly straight up with the **EE ~0.7 m too high** (z≈1.55 vs the demos'
z≈0.85). Generation then has to dive the arm down and reorient from a bad configuration;
the regenerated IK-rel trajectory can't reach the first waypoint and diverges. (Annotation
is unaffected because `reset_to` overrides the joint state after the events.)

Per-step logging (`lab_mimic_env.py`, `LAB_MIMIC_DEBUG=1`) confirmed it: at the broken
reset the rotation error `|Δrot|` grew monotonically 0.04 → 2.0+ rad with the rotation
action pinned at ±1, while the arm sat ~0.4 m from every waypoint.

**Fix** (`lab_mimic_cfg.py::reset_arm_to_home`): replace those events with one reset
event that **teleports** the arm to the demos' home pose via `write_joint_state_to_sim`:

```
FR3_HOME_JOINT_POSE = [0.0, -0.569, 0.0, -2.810, 0.0, 2.25, 0.741, 0.04, 0.04]
```

(joint order fr3_joint1..7 then the two fingers — exactly the demos' `states[0]`.)

**Verified** (`probe_frames.py`): `env.reset()` now lands at
`joints=[0,-0.569,0,-2.81,0,2.25,0.741,0.04,0.04]` and `get_robot_eef_pose=[0.379,
0.138,0.982]` — i.e. exactly the source-trajectory start. With this, the per-step
generation log shows the robot starting on the first waypoint, `|Δpos|≈0.03–0.10` and
`|Δrot|≈0.04–0.15` (no divergence), and **DGR jumps 0% → ~13%**.

Side mystery solved: jake's init says `joint6=3.037` but the demos start at `2.25`
because `randomize_joint_by_gaussian_offset` **clamps to the FR3 soft joint limits**
(3.037 is above the FR3 joint-6 soft limit → 2.25).

## Secondary finding — action scale

The official Franka IK-rel uses `scale=0.5`; jake's teleop (and so our annotation) uses
`scale=1.0`. Generation re-derives actions, so its scale is independent of the recorded
demos. Lowering generation's scale to the official 0.5 gives **gentler IK tracking and a
higher DGR**: fwd `scale=1.0 → 12.8%` (5/39) vs `scale=0.5 → 20.5%` (8/39). Exposed via
`LAB_ARM_SCALE` (teleop/annotation keep 1.0; generation overrides to 0.5).

## Result

- fwd DGR: **0% → 12.8% (scale 1.0) → 20.5% (scale 0.5)**.
- Generated demos are **real** order-correct stacks — verified 4/5 with
  z=0.745/0.795/0.845, 5 cm gaps, xy alignment ~1 cm. (One borderline 2-stack false
  positive, `demo_0`, flagged for a success-criterion check.)

## Still open (DGR optimization — "why not ~60–70%?")

13–20% is well below the paper's 71.3% for Stack Three D0. Under investigation:
per-subtask failure distribution (where the 4-subtask chain breaks), `action_noise`
level, success-criterion tightness, cube-randomization range vs the paper's D0,
source-demo count/quality (13 fwd seeds), and the FR3 wrist near-singularity for some
target orientations. Tracked separately.

## Artifacts

- `lab_mimic_env.py` — base-frame IK-rel subclass (fix 1) + `LAB_MIMIC_DEBUG` logging.
- `lab_mimic_cfg.py` — `reset_arm_to_home` teleport event (fix 2), `LAB_ARM_SCALE`.
- `probe_action_frame.py` — closed-loop reach/rotation control test (verifies fix 1).
- `probe_frames.py` — reset-pose / eef-frame measurement (verifies fix 2).
