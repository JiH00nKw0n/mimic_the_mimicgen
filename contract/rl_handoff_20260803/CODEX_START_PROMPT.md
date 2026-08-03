# Copy/paste prompt for the receiving Codex agent

You are receiving the `fr3_mimicgen_control_handoff_20260803` package.  Work
from the package contents rather than guessing from a stock Franka/Panda setup.

First read `START_HERE.md`, `MANIFEST.yaml`,
`contracts/control_contract.yaml`, `runtime/fr3_panda_gripper.py`,
`runtime/cube_actions.py`, `sysid/SYSTEM_CALIBRATION_CONTRACT.yaml`, and
`contracts/mimicgen_inputs_required.yaml`.  Run `python3 verify_package.py` and
`python3 tools/controller_adapter.py --self-test` before editing your pipeline.

Establish and report whether your robot layer composes the exact NVIDIA Isaac
5.1 FR3 asset URL recorded in `assets/fr3_research3.usda`.  Treat that USDA as
an official-asset wrapper, not as a custom dynamics file.  Compare all runtime
IsaacLab spawn/actuator settings and OSC/action semantics separately.

For Cube MimicGen replay, preserve contract
`fr3_cube_stage1_model4500_legacyosc_v1`: 7D raw unclipped actions, robot-base
frame, quaternion `wxyz`, left-multiplied axis-angle delta, current actual EE
pose as the per-step reference, positive=open, non-positive=close, 10 Hz policy,
120 Hz physics/OSC, and 12-step target hold.  Use
`FR3_CUBE_STAGE1_RELATIVE_OSC`; do not select another controller profile merely
because it appears in the same source file.

Apply SysID dynamics/delay ranges around the frozen controller only.  Do not
retune the OSC, change action scales, add clipping, or infer missing frame
transforms.  Before exporting demonstrations, enumerate unresolved human-demo
frames/timestamps/gripper/cube-identity/subtask inputs.  Your first deliverable
is an audit table, pose/action round-trip, and one closed-loop smoke result with
tracking error, action percentiles, sampled SysID row/seed, and files changed.

