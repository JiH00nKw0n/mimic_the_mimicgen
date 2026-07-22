# FR3 Portable Camera Overlay

This directory is a thin calibration overlay for team-provided FR3 Facelift,
Franka Hand, table, and threaded-plate assets. It does not contain those assets.

## Contents

- `overlay.yaml`: canonical runtime transforms, intrinsics, environment frames, and validation-scene labels.
- `manifest.yaml`: bundle files, checksums, entry points, and external asset requirements.
- `TEAM_HANDOFF_KO.md`: Korean handoff guide for simulation-team members.
- `LLM_SETUP_PROMPT_KO.md`: ready-to-paste prompt for configuring another Isaac environment.
- `ASSET_BINDING_TEMPLATE.yaml`: record of team-specific prim and frame adapters.
- `requirements-validator.txt`: dependencies for bundle-only validation.
- `validation/`: portable figures and metrics referenced by `overlay.yaml`.
- `tools/check_runtime.py`: checks the selected entry point's Python runtime.
- `tools/validate_overlay.py`: dependency-light integrity and transform validator.
- `tools/apply_overlay_to_isaac.py`: authors four Camera prims and optional calibration frames into an existing USD.
- `tools/smoke_render.py`: opens an applied stage and renders all four RGB sensors.

`validation/isaac_5_1_smoke/` contains the passing reference-scene render from
Isaac Sim 5.1. It proves the packaged sensor pipeline ran once; each team must
rerun the smoke tool against its own scene and prim bindings.

## Validate

Use Python 3.10 or newer. Install the bundle-only validator dependencies into a
normal project environment; do not modify the Isaac bundled Python just for this check.

```bash
python -m pip install -r requirements-validator.txt
python tools/check_runtime.py --mode validate
python tools/validate_overlay.py .
```

## Apply In Isaac Sim

The apply and render tools were tested with Isaac Sim 5.1 and its bundled Python.
Other Isaac versions are unverified. Supply the prims that represent the calibrated
robot base and rigid hand TCP in the team's USD scene:

```bash
ISAAC_PYTHON tools/check_runtime.py --mode apply
ISAAC_PYTHON tools/apply_overlay_to_isaac.py --overlay overlay.yaml --stage TEAM_SCENE.usd --output TEAM_SCENE_WITH_CAMERAS.usda --base-prim ROBOT_BASE_PRIM --hand-tcp-prim HAND_TCP_PRIM --validate-only
ISAAC_PYTHON tools/apply_overlay_to_isaac.py --overlay overlay.yaml --stage TEAM_SCENE.usd --output TEAM_SCENE_WITH_CAMERAS.usda --base-prim ROBOT_BASE_PRIM --hand-tcp-prim HAND_TCP_PRIM
```

`--validate-only` opens the USD and validates stage units/up-axis, required prims,
articulation ancestry, dependencies, and target-prim conflicts. The default conflict
policy is `error`. Use `--on-conflict reuse`, `replace`, or `rename` only after inspecting
the existing prims. `--plan-only` is a lightweight path preview and does not inspect USD.

The three fixed cameras are authored below the supplied base prim. The wrist
D405 is authored below the supplied hand-TCP prim, so its view follows every
robot joint configuration through the existing articulation. The tool also
authors table-task, nominal-table-alignment, and threaded-plate calibration
frames; it does not copy or reference the team's geometry automatically.

## RGB Sensor Smoke Test

After applying the overlay, run the sensor path through Isaac's Camera API. Supplying
an articulation and a second safe joint pose also checks that the wrist view follows q.

```bash
ISAAC_PYTHON tools/check_runtime.py --mode render
ISAAC_PYTHON tools/smoke_render.py --overlay overlay.yaml --stage TEAM_SCENE_WITH_CAMERAS.usda --output-dir smoke_render --base-prim ROBOT_BASE_PRIM --hand-tcp-prim HAND_TCP_PRIM
```

## Important

- `fixed_camera_board_scene` and `wrist_handeye_board_scene` are different physical board placements used only for validation.
- The wrist calibration is the depth-refined result. The earlier non-depth-refined estimate is history only.
- Reinstalling or moving the taped wrist camera invalidates the wrist transform.
- Moving a fixed camera, the table, or the robot base invalidates the corresponding calibration.
