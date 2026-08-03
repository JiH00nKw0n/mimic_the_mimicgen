# FR3 MimicGen control handoff — start here

This package freezes the robot-asset identity and controller boundary that the
FR3 three-cube MimicGen pipeline must match.  It is written so an agent can
audit the handoff without access to the sender's repository.

## Answer to the asset question

`assets/fr3_research3.usda` is the canonical robot layer used by UWLab.  It is
not a forked robot model: it is a thin USD wrapper whose only composition arc
references NVIDIA's versioned Isaac 5.1 Franka Research 3 asset:

```text
https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Robots/FrankaRobotics/FrankaFR3/fr3.usd
```

The wrapper SHA-256 is:

```text
6d5a3026a150b2c18a7cdbdefb31135319f005e563b9ce42e01090eabfac13ad
```

If another setup uses the same URL/version, the composed NVIDIA robot asset is
the same.  Differences seen in simulation should first be traced to the
IsaacLab spawn, actuator, controller, timestep, or SysID overrides below—not to
an assumed custom FR3 mesh or kinematic tree.

## Read order for Codex

1. `MANIFEST.yaml`
2. `contracts/control_contract.yaml`
3. `runtime/fr3_panda_gripper.py`
4. `runtime/cube_actions.py`
5. `runtime/actions_cfg.py` and `runtime/task_space_actions.py`
6. `sysid/SYSTEM_CALIBRATION_CONTRACT.yaml`
7. `sysid/nominal_and_ranges.yaml`
8. `contracts/mimicgen_inputs_required.yaml`
9. `contracts/dataset_schema.yaml`

Then run:

```bash
python3 verify_package.py
python3 tools/controller_adapter.py --self-test
```

## What is authored outside the USD

The effective runtime robot is the official NVIDIA asset plus the following
IsaacLab configuration:

- gravity disabled for the robot;
- self-collision enabled;
- 36 position solver iterations and 0 velocity solver iterations;
- arm command is joint effort from the task-space OSC;
- arm actuator stiffness and damping are both zero;
- effort limits are `[87, 87, 87, 87, 12, 12, 12]` Nm;
- velocity limits are `[2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61]` rad/s;
- the sim2real actuator can apply a measured/randomized motor delay;
- policy rate is 10 Hz, physics/OSC rate is 120 Hz, decimation is 12.

These are execution semantics, not edits embedded in
`fr3_research3.usda`.  Copying only the USDA is therefore insufficient for
controller-equivalent MimicGen replay.

## Canonical Cube/MimicGen action contract

Use contract `fr3_cube_stage1_model4500_legacyosc_v1`:

```text
action = [dx, dy, dz, drx, dry, drz, gripper]
scale  = [0.02, 0.02, 0.02, 0.02, 0.02, 0.2]
frame  = robot base
quat   = wxyz, q_target = q_delta * q_current_actual
open   = gripper > 0
close  = gripper <= 0
clip   = none
```

The relative target must be constructed from the current **actual** end-effector
pose at every 10 Hz policy step.  Do not accumulate from the previous desired
target and do not add a hidden `[-1, 1]` clip.

`runtime/cube_actions.py` contains multiple controller profiles.  For the
Stage-1/MimicGen contract, the authoritative entry is
`FR3_CUBE_STAGE1_RELATIVE_OSC` / `Fr3CubeStage1RelativeOSCAction`.  Do not
silently substitute `FR3_RELATIVE_OSC`, `FR3_RELATIVE_OSC_EVAL`, or IsaacLab's
built-in OSC.

## SysID separation

`sysid/` describes plant uncertainty and motor delay used for sim2real.  It must
be applied around the frozen action/controller contract; it does not authorize
retuning the OSC or changing action semantics.  Log the sampled dynamics row
and seed for every generated episode.

## Deliberately excluded files

- `fr3_panda.usd`: a different legacy Panda wrapper; it is not the active FR3.
- `fr3_research3_massfix.usda`: an experimental auxiliary-frame mass override;
  it is not used by the referenced Cube training configuration.
- RL checkpoint weights: not needed to implement or validate the MimicGen
  action/controller boundary.
- NVIDIA's binary `fr3.usd`: not redistributed; the exact official versioned
  URL is preserved in the wrapper.

## Required first report from the receiving agent

Report:

1. wrapper hash and referenced NVIDIA URL;
2. mapped robot base, controller EE, hand TCP, and wrist-camera frames;
3. action order, scale, frame, quaternion convention, gripper sign, and rates;
4. whether any local actuator/spawn setting differs from this package;
5. one pose/action round-trip result;
6. one closed-loop replay result with tracking error and success;
7. every missing external input from `mimicgen_inputs_required.yaml`.

