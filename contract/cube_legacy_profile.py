#!/usr/bin/env python3
"""Apply the immutable Cube model-4500 training controller to one env config.

Use this on a task-local config instance. Do not edit the global FR3 action
constants, because other trained policies use a later calibrated profile.
"""

MOTION_STIFFNESS = (200.0, 200.0, 200.0, 3.0, 3.0, 3.0)
MOTION_DAMPING_RATIO = (3.0, 3.0, 3.0, 1.0, 1.0, 1.0)
NULLSPACE_STIFFNESS = 10.0
NULLSPACE_DAMPING_RATIO = 1.0
SCALE_XYZ_AXISANGLE = (0.02, 0.02, 0.02, 0.02, 0.02, 0.2)
TORQUE_LIMIT = (87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0)


def apply_to_env_cfg(env_cfg):
    """Mutate and return a Cube env config instance before `gym.make`."""

    arm = env_cfg.actions.arm
    arm.scale_xyz_axisangle = SCALE_XYZ_AXISANGLE
    arm.input_clip = None
    arm.motion_stiffness = MOTION_STIFFNESS
    arm.motion_damping_ratio = MOTION_DAMPING_RATIO
    arm.torque_limit = TORQUE_LIMIT
    arm.use_physx_jacobian = True
    arm.nullspace_stiffness = NULLSPACE_STIFFNESS
    arm.nullspace_damping_ratio = NULLSPACE_DAMPING_RATIO
    arm.nullspace_regulate_to_reset = True
    arm.jacobian_damping = 0.01
    env_cfg.sim.dt = 1.0 / 120.0
    env_cfg.decimation = 12
    return env_cfg


def describe() -> dict[str, object]:
    return {
        "motion_stiffness": MOTION_STIFFNESS,
        "motion_damping_ratio": MOTION_DAMPING_RATIO,
        "nullspace_stiffness": NULLSPACE_STIFFNESS,
        "nullspace_damping_ratio": NULLSPACE_DAMPING_RATIO,
        "scale_xyz_axisangle": SCALE_XYZ_AXISANGLE,
        "torque_limit": TORQUE_LIMIT,
        "physics_dt": 1.0 / 120.0,
        "decimation": 12,
    }


if __name__ == "__main__":
    print(describe())

