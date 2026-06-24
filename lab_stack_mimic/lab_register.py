"""Register the lab FR3 3-cube-stack Mimic gym tasks (forward + reverse order).

Imported (after Isaac Sim launches) by the annotate/generate runners so that
`gym.make("Isaac-Stack-Cube-LabFR3-{Fwd,Rev}-IK-Rel-Mimic-v0")` resolves to our
LabFR3CubeStackIKRelMimicEnv (a FrankaCubeStackIKRelMimicEnv subclass that fixes
the IK-rel action frame for the yaw-180 FR3 base — see lab_mimic_env.py) with our
lab env config.
"""

import gymnasium as gym

ENTRY = "lab_mimic_env:LabFR3CubeStackIKRelMimicEnv"

_TASKS = {
    "Isaac-Stack-Cube-LabFR3-Fwd-IK-Rel-Mimic-v0": "lab_mimic_cfg:LabFR3CubeStackFwdMimicEnvCfg",
    "Isaac-Stack-Cube-LabFR3-Rev-IK-Rel-Mimic-v0": "lab_mimic_cfg:LabFR3CubeStackRevMimicEnvCfg",
}

for task_id, cfg_entry in _TASKS.items():
    if task_id not in gym.registry:
        gym.register(
            id=task_id,
            entry_point=ENTRY,
            kwargs={"env_cfg_entry_point": cfg_entry},
            disable_env_checker=True,
        )
        print(f"[lab_register] registered {task_id}")
