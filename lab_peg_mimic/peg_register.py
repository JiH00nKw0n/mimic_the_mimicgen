"""Register the lab FR3 peg-insert Mimic gym task.

Imported (after Isaac Sim launches) by the annotate/generate runners so that
``gym.make("Isaac-PegInsert-LabFR3-IK-Rel-Mimic-v0")`` resolves to our
LabFR3PegInsertIKRelMimicEnv (a FrankaCubeStackIKRelMimicEnv subclass that fixes the IK-rel
action frame for the yaw-180 FR3 base and returns peg/socket object poses — see peg_mimic_env.py)
with our lab peg env config (peg_mimic_cfg.py).
"""

import gymnasium as gym

ENTRY = "peg_mimic_env:LabFR3PegInsertIKRelMimicEnv"

_TASKS = {
    "Isaac-PegInsert-LabFR3-IK-Rel-Mimic-v0": "peg_mimic_cfg:LabFR3PegInsertMimicEnvCfg",
}

for task_id, cfg_entry in _TASKS.items():
    if task_id not in gym.registry:
        gym.register(
            id=task_id,
            entry_point=ENTRY,
            kwargs={"env_cfg_entry_point": cfg_entry},
            disable_env_checker=True,
        )
        print(f"[peg_register] registered {task_id}")
