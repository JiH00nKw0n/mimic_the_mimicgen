"""Register the lab FR3 peg-insert Mimic gym task.

Imported (after Isaac Sim launches) by the annotate/generate runners so that
``gym.make("Isaac-PegInsert-LabFR3-IK-Rel-Mimic-v0")`` resolves to our
LabFR3PegInsertIKRelMimicEnv (a FrankaCubeStackIKRelMimicEnv subclass that fixes the IK-rel
action frame for the yaw-180 FR3 base and returns peg/socket object poses — see peg_mimic_env.py)
with our lab peg env config (peg_mimic_cfg.py).
"""

import os

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

# 청크마다 다른 난수 흐름을 쓰게 한다. 오케스트레이터가 LAB_GEN_SEED를 넣어 주는데 이걸
# 읽는 곳이 없으면 청크마다, 컨테이너마다 같은 초기 배치를 뽑아 같은 데이터를 여러 벌
# 만든다. 큐브 쪽 lab_register.py에는 이 블록이 있고 peg 쪽에는 없었다.
_seed = os.environ.get("LAB_GEN_SEED", "").strip()
if _seed:
    import random

    import numpy as np
    import torch

    _s = int(_seed)
    random.seed(_s)
    np.random.seed(_s % (2 ** 32))
    torch.manual_seed(_s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(_s)
    print(f"[peg_register] seeded random/numpy/torch with LAB_GEN_SEED={_s}")
