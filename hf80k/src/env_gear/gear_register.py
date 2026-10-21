"""기어 조립 Mimic gym 태스크를 등록한다.

Isaac Sim이 뜬 뒤에 어노테이션과 생성 실행기가 불러온다. 그래야
``gym.make("Isaac-MultiGear-LabFR3-IK-Rel-Mimic-v0")``가 우리 환경 클래스와 설정으로
풀린다. 핀 꽂기의 peg_register와 같은 짜임이다.
"""

import os

import gymnasium as gym

ENTRY = "gear_mimic_env:LabFR3MultiGearIKRelMimicEnv"

_TASKS = {
    "Isaac-MultiGear-LabFR3-IK-Rel-Mimic-v0": "gear_mimic_cfg:LabFR3MultiGearMimicEnvCfg",
}

for task_id, cfg_entry in _TASKS.items():
    if task_id not in gym.registry:
        gym.register(
            id=task_id,
            entry_point=ENTRY,
            kwargs={"env_cfg_entry_point": cfg_entry},
            disable_env_checker=True,
        )
        print(f"[gear_register] registered {task_id}")

# 청크마다 다른 난수 흐름을 쓰게 한다. 오케스트레이터가 LAB_GEN_SEED를 넣어 주는데 이걸
# 읽는 곳이 없으면 청크마다, 컨테이너마다 같은 초기 배치를 뽑아 같은 데이터를 여러 벌
# 만든다. 핀 꽂기에서 이 블록이 빠져 있었던 적이 있다.
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
    print(f"[gear_register] seeded random/numpy/torch with LAB_GEN_SEED={_s}")
