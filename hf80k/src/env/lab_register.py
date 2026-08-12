"""Register the hf80k lab FR3 3-cube-stack Mimic gym tasks (forward + reverse order).

Imported (after Isaac Sim launches) by the generate runner so that
`gym.make("Isaac-Stack-Cube-LabFR3-HF80K-Fwd-IK-Rel-Mimic-v0")` resolves to our
LabFR3CubeStackIKRelMimicEnv (a FrankaCubeStackIKRelMimicEnv subclass that fixes the
IK-rel action frame for the yaw-180 FR3 base — see lab_mimic_env.py) with our lab env
config.

WHY THE HF80K NAMES
-------------------
The originals in lab_stack_mimic/ register `Isaac-Stack-Cube-LabFR3-{Fwd,Rev}-IK-Rel-
Mimic-v0`, and the hf80k configs are NOT the same environment any more: the work surface
is a kinematic rigid body, the gripper actuator group is renamed, and (unless
PHYSICS_PROFILE=off) a calibrated SysID event randomizes the plant. gym.register keeps
the first registration of an id and silently ignores later ones, so if both modules ever
land in one interpreter the ids must differ or a run would quietly use the wrong scene.
Hence the HF80K infix, and hence we do NOT re-register the original ids here.

WHAT THIS FILE DOES NOT DO
--------------------------
Nothing here depends on NUM_ENVS: the number of parallel envs is a `--num_envs` CLI
argument to generate_dataset.py, applied to the cfg after `gym.make`, so a registration
is valid for every NUM_ENVS. Everything else (PHYSICS_PROFILE, SOURCE_DEMO_FILTER,
SUBTASK_OFFSETS, LAB_ARM_SCALE, …) is read inside the env cfg's `__post_init__`, which
runs per `gym.make`, so those knobs are honoured no matter how the task is instantiated.
We only echo the resolved values so every run log records the config it actually used.
"""

import os

import gymnasium as gym

ENTRY = "lab_mimic_env:LabFR3CubeStackIKRelMimicEnv"

_PREFIX = "Isaac-Stack-Cube-LabFR3-HF80K-"
_SUFFIX = "-IK-Rel-Mimic-v0"

_TASKS = {
    _PREFIX + "Fwd" + _SUFFIX: "lab_mimic_cfg:LabFR3CubeStackFwdMimicEnvCfg",
    _PREFIX + "Rev" + _SUFFIX: "lab_mimic_cfg:LabFR3CubeStackRevMimicEnvCfg",
}

# Echoed for the run log; the cfg itself is the single place that reads them.
_KNOBS = ("PHYSICS_PROFILE", "SOURCE_DEMO_FILTER", "SUBTASK_OFFSETS", "LAB_ARM_SCALE",
          "LAB_SYSID_BUNDLE_ROOT", "LAB_TABLE_USD")

for task_id, cfg_entry in _TASKS.items():
    if task_id not in gym.registry:
        gym.register(
            id=task_id,
            entry_point=ENTRY,
            kwargs={"env_cfg_entry_point": cfg_entry},
            disable_env_checker=True,
        )
        print(f"[lab_register] registered {task_id}")

# 청크마다 다른 난수 흐름을 쓰게 한다. 오케스트레이터가 LAB_GEN_SEED를 넣어 주는데,
# 이걸 읽는 곳이 없으면 컨테이너 4개가 같은 초기 배치를 뽑아 같은 데이터를 만든다.
# Isaac Lab의 generate_dataset.py에 --seed가 있으면 그쪽이 이기고, 없으면 여기가 유일한
# 경로다. 그래서 실패하면 조용히 넘어가지 않고 예외를 낸다.
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
    print(f"[lab_register] seeded random/numpy/torch with LAB_GEN_SEED={_s}")

print("[lab_register] " + " ".join(f"{k}={os.environ.get(k, '<default>')}" for k in _KNOBS))
