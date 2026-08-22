#!/usr/bin/env python3
"""물리 항이 태스크마다 넘기는 인자를 실제로 받는지 확인한다.

왜 필요한가. Isaac의 이벤트 관리자는 이 항을 `func(env, env_ids, **params)`로 부른다.
그래서 `__init__`이 `cfg.params`에서 어떤 이름을 읽더라도, 같은 이름이 `__call__`의
인자 목록에 없으면 넘기는 순간 TypeError로 죽는다. 물리는 한 줄도 적용되지 않는다.

실제로 한 번 그렇게 됐다. 장면 물체를 역할로 받게 고치면서 `__init__`만 고치고
`__call__`을 그대로 둬서, 새 이름을 넘기면 죽는 상태가 커밋됐다. 이 시험이 그것을
Isaac 없이 잡는다.

    python3 src/tests/test_task_roles.py
"""
import inspect
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from test_physics_equivalence import load_module  # noqa: E402

TARGET = os.path.join(HERE, "..", "env", "calibrated_sysid.py")

# 태스크 프로필이 넘길 수 있는 이름. 하나라도 __call__에 없으면 그 태스크는 죽는다.
ROLE_PARAMS = (
    "object_names",
    "surface_name",
    "object_masses",
    "object_size_m",
    "gripper_actuator_name",
)
# 예전부터 쓰던 이름. 계속 받아야 기존 호출부가 그대로 돈다.
LEGACY_PARAMS = (
    "bundle_root",
    "profile",
    "robot_cfg",
    "cube_names",
    "work_surface_name",
    "arm_actuator_name",
    "sample_seed_offset",
    "log_samples",
)


def main() -> int:
    module = load_module(TARGET, "sysid_roles")
    call = module.apply_fr3_cube_calibration_bundle.__call__
    accepted = set(inspect.signature(call).parameters)

    failures = []
    for name in ROLE_PARAMS + LEGACY_PARAMS:
        mark = "받는다" if name in accepted else "없다"
        print(f"  {name:24s} {mark}")
        if name not in accepted:
            failures.append(name)

    # __init__이 cfg.params에서 읽는 이름도 전부 __call__에 있어야 한다. 둘이 어긋나면
    # 읽기는 하는데 넘길 수 없는 죽은 설정이 된다.
    source = open(TARGET, encoding="utf-8").read()
    init_body = source.split("def __init__", 1)[1].split("def ", 1)[0]
    read_names = set()
    for token in init_body.split('cfg.params.get("')[1:]:
        read_names.add(token.split('"')[0])
    print()
    for name in sorted(read_names):
        if name not in accepted:
            print(f"  __init__이 읽지만 __call__이 못 받는 이름: {name}")
            failures.append(name)

    print()
    if failures:
        print(f"어긋난 이름 {len(failures)}개: {', '.join(sorted(set(failures)))}")
        return 1
    print("프로필이 넘길 이름을 물리 항이 모두 받는다")
    return 0


if __name__ == "__main__":
    sys.exit(main())
