#!/usr/bin/env python3
"""물리 값을 코드에서 파일로 옮긴 뒤에도 큐브 결과가 그대로인지 확인한다.

왜 필요한가. `calibrated_sysid.py`의 표본 추출에는 숫자가 직접 적혀 있었다. 그 숫자를
번들의 `parameters.json`에서 읽도록 바꿨는데, 뽑는 순서나 인자가 하나라도 달라지면 같은
시드에서도 다른 물리가 나온다. 그러면 지금까지 만든 큐브 데이터와 앞으로 만들 데이터가
조용히 달라진다. 그래서 옛 코드와 새 코드의 표본을 같은 시드로 뽑아 비트까지 비교한다.

Isaac Sim이 없어도 돌아야 하므로 표본 추출 함수만 떼어 내 부른다. 장면 물체나 PhysX는
건드리지 않는다.

    python3 src/tests/test_physics_equivalence.py <표준_번들_경로> [옛_코드_경로]
"""
import importlib.util
import os
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NEW_SRC = os.path.join(HERE, "..", "env", "calibrated_sysid.py")


def load_module(path: str, name: str):
    """Isaac 의존성을 가짜로 채워 넣고 모듈을 불러온다."""
    for missing in ("carb", "warp", "torch"):
        if missing not in sys.modules:
            sys.modules[missing] = types.ModuleType(missing)
    for pkg, attrs in (
        ("isaaclab.assets", ("Articulation", "RigidObject")),
        ("isaaclab.envs", ("ManagerBasedEnv",)),
        ("isaaclab.managers", ("EventTermCfg", "ManagerTermBase", "SceneEntityCfg")),
    ):
        module = types.ModuleType(pkg)
        for attr in attrs:
            # 기본 인자에서 SceneEntityCfg("robot") 처럼 불리므로 인자를 받아 삼켜야 한다.
            setattr(module, attr, type(attr, (), {
                "__init__": lambda self, *a, **k: None,
                "name": "robot",
            }))
        sys.modules[pkg] = module
        parent = pkg.split(".")[0]
        sys.modules.setdefault(parent, types.ModuleType(parent))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_instance(module, bundle_root: str, profile: str):
    """__init__을 거치지 않고 표본 추출에 필요한 상태만 채운 객체를 만든다."""
    cls = module.apply_fr3_cube_calibration_bundle
    obj = cls.__new__(cls)
    obj.profile = profile
    from pathlib import Path
    root = Path(bundle_root)
    obj._dynamics_rows = module._read_csv(
        root / "modules/dynamics_controller/domain_randomization_samples.csv")
    obj._contact_rows = module._read_csv(root / "modules/contact/posterior_samples.csv")
    if hasattr(module, "_read_params"):
        params = module._read_params(root / "parameters.json")
        obj._params = params
        obj._nominal = params["contact"].get("nominal", {})
        obj._range = params["contact"].get("range", {})
        obj._constraint = params["contact"].get("constraint", {})
        obj._sampling = params.get("sampling", {})
        obj._joint_nominal = (params.get("joint") or {}).get("nominal", {})
    return obj


def main() -> int:
    if len(sys.argv) < 2:
        print("사용법: test_physics_equivalence.py <표준_번들> [옛_코드]", file=sys.stderr)
        return 2
    bundle = sys.argv[1]
    old_path = sys.argv[2] if len(sys.argv) > 2 else "/tmp/old_sysid.py"
    if not os.path.isfile(old_path):
        print(f"옛 코드가 없다: {old_path}. git show 로 꺼내 둔다.", file=sys.stderr)
        return 2

    # 옛 코드는 원래 큐브 번들을 그대로 읽는다. 표준 번들에는 열 이름이 다르므로,
    # 옛 코드에는 원본 큐브 번들을 주고 새 코드에는 표준 번들을 준다.
    original = sys.argv[3] if len(sys.argv) > 3 else bundle
    new_mod = load_module(NEW_SRC, "sysid_new")
    old_mod = load_module(old_path, "sysid_old")

    failures = 0
    for profile in ("nominal", "posterior_stochastic", "robust_stochastic"):
        for count, seed in ((16, 42000), (256, 7), (4, 123456)):
            new_obj = make_instance(new_mod, bundle, profile)
            old_obj = make_instance(old_mod, original, profile)
            a = new_obj._sample(count, seed)
            b = old_obj._sample(count, seed)
            if set(a) != set(b):
                print(f"  키가 다르다 {profile} n={count}: {set(a) ^ set(b)}")
                failures += 1
                continue
            for key in sorted(a):
                x, y = np.asarray(a[key]), np.asarray(b[key])
                if x.shape != y.shape or not np.array_equal(x, y):
                    diff = float(np.max(np.abs(x.astype(float) - y.astype(float)))) \
                        if x.shape == y.shape else float("nan")
                    print(f"  다름 {profile} n={count} seed={seed} 항목={key} 최대차={diff}")
                    failures += 1
            if failures == 0:
                print(f"  같음 {profile:22s} n={count:4d} seed={seed} "
                      f"({len(a)}개 항목 전부 비트까지 일치)")
    print()
    if failures:
        print(f"어긋난 비교 {failures}건")
        return 1
    print("옛 코드와 새 코드의 표본이 모든 프로파일에서 비트까지 같다")
    return 0


if __name__ == "__main__":
    sys.exit(main())
