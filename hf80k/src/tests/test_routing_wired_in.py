#!/usr/bin/env python3
"""등록된 모든 태스크의 환경 설정이 소스 선택 휴리스틱을 실제로 부르는지 확인한다.

왜 필요한가. 선택 휴리스틱은 환경 설정 클래스의 __post_init__ 안에서 붙는다. 같은 파일에
설정 클래스가 둘 이상 있으면 한쪽에만 적어 두기 쉽고, 그러면 설정 파일에는 켜져 있는데
실제로 쓰는 태스크에서는 기본 선택으로 도는 상태가 된다. 실제로 그런 일이 있었다. 큐브
파일의 역순 클래스에만 적었고 파이프라인이 쓰는 것은 정순 클래스라, 100편을 만드는 동안
휴리스틱이 한 번도 돌지 않았다. 로그에 아무 줄도 남지 않아 알아채기까지 시간이 걸렸다.

어떻게 확인하는가. 시뮬레이터를 띄우지 않는다. src/env/lab_register.py와
src/env_peg/peg_register.py가 태스크 이름마다 적어 둔 설정 클래스를 읽고, 그 클래스의
__post_init__에서 출발해 같은 파일의 모듈 수준 함수를 따라가며 source_routing.apply를
부르는 자리가 있는지 파이썬 구문 나무로 본다. 그래서 Isaac Lab이 없는 자리에서도 돈다.

    python3 src/tests/test_routing_wired_in.py
"""
import ast
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, ".."))

FAILURES = []


def check(name, condition, detail=""):
    if condition:
        print(f"  통과  {name}")
    else:
        print(f"  실패  {name}  {detail}")
        FAILURES.append(name)


def registered_classes(register_path):
    """등록 파일의 _TASKS 표에서 (태스크 이름, 모듈, 클래스) 세 쌍을 뽑는다.

    gym을 들여오지 않는다. 등록 파일을 그대로 실행하면 Isaac Lab이 필요하므로, 표만
    구문 나무에서 읽는다.
    """
    tree = ast.parse(open(register_path, encoding="utf-8").read())
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if "_TASKS" not in names or not isinstance(node.value, ast.Dict):
            continue
        for key, value in zip(node.value.keys, node.value.values):
            if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
                continue
            module, _, cls = value.value.partition(":")
            # 태스크 이름은 접두사와 접미사를 이어 붙여 적는 자리가 있어서 글자 그대로
            # 복원하지 않는다. 확인하려는 것은 클래스이므로 이름은 읽을 수 있는 만큼만 적는다.
            label = ast.unparse(key)
            out.append((label, module, cls))
    return out


def calls_in(node):
    """구문 나무 한 조각에서 부르는 이름을 모은다. 점이 붙은 이름은 마지막 두 조각으로 적는다."""
    found = set()
    for sub in ast.walk(node):
        if not isinstance(sub, ast.Call):
            continue
        fn = sub.func
        if isinstance(fn, ast.Name):
            found.add(fn.id)
        elif isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name):
            found.add(f"{fn.value.id}.{fn.attr}")
    return found


def reaches_routing(module_path, class_name):
    """클래스의 __post_init__에서 출발해 source_routing.apply에 닿는지 본다.

    같은 파일의 모듈 수준 함수까지 한 단계씩 따라간다. 환경 설정 파일은 공통 처리를
    _apply_... 꼴의 모듈 함수로 빼 두므로, 직접 부르는 자리만 보면 놓친다.
    """
    tree = ast.parse(open(module_path, encoding="utf-8").read())
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    target = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "__post_init__":
                    target = item
    if target is None:
        return False, f"{class_name}에 __post_init__이 없다"
    seen, queue = set(), [target]
    while queue:
        node = queue.pop()
        names = calls_in(node)
        if "source_routing.apply" in names:
            return True, ""
        for name in names:
            if name in functions and name not in seen:
                seen.add(name)
                queue.append(functions[name])
    return False, f"{class_name}.__post_init__에서 source_routing.apply에 닿지 않는다"


def main():
    print(__doc__.splitlines()[0])
    pairs = [
        ("큐브 쌓기", os.path.join(SRC, "env", "lab_register.py"), os.path.join(SRC, "env")),
        ("핀 꽂기", os.path.join(SRC, "env_peg", "peg_register.py"), os.path.join(SRC, "env_peg")),
    ]
    total = 0
    for label, register_path, env_dir in pairs:
        print(f"\n[{label}] {register_path}")
        entries = registered_classes(register_path)
        check(f"{label}의 등록 표에서 태스크를 하나 이상 읽었다", bool(entries),
              f"읽은 개수 {len(entries)}")
        for task_label, module, cls in entries:
            total += 1
            module_path = os.path.join(env_dir, f"{module}.py")
            ok, detail = reaches_routing(module_path, cls)
            check(f"{cls}가 선택 휴리스틱을 부른다 (태스크 {task_label})", ok, detail)

    print("\n[공통] 등록된 클래스가 빠짐없이 검사됐다")
    check("검사한 클래스가 3개 이상이다", total >= 3, f"실제 {total}개")

    print()
    if FAILURES:
        print(f"실패 {len(FAILURES)}건: {', '.join(FAILURES)}")
        return 1
    print("모두 통과했다")
    return 0


if __name__ == "__main__":
    sys.exit(main())
