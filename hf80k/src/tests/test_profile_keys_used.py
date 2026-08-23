#!/usr/bin/env python3
"""프로필에 적힌 값을 코드가 실제로 읽는지 확인한다.

왜 필요한가. 프로필은 "태스크마다 다른 값은 전부 여기 있다"고 약속하는 파일이다. 그런데
적어 두기만 하고 읽는 곳이 없으면, 값을 고쳐도 아무 일이 일어나지 않는다. 그러면 고친
사람은 반영됐다고 믿고, 파이프라인은 코드에 박힌 옛 값으로 계속 돈다.

실제로 그런 키가 열 개 있었다. 핀 삽입 프로필의 `physics.bundle_dir`이 실측 물리 번들을
가리키는데 아무도 읽지 않아서 핀 환경은 물리를 한 줄도 적용하지 않고 돌았다.
`render.success`도 읽히지 않아 렌더가 성공 표시를 남기지 않았고, 생성과 변환과 렌더를 다
통과한 10편이 기록 단계에서 전부 버려졌다. 큐브 프로필의 `arm_actuator`는 `arm`이라고
적혀 있었지만 실제 이름은 `a1`이었고, 읽는 곳이 없어 아무도 몰랐다.

방법은 단순하다. 프로필의 모든 잎 키를 점으로 이은 경로로 만들고, 그 문자열이 src 아래
파이썬 파일 어딘가에 따옴표로 감싸여 나타나는지 본다. 문자열을 찾는 방식이라 완벽하지는
않지만, 아무 데서도 읽지 않는 키는 확실히 잡는다.

    python3 src/tests/test_profile_keys_used.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, ".."))
PROFILE_DIR = os.path.join(SRC, "profiles")

# 값 자체가 이름인 절. 여기 아래 키는 코드가 이름으로 읽지 않고 통째로 넘긴다.
PASSTHROUGH_SECTIONS = ("generate.extra_env",)
# 절 이름만 있고 잎이 없는 경우를 위한 예외. 지금은 없다.
EXEMPT: set = set()


def source_text() -> str:
    body = []
    for root, _, files in os.walk(SRC):
        for name in files:
            if name.endswith(".py"):
                with open(os.path.join(root, name), encoding="utf-8") as fh:
                    body.append(fh.read())
    return "\n".join(body)


def leaves(node, prefix=""):
    if isinstance(node, dict):
        for key, value in node.items():
            yield from leaves(value, f"{prefix}.{key}" if prefix else key)
    elif isinstance(node, list) and node and isinstance(node[0], dict):
        for item in node:
            yield from leaves(item, prefix)
    elif prefix:
        yield prefix


def main() -> int:
    try:
        import yaml
    except ImportError:
        print("  건너뜀  PyYAML이 없다")
        return 0

    body = source_text()
    failures = []
    for filename in sorted(os.listdir(PROFILE_DIR)):
        if not filename.endswith(".yaml"):
            continue
        name = filename[:-5]
        with open(os.path.join(PROFILE_DIR, filename), encoding="utf-8") as fh:
            doc = yaml.safe_load(fh)
        doc.pop("schema_version", None)
        doc.pop("id", None)
        unread = set()
        for path in leaves(doc):
            if any(path.startswith(prefix) for prefix in PASSTHROUGH_SECTIONS):
                path = ".".join(path.split(".")[:2])
            parts = path.split(".")
            # 세 단계(render.success.module)와 두 단계(generate.task_id)를 모두 본다.
            for depth in (3, 2):
                if f'"{".".join(parts[:depth])}"' in body:
                    break
            else:
                unread.add(".".join(parts[:2]))
        unread -= EXEMPT
        if unread:
            print(f"  실패  {name}: 코드가 읽지 않는 키 {sorted(unread)}")
            failures.append(name)
        else:
            print(f"  통과  {name}: 모든 키를 코드가 읽는다")

    print()
    if failures:
        print("프로필 %d개에 죽은 키가 있다: %s" % (len(failures), ", ".join(failures)))
        return 1
    print("모든 프로필의 모든 키를 코드가 읽는다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
