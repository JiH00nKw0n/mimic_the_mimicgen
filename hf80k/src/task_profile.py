#!/usr/bin/env python3
"""태스크마다 다른 것을 파일 하나로 모아 읽는다.

왜 필요한가. 파이프라인이 "큐브 쌓기"만 알고 있었다. Isaac 태스크 이름, 소스 데모 파일,
카메라 목록, 데이터셋에 적을 문장, 성공 판정, 장면 물체 이름이 코드 여기저기에 상수로
박혀 있어서, 다른 태스크를 넣으려면 코드를 고쳐야 했다.

이제 그 값들이 `src/profiles/<이름>.yaml` 한 장에 모인다. `.env`의 `TASK_PROFILE`로 고른다.
파이프라인 다섯 단계는 그대로 두고 태스크만 바뀐다.

이 로더는 **절대 예외를 던지지 않는다.** `preflight.py`가 모듈을 불러올 때 이 코드가 함께
불려 나오는데, 여기서 죽으면 "왜 실행이 거부됐는지" 알려 주는 검사 표 자체가 안 나온다.
파일이 없거나 형식이 틀리면 기본 프로필로 돌아가고 이유를 `error`에 한 줄로 남긴다.
프로필 안의 모르는 키는 경고가 아니라 오류로 본다. 오타 하나가 8만 편을 조용히 바꾼다.

    python3 src/task_profile.py --list
    python3 src/task_profile.py --print generate.task_id
    python3 src/task_profile.py --validate
"""
from __future__ import annotations

import argparse
import json
import os
import sys

PROFILE_SCHEMA = "fr3.hf80k.task_profile.v1"
PROFILE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "profiles")
DEFAULT_PROFILE = "cube_stack_fr3"

# 절마다 받을 수 있는 키. 여기 없는 키가 파일에 있으면 오타로 보고 거부한다.
ALLOWED = {
    "generate": {"task_id", "register_modules", "module_dir", "source_hdf5",
                 "source_yield_json", "arm_scale", "subtask_offsets", "extra_env"},
    "convert": {"task_id", "register_modules"},
    "render": {"task_id", "register_modules", "cameras", "overlay_yaml", "binding_yaml",
               "table_usd_env", "success"},
    "physics": {"bundle_dir", "primary_objects", "surface", "arm_actuator",
                "gripper_actuator", "object_masses_kg", "object_size_m"},
    "visual": {"package_dir", "object_prims"},
    "dataset": {"task_string", "robot_type", "fps", "schema_prefix"},
    "assets": {"required"},
}
REQUIRED = {
    "generate": {"task_id", "register_modules", "source_hdf5"},
    "render": {"cameras"},
    "physics": {"bundle_dir", "primary_objects"},
    "dataset": {"task_string", "robot_type"},
}


class Profile:
    """읽어 들인 프로필 한 장. 잘못됐으면 error에 이유가 들어 있다."""

    def __init__(self, name: str, doc: dict, error: str = "", path: str = ""):
        self.name = name
        self.doc = doc
        self.error = error
        self.path = path

    def get(self, dotted: str, default=None):
        """`generate.task_id`처럼 점으로 이어 붙인 경로로 값을 꺼낸다."""
        node = self.doc
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def __repr__(self) -> str:
        return f"<Profile {self.name}{' 오류: ' + self.error if self.error else ''}>"


def available() -> list[str]:
    if not os.path.isdir(PROFILE_DIR):
        return []
    return sorted(f[:-5] for f in os.listdir(PROFILE_DIR) if f.endswith(".yaml"))


def _validate(name: str, doc) -> str:
    """형식을 본다. 통과하면 빈 문자열, 아니면 이유 한 줄."""
    if not isinstance(doc, dict):
        return "파일 내용이 사전이 아니다"
    if doc.get("schema_version") != PROFILE_SCHEMA:
        return f"schema_version이 {PROFILE_SCHEMA}가 아니다: {doc.get('schema_version')!r}"
    if doc.get("id") != name:
        return f"파일 안의 id({doc.get('id')!r})가 파일 이름({name!r})과 다르다"
    for section, keys in ALLOWED.items():
        block = doc.get(section)
        if block is None:
            if section in REQUIRED:
                return f"{section} 절이 없다"
            continue
        if not isinstance(block, dict):
            return f"{section} 절이 사전이 아니다"
        unknown = set(block) - keys
        if unknown:
            return f"{section} 절에 모르는 키가 있다: {', '.join(sorted(unknown))}"
        missing = REQUIRED.get(section, set()) - set(block)
        if missing:
            return f"{section} 절에 필요한 키가 없다: {', '.join(sorted(missing))}"
    return ""


def load(name: str = "") -> Profile:
    """프로필을 읽는다. 무슨 일이 있어도 Profile을 돌려준다."""
    name = (name or os.environ.get("TASK_PROFILE", "") or DEFAULT_PROFILE).strip()
    path = os.path.join(PROFILE_DIR, f"{name}.yaml")
    fallback_reason = ""
    try:
        import yaml
    except ImportError:
        fallback_reason = "PyYAML이 없다"
    else:
        if not os.path.isfile(path):
            fallback_reason = f"파일이 없다: {path}"
        else:
            try:
                with open(path, encoding="utf-8") as stream:
                    doc = yaml.safe_load(stream)
            except Exception as exc:                    # noqa: BLE001
                fallback_reason = f"읽지 못했다: {type(exc).__name__}: {exc}"
            else:
                problem = _validate(name, doc)
                if problem:
                    fallback_reason = problem
                else:
                    return Profile(name, doc, "", path)

    if name != DEFAULT_PROFILE:
        base = load(DEFAULT_PROFILE)
        if not base.error:
            return Profile(DEFAULT_PROFILE, base.doc,
                           f"{name} 프로필을 쓰지 못해 {DEFAULT_PROFILE}로 되돌렸다. "
                           f"이유: {fallback_reason}", base.path)
    return Profile(name, {}, fallback_reason or "기본 프로필도 읽지 못했다", path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="", help="볼 프로필 이름. 비우면 TASK_PROFILE")
    ap.add_argument("--list", action="store_true", help="있는 프로필을 모두 찍는다")
    ap.add_argument("--print", dest="key", default="", help="점으로 이어 붙인 키 하나를 찍는다")
    ap.add_argument("--validate", action="store_true", help="형식만 확인하고 끝낸다")
    args = ap.parse_args(argv)

    if args.list:
        for name in available():
            one = load(name)
            print(f"  {name:22s} {'정상' if not one.error else '오류: ' + one.error}")
        return 0

    profile = load(args.profile)
    if args.validate:
        print(f"{profile.name}: {'정상' if not profile.error else profile.error}")
        return 1 if profile.error else 0
    if args.key:
        value = profile.get(args.key)
        print(json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value)
        return 0 if value is not None else 1
    print(json.dumps(profile.doc, ensure_ascii=False, indent=1))
    if profile.error:
        print(f"경고: {profile.error}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
