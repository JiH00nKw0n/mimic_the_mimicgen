#!/usr/bin/env python3
"""프로필 로더가 잘못된 입력에도 죽지 않고 값을 제대로 꺼내는지 확인한다.

로더가 예외를 던지면 preflight의 검사 표가 통째로 안 나온다. 그래서 "무슨 일이 있어도
Profile을 돌려준다"가 이 코드의 계약이고, 이 시험이 그 계약을 지킨다.

    python3 src/tests/test_task_profile.py
"""
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

import task_profile as tp  # noqa: E402

failures = []


def check(label: str, condition: bool, detail: str = "") -> None:
    print(f"  {'통과' if condition else '어긋남'}  {label}{'  ' + detail if detail else ''}")
    if not condition:
        failures.append(label)


print("있는 프로필이 전부 형식을 통과한다")
names = tp.available()
check("프로필이 두 장 이상 있다", len(names) >= 2, str(names))
for name in names:
    one = tp.load(name)
    check(f"{name} 형식", not one.error, one.error)

print("\n큐브 프로필이 예전 상수와 같은 값을 준다")
cube = tp.load("cube_stack_fr3")
check("태스크 아이디",
      cube.get("generate.task_id") == "Isaac-Stack-Cube-LabFR3-HF80K-Fwd-IK-Rel-Mimic-v0")
check("소스 데모", cube.get("generate.source_hdf5") == "fwd_annotated.hdf5")
check("카메라 셋", cube.get("render.cameras") == ["third_person_0", "third_person_1", "wrist"])
check("태스크 문장",
      cube.get("dataset.task_string") == "Stack three cubes into a three-level tower")
check("끼워 넣을 모듈",
      cube.get("generate.register_modules") == ["lab_register", "clean_success_hook",
                                                "provenance_hooks"])
check("주된 물체", cube.get("physics.primary_objects") == ["cube_1", "cube_2", "cube_3"])
check("작업면", cube.get("physics.surface") == "work_surface")
check("질량 표 세 개", len(cube.get("physics.object_masses_kg") or {}) == 3)

print("\npeg 프로필이 큐브와 다른 값을 준다")
peg = tp.load("peg_insert_fr3")
check("태스크 아이디가 다르다",
      peg.get("generate.task_id") != cube.get("generate.task_id"))
check("그리퍼 액추에이터가 h", peg.get("physics.gripper_actuator") == "h")
check("주된 물체가 핀 하나", peg.get("physics.primary_objects") == ["peg"])
check("렌더가 태스크를 명시한다", bool(peg.get("render.task_id")))
check("성공 판정 모듈이 다르다",
      peg.get("render.success.module") != cube.get("render.success.module"))

print("\n잘못된 입력에도 죽지 않는다")
check("없는 이름", tp.load("이런건없다").error != "")
check("없는 이름이어도 기본 프로필로 돌아온다",
      tp.load("이런건없다").get("generate.task_id") == cube.get("generate.task_id"))

with tempfile.TemporaryDirectory() as tmp:
    original = tp.PROFILE_DIR
    tp.PROFILE_DIR = tmp
    try:
        open(os.path.join(tmp, "broken.yaml"), "w").write("[[[ 이건 야믈이 아니다")
        check("형식이 깨진 파일", tp.load("broken").error != "")
        open(os.path.join(tmp, "typo.yaml"), "w").write(
            "schema_version: fr3.hf80k.task_profile.v1\nid: typo\n"
            "generate:\n  task_id: x\n  register_modules: []\n  source_hdf5: y\n"
            "  taskid_오타: 1\n")
        one = tp.load("typo")
        check("모르는 키는 오류다", "모르는 키" in one.error, one.error)
        open(os.path.join(tmp, "noid.yaml"), "w").write(
            "schema_version: fr3.hf80k.task_profile.v1\nid: 다른이름\n")
        check("id가 파일 이름과 다르면 오류다", "id" in tp.load("noid").error)
    finally:
        tp.PROFILE_DIR = original

print()
if failures:
    print(f"어긋난 항목 {len(failures)}개: {', '.join(failures)}")
    sys.exit(1)
print("프로필 로더가 계약을 지킨다")
