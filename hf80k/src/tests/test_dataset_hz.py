#!/usr/bin/env python3
"""초당 프레임 수를 설정 한 곳에서 바꾸면 세 단계가 함께 따라가는지 확인한다.

왜 필요한가. 최종 데이터셋의 초당 프레임 수는 세 단계에 동시에 나타난다. 변환이 손끝
궤적을 다시 뽑는 주기, 렌더가 원본에서 몇 스텝에 하나씩 뽑을지, 기록이 데이터셋에 적는
값이다. 예전에는 세 자리에 10이 각각 박혀 있었고, 렌더의 건너뛰기 간격은 아예 코드에
숫자 2로 적혀 있었다. 그 상태에서 한 곳만 바꾸면 영상과 행동이 서로 다른 시각을 가리키게
된다. 프레임 번호가 아니라 시간으로 맞추는 기록 단계가 그 어긋남을 조용히 흡수해서,
학습 자료에 잘못된 짝이 들어간 뒤에야 드러난다.

여기서 확인하는 것은 네 가지다. 기본값이 10이라는 것, DATASET_HZ를 주면 세 단계가 모두
그 값을 받는다는 것, 나누어떨어지지 않는 값은 실행 전에 거부한다는 것, 그리고 설정
파일에 값을 적어도 같은 결과가 나온다는 것이다.

    python3 src/tests/test_dataset_hz.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, SRC)

FAILURES = []


def check(name, condition, detail=""):
    if condition:
        print(f"  통과  {name}")
    else:
        print(f"  실패  {name}  {detail}")
        FAILURES.append(name)


def build_cfg(env_overrides):
    """오케스트레이터를 새로 읽어 설정을 만든다.

    환경변수를 프로세스 단위로 읽는 모듈이라, 값을 바꿀 때마다 다시 읽어야 한다.
    """
    saved = {k: os.environ.get(k) for k in env_overrides}
    os.environ.update({k: str(v) for k, v in env_overrides.items()})
    for mod in ("orchestrate", "task_profile"):
        sys.modules.pop(mod, None)
    try:
        import orchestrate
        return orchestrate.load_config()
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        for mod in ("orchestrate", "task_profile"):
            sys.modules.pop(mod, None)


def main():
    print(__doc__.splitlines()[0])
    base = {"TASK_PROFILE": "peg_insert_fr3", "WORK_DIR": "/tmp/hztest", "HF_UPLOAD": "0"}

    print("\n[1] 기본값")
    cfg = build_cfg(base)
    check("초당 프레임 수의 기본값이 10이다", cfg["dataset_hz"] == 10, f"실제 {cfg['dataset_hz']}")
    check("원본 초당 스텝 수의 기본값이 20이다", cfg["source_hz"] == 20, f"실제 {cfg['source_hz']}")
    check("렌더가 두 스텝에 하나씩 뽑는다", cfg["render_every"] == 2, f"실제 {cfg['render_every']}")

    print("\n[2] 값을 바꾸면 건너뛰기 간격이 따라간다")
    for hz, every in ((20, 1), (10, 2), (5, 4), (4, 5), (2, 10), (1, 20)):
        cfg = build_cfg({**base, "DATASET_HZ": hz})
        check(f"초당 {hz}장이면 {every}스텝에 하나씩 뽑는다",
              cfg["render_every"] == every and cfg["dataset_hz"] == hz,
              f"실제 간격 {cfg['render_every']}")

    print("\n[3] 나누어떨어지지 않는 값은 거부한다")
    for hz in (3, 7, 12, 30):
        try:
            build_cfg({**base, "DATASET_HZ": hz})
            stopped = False
        except SystemExit:
            stopped = True
        check(f"초당 {hz}장은 실행 전에 멈춘다", stopped)

    print("\n[4] 원본 초당 스텝 수도 바꿀 수 있다")
    cfg = build_cfg({**base, "SOURCE_HZ": 60, "DATASET_HZ": 30})
    check("원본 60에 목표 30이면 두 스텝에 하나씩 뽑는다",
          cfg["render_every"] == 2, f"실제 {cfg['render_every']}")

    print("\n[5] 0 이하는 거부한다")
    for name, val in (("DATASET_HZ", 0), ("DATASET_HZ", -1), ("SOURCE_HZ", 0)):
        try:
            build_cfg({**base, name: val})
            stopped = False
        except SystemExit:
            stopped = True
        check(f"{name}={val}은 실행 전에 멈춘다", stopped)

    print("\n[6] 설정 파일에 값이 적혀 있다")
    # PyYAML이 없는 환경에서도 돌아야 하므로 두 줄을 글자로 찾는다. 앞의 검사들은
    # PyYAML이 없으면 설정 파일 대신 코드 기본값을 확인하게 되는데, 기본값이 10이어야
    # 한다는 것 자체가 확인할 내용이므로 그대로 의미가 있다.
    import re
    for task in ("peg_insert_fr3", "cube_stack_fr3"):
        path = os.path.join(SRC, "profiles", f"{task}.yaml")
        text = open(path, encoding="utf-8").read()
        hz = re.search(r"^\s*hz:\s*(\d+)\s*$", text, re.M)
        src_hz = re.search(r"^\s*source_hz:\s*(\d+)\s*$", text, re.M)
        check(f"{task}에 hz가 10으로 적혀 있다",
              hz is not None and hz.group(1) == "10",
              f"실제 {hz.group(1) if hz else '없음'}")
        check(f"{task}에 source_hz가 20으로 적혀 있다",
              src_hz is not None and src_hz.group(1) == "20",
              f"실제 {src_hz.group(1) if src_hz else '없음'}")
        check(f"{task}에 옛 이름 fps가 남아 있지 않다",
              re.search(r"^\s*fps:", text, re.M) is None)

    print()
    if FAILURES:
        print(f"실패 {len(FAILURES)}건: {', '.join(FAILURES)}")
        return 1
    print("전부 통과했다")
    return 0


if __name__ == "__main__":
    sys.exit(main())
