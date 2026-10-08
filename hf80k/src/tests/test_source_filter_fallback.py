#!/usr/bin/env python3
"""소스별 수율 표가 없는 태스크에서도 설정을 한 줄도 고치지 않고 시작되는지 확인한다.

왜 필요한가. 기본 설정 SOURCE_DEMO_FILTER=exclude_zero_yield는 시연마다 측정한 수율을
적어 둔 표를 읽어, 성공이 한 번도 없던 시연을 통째로 뺀다. 큐브 쌓기에는 그 표가 있지만
핀 꽂기에는 없다. 태스크 프로필의 generate.source_yield_json이 빈 문자열이다. 그 상태로
기본값을 그대로 두면 실행 전 검사가 두 가지로 막는다. 없는 자산 목록에 빈 이름이 들어가
"없는 자산: " 뒤에 아무것도 없는 문구가 뜨고, 소스 거르기 검사가 표를 찾지 못했다고
적는다. 설정을 건드리지 않은 사람이 시작조차 못 하는 상태였다.

무엇을 확인하는가. 네 가지다. 표가 없는 태스크에서는 설정이 "all"로 내려간다는 것,
그 값이 환경변수로도 바뀌어 자식 프로세스가 같은 값을 읽는다는 것, 사람이 직접 준 값은
내리지 않는다는 것, 그리고 표가 있는 태스크에서는 기본값이 그대로 남는다는 것이다.

    python3 src/tests/test_source_filter_fallback.py
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


def have_yaml():
    """프로필 파일을 읽을 수 있는 자리인지 본다.

    프로필은 YAML 한 장이고, 읽는 데 PyYAML이 필요하다. 컨테이너 안에는 있고 개발용
    맥에는 없을 수 있다. 없으면 task_profile.load가 빈 프로필을 돌려주고, 그때는 어떤
    태스크 이름을 줘도 코드에 적힌 기본값(큐브 쪽 값)이 쓰인다. 그래서 프로필에 의존하는
    확인은 PyYAML이 있을 때만 한다. 되돌림 동작 자체는 아래 [1]에서 프로필과 무관하게
    확인하므로, PyYAML이 없는 자리에서도 이 시험의 핵심은 그대로 돈다.
    """
    try:
        import yaml                       # noqa: F401
    except ImportError:
        return False
    return True


def build_cfg(env_overrides, yield_json=None):
    """오케스트레이터를 새로 읽어 설정을 만들고, 함께 환경변수가 어떻게 됐는지 돌려준다.

    load_config은 표가 없을 때 os.environ도 함께 바꾼다. 자식 프로세스가 그 값을 다시
    읽기 때문이다. 그래서 되돌리기 전에 값을 읽어 둔다.

    yield_json을 주면 모듈이 정한 수율 표 경로를 그 값으로 바꾼 뒤 설정을 만든다. 빈
    문자열을 주면 "표가 없는 태스크"와 같은 상태가 되므로, 프로필을 읽을 수 없는 자리에서도
    되돌림 동작을 확인할 수 있다.
    """
    watched = ("SOURCE_DEMO_FILTER",)
    saved = {k: os.environ.get(k) for k in list(env_overrides) + list(watched)}
    for k in watched:
        os.environ.pop(k, None)
    os.environ.update({k: str(v) for k, v in env_overrides.items()})
    for mod in ("orchestrate", "task_profile", "preflight"):
        sys.modules.pop(mod, None)
    try:
        import orchestrate
        if yield_json is not None:
            orchestrate.SOURCE_YIELD_JSON = yield_json
        cfg = orchestrate.load_config()
        return cfg, {k: os.environ.get(k) for k in watched}, orchestrate.SOURCE_YIELD_JSON
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        for mod in ("orchestrate", "task_profile", "preflight"):
            sys.modules.pop(mod, None)


def main():
    print(__doc__.splitlines()[0])
    base = {"WORK_DIR": "/tmp/filtertest", "HF_UPLOAD": "0"}

    print("\n[1] 수율 표 경로가 비어 있으면 시연을 하나도 빼지 않는다")
    cfg, env, _ = build_cfg({**base, "TASK_PROFILE": "peg_insert_fr3"}, yield_json="")
    check("설정이 'all'로 내려갔다", cfg["source_demo_filter"] == "all",
          f"실제 {cfg['source_demo_filter']!r}")
    check("환경변수도 'all'이 되어 자식 프로세스가 같은 값을 읽는다",
          env["SOURCE_DEMO_FILTER"] == "all", f"실제 {env['SOURCE_DEMO_FILTER']!r}")
    cfg, env, _ = build_cfg({**base, "TASK_PROFILE": "peg_insert_fr3"},
                            yield_json="/does/not/exist/source_yield.json")
    check("경로가 있어도 파일이 없으면 'all'로 내려간다",
          cfg["source_demo_filter"] == "all", f"실제 {cfg['source_demo_filter']!r}")

    print("\n[2] 표를 읽을 수 있으면 기본값이 그대로다")
    cfg, env, yield_path = build_cfg({**base, "TASK_PROFILE": "cube_stack_fr3"})
    check("큐브 프로필에는 수율 표 경로가 있다", bool(yield_path), f"실제 {yield_path!r}")
    check("설정이 'exclude_zero_yield'로 남아 있다",
          cfg["source_demo_filter"] == "exclude_zero_yield",
          f"실제 {cfg['source_demo_filter']!r}")

    print("\n[3] 사람이 직접 준 값은 내리지 않는다")
    for given in ("exclude_zero_yield", "all", "0,1,2"):
        cfg, env, _ = build_cfg({**base, "TASK_PROFILE": "peg_insert_fr3",
                                 "SOURCE_DEMO_FILTER": given}, yield_json="")
        check(f"SOURCE_DEMO_FILTER={given!r}를 주면 그 값이 남는다",
              cfg["source_demo_filter"] == given, f"실제 {cfg['source_demo_filter']!r}")

    print("\n[3-1] 핀 꽂기 프로필이 실제로 표를 적지 않았다")
    if have_yaml():
        cfg, env, yield_path = build_cfg({**base, "TASK_PROFILE": "peg_insert_fr3"})
        check("핀 꽂기 프로필의 수율 표 경로가 비어 있다", not yield_path,
              f"실제 {yield_path!r}")
        check("그래서 설정이 'all'로 내려간다", cfg["source_demo_filter"] == "all",
              f"실제 {cfg['source_demo_filter']!r}")
    else:
        print("  건너뜀  PyYAML이 없어 프로필을 읽을 수 없다. "
              "컨테이너 안에서 다시 돌리면 이 두 줄도 확인된다")

    print("\n[4] 실행 전 검사가 두 항목 모두 통과한다 (핀 꽂기 프로필)")
    if not have_yaml():
        print("  건너뜀  PyYAML이 없어 프로필을 읽을 수 없다")
        print()
        if FAILURES:
            print(f"실패 {len(FAILURES)}건: {', '.join(FAILURES)}")
            return 1
        print("모두 통과했다 (프로필이 필요한 확인은 건너뛰었다)")
        return 0

    saved = {k: os.environ.get(k) for k in ("TASK_PROFILE", "WORK_DIR", "HF_UPLOAD",
                                            "SOURCE_DEMO_FILTER")}
    os.environ.pop("SOURCE_DEMO_FILTER", None)
    os.environ.update({"TASK_PROFILE": "peg_insert_fr3", "WORK_DIR": "/tmp/filtertest",
                       "HF_UPLOAD": "0"})
    for mod in ("orchestrate", "task_profile", "preflight"):
        sys.modules.pop(mod, None)
    try:
        import orchestrate
        import preflight
        cfg = orchestrate.load_config()
        # 자산 검사는 이미지 안의 실제 파일을 보므로, 여기서는 빈 이름이 목록에 들어가지
        # 않는 것만 본다. 자산이 없는 자리에서도 "없는 자산" 문구에 빈 칸이 생기면 안 된다.
        status, detail = preflight.check_assets(cfg)
        empty_name = detail.endswith(": ") or ", ," in detail or detail.endswith(", ")
        check("자산 검사 문구에 이름이 빈 항목이 없다", not empty_name, f"실제 {detail!r}")
        status, detail = preflight.check_source_filter(cfg)
        check("소스 거르기 검사가 통과한다", status == "PASS", f"{status} {detail}")
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        for mod in ("orchestrate", "task_profile", "preflight"):
            sys.modules.pop(mod, None)

    print()
    if FAILURES:
        print(f"실패 {len(FAILURES)}건: {', '.join(FAILURES)}")
        return 1
    print("모두 통과했다")
    return 0


if __name__ == "__main__":
    sys.exit(main())
