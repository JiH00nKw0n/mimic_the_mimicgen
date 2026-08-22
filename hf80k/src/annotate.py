#!/usr/bin/env python3
"""사람 데모에 구간 경계 표시를 붙인다. 컨테이너 안에서 도는 파이프라인 단계다.

왜 필요한가. MimicGen은 아무 데모나 받지 않는다. 데모마다 "여기까지가 집는 구간,
여기부터가 넣는 구간"이라는 표시가 `obs/datagen_info/subtask_term_signals` 아래에 있어야
구간별로 궤적을 옮겨 붙일 수 있다. 텔레오퍼레이션팀이 주는 파일에는 그 표시가 없다.
큐브도 peg도 없다.

지금까지 큐브가 돌았던 이유는 누군가 자기 기계에서 `lab_stack_mimic/run_annotate.sh`를
한 번 돌려 표시된 파일을 만들어 두었기 때문이다. 그 스크립트는 특정 경로와 가상환경에
묶여 있어 컨테이너 안에서는 돌지 않는다. 즉 원본에서 다시 만들라고 하면 재현할 수 없었다.
이 파일이 그 단계를 파이프라인 안으로 들인다.

방식은 생성 단계와 같다. Isaac Lab의 `annotate_demos.py`를 복사해 태스크 등록 모듈을
import 한 줄로 끼워 넣고 실행한다. 태스크마다 다른 것(태스크 이름, 끼워 넣을 모듈,
환경 디렉터리)은 전부 태스크 프로필에서 온다.

    annotate.py --input /work/annotate/peg_source.hdf5 \
                --output /opt/hf80k/assets/peg_annotated.hdf5

주의. `annotate_demos.py`는 **표시에 성공한 편수를 종료 코드로 돌려준다.** 12편을 다
성공하면 12로 끝난다. 보통의 "0이면 성공"으로 판정하면 성공이 실패로 뒤집히고 완전한
실패가 성공으로 보인다. 그래서 종료 코드가 아니라 만들어진 파일을 열어 판정한다.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import orchestrate as orch  # noqa: E402

ANNOTATE_SRC = ("/workspace/isaaclab/scripts/imitation_learning/isaaclab_mimic/"
                "annotate_demos.py")


def count_annotated(path: str) -> tuple[int, int, list[str]]:
    """(전체 데모, 표시가 붙은 데모, 구간 신호 이름들)."""
    try:
        import h5py
    except ImportError:
        return -1, -1, []
    if not os.path.isfile(path):
        return 0, 0, []
    with h5py.File(path, "r") as handle:
        if "data" not in handle:
            return 0, 0, []
        names = sorted(handle["data"].keys())
        marked, signals = 0, []
        for name in names:
            group = handle["data"][name]
            obs = group.get("obs")
            info = obs.get("datagen_info") if obs is not None else None
            terms = info.get("subtask_term_signals") if info is not None else None
            if terms is not None and len(terms.keys()):
                marked += 1
                if not signals:
                    signals = sorted(terms.keys())
        return len(names), marked, signals


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="표시가 없는 원본 데모 HDF5")
    ap.add_argument("--output", required=True,
                    help="표시된 데모를 쓸 곳. 반드시 절대 경로여야 한다")
    ap.add_argument("--device", default="cpu", help="cpu 또는 cuda")
    ap.add_argument("--json", action="store_true", help="결과를 JSON 한 줄로 찍는다")
    args = ap.parse_args(argv)

    src = os.path.abspath(args.input)
    # annotate_demos.py는 디렉터리 성분이 없는 출력 경로에서 시작도 못 하고 죽는다.
    dst = os.path.abspath(args.output)
    if not os.path.isfile(src):
        raise SystemExit(f"[annotate] 원본이 없다: {src}")
    os.makedirs(os.path.dirname(dst), exist_ok=True)

    total_in, marked_in, _ = count_annotated(src)
    if marked_in > 0:
        print(f"[annotate] {src}에 이미 {marked_in}/{total_in}편에 표시가 있다. "
              f"그래도 다시 붙인다.")

    cfg = orch.load_config()
    log_path = os.path.join(os.path.dirname(dst), "annotate.log")
    log = orch.make_logger(log_path)

    # 생성 단계와 같은 방식으로 스크립트를 복사하고 등록 모듈을 끼워 넣는다.
    patched = os.path.join(cfg["work_dir"], "annotate_lab.py")
    os.makedirs(cfg["work_dir"], exist_ok=True)
    inject = "".join(f"\nimport {name}" for name in orch.REGISTER_MODULES
                     if name.endswith("_register"))
    with open(ANNOTATE_SRC, encoding="utf-8") as handle:
        lines = handle.readlines()
    done = False
    for index, line in enumerate(lines):
        if not done and line.startswith("import isaaclab_mimic.envs"):
            lines[index] = line.rstrip("\n") + inject + "\n"
            done = True
    if not done:
        raise SystemExit("[annotate] 'import isaaclab_mimic.envs' 줄을 찾지 못했다")
    with open(patched, "w", encoding="utf-8") as handle:
        handle.writelines(lines)
    log(f"annotate: patched script -> {patched} (끼워 넣은 것: {inject.strip().splitlines()})")

    cmd = [orch.ISAACLAB_SH, "-p", patched,
           "--task", orch.TASK_ID, "--auto", "--headless",
           "--device", args.device,
           "--input_file", src, "--output_file", dst]
    env = orch.base_env(cfg, [orch.ENV_DIR])
    for key, value in (orch.PROFILE.get("generate.extra_env", {}) or {}).items():
        env[str(key)] = str(value)
    # 어노테이션은 기록된 데모를 그대로 재생해야 한다. 리셋 때 물체를 무작위로 옮기는
    # 이벤트가 남아 있으면 기록된 초기 배치가 지워지고, 재생하는 로봇이 빈 자리를 집는다.
    # peg에서 실제로 그렇게 되어 12편 전부 실패했다.
    env["LAB_DISABLE_RESET_RANDOMIZATION"] = "1"

    log(f"annotate: task={orch.TASK_ID} input={src} output={dst}")
    proc = subprocess.run(cmd, env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True)
    with open(log_path, "a") as handle:
        handle.write(proc.stdout or "")

    # 종료 코드는 성공 편수다. 판정은 파일을 열어서 한다.
    total, marked, signals = count_annotated(dst)
    ok = marked > 0
    summary = {
        "schema_version": "fr3.hf80k.annotate.v1",
        "task_id": orch.TASK_ID,
        "input": src, "output": dst,
        "demos_in": total_in, "demos_out": total, "demos_annotated": marked,
        "subtask_signals": signals,
        "exit_code_reported_as_count": proc.returncode,
        "ok": ok,
    }
    log(f"annotate: {marked}/{total}편에 표시가 붙었다. 구간 신호: {signals}")
    if args.json:
        print(json.dumps(summary, ensure_ascii=False))
    if not ok:
        tail = "\n".join((proc.stdout or "").splitlines()[-25:])
        print(f"[annotate] 표시가 하나도 붙지 않았다. 마지막 출력:\n{tail}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
