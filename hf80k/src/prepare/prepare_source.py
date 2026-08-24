#!/usr/bin/env python3
"""녹화 원본에서 파이프라인이 받을 수 있는 시연 파일까지 만든다.

파이프라인 여섯 단계는 "구간 표시가 붙은 시연 파일"에서 시작한다. 여기서 구간 표시란
시연 안에서 어디까지가 집는 동작이고 어디부터가 놓는 동작인지를 적어 둔 것이다. 사람이
조종해 녹화한 원본 파일에는 그 표시가 없다. 이 파일이 원본에서 표시가 붙은 파일까지
가는 두 단계를 한 번에 수행한다.

    원본 시연
      ↓ [1] 다시 재생해 성공한 편만 남긴다   replay_filter.py
    재생에 성공한 시연
      ↓ [2] 구간 표시를 붙인다               annotate.py
    표시가 붙은 시연  ← 파이프라인이 여기서 시작한다

두 단계를 나눠 둔 이유가 있다. 재생에 실패하는 시연은 표시를 붙여도 쓸모가 없고, 표시를
붙이는 일이 재생보다 오래 걸린다. 먼저 걸러 내면 그만큼 시간을 아낀다. 실제로 큐브 쌓기에서
원본 50편 중 13편만 재생에 성공했다.

건너뛰기도 된다. 이미 재생을 마친 파일이 있으면 `--skip-replay`를 주고, 이미 표시가 붙어
있으면 `--skip-annotate`를 준다.

컨테이너 안에서 이렇게 돌린다. 보통은 `make prepare`가 대신 불러 준다.

    prepare_source.py --input /work/prepare/raw.hdf5 \
                      --output /opt/hf80k/assets/fwd_annotated.hdf5
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import orchestrate as orch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPLAY_SCRIPT = os.path.join(HERE, "replay_filter.py")
ANNOTATE_SCRIPT = os.path.join(os.path.dirname(HERE), "annotate.py")


def run(cmd: list, env: dict, label: str) -> int:
    """한 단계를 실행하고 걸린 시간을 찍는다."""
    print(f"[prepare] {label} 시작", flush=True)
    started = time.time()
    code = subprocess.call(cmd, env=env)
    print(f"[prepare] {label} 끝, 종료 코드 {code}, {time.time() - started:.1f}초", flush=True)
    return code


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="녹화 원본 시연 파일")
    ap.add_argument("--output", required=True,
                    help="표시가 붙은 시연을 쓸 곳. 절대 경로여야 한다")
    ap.add_argument("--work", default="",
                    help="중간 파일을 둘 폴더. 비우면 출력 파일 옆에 만든다")
    ap.add_argument("--skip-replay", dest="skip_replay", action="store_true",
                    help="재생 거르기를 건너뛴다. 입력이 이미 걸러진 파일일 때 쓴다")
    ap.add_argument("--skip-annotate", dest="skip_annotate", action="store_true",
                    help="표시 붙이기를 건너뛴다. 입력에 이미 표시가 있을 때 쓴다")
    ap.add_argument("--device", default="cpu", help="cpu 또는 cuda")
    ap.add_argument("--count", type=int, default=-1,
                    help="원본에서 앞에서 몇 편만 쓸지. -1이면 전부")
    ap.add_argument("--json", action="store_true", help="결과를 JSON 한 줄로 찍는다")
    args = ap.parse_args(argv)

    src = os.path.abspath(args.input)
    dst = os.path.abspath(args.output)
    if not os.path.isfile(src):
        raise SystemExit(f"[prepare] 원본이 없다: {src}")
    work = os.path.abspath(args.work) if args.work else os.path.join(
        os.path.dirname(dst), "prepare_work")
    os.makedirs(work, exist_ok=True)
    os.makedirs(os.path.dirname(dst), exist_ok=True)

    cfg = orch.load_config()
    profile_name = orch.PROFILE.name
    task = orch.PROFILE.get("render.task_id", "") or orch.TASK_ID
    register = orch.PROFILE.get("render.register_modules", []) or [
        m for m in orch.REGISTER_MODULES if m.endswith("_register")]

    summary = {"profile": profile_name, "input": src, "output": dst,
               "task": task, "steps": {}}

    # ------------------------------------------------------------ [1] 재생 거르기
    replayed = os.path.join(work, "replayed.hdf5")
    if args.skip_replay:
        replayed = src
        summary["steps"]["replay"] = {"skipped": True}
        print("[prepare] 재생 거르기를 건너뛴다", flush=True)
    else:
        for stale in (replayed, replayed + ".report.json"):
            if os.path.isfile(stale):
                os.remove(stale)
        cmd = [orch.ISAACLAB_SH, "-p", REPLAY_SCRIPT,
               "--device", args.device, "--headless",
               "--task", task,
               "--register", ",".join(register),
               "--input", src, "--output", replayed,
               "--report", os.path.join(work, "replay_report.json"),
               "--success-module", orch.RENDER_SUCCESS_MODULE,
               "--success-function", orch.RENDER_SUCCESS_FUNCTION,
               "--objects", ",".join(orch.CONVERT_OBJECTS)]
        if args.count >= 0:
            cmd += ["--count", str(args.count)]
        env = orch.base_env(cfg, [HERE, os.path.dirname(HERE), orch.ENV_DIR,
                                  orch.RENDER_DIR, orch.CONVERT_DIR])
        code = run(cmd, env, "재생 거르기")
        report = orch.read_json(os.path.join(work, "replay_report.json")) or {}
        summary["steps"]["replay"] = {
            "skipped": False, "exit_code": code,
            "replayed": report.get("replayed"), "passed": report.get("passed"),
            "pass_rate": report.get("pass_rate"), "output": replayed}
        # 종료 코드만 믿으면 안 된다. Isaac Sim 위에서 도는 스크립트는 파이썬이 예외로
        # 죽어도 종료 코드가 0으로 나오는 경우가 있다. 그러면 잘려 나간 파일을 들고 다음
        # 단계로 넘어가고, 거기서 "truncated file" 같은 엉뚱한 오류가 나서 진짜 원인이
        # 가려진다. 그래서 보고서가 쓰였는지, 통과한 편이 있는지를 직접 본다.
        if not report:
            summary["ok"] = False
            summary["reason"] = ("재생 단계가 보고서를 쓰지 못하고 끝났다. 종료 코드는 "
                                 f"{code}지만 실제로는 도중에 죽었다는 뜻이다. 위 로그에서 "
                                 "마지막 예외를 확인해라")
            print("[prepare] " + summary["reason"], flush=True)
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return 1
        if not report.get("passed"):
            summary["ok"] = False
            summary["reason"] = (f"{report.get('replayed')}편을 재생했는데 성공한 편이 "
                                 "하나도 없다. 원본의 명령이 그 파일 자신의 궤적을 만들어 낸 "
                                 "명령이 맞는지 확인해야 한다")
            print("[prepare] " + summary["reason"], flush=True)
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return 1
        if code != 0 or not os.path.isfile(replayed):
            summary["ok"] = False
            summary["reason"] = ("재생에 성공한 편이 하나도 없다. 원본의 명령이 그 파일 자신의 "
                                 "궤적을 만들어 낸 명령이 맞는지 확인해야 한다")
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return 1
        print(f"[prepare] {report.get('passed')}편이 재생에 성공했다", flush=True)

    # ------------------------------------------------------------ [2] 표시 붙이기
    if args.skip_annotate:
        summary["steps"]["annotate"] = {"skipped": True}
        if os.path.abspath(replayed) != dst:
            import shutil
            shutil.copyfile(replayed, dst)
        print("[prepare] 표시 붙이기를 건너뛴다", flush=True)
    else:
        cmd = [sys.executable, ANNOTATE_SCRIPT,
               "--input", replayed, "--output", dst,
               "--device", args.device, "--json"]
        code = run(cmd, os.environ.copy(), "표시 붙이기")
        summary["steps"]["annotate"] = {"skipped": False, "exit_code": code}
        if code != 0:
            summary["ok"] = False
            summary["reason"] = "표시 붙이기가 실패했다"
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return 1

    # ------------------------------------------------------------ 결과 확인
    sys.path.insert(0, os.path.dirname(HERE))
    import annotate as annotate_mod  # noqa: E402

    total, marked, signals = annotate_mod.count_annotated(dst)
    summary["result"] = {"demos": total, "annotated": marked, "signals": signals}
    summary["ok"] = marked > 0
    if not summary["ok"]:
        summary["reason"] = "표시가 붙은 편이 하나도 없다"
    print(f"[prepare] 결과: {dst} 에 {total}편, 그중 {marked}편에 표시가 있다. "
          f"신호 {signals}", flush=True)
    if args.json:
        print(json.dumps(summary, ensure_ascii=False))
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
