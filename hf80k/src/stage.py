#!/usr/bin/env python3
"""단계 하나만 따로 돌린다.

왜 필요한가. 지금까지 이 파이프라인의 실행 단위는 컨테이너 하나였다. `orchestrate.py`가
다섯 단계를 순서대로 다 돌리고 허깅페이스에 올리는 것까지 한 덩어리로 했다. 남의
파이프라인 안에 한 모듈로 끼워 넣으려면 "생성만 해 줘"나 "렌더만 해 줘"처럼 단계 하나를
부를 수 있어야 한다.

단계 함수는 이미 `stage_generate(cfg, chunk, log, log_path)`처럼 모양이 같다. 이 파일은
그 함수들에 명령줄을 붙여 주는 얇은 껍데기다. 오케스트레이터를 고치지 않는다.

    # 사람 데모에서 성공 궤적 50편을 만든다
    stage.py generate --chunk-dir /work/c0 --episodes 50 --profile nominal_lab --seed 42000

    # 그 궤적을 로봇 명령 형식으로 바꾼다
    stage.py convert --chunk-dir /work/c0

    # 카메라 세 대로 찍는다
    stage.py render --chunk-dir /work/c0 --profile nominal_lab

    # LeRobot 데이터셋으로 쓴다
    stage.py lerobot --chunk-dir /work/c0 --profile nominal_lab

    # 허깅페이스에 올린다 (HF_TOKEN과 HF_REPO_ID가 있어야 한다)
    stage.py upload --chunk-dir /work/c0 --chunk-index 0

단계마다 무엇이 들어가고 무엇이 나오는지는 `--io`로 확인한다. 설정은 `orchestrate.py`와
같은 환경변수를 쓰므로, 부르는 쪽은 `.env`를 그대로 재사용하면 된다.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import orchestrate as orch  # noqa: E402

# 단계 이름, 함수, 그리고 그 단계의 입출력. `--io`가 이 표를 그대로 찍는다. 남의
# 파이프라인이 우리를 부를 때 무엇을 준비하고 무엇을 받아 갈지를 여기서 읽으면 된다.
STAGES = {
    "generate": {
        "func": "stage_generate",
        "inputs": ["<작업디렉터리>/source_filtered.hdf5 (없으면 자산의 소스 데모에서 자동 생성)"],
        "outputs": ["<청크>/gen.hdf5", "<청크>/gen.provenance.json"],
        "needs": ["--episodes", "--seed"],
        "note": "Isaac Sim 안에서 MimicGen이 돈다. 성공한 궤적만 남는다.",
    },
    "sart": {
        "func": "stage_sart",
        "inputs": ["<청크>/gen.hdf5"],
        "outputs": ["<청크>/gen_sart.hdf5 (생성분 + 증강분을 묶은 링크 파일)",
                    "<청크>/sart_report.json"],
        "needs": [],
        "note": ("생성된 에피소드마다 장면을 되돌려 다시 굴리되 접근 구간만 다양화하고, "
                 "성공한 것을 더한다. 태스크 프로필이 켠 태스크에서만 돈다."),
    },
    "convert": {
        "func": "stage_convert",
        "inputs": ["<청크>/gen.hdf5 또는 <청크>/gen_sart.hdf5"],
        "outputs": ["<청크>/contract.hdf5", "<청크>/contract_report.json"],
        "needs": [],
        "note": "손끝 상대 명령 7개를 초당 10개로 다시 계산한다. 물리는 돌리지 않는다.",
    },
    "render": {
        "func": "stage_render",
        "inputs": ["<청크>/gen.hdf5 또는 <청크>/gen_sart.hdf5"],
        "outputs": ["<청크>/rgb.hdf5", "<청크>/vrand_log.json"],
        "needs": ["--profile"],
        "note": "궤적을 다시 재생하며 카메라 3대로 찍고, 성공 판정을 한 번 더 한다.",
    },
    "lerobot": {
        "func": "stage_lerobot",
        "inputs": ["<청크>/contract.hdf5", "<청크>/rgb.hdf5", "<청크>/vrand_log.json"],
        "outputs": ["<청크>/lerobot/ (LeRobot v3 데이터셋)"],
        "needs": ["--profile"],
        "note": "시각으로 영상과 명령을 맞추고 성공 표시 두 개를 다시 확인한다.",
    },
    "upload": {
        "func": "stage_upload",
        "inputs": ["<청크>/lerobot/"],
        "outputs": ["허깅페이스 저장소 chunks/<몫>/chunk_NNNNN"],
        "needs": ["--chunk-index"],
        "note": "HF_TOKEN과 HF_REPO_ID가 필요하다. 이 단계만 자격증명을 쓴다.",
    },
}


def count_demos(path: str) -> int:
    """HDF5 안의 데모 수. 단계를 따로 부를 때 앞 단계의 결과를 이어받는 데 쓴다."""
    if not os.path.isfile(path):
        return 0
    try:
        import h5py
        with h5py.File(path, "r") as handle:
            return len(handle["data"].keys()) if "data" in handle else 0
    except Exception:
        return 0


def print_io() -> int:
    print("단계별 입출력. <청크>는 --chunk-dir로 준 디렉터리다.\n")
    for name, spec in STAGES.items():
        print(f"[{name}] {spec['note']}")
        for item in spec["inputs"]:
            print(f"   입력  {item}")
        for item in spec["outputs"]:
            print(f"   출력  {item}")
        if spec["needs"]:
            print(f"   필요한 인자  {' '.join(spec['needs'])}")
        print()
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", nargs="?", choices=sorted(STAGES),
                    help="돌릴 단계 하나")
    ap.add_argument("--chunk-dir", dest="chunk_dir", default="",
                    help="이 단계가 읽고 쓸 디렉터리")
    ap.add_argument("--episodes", type=int, default=0, help="generate가 만들 성공 편수")
    ap.add_argument("--seed", type=int, default=-1, help="generate 난수 시드")
    ap.add_argument("--profile", default="nominal_lab",
                    choices=sorted(orch.PROFILE_IDS), help="시각 프로파일")
    ap.add_argument("--chunk-index", dest="chunk_index", type=int, default=-1,
                    help="upload가 저장소에서 쓸 청크 번호")
    ap.add_argument("--io", action="store_true", help="단계별 입출력만 찍고 끝낸다")
    ap.add_argument("--json", action="store_true", help="결과를 JSON 한 줄로 찍는다")
    args = ap.parse_args(argv)

    if args.io:
        return print_io()
    if not args.stage:
        ap.error("돌릴 단계를 하나 준다. 목록은 --io로 본다")
    if not args.chunk_dir:
        ap.error("--chunk-dir가 필요하다")

    spec = STAGES[args.stage]
    if args.stage == "generate" and args.episodes <= 0:
        ap.error("generate에는 --episodes가 필요하다")
    if args.stage == "upload" and args.chunk_index < 0:
        ap.error("upload에는 --chunk-index가 필요하다")

    cfg = orch.load_config()
    chunk_dir = os.path.abspath(args.chunk_dir)
    os.makedirs(chunk_dir, exist_ok=True)
    # 단계 함수는 청크 딕셔너리에서 이 여섯 개만 읽는다. 오케스트레이터가 만드는 것과
    # 같은 모양을 손으로 만들어 준다.
    index = args.chunk_index if args.chunk_index >= 0 else 0
    chunk = {
        "chunk_index": index,
        "profile": args.profile,
        "profile_id": orch.PROFILE_IDS[args.profile],
        "episodes": args.episodes,
        "seed": args.seed if args.seed >= 0 else cfg["seed_base"] + index,
        "dir": chunk_dir,
    }

    log_path = os.path.join(chunk_dir, f"stage_{args.stage}.log")
    log = orch.make_logger(log_path)
    log(f"stage {args.stage} start dir={chunk_dir}")

    if args.stage == "generate":
        # 오케스트레이터는 main()에서 이 두 가지를 미리 해 두고 단계 함수를 부른다.
        # 단계만 따로 부를 때도 같은 준비가 필요하다. 안 하면 cfg["gen_script"]가
        # 아직 만들어지지 않은 자리표시 경로라 생성이 파일을 못 찾고 죽는다.
        orch.prepare_source_dataset(cfg, log)
        orch.prepare_generate_script(cfg, log)

    # generate가 청크에 남기는 값을 뒤 단계가 읽는다. 단계를 따로 부르면 그 값이 없으므로,
    # 이미 만들어져 있는 gen.hdf5를 열어 몇 편인지 세서 채운다. 이것이 없으면 convert가
    # "이미 다 됐다"를 잘못 판정하거나 render가 0편을 렌더한다.
    if args.stage in ("sart", "convert", "render"):
        # sart는 gen.hdf5를 읽고, convert와 render는 증강이 끝났으면 gen_sart.hdf5를
        # 읽는다. 어느 쪽인지는 orchestrate가 보고서를 보고 정한다.
        source = (os.path.join(chunk_dir, "gen.hdf5") if args.stage == "sart"
                  else orch.gen_dataset_path(chunk))
        chunk["produced"] = args.episodes if args.episodes > 0 else count_demos(source)
        if chunk["produced"] <= 0:
            log(f"stage {args.stage}: {chunk_dir}/gen.hdf5 에 데모가 없다")
            return 1
        log(f"stage {args.stage}: 앞 단계가 만든 {chunk['produced']}편을 이어받는다")

    if args.stage == "sart" and chunk["episodes"] <= 0:
        # 증강 단계는 청크 할당량에서 이미 만든 편수를 뺀 만큼만 더 만든다. 단계를 따로
        # 부르면 할당량(episodes)이 0이라 남은 자리도 0으로 계산되고, 시뮬레이터를 띄우지도
        # 않은 채 "이미 할당량을 채웠다"로 끝나 버린다. 겉으로는 성공인데 한 편도 안 는다.
        #
        # 그래서 소스 비율에서 원래 할당량을 되짚어 준다. 생성이 할당량의 source_frac만큼
        # 만들었으므로, 지금 있는 편수를 그 비율로 나누면 원래 할당량이 나온다. 비율이
        # 1.0이면 상한 자체가 없으므로 지금 편수를 그대로 두면 된다.
        frac = float(cfg.get("sart_source_frac", 1.0))
        if 0.0 < frac < 1.0:
            chunk["episodes"] = int(round(chunk["produced"] / frac))
        else:
            chunk["episodes"] = chunk["produced"]
        log(f"stage sart: --episodes를 주지 않아 소스 비율 {frac}에서 청크 할당량을 "
            f"{chunk['episodes']}편으로 되짚었다 (증강으로 더 만들 자리는 "
            f"{max(0, chunk['episodes'] - chunk['produced'])}편이다)")

    func = getattr(orch, spec["func"])
    try:
        seconds = func(cfg, chunk, log, log_path)
    except Exception as exc:
        log(f"stage {args.stage} failed: {type(exc).__name__}: {exc}")
        if args.json:
            print(json.dumps({"stage": args.stage, "ok": False,
                              "error": f"{type(exc).__name__}: {exc}"}))
        return 1

    log(f"stage {args.stage} done {seconds:.1f}s")
    if args.json:
        print(json.dumps({"stage": args.stage, "ok": True,
                          "seconds": round(float(seconds), 1),
                          "chunk_dir": chunk_dir,
                          "outputs": spec["outputs"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
