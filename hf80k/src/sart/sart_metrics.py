#!/usr/bin/env python3
"""증강이 실제로 새 데이터를 만들었는지 재는 계산.

왜 이 파일이 따로 있는가. 증강의 성공률만 보면 무너진 증강이 오히려 좋아 보인다. 접근
경로를 다양하게 만들지 못하고 원본을 그대로 복사하면, 원본이 성공한 궤적이니 복사본도
전부 성공해서 성공률이 100%에 가까워진다. 그러니 성공률로는 무너진 것을 알 수 없다.

여기서 재는 값은 이렇다. 같은 소스 에피소드에서 나온 증강 편들을 모아, 에피소드의
**처음**을 기준으로 줄을 맞춘 뒤, 같은 스텝에서 손끝 좌표가 서로 얼마나 다른지를
표준편차로 구한다. 처음을 기준으로 맞추는 것이 정확한 이유는, 같은 소스에서 나온 증강
편들은 갈라지는 지점과 각 구간의 길이가 모두 같아서 스텝 번호가 그대로 대응하기
때문이다.

읽는 법은 두 숫자의 대비다. 접근 구간에서는 편마다 다른 자리로 벗어났다 오므로 표준편차가
크고, 삽입 구간에서는 원본을 그대로 재생하므로 거의 0이다. 실제로 잰 값은 접근 구간
최대 18.3 mm에 삽입 구간 0.2 mm였다. 약 90배 차이다. 이 대비가 1에 가까우면 증강이
원본 복사로 무너진 것이고, 성공률이 아무리 좋아도 그 실행은 멈춰야 한다.

예전에는 에피소드 끝에서 90스텝만 봤는데, 그러면 삽입 구간이 90스텝보다 긴 태스크에서
접근 구간이 창 밖으로 나간다. 핀 삽입이 그런 경우다. 222스텝짜리 증강 편에서 다양성이
있는 자리는 88번째에서 110번째 사이인데, 끝에서 90스텝이면 132번째부터만 본다. 그래서
멀쩡한 증강이 0.99 mm로 보고됐다. 지금은 전체 구간을 본다.

노트북에서 내려받은 조각 파일 하나에도 그대로 돌아간다. Isaac Lab을 부르지 않는다.

    python3 src/sart/sart_metrics.py /path/to/sart_00.hdf5
"""
from __future__ import annotations

import json
import sys

import numpy as np

# 소스 에피소드 이름을 적어 두는 속성. sart_augment.py가 증강 편마다 붙인다.
SOURCE_ATTR = "sart_src_demo"
# 삽입 구간으로 볼 꼬리 길이. 에피소드 끝에서 이만큼은 원본을 그대로 재생하는 자리로 본다.
# 이 구간의 표준편차가 비교 기준이 된다.
DEFAULT_TAIL = 30
# 프로파일을 찍어 볼 자리를 에피소드 길이의 비율로 적는다. 0.0이 처음, 1.0이 끝이다.
PROFILE_FRACTIONS = (0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0)


def group_by_source(shard_path: str) -> dict:
    """조각 파일을 열어 손끝 좌표 궤적을 소스 에피소드별로 모은다.

    두 편 이상 모인 소스만 남긴다. 한 편뿐이면 서로 비교할 상대가 없어 표준편차를
    구할 수 없다.
    """
    import h5py

    groups = {}
    with h5py.File(shard_path, "r") as fh:
        data = fh.get("data")
        if data is None:
            return {}
        for name in data:
            grp = data[name]
            source = grp.attrs.get(SOURCE_ATTR)
            if source is None:
                continue
            if isinstance(source, bytes):
                source = source.decode("utf-8", "replace")
            obs = grp.get("obs")
            if obs is None or "eef_pos" not in obs:
                continue
            groups.setdefault(str(source), []).append(np.asarray(obs["eef_pos"][()], dtype=float))
    return {k: v for k, v in groups.items() if len(v) >= 2}


def approach_std_profile(tracks: list, tail: int = DEFAULT_TAIL) -> dict:
    """한 소스에서 나온 궤적 묶음의 표준편차 프로파일을 구한다.

    모든 궤적의 앞에서부터 공통 길이만큼 잘라 (편수, 길이, 3) 모양으로 쌓고, 스텝마다
    편들 사이의 표준편차를 구한 뒤 세 축 평균을 낸다. 앞에서부터 맞추는 것이 맞는
    이유는, 같은 소스에서 나온 증강 편들은 갈라지는 지점과 구간 길이가 모두 같기
    때문이다.

    peak_m은 전체에서 가장 큰 표준편차이고 접근 구간에서 나온다. tail_m은 마지막
    tail스텝의 평균 표준편차이고 원본을 그대로 재생하는 삽입 구간의 값이다. 이 둘의
    비가 증강이 살아 있는지를 말한다.
    """
    arrays = [np.asarray(a, dtype=float) for a in tracks if np.asarray(a).ndim == 2]
    arrays = [a for a in arrays if a.shape[0] > 0]
    if len(arrays) < 2:
        return {"peak_m": None, "profile": {}, "n_episodes": len(arrays), "length": 0}
    length = int(min(a.shape[0] for a in arrays))
    stack = np.stack([a[:length] for a in arrays], axis=0)      # (편수, 길이, 3)
    std_k = stack.std(axis=0).mean(axis=-1)                     # (길이,)
    t = int(min(max(1, int(tail)), length))
    tail_m = float(std_k[-t:].mean())
    peak_m = float(std_k.max())
    # 삽입 구간이 완전히 똑같으면 tail_m이 정확히 0이 되어 나눗셈이 안 된다. 그런데 그것은
    # "잴 수 없다"가 아니라 "원본을 가장 잘 그대로 재생했다"는 뜻이라 대비가 가장 커야 할
    # 경우다. 그래서 분모를 1 마이크로미터로 받쳐 준다. 이 길이는 물리적으로 의미가 없는
    # 크기라서 값을 왜곡하지 않는다.
    denom = max(tail_m, 1e-6)
    profile = {}
    for frac in PROFILE_FRACTIONS:
        idx = min(length - 1, int(round(frac * (length - 1))))
        profile[str(frac)] = round(float(std_k[idx]), 6)
    return {
        "peak_m": round(peak_m, 6),
        "peak_index": int(std_k.argmax()),
        "tail_m": round(tail_m, 6),
        # 접근 구간이 삽입 구간보다 몇 배 흩어져 있는지. 1에 가까우면 증강이 무너진 것이다.
        "peak_over_tail": round(peak_m / denom, 1),
        "profile": profile,
        "n_episodes": len(arrays),
        "length": length,
    }


def report(shard_path: str, tail: int = DEFAULT_TAIL) -> dict:
    """조각 파일 하나를 통째로 재서 보고서에 넣을 사전을 돌려준다.

    peak_m은 소스 묶음별 접근 구간 최대 표준편차의 평균이고, tail_m은 삽입 구간 표준편차의
    평균이다. peak_over_tail이 이 둘의 비이고, 증강이 살아 있는지를 이 값으로 판단한다.
    """
    groups = group_by_source(shard_path)
    if not groups:
        # 0.0을 돌려주면 안 된다. 0.0은 "증강이 복사로 무너졌다"는 뜻이고, 여기는
        # "잴 수 없었다"는 뜻이다. 두 경우가 같은 숫자로 보이면 무너진 실행을 놓친다.
        return {"n_groups": 0, "peak_m": None, "tail_m": None, "peak_over_tail": None,
                "profile": {},
                "note": "같은 소스에서 두 편 이상 성공한 묶음이 없어 잴 수 없다. "
                        "편당 시도 수를 늘리면 잴 수 있다"}
    peaks, tails, profiles = [], [], []
    for tracks in groups.values():
        one = approach_std_profile(tracks, tail)
        if one["length"] == 0 or one["peak_m"] is None:
            continue
        peaks.append(one["peak_m"])
        tails.append(one["tail_m"])
        profiles.append(one["profile"])
    if not peaks:
        return {"n_groups": 0, "peak_m": None, "tail_m": None, "peak_over_tail": None,
                "profile": {}, "note": "잴 수 있는 묶음이 없다"}
    merged = {}
    for frac in PROFILE_FRACTIONS:
        values = [p[str(frac)] for p in profiles if str(frac) in p]
        if values:
            merged[str(frac)] = round(float(np.mean(values)), 6)
    peak = float(np.mean(peaks))
    tail_mean = float(np.mean(tails))
    return {"n_groups": len(peaks),
            "peak_m": round(peak, 6),
            "tail_m": round(tail_mean, 6),
            "peak_over_tail": round(peak / max(tail_mean, 1e-6), 1),
            "profile": merged}


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print("쓰는 법: python3 src/sart/sart_metrics.py <조각 파일.hdf5> [꼬리 길이]")
        return 2
    tail = int(argv[1]) if len(argv) > 1 else DEFAULT_TAIL
    doc = report(argv[0], tail)
    print(json.dumps(doc, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
