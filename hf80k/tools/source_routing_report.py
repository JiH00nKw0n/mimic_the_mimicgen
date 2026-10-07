#!/usr/bin/env python3
"""생성 기록을 읽어 (하위 작업, 소스)별 성공을 세고, 뺄 짝과 다양성 관문을 알려준다.

왜 필요한가. 소스 시연 중 어느 것이 실제로 성공을 만드는지는 재 봐야 안다. 핀 꽂기에서
301회를 시도해 보니 성공 26편이 전부 한 편에서 나왔고 나머지 세 편은 235회 시도에 0편이었다.
큐브 쌓기에서는 13편 중 아홉 편이 한 번도 뽑히지 않았다. 둘 다 숫자를 보기 전에는 몰랐다.

무엇을 하는가. 생성이 남긴 기록 파일을 읽어 세 가지를 찍는다. 첫째, (하위 작업, 소스)마다
시도와 성공과 성공률을 센다. 둘째, 반복해서 성공이 없는 짝을 골라 태스크 프로필에 그대로
붙여 넣을 수 있는 형태로 적어 준다. 셋째, 다양성 관문을 통과하는지 본다.

다양성 관문이란 소스가 골고루 쓰였는지 보는 기준이다. 넷을 본다. 실제로 쓰인 소스의 개수,
하위 작업마다 몇 개의 소스가 유효하게 쓰였는지, 한 소스가 차지한 최대 비율, 그리고 전체
성공률이다. 기준값은 태스크마다 다르므로 처음 실행에서 직접 정해야 한다. 다른 태스크의
기준값을 그대로 옮겨 쓰면 안 된다.

쓰는 법.

    python3 tools/source_routing_report.py <생성기록.json> [<생성기록.json> ...]
    python3 tools/source_routing_report.py /work/chunks/chunk_*/gen.provenance.json

기록 파일은 생성 단계가 청크 폴더에 남기는 gen.provenance.json이다. 그 안에 시도마다
어느 하위 작업에서 어느 소스를 썼는지가 들어 있다.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict

# 이 횟수 이상 시도하고도 성공이 하나도 없는 짝만 뺄 후보로 올린다. 시도가 적으면
# 성공이 0인 것이 우연일 수 있다. 20회에서 한 번도 성공하지 않았다면 진짜 성공률의
# 95% 상한이 약 16%이고, 50회면 약 7%다.
MIN_ATTEMPTS_TO_DENY = 20


def wilson_upper(successes: int, attempts: int, z: float = 1.96) -> float:
    """성공률의 95% 상한을 백분율로 돌려준다. 시도가 0이면 100을 돌려준다."""
    if attempts <= 0:
        return 100.0
    p = successes / attempts
    denom = 1 + z * z / attempts
    center = (p + z * z / (2 * attempts)) / denom
    half = z * math.sqrt(p * (1 - p) / attempts + z * z / (4 * attempts * attempts)) / denom
    return min(1.0, center + half) * 100.0


def read_counts(paths):
    """(하위 작업, 소스)별 시도 수와 성공 수를 합산한다.

    돌려주는 값은 (시도 사전, 성공 사전, 전체 시도, 전체 성공)이고, 두 사전의 열쇠는
    (하위 작업 번호, 소스 번호)다.
    """
    attempts = defaultdict(int)
    successes = defaultdict(int)
    total_attempts = total_successes = 0
    combos = set()
    for path in paths:
        try:
            with open(path, encoding="utf-8") as handle:
                doc = json.load(handle)
        except Exception as exc:                                  # noqa: BLE001
            print(f"  건너뜀  {path}를 읽지 못했다: {type(exc).__name__}: {exc}")
            continue
        total_attempts += int(doc.get("n_attempts", 0))
        total_successes += int(doc.get("n_success", 0))
        for row in doc.get("counts_all", []):
            attempts[(int(row.get("subtask", 0)), int(row["src_ind"]))] += int(row["count"])
        for row in doc.get("counts_success", []):
            successes[(int(row.get("subtask", 0)), int(row["src_ind"]))] += int(row["count"])
        for demo in doc.get("per_demo", []):
            combos.add(tuple(sorted((int(r.get("subtask", 0)), int(r["src_ind"]))
                                    for r in demo)))
    return attempts, successes, total_attempts, total_successes, combos


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("provenance", nargs="+", help="생성 기록 JSON 파일 하나 이상")
    ap.add_argument("--min-attempts", type=int, default=MIN_ATTEMPTS_TO_DENY,
                    help=f"뺄 후보로 올리는 최소 시도 수 (기본 {MIN_ATTEMPTS_TO_DENY})")
    args = ap.parse_args(argv)

    attempts, successes, total_a, total_s, combos = read_counts(args.provenance)
    if not attempts:
        print("읽을 수 있는 기록이 없다.")
        return 1

    print(f"기록 파일 {len(args.provenance)}개")
    print(f"전체 시도 {total_a}회, 성공 {total_s}편, 성공률 "
          f"{total_s / max(total_a, 1) * 100:.1f}%")
    print()

    subtasks = sorted({st for st, _ in attempts})
    print("하위 작업 | 소스 | 시도 | 성공 | 성공률 | 성공률 95% 상한")
    deny = defaultdict(list)
    for st in subtasks:
        for src in sorted({s for t, s in attempts if t == st}):
            a = attempts[(st, src)]
            s = successes.get((st, src), 0)
            upper = wilson_upper(s, a)
            mark = ""
            if s == 0 and a >= args.min_attempts:
                deny[st].append(src)
                mark = "   <- 뺄 후보"
            print(f"    {st:5d} | {src:4d} | {a:4d} | {s:4d} | {s / max(a, 1) * 100:5.1f}% "
                  f"| {upper:5.1f}%{mark}")
    print()

    if deny:
        print("태스크 프로필의 generate.heuristic_selection.deny에 넣을 값이다.")
        print("    deny:")
        for st in sorted(deny):
            print(f"      {st}: {deny[st]}")
        print()
        print(f"기준은 시도 {args.min_attempts}회 이상에 성공 0편이다. 이 번호는 지금 쓰는")
        print("시연 파일의 순서에만 유효하다. 파일을 다시 정렬하거나 시연을 더하면 다시 재야 한다.")
    else:
        print("뺄 후보가 없다. 모든 소스가 적어도 한 번은 성공했거나 시도가 모자라다.")
    print()

    print("다양성 관문")
    used = {src for (_, src), a in attempts.items() if a > 0}
    print(f"  실제로 쓰인 소스 {len(used)}개: {sorted(used)}")
    if combos:
        print(f"  소스 조합의 가짓수 {len(combos)}개")
    else:
        print("  소스 조합의 가짓수는 기록에 per_demo가 없어 세지 못했다")
    for st in subtasks:
        per = {src: attempts[(st, src)] for t, src in attempts if t == st}
        tot = sum(per.values()) or 1
        shares = sorted((v / tot for v in per.values()), reverse=True)
        # 유효 개수란 쏠림을 반영한 개수다. 넷이 똑같이 쓰이면 4가 되고, 하나가 전부
        # 차지하면 1이 된다. 점유율 제곱의 합의 역수로 센다.
        effective = 1.0 / sum(x * x for x in shares) if shares else 0.0
        if shares:
            print(f"  하위 작업 {st}: 쓰인 소스 {len(per)}개, 유효 개수 {effective:.1f}개, "
                  f"최대 점유율 {shares[0] * 100:.1f}%")
        else:
            print(f"  하위 작업 {st}: 기록 없음")
    print()
    print("관문의 기준값은 태스크마다 직접 정한다. 참고로 다른 팀이 큐브 28편으로 쓴 값은")
    print("쓰인 소스 14개 이상, 소스 조합 20개 이상, 하위 작업별 유효 개수 6개 이상,")
    print("한 소스의 최대 점유율 35% 이하, 전체 성공률 2% 이상이었다. 그대로 옮겨 쓰지 않는다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
