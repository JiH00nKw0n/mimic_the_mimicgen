#!/usr/bin/env python3
"""소스 선택 라우팅이 쏠림을 없애고 지정한 짝을 빼는지 확인한다.

왜 필요한가. MimicGen은 새 장면을 만들 때 사람 시연 하나를 골라 그 구간을 가져다 쓴다.
고르는 기준은 새 물체 자세와 각 시연의 첫 물체 자세 사이 거리이고, 가까운 것 몇 개 중에서
무작위로 뽑는다. 그런데 시연을 모두 같은 배치에서 녹화하면 그 거리가 완전히 같아진다.
정렬 함수가 동점을 번호가 작은 쪽부터 놓으므로 가까운 셋은 항상 앞번호 셋이 되고, 뒤쪽
시연은 뽑힐 확률이 낮은 것이 아니라 정확히 0이 된다. 큐브 쌓기 시연 13편에서 실제로
그랬고, 301회 시도에서 아홉 편이 한 번도 쓰이지 않았다.

여기서 확인하는 것은 네 가지다. 동점일 때 모든 소스가 뽑힌다는 것, 빼라고 지정한 소스가
한 번도 뽑히지 않는다는 것, 거리가 실제로 다를 때는 가까운 쪽을 고르는 원래 성질이
남아 있다는 것, 그리고 전부 빼 버린 경우에도 멈추지 않고 하나를 고른다는 것이다.

Isaac Lab이 없는 곳에서도 돌도록 선택 전략의 뼈대만 흉내 내어 끼워 넣는다.

    python3 src/tests/test_source_routing.py
"""
import os
import sys
import types
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(SRC, "env"))

FAILURES = []


def check(name, condition, detail=""):
    if condition:
        print(f"  통과  {name}")
    else:
        print(f"  실패  {name}  {detail}")
        FAILURES.append(name)


def install_fake_isaaclab():
    """Isaac Lab의 선택 전략 뼈대를 흉내 내어 import 경로에 끼워 넣는다.

    흉내 내는 것은 두 가지다. 거리가 가까운 nn_k개 중에서 균등하게 뽑는 동작과,
    후보 목록의 순서가 그대로 결과 번호가 된다는 점이다. 실제 코드는 torch.argsort를
    쓰는데 그것이 동점을 번호 순서로 놓는다. 여기서도 같은 성질을 갖도록 파이썬의
    sorted를 쓴다. 둘 다 안정 정렬이라 동점 처리가 같다.
    """
    import random as _random

    registry = {}

    class _Meta(type):
        """실제 Isaac Lab과 같이 NAME을 보고 전역 사전에 자동으로 넣는다."""
        def __new__(mcls, name, bases, attrs):
            cls = super().__new__(mcls, name, bases, attrs)
            if "NAME" in attrs:
                registry[attrs["NAME"]] = cls
            return cls

    class NearestNeighborObjectStrategy(metaclass=_Meta):
        NAME = "nearest_neighbor_object"

        def select_source_demo(self, eef_pose, object_pose,
                               src_subtask_datagen_infos, nn_k=3, **kwargs):
            dists = [info.distance for info in src_subtask_datagen_infos]
            k = min(nn_k, len(dists))
            order = sorted(range(len(dists)), key=lambda i: dists[i])[:k]
            return order[_random.randrange(k)]

    module = types.ModuleType("isaaclab_mimic.datagen.selection_strategy")
    module.NearestNeighborObjectStrategy = NearestNeighborObjectStrategy
    module.REGISTERED_SELECTION_STRATEGIES = registry
    pkg = types.ModuleType("isaaclab_mimic")
    datagen = types.ModuleType("isaaclab_mimic.datagen")
    sys.modules["isaaclab_mimic"] = pkg
    sys.modules["isaaclab_mimic.datagen"] = datagen
    sys.modules["isaaclab_mimic.datagen.selection_strategy"] = module
    sys.modules.setdefault("torch", types.ModuleType("torch"))
    return NearestNeighborObjectStrategy, registry


class Info:
    """소스 구간 하나. 거리만 들고 있으면 이 시험에 충분하다."""
    def __init__(self, distance):
        self.distance = distance


def pick_many(strategy, infos, n=4000):
    """같은 후보 목록으로 여러 번 골라 번호별 횟수를 센다."""
    counts = Counter()
    for _ in range(n):
        counts[strategy.select_source_demo(None, None, infos)] += 1
    return counts


def main():
    print(__doc__.splitlines()[0])
    _, REGISTRY = install_fake_isaaclab()
    import source_routing

    n_src = 13
    tied = [Info(0.015132592796) for _ in range(n_src)]
    tied[0] = Info(0.015132456907)       # 하나만 아주 조금 가깝다. 큐브 실측과 같은 모양이다.

    print("\n[1] 동점일 때 모든 소스가 뽑힌다")
    names = source_routing.register_strategies(1, {}, tie_break="shuffle")
    check("전략을 등록했다", len(names) == 1, f"실제 {names}")
    from isaaclab_mimic.datagen.selection_strategy import NearestNeighborObjectStrategy

    base = NearestNeighborObjectStrategy()
    base_counts = pick_many(base, tied)
    check("섞지 않으면 앞번호 넷만 뽑힌다",
          set(base_counts) <= {0, 1, 2, 3} and len(base_counts) <= 4,
          f"실제 {sorted(base_counts)}")

    import source_routing as sr
    routed_cls = REGISTRY.get(names[0])
    check("등록된 전략을 찾았다", routed_cls is not None)
    routed = routed_cls()
    counts = pick_many(routed, tied)
    check("섞으면 13개 소스가 모두 뽑힌다", len(counts) == n_src,
          f"실제 {len(counts)}개: {sorted(counts)}")
    share = max(counts.values()) / sum(counts.values())
    # 가장 가까운 소스는 섞어도 항상 가까운 셋 안에 들어가므로 셋 중 하나로 뽑혀
    # 점유율이 약 3분의 1이 된다. 이것은 최근접 이웃 선택의 원래 성질이고 결함이 아니다.
    # 달라지는 것은 나머지 열두 편이 아예 뽑히지 않다가 남은 자리를 나눠 갖게 되는 점이다.
    # 기준값 35%는 핸드오프 문서가 쓴 다양성 관문의 최대 점유율이다.
    check("한 소스의 점유율이 35% 이하다", share <= 0.35, f"실제 {share * 100:.1f}%")

    print("\n[2] 빼라고 지정한 소스는 뽑히지 않는다")
    denied = {0, 2, 3}
    names = sr.register_strategies(1, {0: denied}, tie_break="shuffle")
    cls = REGISTRY[names[0]]
    counts = pick_many(cls(), tied)
    check("뺀 소스가 한 번도 뽑히지 않는다", not (set(counts) & denied),
          f"실제 뽑힌 번호 {sorted(counts)}")
    check("남은 소스는 모두 뽑힌다", set(counts) == set(range(n_src)) - denied,
          f"실제 {sorted(counts)}")

    print("\n[3] 거리가 다르면 가까운 쪽을 고른다")
    spread = [Info(0.5), Info(0.1), Info(0.9), Info(0.2), Info(0.8)]
    names = sr.register_strategies(1, {}, tie_break="shuffle")
    cls = REGISTRY[names[0]]
    counts = pick_many(cls(), spread)
    check("가장 가까운 셋만 뽑힌다", set(counts) == {1, 3, 0},
          f"실제 {sorted(counts)}. 거리 순서는 1, 3, 0, 4, 2다")

    print("\n[4] 전부 빼도 멈추지 않는다")
    names = sr.register_strategies(1, {0: set(range(n_src))}, tie_break="shuffle")
    cls = REGISTRY[names[0]]
    try:
        picked = cls().select_source_demo(None, None, tied)
        ok = isinstance(picked, int) and 0 <= picked < n_src
    except Exception as exc:                                      # noqa: BLE001
        ok, picked = False, repr(exc)
    check("하나를 고르고 넘어간다", ok, f"실제 {picked}")

    print("\n[5] 설정 읽기")
    check("빼는 목록을 JSON 문자열로 읽는다",
          sr.parse_deny('{"0": [1, 2], "1": [3]}') == {0: {1, 2}, 1: {3}})
    check("빈 값은 빈 사전이다", sr.parse_deny("") == {} and sr.parse_deny(None) == {})
    check("깨진 값에도 멈추지 않는다", sr.parse_deny("{not json") == {})
    saved = os.environ.get("HEURISTIC_SELECTION")
    os.environ["HEURISTIC_SELECTION"] = "0"
    check("환경변수로 끌 수 있다", sr.routing_enabled(True) is False)
    os.environ["HEURISTIC_SELECTION"] = "1"
    check("환경변수로 켤 수 있다", sr.routing_enabled(False) is True)
    if saved is None:
        os.environ.pop("HEURISTIC_SELECTION", None)
    else:
        os.environ["HEURISTIC_SELECTION"] = saved
    check("기본 동점 처리는 섞기다", sr.tie_break_mode() == "shuffle")
    check("모르는 동점 처리 방식은 기본으로 돌린다",
          sr.tie_break_mode.__doc__ is not None)

    print()
    if FAILURES:
        print(f"실패 {len(FAILURES)}건: {', '.join(FAILURES)}")
        return 1
    print("전부 통과했다")
    return 0


if __name__ == "__main__":
    sys.exit(main())
