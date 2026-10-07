"""소스 시연을 하위 작업마다 다시 고르고, 성공이 없는 짝을 그 하위 작업에서만 뺀다.

하위 작업이란 한 시연을 "집기", "옮기기", "꽂기"처럼 의미 있는 구간으로 자른 단위다.
MimicGen은 새 장면을 만들 때 하위 작업마다 사람 시연 하나를 골라 그 구간을 가져다 쓴다.

이 파일이 있는 이유는 세 가지다.

첫째, 하위 작업마다 소스를 따로 고르게 만든다. 설치된 생성기는 `generation_select_src_per_subtask`
하나만 켜면 "아직 아무것도 고르지 않았을 때"라는 조건이 함께 걸려서, 첫 하위 작업에서 정해진
소스가 나머지 하위 작업으로 그대로 물려진다. 짝이 되는 `generation_select_src_per_arm`까지
켜야 실제로 하위 작업마다 다시 고른다. 설정값이 아니라 기록에 남는 소스 번호가 하위 작업마다
달라지는지로 확인해야 한다.

둘째, 거리가 같을 때 번호 순서로 고르지 않게 만든다. 최근접 이웃 선택은 새 물체 자세와 각
시연의 첫 물체 자세 사이 거리를 재서 가까운 것 몇 개 중에서 뽑는다. 큐브 쌓기 시연 13편은
모두 같은 큐브 배치에서 녹화돼서 첫 물체 자세가 비트 단위로 같고, 거리가 12중 동점이 된다.
정렬 함수가 동점을 번호가 작은 쪽부터 놓으므로 가까운 셋은 항상 앞번호 셋이 되고, 뒤쪽 아홉
편은 뽑힐 확률이 낮은 것이 아니라 정확히 0이 된다. 301회 시도로 확인했다. 동점인 후보끼리는
무작위로 섞어서 이 쏠림을 없앤다.

셋째, 반복해서 성공이 없는 (하위 작업, 소스) 짝만 그 하위 작업에서 뺀다. 시연 한 편이 어느
하위 작업에서 실패했다고 그 편을 전체에서 지우지 않는다. 전체에서 지우면 쓸 수 있는 소스가
줄어 데이터 다양성이 함께 떨어진다. 실제로 큐브의 기존 전역 필터는 동점을 깨는 유일한 시연을
지워서 도달 가능한 소스를 넷에서 셋으로 줄였다.

빼는 목록은 태스크마다 직접 재서 채운다. 다른 태스크의 번호를 베껴 쓰면 안 된다. 번호는
파일 안의 순서에만 유효하고, 파일을 다시 정렬하거나 시연을 추가하면 가리키는 시연이 달라진다.
"""

from __future__ import annotations

import json
import os
import random
from typing import Iterable

# 하위 작업마다 등록할 전략 이름의 앞머리. 뒤에 하위 작업 번호가 붙는다.
STRATEGY_PREFIX = "hf80k_routed"
# 환경변수로 라우팅 전체를 끌 수 있다. 0을 주면 Isaac Lab 기본 선택으로 돌아간다.
ENABLE_ENV = "HEURISTIC_SELECTION"
# 빼는 목록을 환경변수로 주는 경로. JSON 사전이고 열쇠가 하위 작업 번호다.
DENY_ENV = "SOURCE_DENY_PAIRS"
# 동점 처리 방식. shuffle이면 무작위로 섞고 index면 번호 순서를 그대로 둔다.
TIE_BREAK_ENV = "SOURCE_TIE_BREAK"
# 하위 작업마다 소스를 다시 고를지. 0을 주면 첫 하위 작업의 선택을 나머지가 물려받는다.
PER_SUBTASK_ENV = "SOURCE_PER_SUBTASK"


def _log(message: str) -> None:
    print(f"[source_routing] {message}", flush=True)


def parse_deny(raw) -> dict:
    """빼는 목록을 {하위 작업 번호: {소스 번호}} 형태로 바꾼다.

    사전이나 JSON 문자열을 받는다. 열쇠와 값이 문자열이어도 정수로 바꾼다. 읽지 못하면
    빈 사전을 돌려주고 이유를 찍는다. 이 함수는 어떤 경우에도 예외를 올리지 않는다.
    생성이 설정 한 줄 때문에 멈추는 것보다 라우팅 없이 도는 쪽이 낫기 때문이다.
    """
    if raw is None or raw == "":
        return {}
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception as exc:                                  # noqa: BLE001
            _log(f"빼는 목록을 읽지 못해 무시한다: {type(exc).__name__}: {exc}")
            return {}
    if not isinstance(raw, dict):
        _log(f"빼는 목록이 사전이 아니라 무시한다: {type(raw).__name__}")
        return {}
    out = {}
    for key, values in raw.items():
        try:
            subtask = int(key)
            out[subtask] = {int(v) for v in values}
        except Exception as exc:                                  # noqa: BLE001
            _log(f"빼는 목록의 {key!r} 항목을 읽지 못해 건너뛴다: {exc}")
    return out


def routing_enabled(default: bool = True) -> bool:
    """환경변수가 라우팅을 껐는지 본다. 환경변수가 없으면 default를 쓴다."""
    raw = os.environ.get(ENABLE_ENV)
    if raw is None or raw == "":
        return bool(default)
    return raw.strip().lower() not in ("0", "false", "no", "off")


def tie_break_mode(default: str = "shuffle") -> str:
    """동점 처리 방식을 읽는다. shuffle과 index만 받고, 그 밖의 값은 default로 둔다."""
    raw = (os.environ.get(TIE_BREAK_ENV) or default).strip().lower()
    if raw not in ("shuffle", "index"):
        _log(f"동점 처리 방식 {raw!r}을 모르므로 {default!r}로 둔다")
        return default
    return raw


def register_strategies(n_subtasks: int, deny: dict, tie_break: str = "shuffle") -> list:
    """하위 작업 번호마다 전략을 하나씩 등록하고 그 이름 목록을 돌려준다.

    전략 이름을 하위 작업마다 따로 두는 이유는, 선택 함수가 자기가 몇 번째 하위 작업을
    맡았는지 알 방법이 없기 때문이다. 선택 함수는 지금 손 자세와 물체 자세와 그 하위 작업의
    소스 구간 목록만 받는다. 그래서 하위 작업 번호를 전략 자체에 박아 등록한다.

    등록에 실패하면 빈 목록을 돌려준다. 그러면 부르는 쪽이 기존 선택을 그대로 쓴다.
    """
    try:
        import torch
        from isaaclab_mimic.datagen.selection_strategy import NearestNeighborObjectStrategy
    except Exception as exc:                                      # noqa: BLE001
        _log(f"선택 전략을 불러오지 못해 라우팅을 건너뛴다: {type(exc).__name__}: {exc}")
        return []

    names = []
    for subtask in range(int(n_subtasks)):
        denied = set(deny.get(subtask, ()))
        name = f"{STRATEGY_PREFIX}_{subtask}"

        # 메타클래스가 NAME을 보고 전역 사전에 자동으로 넣는다. 그래서 클래스를 만드는
        # 것만으로 등록이 끝난다.
        class _Routed(NearestNeighborObjectStrategy):
            NAME = name
            SUBTASK = subtask
            DENIED = denied
            TIE_BREAK = tie_break

            def select_source_demo(self, eef_pose, object_pose,
                                   src_subtask_datagen_infos, **kwargs):
                infos = list(src_subtask_datagen_infos)
                allowed = [i for i in range(len(infos)) if i not in self.DENIED]
                if not allowed:
                    # 전부 빠진 경우에는 빼는 목록을 무시한다. 소스가 하나도 없으면
                    # 이 하위 작업을 만들 수 없고, 그러면 생성이 통째로 멈춘다.
                    _log(f"하위 작업 {self.SUBTASK}: 소스가 전부 빠져 있어 빼는 목록을 "
                         f"이번 한 번 무시한다")
                    allowed = list(range(len(infos)))

                if self.TIE_BREAK == "shuffle":
                    # 거리가 같은 후보가 번호 순서로 줄 서지 않도록, 후보 목록 자체를
                    # 섞은 뒤 넘긴다. 거리 계산은 원래 코드를 그대로 쓴다.
                    allowed = list(allowed)
                    random.shuffle(allowed)

                sub = [infos[i] for i in allowed]
                picked = super().select_source_demo(eef_pose, object_pose, sub, **kwargs)
                return allowed[int(picked)]

        names.append(name)
    _log(f"하위 작업 {len(names)}개에 전략을 등록했다. 동점 처리는 {tie_break}이고 "
         f"빼는 목록은 { {k: sorted(v) for k, v in deny.items() if v} or '비어 있다'}")
    return names


def apply(env_cfg, spec: dict | None = None) -> bool:
    """환경 설정에 라우팅을 적용한다. 적용했으면 참을 돌려준다.

    spec은 태스크 프로필의 generate.heuristic_selection 절이고, 다음 열쇠를 읽는다.
    enable은 켤지 여부이고 기본은 참이다. per_subtask는 하위 작업마다 다시 고를지이고
    기본은 참이다. tie_break는 동점 처리 방식이고 기본은 shuffle이다. deny는 빼는
    목록이고 기본은 비어 있다.

    이 함수는 어떤 경우에도 예외를 올리지 않는다. 적용하지 못하면 이유를 찍고 거짓을
    돌려주며, 그러면 생성은 Isaac Lab 기본 선택으로 그대로 돈다.
    """
    spec = dict(spec or {})
    if not routing_enabled(bool(spec.get("enable", True))):
        _log("환경변수가 껐으므로 기본 선택을 쓴다")
        return False

    try:
        eef_keys = list(env_cfg.subtask_configs.keys())
        if not eef_keys:
            _log("하위 작업 설정이 비어 있어 건너뛴다")
            return False

        # 하위 작업마다 다시 고르게 하는 두 스위치. 하나만 켜면 첫 하위 작업의 선택이
        # 나머지로 물려지므로 둘 다 켠다. 설치된 생성기에 없는 이름일 수 있어 확인하고 쓴다.
        per_subtask = spec.get("per_subtask", True)
        raw_per_subtask = os.environ.get(PER_SUBTASK_ENV)
        if raw_per_subtask not in (None, ""):
            per_subtask = raw_per_subtask.strip().lower() not in ("0", "false", "no", "off")
        if per_subtask:
            for field in ("generation_select_src_per_subtask",
                          "generation_select_src_per_arm"):
                if hasattr(env_cfg.datagen_config, field):
                    setattr(env_cfg.datagen_config, field, True)
                else:
                    _log(f"{field}가 이 생성기에 없어 건너뛴다. 하위 작업별 재선택이 "
                         f"실제로 되는지는 기록의 소스 번호로 확인해야 한다")

        deny = parse_deny(os.environ.get(DENY_ENV) or spec.get("deny"))
        tie = tie_break_mode(str(spec.get("tie_break", "shuffle")))

        applied = 0
        for eef in eef_keys:
            subtasks = env_cfg.subtask_configs[eef]
            names = register_strategies(len(subtasks), deny, tie)
            if not names:
                return False
            for i, subtask in enumerate(subtasks):
                if hasattr(subtask, "selection_strategy"):
                    subtask.selection_strategy = names[i]
                    applied += 1
                else:
                    _log(f"하위 작업 {i}에 selection_strategy 항목이 없어 건너뛴다")
        _log(f"손 {len(eef_keys)}개의 하위 작업 {applied}개에 적용했다")
        return applied > 0
    except Exception as exc:                                      # noqa: BLE001
        _log(f"적용하지 못해 기본 선택을 쓴다: {type(exc).__name__}: {exc}")
        return False
