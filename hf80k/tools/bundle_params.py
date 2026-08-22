#!/usr/bin/env python3
"""받은 번들에서 물리 대표값과 범위를 뽑아 표준 파일 하나로 만든다.

왜 필요한가. 지금 물리 항(`src/env/calibrated_sysid.py`)에는 숫자가 직접 적혀 있다.
책상 마찰을 1.25에서 2.6 사이에서 뽑고, 책상 운동 마찰을 최빈값 1.6753858178인
삼각분포에서 뽑는다. 이 숫자들은 전부 큐브 번들의
`modules/contact/task_contact_randomization.yaml`과
`modules/dynamics_controller/nominal_and_ranges.yaml`에 그대로 있는 값이다. 번들 내용을
코드에 베껴 둔 상태라, 다른 번들이 다른 값을 줘도 코드가 무시한다.

이 파일이 그 숫자들을 번들에서 읽어 `parameters.json` 하나로 만든다. 물리 항은 그
파일만 읽는다. 번들을 바꾸면 물리가 바뀐다.

번들마다 담고 있는 것이 다르다. peg 번들에는 로봇 관절 물리도, 작업면 반발도, 물체
감쇠도 없다. 없는 값은 세 가지 중 하나로 처리하고 무엇을 했는지 `provenance`에 남긴다.

  measured        그 번들이 실제로 잰 값이다.
  borrowed:<곳>   태스크가 아니라 로봇이나 작업대의 성질이라 다른 번들에서 가져왔다.
  fallback:<이유> 아무도 재지 않아 정해 둔 값을 쓴다.

용어. `surface`는 물체가 놓인 작업면이다(큐브에서는 책상, peg에서는 데스크).
`pair_primary`는 그 태스크의 주된 접촉 쌍이다(큐브끼리, 핀과 구멍). `finger`는
그리퍼 손가락과 물체 사이다.
"""
from __future__ import annotations

import copy

PARAMS_SCHEMA = "fr3_cube.hf80k.normalized_params.v1"

# 표준 접촉 항목. 물리 항이 이 이름으로만 값을 찾는다.
CONTACT_KEYS = (
    "surface_static_friction",
    "surface_dynamic_friction",
    "surface_restitution",
    "pair_primary_static_friction",
    "pair_primary_dynamic_friction",
    "pair_primary_dynamic_ratio",
    "object_linear_damping",
    "object_angular_damping",
    "finger_static_friction",
    "finger_dynamic_friction",
    "finger_dynamic_ratio",
    "gripper_force_scale",
)

# 큐브 번들의 접촉 계약에서 표준 이름으로 옮기는 표. 값은 (그룹, 항목)이다.
CUBE_CONTACT_PATHS = {
    "surface_static_friction": ("table_cube", "static_friction"),
    "surface_dynamic_friction": ("table_cube", "dynamic_friction"),
    "surface_restitution": ("table_cube", "restitution"),
    "pair_primary_static_friction": ("cube_cube", "static_friction"),
    "pair_primary_dynamic_friction": ("cube_cube", "dynamic_friction"),
    "pair_primary_dynamic_ratio": ("cube_cube", "dynamic_ratio"),
    "object_linear_damping": ("cube_rigid_body", "linear_damping"),
    "object_angular_damping": ("cube_rigid_body", "angular_damping"),
    "finger_static_friction": ("finger_cube", "static_friction"),
    "finger_dynamic_friction": ("finger_cube", "dynamic_friction"),
    "finger_dynamic_ratio": ("finger_cube", "dynamic_ratio"),
    "gripper_force_scale": ("finger_cube", "gripper_force_scale"),
}

# peg 번들의 파라미터 이름에서 표준 이름으로 옮기는 표.
PEG_CONTACT_IDS = {
    "pair_primary_static_friction": "contact.peg_hole.static_friction",
    "pair_primary_dynamic_friction": "contact.peg_hole.dynamic_friction",
    "finger_static_friction": "contact.finger_peg.static_friction",
}
# peg 번들에만 있는 값. 표준 항목은 아니지만 버리지 않고 실어 둔다.
PEG_EXTRA_IDS = {
    "pair_primary_contact_damping": "contact.peg_hole.contact_damping",
    "object_mass_kg": "object.peg.mass_kg",
    "pair_radial_clearance_m": "task.peg_hole.radial_clearance_m",
}


def _range_of(entry: dict) -> dict | None:
    """계약 항목 하나에서 뽑을 범위. 분포 종류까지 같이 싣는다."""
    low_high = entry.get("training_range") or entry.get("range")
    if not low_high or len(low_high) != 2:
        return None
    out = {"dist": "uniform", "low": float(low_high[0]), "high": float(low_high[1])}
    if str(entry.get("distribution", "")).startswith("triangular"):
        mode = entry.get("distribution_mode")
        if mode is not None:
            out = {"dist": "triangular", "low": float(low_high[0]),
                   "mode": float(mode), "high": float(low_high[1])}
    return out


def from_cube_bundle(contact_doc: dict, dynamics_doc: dict) -> dict:
    """큐브 번들의 두 계약 파일에서 표준 파라미터를 만든다."""
    params = contact_doc.get("parameters", {})
    nominal, ranges, provenance = {}, {}, {}
    for key, (group, item) in CUBE_CONTACT_PATHS.items():
        entry = (params.get(group) or {}).get(item)
        if not isinstance(entry, dict):
            continue
        if entry.get("nominal") is not None:
            nominal[key] = float(entry["nominal"])
        got = _range_of(entry)
        if got:
            ranges[key] = got
        provenance[key] = f"measured:contact/{group}.{item}"

    constraint = ((params.get("finger_cube") or {}).get("joint_constraint") or {})
    contact = {"nominal": nominal, "range": ranges}
    if constraint.get("expression"):
        contact["constraint"] = {
            "expression": constraint["expression"],
            # 표현식을 문자열로만 두면 코드가 쓸 수 없다. 곱의 하한을 숫자로 뽑아 둔다.
            "finger_force_product_min": _product_min(constraint["expression"]),
            "source": constraint.get("source", ""),
        }
    return {
        "joint": _joint_from_dynamics(dynamics_doc),
        "contact": contact,
        "sampling": {
            # 번들은 이 둘을 문장으로만 적어 두었다(robust_stochastic.recommendation).
            # 숫자로 옮겨 코드가 읽게 한다. 어디서 왔는지는 provenance에 남긴다.
            "posterior_near_fraction": 0.8,
            "material_buckets": 256,
        },
        "provenance": dict(provenance, **{
            "sampling.posterior_near_fraction":
                "measured:contact/usage_profiles.robust_stochastic.recommendation (문장)",
            "sampling.material_buckets":
                "fallback:PhysX 재질 수 상한을 넘지 않도록 파이프라인이 정한 값",
        }),
    }


def _product_min(expression: str) -> float | None:
    """`a * b >= 1.241946` 같은 표현식에서 오른쪽 숫자만 뽑는다."""
    for token in (">=", ">"):
        if token in expression:
            try:
                return float(expression.split(token)[1].strip())
            except ValueError:
                return None
    return None


def _joint_from_dynamics(doc: dict) -> dict:
    """로봇 관절 물리의 대표값. 태스크가 아니라 로봇의 성질이다."""
    if not doc:
        return {}
    params = doc.get("parameters", {})

    def nom(name):
        entry = params.get(name) or {}
        return entry.get("nominal")

    return {
        "order": doc.get("joint_order", []),
        "payload": doc.get("payload", {}),
        "nominal": {
            "armature": nom("armature_kg_m2"),
            "static_friction": nom("static_friction"),
            "dynamic_friction": nom("dynamic_friction"),
            "viscous_friction": nom("viscous_friction"),
            "motor_delay_steps": nom("motor_delay_simulation_steps"),
        },
    }


def from_peg_bundle(contract_doc: dict, donor: dict | None) -> dict:
    """peg 번들에서 표준 파라미터를 만들고, 없는 값은 기증 번들에서 채운다.

    peg 번들은 핀과 구멍 사이의 접촉만 담는다. 로봇 관절 물리, 작업면 마찰과 반발, 물체
    감쇠, 그리퍼 힘 배율은 들어 있지 않다. 그 값들은 태스크가 아니라 로봇과 작업대의
    성질이므로 큐브 번들 것을 그대로 쓴다. 두 번들이 같은 제어기 계약
    (fr3_cube_stage1_model4500_legacyosc_v1) 아래에서 맞춰졌다는 근거가 peg 번들
    자신에 적혀 있다.
    """
    by_id = {p["parameter_id"]: p for p in contract_doc.get("parameters", [])
             if isinstance(p, dict) and "parameter_id" in p}
    nominal, ranges, provenance = {}, {}, {}

    for key, pid in list(PEG_CONTACT_IDS.items()) + list(PEG_EXTRA_IDS.items()):
        entry = by_id.get(pid)
        if not entry:
            continue
        if entry.get("nominal") is not None:
            nominal[key] = float(entry["nominal"])
        got = _range_of(entry)
        if got:
            ranges[key] = got
        provenance[key] = f"measured:{pid} (confidence {entry.get('confidence', '?')})"

    # 운동 마찰 비율. peg 번들은 비율을 따로 싣지 않고 "Isaac 어댑터가 정지 마찰의 0.9배로
    # 둔다"고 해석에 적어 두었다. 정지와 운동 대표값에서 되돌린다.
    if "pair_primary_static_friction" in nominal and "pair_primary_dynamic_friction" in nominal:
        static = nominal["pair_primary_static_friction"]
        if static:
            ratio = nominal["pair_primary_dynamic_friction"] / static
            nominal["pair_primary_dynamic_ratio"] = ratio
            ranges["pair_primary_dynamic_ratio"] = {"dist": "uniform", "low": ratio, "high": ratio}
            provenance["pair_primary_dynamic_ratio"] = (
                "derived:운동/정지 대표값의 비. 번들 해석에 '정지 마찰의 0.9배'로 적혀 있다")

    result = {
        "joint": {},
        "contact": {"nominal": nominal, "range": ranges},
        "sampling": {"posterior_near_fraction": 0.8, "material_buckets": 256},
        "provenance": provenance,
    }

    if donor:
        _fill_from_donor(result, donor)
    result["provenance"]["sampling.posterior_near_fraction"] = (
        "fallback:큐브 번들의 권고를 그대로 쓴다. peg 번들에는 혼합 비율이 없다")
    result["provenance"]["sampling.material_buckets"] = (
        "fallback:PhysX 재질 수 상한을 넘지 않도록 파이프라인이 정한 값")
    return result


def _fill_from_donor(result: dict, donor: dict) -> None:
    """비어 있는 표준 항목을 기증 번들로 채우고 무엇을 빌렸는지 남긴다."""
    if not result.get("joint") and donor.get("joint"):
        result["joint"] = copy.deepcopy(donor["joint"])
        result["provenance"]["joint"] = (
            "borrowed:같은 FR3 로봇이고 두 번들이 같은 제어기 계약 아래에서 맞춰졌다")
    donor_contact = donor.get("contact", {})
    for key in CONTACT_KEYS:
        if key not in result["contact"]["nominal"] and key in donor_contact.get("nominal", {}):
            result["contact"]["nominal"][key] = donor_contact["nominal"][key]
            result["provenance"][key] = "borrowed:이 번들이 재지 않아 기증 번들 값을 쓴다"
        if key not in result["contact"]["range"] and key in donor_contact.get("range", {}):
            result["contact"]["range"][key] = copy.deepcopy(donor_contact["range"][key])
    if "constraint" not in result["contact"] and "constraint" in donor_contact:
        result["contact"]["constraint"] = copy.deepcopy(donor_contact["constraint"])
        result["provenance"]["contact.constraint"] = (
            "borrowed:그리퍼 쥐는 힘 조건은 로봇 쪽 성질이라 그대로 쓴다")


def missing_keys(params: dict) -> list[str]:
    """물리 항이 찾을 표준 항목 중 대표값이 비어 있는 것."""
    nominal = params.get("contact", {}).get("nominal", {})
    return [key for key in CONTACT_KEYS if key not in nominal]
