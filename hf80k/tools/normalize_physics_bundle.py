#!/usr/bin/env python3
"""받은 물리 번들을 파이프라인이 그대로 읽는 배치로 바꾼다.

왜 필요한가. SysID팀이 주는 번들은 태스크마다 모양이 다르다. 큐브 번들은 `modules/`
아래에 관절 물리와 접촉 물리를 나눠 담고, peg 번들은 파일을 최상위에 평평하게 두며 열
이름도 겹치지 않는다. 파이프라인이 번들 모양을 하나하나 알게 만드는 대신, 번들을 하나의
표준 배치로 옮겨 놓고 파이프라인은 그 배치만 읽게 한다. 새 번들이 오면 이 파일에 매핑을
하나 추가하는 것으로 끝나고 파이프라인은 건드리지 않는다.

표준 배치는 아래와 같다. 경로와 이름은 파이프라인이 이미 읽고 있는 것이다.

    <출력>/
      modules/dynamics_controller/domain_randomization_samples.csv   로봇 관절 물리
      modules/contact/posterior_samples.csv                          접촉 물리
      NORMALIZED.yaml                                                무엇을 어디서 가져왔는지

접촉 물리의 열 이름은 태스크 이름이 아니라 역할로 적는다. 큐브에서는 "큐브끼리"이고
peg에서는 "핀과 구멍 사이"인데, 파이프라인 입장에서는 둘 다 "주된 접촉 쌍"이다.

    pair_primary_static_friction    주된 접촉 쌍의 정지 마찰
    pair_primary_dynamic_friction   같은 쌍의 운동 마찰
    surface_restitution             물체와 작업면 사이 반발
    object_linear_damping           물체의 직선 감쇠
    object_angular_damping          물체의 회전 감쇠
    contact_damping                 접촉 감쇠. 없는 번들은 빈칸
    object_mass_kg                  물체 질량. 없는 번들은 빈칸

사용법

    tools/normalize_physics_bundle.py --in <받은_번들> --out <표준_번들> \
        [--dynamics-from <관절_물리를_가져올_번들>]

관절 물리(`--dynamics-from`)를 따로 받는 이유가 있다. peg 번들에는 관절 물리가 없다.
로봇은 같은 FR3이고 두 번들이 같은 제어기 계약(fr3_cube_stage1_model4500_legacyosc_v1)
아래에서 맞춰졌으므로, 큐브 번들의 관절 물리를 그대로 쓴다. 무엇을 빌려 왔는지는
NORMALIZED.yaml에 남는다.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bundle_params  # noqa: E402

CONTACT_COLUMNS = [
    "sample_id",
    "pair_primary_static_friction",
    "pair_primary_dynamic_friction",
    # 운동 마찰을 정지 마찰로 나눈 비율. 나눗셈으로 되돌리면 될 것 같지만 안 된다.
    # 큐브 번들 2048행 중 278행이 dynamic/static을 다시 계산했을 때 원본 비율과
    # 비트까지 같지 않다. 물리 항이 이 비율을 그대로 쓰므로, 옮길 때 잃으면 큐브
    # 결과가 미세하게 달라진다. 그래서 별도 열로 싣는다.
    "pair_primary_dynamic_ratio",
    "surface_restitution",
    "object_linear_damping",
    "object_angular_damping",
    "contact_damping",
    "object_mass_kg",
]

# 받은 번들의 열 이름을 표준 열 이름으로 옮기는 표. 새 번들이 오면 여기에 한 줄 묶음을
# 더한다. 값이 없는 표준 열은 빈칸으로 남고, 파이프라인은 빈칸을 "이 번들은 그 값을
# 재지 않았다"로 읽는다.
MAPPINGS = {
    "cube": {
        "match": ("cube_cube_static_friction",),
        "columns": {
            "sample_id": "sample_id",
            "pair_primary_static_friction": "cube_cube_static_friction",
            "pair_primary_dynamic_friction": "cube_cube_dynamic_friction",
            "pair_primary_dynamic_ratio": "cube_cube_dynamic_ratio",
            "surface_restitution": "table_cube_restitution",
            "object_linear_damping": "cube_linear_damping",
            "object_angular_damping": "cube_angular_damping",
        },
    },
    "peg_in_hole": {
        "match": ("contact.peg_hole.static_friction",),
        "columns": {
            "sample_id": "candidate_id",
            "pair_primary_static_friction": "contact.peg_hole.static_friction",
            "pair_primary_dynamic_friction": "contact.peg_hole.dynamic_friction",
            "contact_damping": "contact.peg_hole.contact_damping",
            "object_mass_kg": "object.peg.mass_kg",
        },
    },
}

# 표준 배치가 담지 못하는 값. 버리지 않고 NORMALIZED.yaml에 그대로 적어 둔다. 나중에
# 파이프라인이 이 값을 쓸 수 있게 되면 여기부터 보면 된다.
CARRY_AS_NOTES = {
    "peg_in_hole": [
        "task.peg_hole.radial_clearance_m",
        "contact.finger_peg.static_friction",
    ],
}

PARAMS_REL = "parameters.json"

# 대표값과 범위를 담은 계약 파일이 번들마다 다른 자리에 있다.
CONTACT_CONTRACT_CANDIDATES = (
    os.path.join("modules", "contact", "task_contact_randomization.yaml"),
    "peg_contact_parameter_contract.yaml",
)
DYNAMICS_CONTRACT_REL = os.path.join("modules", "dynamics_controller", "nominal_and_ranges.yaml")

DYNAMICS_REL = os.path.join("modules", "dynamics_controller",
                            "domain_randomization_samples.csv")
CONTACT_REL = os.path.join("modules", "contact", "posterior_samples.csv")


def find_contact_csv(root: str) -> str:
    """접촉 표본 CSV를 찾는다. 표준 위치를 먼저 보고, 없으면 최상위를 본다."""
    candidates = [os.path.join(root, CONTACT_REL),
                  os.path.join(root, "posterior_samples.csv")]
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise SystemExit(
        "접촉 표본 CSV를 찾지 못했다. 찾아본 곳: " + ", ".join(candidates))


def detect_schema(header: list[str]) -> str:
    for name, spec in MAPPINGS.items():
        if all(column in header for column in spec["match"]):
            return name
    raise SystemExit(
        "아는 번들 모양이 아니다. 열 이름: " + ", ".join(header) + "\n"
        "MAPPINGS에 이 번들의 매핑을 추가해야 한다.")


def normalize_contact(src_csv: str):
    with open(src_csv, newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise SystemExit(f"{src_csv}에 행이 없다")
    schema = detect_schema(list(rows[0].keys()))
    columns = MAPPINGS[schema]["columns"]
    out_rows = []
    for index, row in enumerate(rows):
        out = {}
        for standard in CONTACT_COLUMNS:
            source = columns.get(standard)
            out[standard] = row.get(source, "") if source else ""
        if not out["sample_id"]:
            out["sample_id"] = str(index)
        out_rows.append(out)
    notes = {}
    for key in CARRY_AS_NOTES.get(schema, []):
        values = sorted({row[key] for row in rows if key in row})
        if values:
            notes[key] = values
    return schema, out_rows, notes


def load_yaml(path: str):
    try:
        import yaml
    except ImportError:
        raise SystemExit(
            "PyYAML이 필요하다. 이 도구는 번들을 표준 형식으로 옮길 때만 쓰므로 "
            "파이프라인 컨테이너가 아니라 준비하는 기계에서 돌린다. pip install pyyaml")
    with open(path, encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def build_parameters(src: str, schema: str, donor_root: str) -> dict:
    """번들의 계약 파일에서 대표값과 범위를 뽑아 표준 파라미터를 만든다."""
    dynamics_doc = {}
    dyn_contract = os.path.join(src, DYNAMICS_CONTRACT_REL)
    if os.path.isfile(dyn_contract):
        dynamics_doc = load_yaml(dyn_contract) or {}

    if schema == "cube":
        contract = os.path.join(src, CONTACT_CONTRACT_CANDIDATES[0])
        if not os.path.isfile(contract):
            raise SystemExit(f"접촉 계약 파일이 없다: {contract}")
        return bundle_params.from_cube_bundle(load_yaml(contract) or {}, dynamics_doc)

    contract = os.path.join(src, CONTACT_CONTRACT_CANDIDATES[1])
    if not os.path.isfile(contract):
        raise SystemExit(f"접촉 계약 파일이 없다: {contract}")
    donor_params = None
    if donor_root:
        donor_dyn = os.path.join(donor_root, DYNAMICS_CONTRACT_REL)
        donor_contact = os.path.join(donor_root, CONTACT_CONTRACT_CANDIDATES[0])
        if os.path.isfile(donor_contact):
            donor_params = bundle_params.from_cube_bundle(
                load_yaml(donor_contact) or {},
                load_yaml(donor_dyn) if os.path.isfile(donor_dyn) else {})
    return bundle_params.from_peg_bundle(load_yaml(contract) or {}, donor_params)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="src", required=True, help="받은 번들 디렉터리")
    parser.add_argument("--out", dest="out", required=True, help="표준 번들을 만들 위치")
    parser.add_argument("--dynamics-from", dest="donor", default="",
                        help="관절 물리를 가져올 번들. 받은 번들에 관절 물리가 없을 때 준다")
    parser.add_argument("--force", action="store_true", help="출력 위치가 비어 있지 않아도 덮어쓴다")
    args = parser.parse_args()

    src = os.path.abspath(os.path.expanduser(args.src))
    out = os.path.abspath(os.path.expanduser(args.out))
    if not os.path.isdir(src):
        raise SystemExit(f"받은 번들이 디렉터리가 아니다: {src}")
    if os.path.isdir(out) and os.listdir(out) and not args.force:
        raise SystemExit(f"{out}가 비어 있지 않다. --force를 주면 덮어쓴다")

    schema, rows, notes = normalize_contact(find_contact_csv(src))

    # 관절 물리. 받은 번들에 있으면 그것을 쓰고, 없으면 빌려 온다.
    own_dynamics = os.path.join(src, DYNAMICS_REL)
    if os.path.isfile(own_dynamics):
        dynamics_src, borrowed = own_dynamics, False
    elif args.donor:
        dynamics_src = os.path.join(os.path.abspath(os.path.expanduser(args.donor)),
                                    DYNAMICS_REL)
        borrowed = True
        if not os.path.isfile(dynamics_src):
            raise SystemExit(f"빌려 올 관절 물리가 없다: {dynamics_src}")
    else:
        raise SystemExit(
            "받은 번들에 관절 물리가 없다. 같은 로봇의 번들을 --dynamics-from으로 준다.\n"
            f"찾아본 곳: {own_dynamics}")

    os.makedirs(os.path.join(out, "modules", "contact"), exist_ok=True)
    os.makedirs(os.path.join(out, "modules", "dynamics_controller"), exist_ok=True)
    shutil.copy2(dynamics_src, os.path.join(out, DYNAMICS_REL))
    with open(os.path.join(out, CONTACT_REL), "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CONTACT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    # 대표값과 범위를 표준 파일로 만든다. 물리 항은 코드에 숫자를 두지 않고 이 파일만 읽는다.
    params = build_parameters(src, schema, os.path.abspath(os.path.expanduser(args.donor))
                              if args.donor else "")
    params["schema_version"] = bundle_params.PARAMS_SCHEMA
    params["source_schema"] = schema
    with open(os.path.join(out, PARAMS_REL), "w", encoding="utf-8") as stream:
        json.dump(params, stream, ensure_ascii=False, indent=1, sort_keys=True)
    still_missing = bundle_params.missing_keys(params)

    record = {
        "schema_version": "fr3_cube.hf80k.normalized_bundle.v1",
        "detected_source_schema": schema,
        "source_bundle": src,
        "contact_rows": len(rows),
        "dynamics_source": dynamics_src,
        "dynamics_borrowed": borrowed,
        "unmapped_values_kept_as_notes": notes,
        "empty_standard_columns": sorted(
            column for column in CONTACT_COLUMNS
            if all(not row[column] for row in rows)),
        "parameters_file": PARAMS_REL,
        "parameters_missing_after_fill": still_missing,
    }
    with open(os.path.join(out, "NORMALIZED.yaml"), "w", encoding="utf-8") as stream:
        for key, value in record.items():
            stream.write(f"{key}: {json.dumps(value, ensure_ascii=False)}\n")

    print(f"[normalize] {schema} 번들 -> {out}")
    print(f"[normalize] 접촉 표본 {len(rows)}행")
    print(f"[normalize] 관절 물리 {'빌려 옴' if borrowed else '받은 번들의 것'}: {dynamics_src}")
    if record["empty_standard_columns"]:
        print("[normalize] 이 번들이 재지 않은 값: "
              + ", ".join(record["empty_standard_columns"]))
    for key, values in notes.items():
        print(f"[normalize] 표준 배치에 자리가 없어 기록만 남긴 값: {key} = {values}")
    borrowed = sorted(k for k, v in params["provenance"].items()
                      if str(v).startswith("borrowed"))
    if borrowed:
        print(f"[normalize] 기증 번들에서 채운 항목 {len(borrowed)}개: {', '.join(borrowed)}")
    if still_missing:
        print(f"[normalize] 채우지 못한 표준 항목: {', '.join(still_missing)}")
    else:
        print("[normalize] 표준 항목이 모두 채워졌다")
    return 0


if __name__ == "__main__":
    sys.exit(main())
