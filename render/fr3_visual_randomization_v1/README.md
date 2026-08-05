# FR3 Cube Visual Randomization Handoff v1

이 패키지는 연구실 FR3 3-cube stacking RGB distillation에 실제 사용 중인 visual randomization 계약을 다른 팀이 재현하거나 새 task로 이식하기 위한 handoff입니다.

## 현재 production 계약

- 카메라: `third_person_0`, `third_person_1`, `wrist` 3개
- 저장 해상도: 320×180, 16:9, 10 Hz, H.264
- 분포: `nominal_lab 50% / lab_variation 40% / stress_tail 10%`
- 연구실 기하: 책상, 회색 카펫, FR3 facelift, 카메라 nominal frame을 유지
- episode reset마다: 카메라 실측 오차, 물체/그리퍼/책상/커튼 재질을 샘플
- process마다: profile-specific HDRI와 DomeLight를 하나만 선택
- 금지: tiled env 안에서 global DomeLight를 episode마다 변경, 다른 env 로봇/물체 pixel 유입, 물체 identity를 없애는 무작위 색/texture
- physics/SysID randomization은 visual randomization과 별도 계약입니다.

## 먼저 읽을 파일

1. `CODEX_HANDOFF.md`: Codex에게 그대로 전달할 적용 지시서
2. `VISUAL_RANDOMIZATION_SPEC.md`: 설계 원칙과 실제 수치
3. `config/visual_randomization_profiles.yaml`: 기계 판독 가능한 profile 설정
4. `ADOPTION_CHECKLIST.md`: 적용 완료 전 O/X gate
5. `config/camera_nominal_measured_ranges.yaml`: 실측 카메라 nominal/range
6. `DEPENDENCIES.md`: NVIDIA base FR3 USD와 offline 사용 조건

## 포함 내용

- `source/data_collection_rgb_cfg.py`: 현재 FR3 production RGB task config snapshot
- `source/events.py`: 카메라/appearance randomizer 구현이 포함된 MDP event snapshot
- `source/audit_fr3_cube_rgb_isolation_runtime.py`: multi-env pixel isolation gate
- `source/collect_demos_lerobot.py`: 성공 episode LeRobot 수집 예시
- `source/run_fr3_cube_rgb_80k_collection.py`: 50/40/10 profile orchestration 예시
- `assets/fr3`: FR3 base composition, facelift composition과 mesh
- `assets/table`: 연구실 table wrapper와 source scene
- `resources`: 회색 카펫 texture와 profile별 indoor HDRI
- `reference/lab_photos`: visual target으로 사용한 연구실 사진
- `reference/camera_overlay`: 카메라 overlay/calibration 적용 자료

## 적용 원칙

이 패키지를 통째로 덮어쓰지 마십시오. 대상 repository의 scene prim 이름, asset binding, observation key, camera parent frame을 먼저 확인한 뒤 의미 기반으로 매핑해야 합니다. 절대경로는 모두 대상 환경의 경로 또는 환경변수로 치환하십시오.

## 무결성 검사

```bash
python3 tools/verify_bundle.py
python3 tools/preflight_contract.py
```

검사가 통과해도 simulation 적용이 끝난 것은 아닙니다. 마지막으로 `ADOPTION_CHECKLIST.md`의 runtime gate를 실행해야 합니다.
