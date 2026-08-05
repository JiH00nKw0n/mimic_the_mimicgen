# Codex 적용 지시서

아래 내용을 대상 repository의 Codex에게 이 패키지 경로와 함께 전달하십시오.

---

이 패키지의 visual randomization 계약을 현재 task에 이식하라. 먼저 `README.md`, `VISUAL_RANDOMIZATION_SPEC.md`, `config/visual_randomization_profiles.yaml`, `ADOPTION_CHECKLIST.md`를 전부 읽고, 대상 repository의 scene/task/camera/asset 구조를 조사한 뒤 구현하라.

필수 요구사항:

1. 연구실 geometry와 task identity를 유지하고 appearance만 계층적으로 randomize한다.
2. profile mixture는 nominal_lab 50%, lab_variation 40%, stress_tail 10%로 episode label과 함께 분리 저장한다.
3. 카메라 nominal/range는 `config/camera_nominal_measured_ranges.yaml`을 source of truth로 사용한다. 대상 scene의 prim 이름이 다르면 semantic parent frame 기준으로 매핑한다.
4. DomeLight, HDRI, global floor는 tiled env별 randomization을 하지 않는다. profile별 process에서 고정하거나 startup에서 한 번만 샘플한다.
5. object/material/camera pose는 episode reset 경계에서만 바꾸고 episode 중간에는 바꾸지 않는다.
6. arbitrary texture/full RGB randomization으로 object identity를 없애지 않는다. nominal object color 주변의 profile별 scale만 사용한다.
7. FR3 articulation, collision, actuator/controller, physics를 visual asset 교체 때문에 변경하지 않는다. visual composition layer만 연결한다.
8. camera는 third_person_0, third_person_1, wrist 세 개이며 third_person_2는 저장하지 않는다. 기본 저장 계약은 320×180, 10 Hz, H.264다.
9. multi-env에서 다른 env의 robot/object pixel이 하나라도 보이면 실패 처리한다. 제공된 isolation audit를 대상 task에 맞게 포팅하고 정량 결과를 남긴다.
10. physics/SysID randomization은 별도 입력 계약을 따르며 visual profile과 임의로 결합하거나 범위를 새로 만들지 않는다.

작업 순서:

1. 대상 repository에서 scene prim, robot visual layer, table/floor, camera parent frame, observation key를 표로 작성한다.
2. 패키지 source snapshot과 대상 구현의 차이를 정리한다.
3. profile YAML을 대상 config 형식으로 연결한다.
4. calibration YAML을 camera config/event에 연결한다.
5. global randomization과 per-env/per-episode randomization scope를 분리한다.
6. 제공 에셋을 대상 repository의 상대경로로 설치하고 모든 hard-coded source path를 제거한다.
7. 1-env smoke → multi-env isolation audit → profile gallery → short collection smoke 순으로 검증한다.
8. `ADOPTION_CHECKLIST.md`를 O/X와 실제 수치로 채운 HTML 또는 Markdown 보고서를 생성한다.

완료 조건:

- three-camera output shape/codec/fps가 계약과 일치
- foreign pixels = 0
- min RGB std ≥ 10
- profile count와 label 정확
- object identity 유지
- episode 중 appearance jump 없음
- visual layer 적용 전후 articulation/controller/physics 값 동일
- 누락 asset과 unresolved USD reference 없음

불확실한 asset mapping이나 parent frame은 추측하지 말고 증거와 함께 blocker로 보고하라. 구현 후 변경 파일, 실행 명령, gate 수치, 생성된 sample 경로를 구체적으로 보고하라.

---

