# LLM 입력용 Isaac Sim 구성 프롬프트

아래 프롬프트의 `<...>` placeholder만 현재 환경에 맞게 바꾼 뒤 LLM 세션에 입력한다.

```text
나는 FR3 FACELIFT + Franka Hand와 cube-stacking table asset이 이미 들어 있는
Isaac Sim 환경을 가지고 있다. 첨부하거나 지정한 `<BUNDLE_DIR>`는
`stage2.fr3_camera_overlay.v1` 형식의 portable calibration overlay다.
우리 팀의 robot URDF/USD, prim hierarchy, hand TCP frame, table mesh는 calibration을
만든 원본 환경과 이름 또는 asset origin이 다를 수 있다.

목표:
1. `<BUNDLE_DIR>/overlay.yaml`을 canonical calibration으로 사용한다.
2. `<TEAM_SCENE_USD>`에 third-person D435 3대와 wrist D405 1대를 배치한다.
3. 결과는 원본을 덮어쓰지 않고 `<OUTPUT_USD>`로 저장한다.
4. 팀의 기존 robot/table geometry를 재생성하거나 복사하지 않는다.

반드시 다음 순서로 진행해줘.

1. 먼저 `<BUNDLE_DIR>/README.md`, `TEAM_HANDOFF_KO.md`, `manifest.yaml`,
   `overlay.yaml`, `ASSET_BINDING_TEMPLATE.yaml`을 읽고 schema와 asset requirement를 요약한다.
2. Python 3.10+ 환경에서 `requirements-validator.txt`를 설치하고
   `<BUNDLE_DIR>/tools/check_runtime.py --mode validate` 및
   `<BUNDLE_DIR>/tools/validate_overlay.py <BUNDLE_DIR>`를 실행해 checksum,
   내부 상대 경로, transform, 최종 wrist calibration ID를 검증한다.
3. `<TEAM_SCENE_USD>`와 그 robot URDF/USD를 조사하고 다음 표를 먼저 만든다.
   - stage meters-per-unit와 up axis
   - articulation root와 arm joint 이름
   - 팀 robot base prim 및 frame 정의
   - 팀 rigid hand 또는 TCP prim 및 frame 정의
   - 팀 table prim, asset origin, 실제 top-surface frame
   - 기존 Camera prim과 sensor prim
4. 팀 asset을 아래 compatibility class로 분류한다.
   A. `exact_semantic_frame`: 이름은 달라도 원점과 축이 calibrated frame과 동일
   B. `known_adapter_required`: 원점/축이 다르지만 고정 transform을 URDF/USD에서 계산 가능
   C. `unknown_or_incompatible`: 필요한 고정 transform을 입증할 수 없음
5. Robot base와 hand TCP가 A이면 직접 binding한다. B이면 다음 adapter Xform을 팀
   asset 아래에 만들고 그 adapter prim을 binding한다.
   - `team_base_T_calibrated_fr3v2_link0`
   - `team_hand_T_calibrated_fr3v2_hand_tcp`
   변환 방향을 명시하고 identity를 임의로 가정하지 않는다. C이면 적용을 중단하고
   부족한 URDF fixed joint 또는 frame 정보를 보고한다.
6. Table asset이 원본 nominal asset과 다르면 `table_T_nominal_asset`의 `-0.72 m`를
   사용하지 않는다. overlay의 `table_task` frame은 유지하되 팀 table mesh의 실제
   top surface가 `table z=0`에 오도록 별도 `table_task_T_team_table_asset`을 계산한다.
   외형 크기 차이는 허용하지만 cube가 놓이는 surface 높이와 축은 확인한다.
7. `<BUNDLE_DIR>/ASSET_BINDING_TEMPLATE.yaml`을 바탕으로 `<TEAM_BINDING_YAML>`을
   작성한다. prim 경로, compatibility class, adapter transform, table surface alignment,
   근거가 된 URDF joint 또는 USD prim을 기록한다.
8. 결정한 prim과 adapter를 나에게 보여주고 다음을 확인한다.
   - calibrated robot base와 hand TCP semantic이 보존된다.
   - hand adapter는 arm joint q를 따라 움직이는 rigid frame 아래에 있다.
   - 기존 카메라가 있다면 중복 생성 여부를 확인한다.
9. Isaac Sim 5.1 bundled Python에서 `tools/check_runtime.py --mode apply`를 실행한 뒤
   아래 도구를 먼저 `--validate-only`로 실행한다. 다른 Isaac 버전은 검증되지 않았다.

   <ISAAC_PYTHON> <BUNDLE_DIR>/tools/apply_overlay_to_isaac.py --overlay <BUNDLE_DIR>/overlay.yaml --stage <TEAM_SCENE_USD> --output <OUTPUT_USD> --base-prim <ROBOT_BASE_PRIM> --hand-tcp-prim <HAND_TCP_PRIM> --validate-only

   이 preflight는 USD를 실제로 열어 units/up-axis, prim, articulation ancestry,
   unresolved dependency와 target conflict를 검사해야 한다. 기본 `--on-conflict error`를
   유지하고, 기존 prim의 type/transform을 확인한 뒤에만 reuse/replace/rename을 선택한다.

10. 계획된 Camera prim 4대의 경로가 맞으면 `--validate-only`를 제거하고 적용한다.
11. 출력 USD를 다시 열어 다음을 검사한다.
   - third_person_0, third_person_1, third_person_2가 Camera prim이다.
   - wrist_d405_color가 hand TCP 아래의 Camera prim이다.
   - focal length, aperture, aperture offset, clipping range가 overlay와 일치한다.
   - table_task, nominal_table_asset_alignment, threaded_plate frame이 생성됐다.
12. Home q와 Home이 아닌 안전한 q 한 개에서 wrist camera world transform을 비교해
   `hand_tcp_T_camera`는 일정하고 `base_T_camera(q)`는 articulation을 따라 변하는지 확인한다.
13. `tools/check_runtime.py --mode render`를 확인하고 `tools/smoke_render.py`로 네 Camera의
    RGB를 렌더링한다. 가능하면 articulation prim과 안전한 두 번째 q를 전달해 wrist RGB가
    두 자세에서 달라지는지도 확인하고 경로를 알려준다.
14. 마지막에 사용한 prim binding, adapter transform, table alignment, 생성된 USD,
    검증 결과, 남은 제한사항을 Markdown으로 정리한다.

중요한 제한:
- `overlay.yaml`의 runtime 값을 다른 과거 v2/v3 YAML로 교체하지 않는다.
- wrist의 world 기준 reference-pose transform을 고정 pose로 사용하지 않는다.
- `fixed_camera_board_scene`과 `wrist_handeye_board_scene`은 서로 다른 검증 장면이다.
- board pose를 cube task의 object pose로 사용하지 않는다.
- wrist D405는 현재 테이프 마운트용 임시 calibration임을 결과에 명시한다.
- 실제 파일이나 prim을 찾을 수 없다면 임의로 만들지 말고 정확히 무엇이 없는지 보고한다.
- 팀 URDF의 hand frame이 다르면 camera extrinsic을 그대로 그 frame에 붙이지 않는다.
- 팀 table mesh가 다르면 nominal table asset offset을 그대로 재사용하지 않는다.
```

## 짧은 확인용 프롬프트

이미 overlay가 적용된 뒤 검토만 요청할 때 사용한다.

```text
`<BUNDLE_DIR>/overlay.yaml`을 기준으로 `<OUTPUT_USD>`의 카메라 4대와 환경 frame을
검토해줘. 카메라별 parent prim, local transform, intrinsics를 overlay와 비교하고,
wrist D405가 hand TCP의 자식으로서 서로 다른 q를 따라 움직이는지 확인해줘.
원본 asset geometry는 수정하지 말고 발견한 불일치만 파일/prim 경로와 함께 보고해줘.
```
