# FR3 Camera Overlay Team Handoff

## 전달 목적

이 패키지는 FR3 FACELIFT + Franka Hand 환경에 설치된 카메라 4대와 책상/철판
좌표를 팀의 기존 Isaac Sim asset에 적용하기 위한 얇은 calibration overlay입니다.
로봇, 그리퍼, 책상, 철판 geometry는 포함하지 않습니다.

## 포함된 Calibration

- Third-person D435 3대의 `fr3v2_link0` 기준 고정 extrinsic
- Wrist D405의 `fr3v2_hand_tcp` 기준 고정 extrinsic
- 카메라별 color intrinsics와 Isaac USD Camera parameter
- `fr3v2_link0` 기준 table task frame과 threaded-plate frame
- Fixed-camera 및 wrist-camera 검증 그림과 정량 지표

Wrist D405는 hand TCP에 rigid하게 연결해야 합니다. 로봇 관절 자세가 `q`일 때
카메라의 base pose는 다음과 같이 articulation FK를 따라 자동 갱신됩니다.

```text
base_T_wrist_camera(q) = base_T_hand_tcp(q) * hand_tcp_T_wrist_camera
```

## 팀원이 준비할 정보

적용 전에 팀 USD에서 다음 두 prim을 확인하십시오.

1. `fr3v2_link0`에 대응하는 robot base prim
2. `fr3v2_hand_tcp`에 대응하며 articulation을 따라 움직이는 hand TCP prim

그리고 입력 scene USD와 별도의 출력 USD 경로가 필요합니다. 상대 reference가 있는
USD라면 출력 파일을 입력 scene과 같은 폴더에 두는 것이 가장 안전합니다.

팀마다 URDF/USD와 책상 mesh가 다를 수 있으므로 prim 이름이 비슷하다는 이유만으로
직접 binding하면 안 됩니다. 먼저 다음 compatibility를 판정해야 합니다.

- Robot base: 팀 frame이 `fr3v2_link0`과 원점/축까지 같은가?
- Hand frame: 팀 frame이 `fr3v2_hand_tcp`와 원점/축까지 같은가?
- Stage convention: meter 단위이며 Z-up인가?
- Table asset: asset origin에서 실제 top surface까지의 transform이 무엇인가?

Robot frame 정의가 다르면 팀 frame 아래에 calibrated-frame adapter Xform을 만든 뒤
그 adapter prim을 적용 도구에 전달합니다. 책상 mesh가 다르면 `table_T_nominal_asset`
의 `-0.72 m`를 그대로 사용하지 않고, 팀 mesh의 실제 top-surface frame을 새로 맞춥니다.
판정 결과와 adapter transform은 `ASSET_BINDING_TEMPLATE.yaml`을 복사한 팀별 binding
파일에 기록하십시오.

## 빠른 적용 순서

1. 압축을 임의 위치에 풉니다.
2. `overlay.yaml`과 `manifest.yaml`을 읽습니다.
3. `ASSET_BINDING_TEMPLATE.yaml`을 팀 환경 이름으로 복사하고 frame mapping을 기록합니다.
4. NumPy와 PyYAML이 있는 Python에서 bundle 검증을 실행합니다.

```bash
python -m pip install -r requirements-validator.txt
python tools/check_runtime.py --mode validate
python tools/validate_overlay.py .
```

5. Isaac Sim 5.1 bundled Python에서 runtime과 실제 USD preflight를 확인합니다.

```bash
<ISAAC_PYTHON> tools/check_runtime.py --mode apply
<ISAAC_PYTHON> tools/apply_overlay_to_isaac.py --overlay overlay.yaml --stage <TEAM_SCENE_USD> --output <OUTPUT_USD> --base-prim <ROBOT_BASE_PRIM> --hand-tcp-prim <HAND_TCP_PRIM> --validate-only
```

`--validate-only`는 USD를 실제로 열어 단위/Z-up, prim, articulation, reference와
target 충돌을 검사합니다. 기본 충돌 정책은 `error`입니다. 기존 prim을 검토한 뒤에만
`--on-conflict reuse|replace|rename` 중 하나를 명시하십시오. `--plan-only`는 USD를
열지 않는 경로 미리보기이므로 최종 preflight를 대신하지 않습니다.

6. 계획된 prim이 맞으면 `--validate-only`를 제거해 새 USD를 생성합니다.
7. 생성된 USD에서 Camera prim 4대와 calibration frame을 확인합니다.
8. `tools/smoke_render.py`로 Camera sensor RGB 4장을 렌더링합니다.
9. Home 자세와 다른 한 자세에서 wrist view가 로봇과 함께 움직이는지 확인합니다.

패키지의 `validation/isaac_5_1_smoke/`는 기준 scene에서 통과한 참고 결과입니다.
팀 scene의 asset과 prim binding이 다르므로 전달받은 환경에서도 smoke test를 다시
실행해야 합니다.

입력 USD는 직접 덮어쓰지 마십시오. 적용 도구는 공통 asset을 복사하거나 geometry를
변경하지 않고 Camera prim과 calibration frame만 author합니다.

## Validation Scene 주의사항

`fixed_camera_board_scene`과 `wrist_handeye_board_scene`의 ChArUco 보드는 서로 다른
시점과 물리 위치에서 촬영되었습니다. 두 board pose는 calibration 검증용이며 실제
cube task의 object pose나 공통 board pose로 사용하면 안 됩니다.

## 현재 Calibration의 제한

- Wrist D405는 테이프 마운트 상태의 임시 depth-refined extrinsic입니다.
- D405 또는 테이프가 움직이거나 재장착되면 wrist extrinsic이 무효가 됩니다.
- Fixed camera, robot base, table이 서로 상대적으로 움직이면 관련 calibration이 무효가 됩니다.
- RealSense distortion coefficient는 metadata로 제공되지만 기본 USD pinhole camera에는 자동 적용되지 않습니다.
- 조명, 노출, white balance, material appearance는 이 패키지의 calibration 범위가 아닙니다.

## 전달 완료 기준

- `tools/validate_overlay.py`가 성공할 것
- Isaac stage-aware `--validate-only`가 성공할 것
- 카메라 serial과 role이 `overlay.yaml`과 일치할 것
- Third-person 카메라 3대가 robot base frame에 고정될 것
- Wrist D405가 hand TCP의 자식으로 연결될 것
- 서로 다른 두 `q`에서 wrist camera가 articulation을 따라 움직일 것
- `tools/smoke_render.py`가 네 카메라의 nonblank RGB를 생성할 것
- 원본 scene과 별도의 출력 USD가 생성될 것
