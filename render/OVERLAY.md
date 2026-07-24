# OVERLAY.md — FR3 카메라 캘리브레이션 번들 레퍼런스

주상님이 2026년 7월 21일에 공유한 실기 카메라 캘리브레이션 번들
`fr3_camera_overlay_v1` (schema `stage2.fr3_camera_overlay.v1`, revision
`fr3_four_camera_v3_depth_refined`)의 내용과, 이 폴더의 렌더 파이프라인이
그것을 소비하는 방식을 정리한 문서다. 번들 사본은 `fr3_camera_overlay_v1/`에
있고 canonical 데이터는 그 안의 `overlay.yaml` 하나다.

핵심 성격: 실제 랩 FR3 팔에 설치된 카메라 4대의 위치와 렌즈 정보를
시뮬레이션으로 옮기기 위한 "얇은" 오버레이. 로봇, 그리퍼, 책상, 나사판
geometry는 포함하지 않고 좌표 변환과 카메라 파라미터만 담는다.

현재 상태는 **임시**다 (`provisional_wrist_tape_mount`). 손목 카메라가 테이프로
고정된 상태에서 캘리브레이션했고, 새 카메라 마운트가 완성되면 갱신 번들이 온다.

## 카메라 4대

| role | 기종 | serial | 부착 | 기준 프레임 | 상태 |
| --- | --- | --- | --- | --- | --- |
| third_person_0 | RealSense D435 | 405622073775 | 고정 | fr3v2_link0 (로봇 베이스) | 오버레이 검증 통과 |
| third_person_1 | RealSense D435 | 405622072503 | 고정 | fr3v2_link0 | 오버레이 검증 통과 |
| third_person_2 | RealSense D435 | 401622073398 | 고정 | fr3v2_link0 | 오버레이 검증 통과 |
| wrist | RealSense D405 | 352122274598 | 그리퍼에 강결합 | fr3v2_hand_tcp (손끝 중심) | 테이프 마운트 임시 |

배치 감각 (베이스 프레임 top view 기준): third_person_0은 왼쪽 뒤 위에서,
third_person_1은 오른쪽 앞 위에서 작업 공간을 내려다보고, third_person_2는
정면 약 1.5m 거리에서 로봇 쪽을 바라본다. wrist는 손끝 중심에 붙어 팔을
따라 움직인다. 그림: `fr3_camera_overlay_v1/validation/camera_layout.png`.

## 좌표계와 규약

- 단위는 meter와 radian. 행렬은 열벡터 규약 (p_parent = T @ p_child).
  YAML의 matrix는 행 단위로 적혀 있고 translation은 4번째 열에 있다
  (`translation_m` 필드로 한 번 더 중복 기록됨 — 파서 self-check에 쓰기 좋다).
- 카메라 프레임이 두 벌 있다. `parent_T_camera_optical`은 RealSense/OpenCV
  광학 규약 (X 오른쪽, Y 아래, Z가 시선 방향)이고, `parent_T_camera_usd`는
  USD/OpenGL 카메라 규약 (X 오른쪽, Y 위, 시선은 -Z)이다. 관계는
  R_usd = R_optical @ diag(1, -1, -1). Isaac에 넣을 때는 usd 쪽을 그대로 쓴다.
- YAML의 quaternion 필드는 전부 **(x, y, z, w)** 순서다 (`quaternion_xyzw`).
- 기준 프레임 의미: `fr3v2_link0`은 FR3 베이스 링크, `fr3v2_hand_tcp`는
  핸드 플랜지에서 약 10.34cm 앞 손끝 중심 (Franka Hand TCP 규약),
  `fr3v2_table_task`는 책상 위에 표시된 작업 원점, `fr3v2_threaded_plate`는
  나사판 원점이다.

## overlay.yaml 구조 요약

- `runtime.reference_robot_pose` — Franka Home 관절값 (0, -0.785, 0, -2.356,
  0, 1.571, 0.785)과 그때의 `base_T_hand_tcp`. 팀 로봇의 프레임 정의가
  캘리브레이션과 같은지 순기구학으로 검증하라고 주는 기준값이다.
  이 값을 손목 카메라의 고정 pose로 쓰면 안 된다 (관절이 바뀌면 무효).
- `runtime.cameras.<role>` — 카메라별로 extrinsic 두 벌(optical/usd),
  1280x720 color intrinsics (fx, fy, ppx, ppy, 왜곡계수), 그리고
  `isaac_camera_model` (USD 핀홀로 미리 환산된 focal length 20mm,
  가로/세로 aperture, aperture offset, clipping range). USD에 넣을 때는
  이 환산값을 글자 그대로 쓰면 된다 (번들의 자체 적용 도구
  `tools/apply_overlay_to_isaac.py`와 동일한 방식).
- `runtime.environment` — 베이스 기준 책상 프레임(`base_T_table`, 책상 위
  작업 원점이 원점이고 z=0이 상판)과 나사판 프레임. 책상 상판은 베이스보다
  19.4mm 위에 있다. 나사판은 회전 없이 평행이동만 재측정된 임시값.
- `validation_scenes` — ChArUco 보드 두 장면. **검증 전용**이다. 보드 pose를
  큐브 태스크의 물체 pose로 쓰거나 두 장면을 같은 보드로 취급하면 안 된다.
- `invalidation` / `limitations` — 아래 두 절 참조.

## 캘리브레이션 정확도 (번들 기재값)

- 고정 카메라 3대: 위치 편차 최대 약 3.2mm, 회전 편차 최대 약 0.18도
- 손목 카메라: depth 보정 후 거리 오차 중앙값 4.15mm (보정 전 6.77mm에서
  39% 개선), p95 20.8mm
- 보드 재투영: 모서리 평균 약 6.2px, p95 약 7.5px (1280x720 기준)
- 책상 등록: 평면 RMS 0.26mm 수준, 품질 게이트 전부 통과

## 무효가 되는 조건 (번들 명시)

- 고정 카메라 중 하나라도 움직였을 때
- 손목 카메라나 테이프 마운트가 움직이거나 재장착됐을 때
- 책상과 로봇 베이스의 상대 위치가 바뀌었을 때

## 알려진 한계

- 손목 extrinsic은 테이프 마운트 상태의 임시값이다.
- RealSense 왜곡계수는 metadata로만 있고 USD 핀홀 렌더에는 적용되지 않는다
  (렌더는 무왜곡). 여기에 더해 Omniverse 렌더러는 USD의 aperture offset과
  세로 aperture를 무시하므로 (OM-42611), 실제 렌더 픽셀은 fx 기준 정사각
  픽셀에 주점이 정중앙이다. 원본 해상도 기준 최대 약 13px 차이. 우리 출력
  hdf5에는 실렌더용 `K_effective_render`와 실카메라용
  `K_calibrated_at_render_size`를 둘 다 기록한다.
- 조명, 노출, 재질 외형은 캘리브레이션 범위 밖이다.

## 우리 파이프라인이 번들을 쓰는 방식

번들의 어댑터 정책이 핵심이다: 팀 시뮬레이션의 prim 이름이 비슷하다는
이유만으로 캘리브레이션 프레임에 바로 연결하지 말고, 프레임 정의가 같은지
판정한 뒤 다르면 어댑터 변환을 만들어 기록하라는 것. 우리는 이것을
`probe_tcp_binding.py`로 자동화했다.

1. probe가 시뮬레이션 FR3를 번들의 reference home 관절값으로 옮기고,
   순기구학으로 잰 손 위치를 번들의 `base_T_hand_tcp`와 대조한다.
   베이스 프레임 차이는 yaw 후보 (0, 90, 180, 270도)에서 맞춰보고,
   손 어댑터 `fr3_hand_T_fr3v2_hand_tcp`를 풀어낸다. 게이트는 물리 불변량
   (TCP 거리가 0.1034m 근처, 그리퍼 축 정렬)이다.
2. 결과가 `fr3_binding.yaml`로 남는다. 계측 결과 Isaac의 공식
   `FrankaFR3/fr3.usd`는 **베이스 어댑터 identity, hand_T_tcp 정확히
   (0, 0, 0.1034), 회전 0도** — 즉 캘리브레이션 프레임 정의와 완전히 같다.
   binding에는 번들 revision이 함께 박혀서, 번들만 갈아끼우고 probe를 안
   돌리면 렌더러가 거부한다 (무효화 정책의 코드 강제).
3. `overlay_cameras.py`가 overlay.yaml에서 `parent_T_camera_usd`와
   `isaac_camera_model`을 읽어 Isaac Lab 카메라 4대를 만든다. 고정 3대는
   fr3_link0 밑에 (베이스 어댑터 적용), wrist는 fr3_hand 밑에
   (hand_T_tcp 합성) 자식 prim으로 붙는다.
4. `render_viewpoints.py`가 데모의 기록된 상태를 재생하며 매 스텝 4대의
   RGB를 캡처한다. 검증 수치: 리플레이한 손 위치와 데모에 기록된 관측이
   서브밀리미터로 일치 (stack 0.6~1.1mm, peg는 hand 프레임 기준 서브mm).

Isaac Lab 버전별 쿼터니언 규약 주의사항 (파이프라인이 자동 감지해 처리하지만
새 코드를 짤 때 알아야 함)은 `README.md`의 검증 상태 절에 정리돼 있다.

## 번들이 갱신되면 (새 마운트 완성 후)

1. 새 tar를 풀어 `fr3_camera_overlay_v1/`을 통째로 교체한다 (폴더명이 바뀌면
   스크립트들의 `--overlay` 기본 경로도 같이).
2. `tools/validate_overlay.py`로 무결성 확인 (`portable_overlay_validation_ok true`).
3. `bash run_probe.sh` 재실행 — binding revision 검증 때문에 생략 불가.
4. 데모 2~3개로 소규모 렌더를 돌려 프리뷰 mp4를 눈으로 확인한 뒤 본 렌더.

## 파일 포인터

- `fr3_camera_overlay_v1/overlay.yaml` — canonical 캘리브레이션 데이터
- `fr3_camera_overlay_v1/manifest.yaml` — 파일 목록, 체크섬, 필요 asset
- `fr3_camera_overlay_v1/TEAM_HANDOFF_KO.md` — 주상님의 원본 적용 가이드
- `fr3_camera_overlay_v1/LLM_SETUP_PROMPT_KO.md` — LLM에게 붙여넣는 구성 프롬프트
- `fr3_camera_overlay_v1/validation/` — 실기 대 시뮬 비교 그림과 정량 지표
- `overlay_cameras.py`, `probe_tcp_binding.py`, `render_viewpoints.py`,
  `lab_env.py` — 이 폴더의 소비 파이프라인 (사용법은 `README.md`)
- `FR3_CAMERA_OVERLAY_notion.md` — 랩 Notion Archive용 요약 초안
