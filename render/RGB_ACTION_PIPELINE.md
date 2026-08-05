# RGB-Action 데이터 생성 파이프라인 (작업 3)

작성 2026-08-05, MimicGen팀. MimicGen으로 증폭한 3-큐브 쌓기 시연에 실측 카메라 시점의
영상을 붙이고, RL팀의 시각 랜덤화 규격을 적용해 (영상, 계약 행동) 학습 데이터를 만드는
파이프라인의 구현·검증 기록이다. 이 문서만으로 재현할 수 있게 명령과 수치를 모두 적었다.

## 1. 입력물

| 무엇 | 어디서 | 이 작업에서 쓰는 부분 |
|---|---|---|
| 증폭된 시연 | 서버 `~/contract_out/gen_human_25.hdf5` (사람 소스 25개 성공분) | `states/` 전 프레임(로봇 관절·큐브 자세) — 상태 재생 렌더링의 입력 |
| 카메라 캘리브레이션 | 저장소 `render/fr3_camera_overlay_v1/overlay.yaml` (bundle `fr3_four_camera_overlay_depth_refined`) | 고정 D435 3대 + 손목 D405 1대의 실측 장착 위치·렌즈 |
| 씬 바인딩 | 저장소 `render/fr3_binding.yaml` | 캘리브레이션 프레임(`fr3v2_link0`)과 우리 씬 프림(`fr3_link0`)의 대응. `probe_tcp_binding.py`가 시뮬에서 측정해 생성하며 `ready_to_apply: true`여야 렌더러가 받아들인다 |
| 시각 랜덤화 규격 | `fr3_visual_randomization_handoff_v1_320x180` (RL팀, 2026-08-04). 서버 `~/fr3_visual_randomization_v1/`, 저장소에는 경량부만 `render/fr3_visual_randomization_v1/` | 프로파일 혼합·색·재질·HDRI·카메라 오차 범위 |

## 2. 실행 방법

```bash
# 서버(aidas)에서. 경로는 컨테이너 내부 기준(/repo=저장소, /out=~/contract_out,
# /vrand=~/fr3_visual_randomization_v1)
cd ~/mimicgen_jihoonkwon/mimic_the_mimicgen

# (1) 영상: 카메라 4대를 320x180으로 렌더 + 시각 랜덤화
render/run_render_aidas.sh --dataset /out/gen_human_25.hdf5 --count 25 \
    --width 320 --height 180 --vrand mixture --vrand_seed 7 \
    --output /out/rgb_human25.hdf5

# (2) 행동: 같은 시연을 계약 형식(10 Hz)으로 변환
contract/run_convert_aidas.sh --dataset /out/gen_human_25.hdf5 \
    --output /out/gen_human_25_contract.hdf5 --count 25 --source_hz 20

# (3) 결합: 영상(20 Hz)과 계약 행동(10 Hz)을 인덱스로 짝지어 최종 데이터셋
python3 contract/join_rgb_contract.py --rgb /out/rgb_human25.hdf5 \
    --contract /out/gen_human_25_contract.hdf5 \
    --output /out/rgb_action_human25.hdf5
```

## 3. 구현한 것

- `render/run_render_aidas.sh` — 렌더러의 aidas 도커 실행 래퍼. 기존 `run_render.sh`는
  사라진 arpa 서버의 UWLab 가상환경 전용이라 이 서버에서는 쓸 수 없었다.
- `render/visual_randomization.py` — RL팀 규격의 재구현. 그들의 `source/events.py`는
  IsaacLab 매니저 기반 이벤트(에피소드 리셋 시 발화)라 상태 재생 루프인 우리 렌더러에서는
  아예 실행되지 않는다. 그래서 옮긴 것은 코드가 아니라 계약이며, 숫자는 그들의 YAML을
  직접 읽어 쓰므로 손으로 옮겨적은 값이 없다. 적용 범위는 그들의 `scope_contract`를
  따른다: 프로파일·HDRI·바닥은 프로세스 단위, 카메라 자세·초점과 물체 재질은 에피소드
  단위, 에피소드 중간 변경 없음.
- `render/render_viewpoints.py` — `--vrand {nominal_lab|lab_variation|stress_tail|mixture}`
  플래그 추가. `mixture`는 에피소드를 50/40/10으로 정확히 배분한다(이항 추출이 아니라
  최대잉여법이라 25개면 정확히 13/10/2).
- `contract/join_rgb_contract.py` — 영상과 계약 행동의 결합. 두 파일의 길이비가 정수가
  아니면 조용히 어긋나는 대신 그 에피소드를 거부한다.
- `render/diag_camera_world.py` — 렌더 시점 카메라 월드 자세 진단(4절에서 쓴 도구).

## 4. 해결한 문제: 카메라가 작업면 대신 바닥을 보던 원인

첫 렌더는 정상 종료했지만 고정 카메라 2대가 바닥만 비췄다. 원인 추적 순서는 이랬다.

1. 캘리브레이션 값으로 카메라 시선을 손계산해 보니, 로봇 받침이 180° 회전한 상태라면
   세 카메라 모두 큐브를 화면에 담아야 했다. 관측과 반대였다.
2. 받침이 회전하지 않은 경우로 다시 계산하니 관측과 정확히 일치했다(third_person_2만
   큐브가 화면 안, 나머지 둘은 밖).
3. `render/diag_camera_world.py`로 시뮬이 보고하는 카메라 월드 좌표를 직접 읽어 확정:
   카메라 프림은 **매 프레임 기록에서 써넣는 받침 자세가 아니라 스폰 시점의 자세**를
   따르고 있었다. 관절 재생은 써넣은 자세를 쓰므로 로봇과 카메라가 180° 어긋났다.
4. 근본 원인은 Isaac Lab 3.0의 쿼터니언 규약이다. 씬 설정의 회전값 `(0,0,0,1)`은
   wxyz로 읽으면 180° 요yaw지만 이 버전은 스폰에서 xyzw로 읽어 무회전이 된다.
5. 조치: 렌더 경로에서만 스폰 회전을 실제 180°가 되도록 지정한다
   (`LAB_ROBOT_SPAWN_ROT=0,0,1,0`, `render/run_render_aidas.sh`가 기본으로 넘긴다).
   오프라인 FK 변환과 warm-start 실행기는 스폰 자세에서 상수를 유도하므로 영향받지
   않도록 옵트인으로 두었다.

효과는 측정으로 확인했다. 고정 카메라 third_person_0의 시선이 책상면과 만나는 지점과
가장 가까운 큐브의 거리가 **0.99 m → 0.15 m**, 손목 카메라는 **1.01 m → 0.095 m**로
줄었고, 렌더 이미지에 세 큐브가 모두 보인다. 시연 재생 자체의 정합(재생 프레임과 기록
관측의 손끝 위치 차이)은 수정 전후 모두 mm 수준을 유지했다(1.3 mm → 3.2 mm).

## 5. 확인된 사항과 한계

**동작 확인.**
- 카메라 4대(third_person_0/1/2, wrist), 320×180, 계약이 요구하는 16:9.
- 시각 랜덤화가 실제로 적용된다: HDRI·돔라이트 강도·바닥 재질(프로세스 단위), 큐브
  3개·그리퍼·책상의 색과 거칠기/금속성(에피소드 단위), 카메라 위치·자세·초점 흔들기
  (에피소드 단위, `skipped=[]`로 전 항목 적용 확인).
- 큐브 색이 우리 씬 기본값(노랑)이 아니라 계약의 실측 실험실 색(검정 등)으로 바뀌는
  것을 이미지로 확인했다.
- 이미지 대비 게이트: 계약이 요구하는 `min_rgb_std 10.0`에 대해 실측 23.5~76.9로 통과.

**한계와 미해결.**
1. **씬 배경이 실험실이 아니다.** 패키지가 실험실 책상 USD(`assets/table/table_scene.usdc`)를
   제공하지만, 그 에셋은 RL팀 씬의 원점 기준으로 배치돼 있어 우리 씬(로봇이
   `(0.72, 0.138, 0.722)`)에 그대로 놓으면 작업 영역 아래에 오지 않는다. 그 상태로
   렌더하면 다시 바닥이 보인다. 현재는 검증된 대체 슬랩으로 렌더하며, 실험실 책상·카펫·
   커튼을 제자리에 놓는 것은 남은 작업이다. 커튼은 우리 씬에 대응물이 아예 없어
   랜덤화 대상에서 제외했다(코드가 `skipped`로 보고한다).
2. **카메라 캘리브레이션 판본.** 우리가 쓰는 overlay는 7월 21일 판(v3)이고, 시각 랜덤화
   패키지에는 7월 28일 재캘리브레이션(v4) 값이 들어 있다. 고정 카메라는 12~14 cm,
   손목은 8 cm와 158° 차이라 측정 오차가 아니라 재설치로 보인다. 새 값으로 갈아타려면
   바인딩 프로브를 다시 돌려야 하므로 이번 산출물은 v3 기준이며, 두 판본 식별자를 모두
   출력 파일 메타데이터에 적어 두었다.
3. **다중 환경 픽셀 격리 게이트는 해당 없음.** 계약의 `max_foreign_pixels_per_camera_env: 0`은
   여러 환경을 한 화면에 타일로 배치해 학습할 때 옆 환경이 화면에 새어 들어오는지 보는
   검사다. 우리는 한 번에 한 환경만 렌더하므로 구조적으로 발생하지 않는다.
4. **좌표 규약이 데이터셋마다 다르다.** 사람 teleop 원본은 기록된 받침 회전이 wxyz 180°로
   읽히고, MimicGen이 생성한 에피소드는 무회전으로 읽힌다. 같은 검증식을 두 데이터에
   그대로 쓰면 후자에서 91 cm 오차가 난다(실측: 무회전 3.0 cm vs 180° 91.6 cm). 계약
   변환의 좌표 검증이 두 규약을 모두 시도해 맞는 쪽을 보고하도록 고쳤고, 내보내는
   행동 자체는 시뮬의 실제 받침 프레임 기준이라 영향받지 않는다.

## 6. 산출물

- 데이터셋: 서버 `~/contract_out/rgb_action_human25.hdf5` (계약 스키마 + 카메라별
  `obs/<role>_image`), 중간 산출물 `rgb_human25.hdf5`(영상)와
  `gen_human_25_contract.hdf5`(계약 행동).
- 메타데이터: 출력 파일의 `data` 속성에 적용된 시각 랜덤화 전체(프로파일 혼합, 시드,
  에피소드별 프로파일, HDRI·조명·바닥 값), 카메라 overlay/바인딩 식별자, 소스 데이터셋
  경로가 기록된다.
