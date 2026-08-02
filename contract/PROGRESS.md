# FR3 Control-Contract 통합 작업 기록 (PROGRESS)

작성 2026-08-03. 이 문서는 `contract/` 폴더에서 진행한 FR3 3-cube sim2real 통합 작업을,
이 프로젝트를 처음 보는 사람도 따라올 수 있게 처음부터 기록한 것이다. 어떤 배경에서 무엇을
받았고, 무엇을 만들었고, 어떤 검증을 통과했고, 무엇이 아직 안 되는지를 순서대로 담는다.

## 1. 배경: 왜 이 작업을 하는가

연구실의 sim 합성 데이터 프로젝트는 세 팀이 나눠 진행한다. RL팀(김재익)은 Isaac 시뮬레이터에서
FR3 로봇(Franka Research 3)의 3-큐브 쌓기 정책을 강화학습으로 이미 학습했다. MimicGen팀(권지훈,
이 문서의 작업)은 소수의 시연(demonstration)을 MimicGen 계열 방법으로 증폭해 학습 데이터를
만든다. System Identification팀은 실물 FR3와 시뮬레이터의 동역학 차이를 식별한다.

2026년 8월부터 세 팀의 결과물을 실기 이전(sim2real)에 쓰기 위해 로봇 제어 스택을 하나로
통일하기로 했고, 그 기준은 "RL 학습 당시의 제어 방식"이다. RL 정책은 재학습하지 않기로 확정돼
있어서, 다른 팀이 그 방식에 맞춰야 한다. 이 "방식"을 명문화한 것이 control contract다:

- 행동(action)은 7개 숫자 `[dx, dy, dz, drx, dry, drz, gripper]`. 현재 실제
  엔드이펙터(EE, 손끝) 위치 기준의 상대 이동이며, 숫자 1이 xyz는 2 cm, 회전은 0.02/0.02/0.2
  라디안을 뜻한다. 클리핑은 없다.
- 정책은 10 Hz로 명령하고, 120 Hz 토크 제어기가 명령 하나를 12 물리스텝 동안 실행한다.
- 제어기는 `RelCartesianOSCAction`이라는 UWLab(워싱턴대 Isaac Lab 포크) 구현의
  Operational Space Controller다. 강성 (200,200,200,3,3,3), 감쇠비 (3,3,3,1,1,1),
  질량행렬 미사용, 널스페이스 목표는 리셋 직후 관절값. 게인·주기·좌표 규약 변경은 금지 항목이다.
- 쿼터니언은 wxyz 순서, 좌표계는 로봇 베이스, 회전 합성은 왼쪽 곱(`q_delta ⊗ q_current`)이다.

MimicGen팀에 떨어진 요구는 두 가지다. 첫째, 우리가 생성한 시연 데이터를 이 계약의 action
규약으로 변환하고, 계약 제어기로 시뮬레이터에서 재실행해 여전히 성공하는지 검증할 것("계약
정합"). 둘째, 사람 시연 대신 RL 정책이 만든 시연(RL-teacher demo)을 소스로 썼을 때 생성
수율이 어떤지 사람 시연과 비교할 것.

## 2. 받은 입력물 세 가지

**(a) `fr3_cube_stage1_handoff_20260801.tar.gz`** — RL팀의 계약 패키지.
`common/control_contract.yaml`(위 계약 명세), `common/controller_adapter.py`(의존성 없는
pose↔action 변환기, self-test 포함), `common/cube_legacy_profile.py`(환경 설정에 계약 게인을
주입하는 함수), `mimicgen/dataset_schema.yaml`(최종 HDF5 형식: contract_id, 10 Hz 타임스탬프,
actions [T,7] 등), `mimicgen/ACCEPTANCE_CRITERIA.md`(합격 체크리스트),
`frozen_payload/historical_09f7e5b/`(학습 당시 제어기 소스 동결본 — "authoritative execution
contract"라고 명시됨)를 담는다. RL 체크포인트 자체는 배포하지 않는다.
주의사항으로, 원래 학습 런의 `env.yaml`이 유실돼 계약은 "높은 신뢰도의 재구성"이며, 로봇
URDF/USD는 공칭값(실기 시리얼 보정 아님)이라는 한계를 패키지 스스로 명시한다.

**(b) `full_success_hdf5_bundle_20260801.zip`** — RL-teacher 성공 시연.
3-cube 쌓기, peg-in-hole, multi-gear 각 50개. Isaac Lab 에피소드 형식(`states/`에 관절·큐브
상태, `actions [T,7]`, 10 Hz)이고 3-cube의 환경 id는
`OmniReset-Fr3PandaCube-FullStack-RelCartesianOSC-State-Play-v0`이다. 3-cube 50개를 검사한
결과 모두 세 큐브가 테이블 위에서 시작하고, 항상 cube_2를 cube_1 위에, cube_3를 cube_2 위에
쌓는 canonical 순서였다(60~137스텝).

**(c) `fr3_cube_system_calibration_bundle_v1.tar.gz`** — 시스템 캘리브레이션 통합본(공지환).
세 모듈로 구성된다. contact(테이블-큐브·큐브-큐브·핑거-큐브 유효 접촉계수의 nominal/범위/사후
분포 — 앞서 별도 배포된 stage2 v2와 동일 스키마), dynamics_controller(D405 손목 카메라 payload
94.6 g, 관절 armature 0.1 kg·m² 등 관절 동역학 nominal/범위, 동결 OSC 계약 재수록),
camera(D435 3대 + 손목 D405의 실측 extrinsic/intrinsic과 불확실성 범위, 에피소드 단위 샘플러
도구 동봉). 무결성 검증 도구(`validate_bundle.py`)가 통과함을 확인했다.

## 3. 우리가 이미 갖고 있던 것

- **lab_stack_mimic 파이프라인** (`../lab_stack_mimic/`): 실험실 FR3 + 책상 + 3큐브 씬을 Isaac
  Lab Mimic 위에 재현한 기존 자산. 사람 teleop 시연을 annotation(서브태스크 경계 신호 부착) →
  MimicGen 생성 → 재생 판정하는 스크립트 일체가 있다. 제어는 계약과 다른 IK-rel(위치 PD,
  20 Hz)이다. FR3로 annotation을 마친 사람 시연 13개가 `../datasets/fwd_annotated.hdf5`에 있다.
- **서버**: 원래 UWLab 체크아웃과 사람 teleop 원본(29개), 실험실 책상 USD가 있던 arpa 서버는
  더 이상 없다. 현재 쓰는 aidas 서버(L40S GPU)에는 Isaac이 Docker 이미지
  (`nvcr.io/nvidia/isaac-lab:3.0.0-beta2-post1`, isaaclab_mimic 포함)로만 있고 UWLab은 없다.
  공개 GitHub UWLab을 받아 보니 omnireset 태스크는 있지만 FR3 설정 파일들은 비공개 추가분이라
  없었다. 책상 USD가 없는 문제는 씬 빌더의 대체 슬랩(fallback slab)으로 해결했다.

## 4. 만든 것 (`contract/` 파일별)

- `adapter.py` — 핸드오프의 controller_adapter.py를 바이트 그대로 복사(vendored). self-test 통과.
- `control_contract.yaml`, `cube_legacy_profile.py`, `dataset_schema.yaml` — 핸드오프 원본 사본.
- `uwlab_frozen/` — 동결 제어기(`RelCartesianOSCAction`) 소스 2파일을 핸드오프의
  frozen_payload에서 그대로 가져온 패키지. 의존성이 torch와 Isaac Lab 코어뿐이라 UWLab 없이
  Docker에서 돈다. arpa가 사라진 상황에서 계약 실행을 가능하게 만든 핵심 우회로다.
- `traj_tools.py` — 순수 파이썬 궤적 도구: 10 Hz 리샘플(위치 선형, 회전 slerp), "현재 실제
  pose 기준" 계약 action 유도, 왕복(pose→action→pose) 오차 계산, action 분포 통계.
- `schema_io.py` — 계약 HDF5 writer와 검증기.
- `convert_demo.py` — 오프라인 변환기. 시연의 관절 궤적을 씬에 기구학적으로 재생(물리 스텝
  없이 관절값만 써넣고 fr3_hand 바디 pose를 읽음)해 베이스 좌표 EE 궤적을 얻고, 10 Hz로
  리샘플해 계약 action과 스키마 HDF5, 검증 리포트를 낸다. `--time_stretch`로 시연을 느리게
  변환할 수 있다.
- `warmstart_replay.py` — 계약 실행기. 변환된 목표 궤적을 매 10 Hz 스텝마다 "현재 실제 EE
  기준 action"으로 온라인 계산해 동결 OSC로 실행하고, 추적오차·성공 여부·스키마 검증을
  리포트한다. `--bundle`로 캘리브레이션 번들의 관절 동역학(armature·마찰)을 적용할 수 있다.
  Isaac Lab 3.0-beta2의 세 가지 런타임 문제(첫 리셋 시 fabric 프록시 배열, warp 형식 자코비안,
  root pose 읽기)를 동결 파일 무수정 원칙 하에 실행기 쪽 패치로 해결했다. 고정 베이스 로봇이라
  root pose를 스폰 상수로 치환하는 것이 수학적으로 동일하다는 점을 이용했다.
- `rl_to_lab.py` — RL 시연을 lab_stack_mimic 생성 파이프라인의 입력 형식으로 바꾸는 변환기.
  기구학 재생으로 EE 궤적을 복원하고, 큐브 좌표를 실험실 씬으로 옮기고(테이블 높이 차이가
  +0.4 mm에 불과해 사실상 동일), 20 Hz IK-rel action을 합성하고, **annotation을 오프라인으로
  합성**한다(상태 술어로 grasp_1/stack_1/grasp_2 신호 생성 — 사람 시연의 annotation 스키마를
  그대로 따름). MimicGen 생성은 소스의 action 재생 품질이 아니라 이 datagen_info만 소비하므로,
  개루프 action 재생의 누적 오차 문제를 우회한다.
- `replay_lab.py` — 대체 슬랩 씬에서 lab 형식 시연을 재생하고 탑 성공을 판정하는 도구.
- `bundle_integration.py` — 캘리브레이션 번들 로더(contact 모듈은 기존 stage2 로더와 호환,
  dynamics는 관절별 nominal/범위 파싱).
- 실행 래퍼: `run_convert_aidas.sh`, `run_warmstart_aidas.sh`(Docker),
  `run_lab_generate_docker.sh`(기존 run_generate.sh의 Docker 이식; `LAB_SUBTASK_OFFSETS`
  환경변수로 서브태스크 경계 오프셋을 모든 비교군에 동일하게 조정 가능),
  `run_warmstart_arpa.sh`(arpa 소멸로 사실상 폐기).
- `../lab_stack_mimic/lab_mimic_cfg.py`에 책상 USD 부재 시 슬랩 fallback을 추가했다(우리 파일).

## 5. 검증된 결과 (숫자)

**변환 정확도.** 사람 시연 demo_0(349스텝 20 Hz)을 174 action(10 Hz)으로 변환했을 때
pose↔action 왕복 오차는 최대 7e-18 m / 4e-17 rad(부동소수 정밀도)이고 스키마 검증을 통과했다.
독립 검증으로, 복원한 EE 궤적을 기록 당시 월드 좌표로 되돌려 시연에 기록된 obs와 대조하면
평균 1.0 cm(최대 2.2 cm)에서 일치한다. 이 1 cm는 기록 obs가 손이 아니라 TCP(공구 중심점)
좌표라는 사실, 기록 쿼터니언이 wxyz(yaw 180° 베이스)라는 사실을 확정하고서야 얻은 숫자다.
RL 시연 변환도 같은 방식으로 통과했다(60스텝→59 action, 오차 0).

**계약 실행(warm-start).** 동결 OSC로 사람 시연 1개(2배 감속판, 348스텝)를 closed-loop 완주
시켰고 실행 산출물이 계약 스키마를 통과했다. 다만 태스크 성공은 실패다: 추적오차 평균
4.8 cm가 남는데, 분해해 보면 속도 비례 지연(계약 제어기 자체가 감쇠비 3의 과감쇠 특성으로
시정수 약 0.4초)과 베이스 -x 방향 약 4 cm의 정적 편차로 나뉜다. 관절 동역학(번들 armature
0.1, 마찰 0.25/0.5)을 적용해도 편차가 그대로였으므로(4.76→4.97 cm), 남은 원인은 로봇 자산
차이로 좁혀진다: 우리는 NVIDIA 공식 fr3.usd를 쓰는데(링크 하나의 관성이 무효라는 PhysX 경고가
남), RL팀은 이를 감싼 자체 래퍼 `fr3_research3.usda`를 쓰며 이 파일은 핸드오프에 없다.
핸드오프의 자체 감사도 "URDF/USD는 공칭"을 외부 한계로 명시하므로, 이 파일을 받아 재실행하는
것이 종결 조건이다. 계약 문서의 지침("이상치는 숨기지 말고 보고하라")에 따라 이 상태로 보고한다.

**생성 수율 비교(진행 중).** 사람 시연 13개를 소스로 한 생성은 Docker 체인에서 정상 동작했다:
10 성공 / 62 시도 = 수율 16.1%(구 오프셋 프로토콜 기준). RL 시연 소스는 46/50이 annotation
합성을 통과했지만, 생성은 현재 0%로 미해결이다(아래 6절).

## 6. 밟은 지뢰들과 현재 미해결 문제

이 작업의 대부분은 세 코드베이스(robosuite MimicGen의 관례, Isaac Lab의 관례, RL팀 계약의
관례) 사이의 좌표·시간·스키마 규약을 맞추는 일이었다. 실제로 밟은 것들:

1. 기록 obs의 EE는 월드 TCP, 계약의 EE는 베이스 좌표 fr3_hand 바디 — 둘을 혼동하면 정확히
   10.34 cm(hand→TCP 오프셋)씩 어긋난다. 이 오프셋의 부호조차 계산 경로에 따라 달라서, 두
   곳에서 각각 실데이터로 검증해 반대 부호를 확정했다.
2. Isaac Lab 3.0-beta2에서 첫 리셋 중 관절·바디 데이터가 fabric/warp 프록시로 나오는 문제 —
   동결 제어기의 리셋 latch를 미루고(의미 보존 확인), 고정 베이스의 root pose를 상수로 치환.
3. 개루프로 합성한 IK-rel action 재생은 추적오차가 누적돼 3/3 실패 — 생성은 datagen_info만
   쓰므로 annotation을 오프라인 합성하는 쪽으로 설계 변경.
4. RL 시연은 전환이 빨라(잡기→쌓기 0.5초) MimicGen의 서브태스크 경계 검사에 걸림 — 시연을
   2배 감속하는 방법은 생성 궤적도 2배 길어져 에피소드 상한에 걸려 전멸(0/130)했고, 대신
   경계 오프셋을 (10,20)→(0,5)로 줄이는 환경변수 노브를 만들어 사람/RL 양쪽에 동일 적용했다.
   이때 마지막 서브태스크의 오프셋은 (0,0)이어야 한다는 규약도 밟았다.
5. 합성 annotation의 eef_pose를 fr3_hand 바디로 넣었더니 생성이 0/2717 — 사람 시연의
   annotation을 확인해 보니 datagen_info의 eef_pose도 TCP 좌표였다(1번의 재발). TCP로
   고쳤는데도 0%가 계속돼, 사람 시연의 원본은 그대로 두고 datagen_info만 내 합성 코드로
   재작성하는 대조 실험(`identity_test.py`)을 만들었다. 이 실험이 남은 결함 둘을 더 찾아냈다:
   쌓임 판정의 z-갭 상한(0.052 m)이 실측 안착 갭(0.050~0.053 m)을 경계에서 탈락시키는 문제
   (창을 0.035~0.065로 완화), 그리고 사람 시연의 obs `gripper_pos`가 두 손가락을
   (+0.04, −0.04) 부호 대칭으로 기록해 평균 기반 열림/닫힘 술어가 항상 0이 되는 문제(절대값
   평균으로 통일). 수정 후 합성 신호가 원본 annotation과 같은 시점(stack_1 121~136 vs 원본
   123)에 발화함을 확인했고, 이 상태의 identity 생성 실험이 진행 중이다. RL 소스 생성도 이
   수정들이 반영된 재실행으로 판정한다.

## 7. 재현 방법

모든 실행은 aidas 서버의 Isaac Lab Docker에서 한다. 예:

```bash
cd ~/mimicgen_jihoonkwon/mimic_the_mimicgen/contract
# 시연 -> 계약 형식 변환 + 리포트
./run_convert_aidas.sh --dataset /repo/datasets/fwd_annotated.hdf5 \
    --output /out/human_demo0_contract.hdf5 --count 1 \
    --reference /rl_demos/fr3_three_cube_fullstack_success_50.hdf5
# 계약 제어기 closed-loop 재실행 (번들 동역학 적용)
./run_warmstart_aidas.sh --device cuda:0 --bundle /bundle \
    --contract /out/human_demo0_contract_s2.hdf5 \
    --source /repo/datasets/fwd_annotated.hdf5 --demo demo_0 \
    --output /out/human_demo0_executed.hdf5
# RL 시연 -> 생성 소스 변환(annotation 합성 포함)
./run_convert_aidas.sh  # 대신 rl_to_lab.py를 같은 docker 패턴으로 실행
# 사람 vs RL 동일 조건 생성
LAB_SUBTASK_OFFSETS=0,5 ./run_lab_generate_docker.sh fwd <소스.hdf5> <출력.hdf5> cpu 10 4
```

산출물은 서버 `/home/ubuntu/contract_out/`과 로컬 `robot_data/contract_outputs/`에 있고,
각 단계가 `*.report.json`을 남긴다.

## 8. 남은 일

1. RL 소스 생성 0% 원인 규명(실패 attempt 부검 + 사람 시연 identity-변환 대조 실험).
2. RL팀에 `fr3_research3.usda` 요청 → warm-start 태스크 성공 재검증으로 계약 정합 종결.
3. 수율 비교를 의미 있는 규모(성공 50개 또는 고정 시도수)로 확장해 다음 주 미팅 보고.
4. 캘리브레이션 번들의 camera 모듈을 렌더 파이프라인(`../render/`)에 연결해 물리+시각
   랜덤화가 걸린 RGB-Action 증강 데이터 생산(번들의 deterministic→ensemble→robust 프로파일
   순서를 따름).
