# SkillGen 확장 — 코드 작업 내역 (문제 → 원인 → 수정 → 맞춰야 한 것 → 검출 테스트)

MimicGen 파이프라인을 SkillGen으로 올려 FR3 3-큐브 스태킹 합성 데이터셋(2000 demos)을
만들면서 실제로 손댄 코드 작업을, 시간순/영역순으로 정리한다. 각 항목은 "무엇이 문제였고 /
왜 그랬고 / 어떻게 고쳤고 / 무엇을 서로 맞춰야 했고 / 어떤 테스트·관찰로 잡아냈는지"로 적는다.
방법론적 차이는 [SKILLGEN_VS_MIMICGEN.md](SKILLGEN_VS_MIMICGEN.md) 참고.

핵심 제약: 동료(jake)의 공유 Isaac Lab / cuRobo 소스는 건드리지 않는다. 모든 확장은 별도
파일 + 런타임 주입(/tmp 복사 후 sed) + 클래스 몽키패치로 처리했다.

작업은 크게 두 단계다.

- **1단계 (A~J): 환경·어노테이션** — cuRobo를 FR3에 올리고, 시드 데모를 SkillGen이 먹을 수
  있게 어노테이트하기까지. 여기까진 "셋업"이다.
- **2단계 (K~Q): 생성 디버깅** — 막상 생성을 돌리니 데모가 **하나도 안 쌓였다.** 이 단계의
  거의 모든 버그는 **하나의 뿌리**에서 나온다: *cuRobo는 로봇 베이스가 자기 프레임의 원점에
  있다고 가정한다.* 정품 Franka는 env 원점에 서 있어서 이게 안 보이지만, 우리 lab FR3는
  책상 위 월드좌표 **[0.72, 0.138, 0.722]에 yaw=180°로 볼트 고정**돼 있다. 그래서 cuRobo와
  env 사이를 넘나드는 모든 지점이 **~0.79 m, 180°씩 어긋난다.** 이 한 가정이 서로 독립된 세
  군데(K, L, M)에서 각기 다른 증상으로 터졌다.

---

## A. cuRobo 환경 격리 (Docker)

- **문제**: SkillGen은 cuRobo 모션 플래너가 필수인데, jake env에 깔린 cuRobo는
  **flat-API 포크**(`curobo.motion_planner.MotionPlanner`)라서 Isaac Lab 번들 SkillGen
  플래너가 기대하는 **nested API**(`curobo.wrap.reacher.motion_gen.MotionGen`)와 호환 안 됨.
- **원인**: jake의 cuRobo는 다른 계열의 포크. 공유 환경이라 갈아엎을 수도 없음.
- **수정**: Docker 컨테이너에 **정식 NVIDIA cuRobo 핀 커밋**
  (`ebb71702f3f70e767f40fd8e050674af0288abe8`, Isaac Lab 테스트 버전,
  `nvidia-curobo 0.7.7`)을 별도 빌드. cuda-toolkit-12-8 + `TORCH_CUDA_ARCH_LIST=8.9`
  (L40S sm_89). jake env / cgcg 컨테이너 무수정.
- **맞춰야 한 것**: 컨테이너 python은 `python`이 아니라
  `/workspace/isaaclab/_isaac_sim/python.sh`.
- **검출**: import 스모크 — `curobo.wrap.reacher.motion_gen.MotionGen`이 import되면 통과
  (`CUROBO_NESTED_API_OK`).

## B. FR3용 cuRobo 로봇 설정 만들기

- **문제**: cuRobo는 **Franka Panda 설정만** 제공. jake의 franka.yml을 panda→fr3 치환해
  썼더니 `MotionGenConfig.load_from_robot_config`가 `format_version`,
  `grasp_contact_link_names` 등 **모르는 키로 거부**.
- **원인**: 그 franka.yml은 **다른 cuRobo 버전의 스키마**.
- **수정**: **설치된 cuRobo 0.7.7이 번들한 자기 franka.yml**(버전 일치)을 베이스로 삼아
  `make_fr3_curobo.py`로 FR3 설정 생성 — 전체를 재귀적으로 `panda_`→`fr3_` 치환,
  USD 운동학 필드 제거, FR3 URDF로 교체, FR3 충돌 스피어를 inline으로 박음.
- **맞춰야 한 것**: cuRobo 버전 ↔ robot config 스키마는 반드시 같은 버전에서 가져올 것.
- **검출**: `test_fr3_curobo.py` — `load_from_robot_config` + warmup 통과
  (base_link=fr3_link0, 12 충돌 링크).

## C. FR3 URDF 메시 참조 누락

- **문제**: FR3 URDF가 `./meshes/fr3/...`를 참조하는데 그 메시 파일이 없음.
- **수정**: 메시를 작은 박스로 치환한 `fr3_nomesh.urdf` 사용. cuRobo는 충돌을 **스피어**로
  처리하므로 시각 메시가 필요 없음.
- **검출**: cuRobo 운동학 파싱 성공(9 조인트, 12 충돌 링크).

## D. warp 1.14의 `warp.torch` 사라짐

- **문제**: cuRobo 0.7.7이 충돌 체커 생성 시 `wp.torch.device_from_torch(...)` 호출 →
  `AttributeError: module 'warp' has no attribute 'torch'`.
- **원인**: 컨테이너의 **warp 1.14**가 torch interop을 `warp._src.torch`로 옮기고 공개
  `warp.torch` 네임스페이스를 제거. cuRobo는 옛 위치를 기대.
- **수정**: `warp_torch_shim.py` — 플래너 빌드 전에 `warp.torch`를 `warp._src.torch`로
  alias 등록 (공유 설치 무수정, 프로세스 내 별칭만).
- **맞춰야 한 것**: warp 버전 ↔ cuRobo가 기대하는 torch interop API.
- **검출**: `test_fr3_curobo.py`가 이 에러로 죽었다가, shim 후 MotionGen 빌드 성공.

## E. cuRobo DOF / 손가락 잠금(lock_joints)

- **문제**: 테스트가 9-조인트 start state를 넘기자 `shape '[1,7]' invalid for size 9`.
- **원인**: cuRobo가 손가락 2개를 `lock_joints`로 잠가 **팔 7 DOF만** 계획.
- **수정**: start state를 `mg.kinematics.joint_names`(활성 7개)로 구성.
- **검출**: `test_fr3_curobo.py` — 최종 `plan_single success=True`,
  `steps=torch.Size([31, 7])` (31스텝·7DOF 충돌없는 궤적). **cuRobo+FR3 통합 end-to-end 확인.**

## F. 컨테이너에 lab 테이블 USD 없음

- **문제**: env 생성 시 `FileNotFoundError: .../table_scene.usdc` (jake 호스트 경로가
  컨테이너에 없음).
- **수정**: `table_scene.usdc`를 컨테이너 `assets/`로 `docker cp`, run 스크립트에서
  `LAB_TABLE_USD`를 그 경로로 설정. (USD가 참조하는 `./table.usdc`는 호스트에도 없지만
  vanilla 런에서도 없이 성공 → 비치명적 참조라 무시.)
- **맞춰야 한 것**: 컨테이너 ↔ 호스트 에셋 경로. 에셋은 컨테이너로 복사하고 env 변수로 지정.
- **검출**: annotate가 씬 스폰에서 죽었다가, 복사+변수 설정 후 씬 생성 통과.

## G. 공식 annotate 스크립트의 start-신호 버그

- **문제**: auto 모드에서 start 신호를 검사하는 부분이 `torch.any()`를 **raw 파이썬
  리스트**에 호출 → `TypeError`. (term 신호 루프엔 텐서 변환이 있는데 start 루프엔 빠짐.)
- **원인**: start-신호 auto 어노테이션이 거의 안 쓰여 안 잡힌 업스트림 버그.
- **수정**: /tmp 복사본에 한 줄(`signal_flags = torch.tensor(signal_flags, device=...)`)을
  sed로 주입(들여쓰기 16칸, start 루프 내부). 공유 소스 무수정.
- **검출**: annotate가 line 420에서 죽었다가, 패치 후 28개 데모 전부 처리.

## H. grasp_1/grasp_2 종료 신호가 전혀 안 잡힘 ★1단계 최대 문제

- **문제**: 모든 데모에서 grasp 종료 신호 미검출(반면 stack은 검출) → 어노테이트 0개.
- **원인**: **SkillGen 씬 cfg가 ee_frame의 `end_effector` 오프셋을 `[0,0,0]`으로 설정**한다.
  목적은 MimicGen의 eef 기준 프레임을 cuRobo의 tool 프레임(fr3_hand)과 일치시키는 것.
  부작용으로 ee_frame 기준점이 **손목**에 앉아 grasp 지점보다 ~0.10m 위. grasp 검출은
  `||cube − ee_frame|| < 0.06~0.08`을 보는데 손목-큐브 거리가 ~0.10m라 **절대 안 참**.
  (stack 검출은 큐브-큐브 거리만 보므로 멀쩡 → grasp만 실패하는 비대칭이 단서였다.)
- **수정**: SkillGen 설계(cuRobo 일치)를 깨지 않으려 오프셋은 0으로 두고, skillgen cfg에서
  **grasp 근접 임계만 0.13으로** 넓힘(`LAB_SKILLGEN_GRASP_DIFF`). grasp은 여전히
  **그리퍼 닫힘으로 게이팅**되므로 거리를 넓혀도 오검출 없음.
- **맞춰야 한 것**: ee_frame 오프셋(SkillGen=0, 손목) ↔ grasp 거리 임계. 프레임을 손목으로
  옮기면 거리 기준도 그만큼 키워야 한다. (이 offset-0 결정은 뒤의 O 토플 문제까지 영향이 길다.)
- **검출**: mimic vs skillgen 씬 cfg를 비교하는 probe 스크립트로 offset-0 루프를 발견.
  수정 후 annotate가 데모를 export하기 시작.

## I. stack_1/stack_2 시작 신호가 드물게만 발화

- **문제**: grasp 고친 뒤 4/28. 대부분 "Did not detect start for stack_1/stack_2"로 탈락.
- **원인**: 내 시작 신호 정의 = `(직전 종료) AND (손목이 object_ref에 APPROACH_DIST 이내)`.
  offset-0 손목 프레임에서는 올려놓는 순간 손목이 아래 큐브로부터 ~0.15m라,
  `APPROACH_DIST=0.15`로는 터치다운 순간만 겨우 스쳐 대부분 놓침.
- **수정**: `APPROACH_DIST` 0.15 → **0.25** (손목-큐브 거리를 여유 있게 포함).
- **맞춰야 한 것**: ee_frame 오프셋 ↔ 시작 신호 근접 임계도 함께 키울 것 (H와 같은 뿌리).
- **검출**: annotate export 수가 **4 → 26 / 28** 로 급증 (vanilla MimicGen ~17/28보다 높음).

## J. 생성 단계: subtask 경계 겹침(overlap) assertion

- **문제**: 생성(`--use_skillgen`) 시작 직후 cuRobo 플래너는 FR3 config로 정상 초기화·웜업
  됐는데(라우팅·gripper 다 맞음), 데이터젠이
  `AssertionError: subtasks 0 and 1 are overlapped with the largest offsets`로 죽음.
- **원인**: SkillGen 데이터젠은 각 데모를 `term`으로 끝 경계, `start`로 스킬 시작 경계를
  잡고, 풀 전체에서 `term_i + offset_hi < start_{i+1}` 를 강제한다. 내 geometric start
  신호(다음 물체에 APPROACH_DIST 이내)는 **두 큐브가 가깝게 스폰된 데모**에서 직전 grasp
  순간에 이미 "다음 물체 근처"가 참이 되어 start가 term과 같은 시점에 발화 → 순서 위반.
- **수정**: start 신호를 geometry가 아니라 **term 신호에서 결정적으로 유도**하는 오프라인
  후처리 `fix_start_signals.py`. subtask i의 start를 (직전 term → 이번 term) 구간의 비율
  `frac=0.6` 지점에 두고, 항상 유효 구간 안으로 clamp. 결과적으로 **스킬 = 각 subtask의
  뒷부분(접촉: 최종 접근+grasp / 하강+release), 전이 = 앞부분(자유공간, cuRobo가 계획)**.
  모든 데모에서 비겹침이 구조적으로 보장됨(26/26 재작성 → `..._sg_fixed.hdf5`).
- **맞춰야 한 것**: start 경계는 거리 휴리스틱이 아니라 term 경계와의 순서로 정의해야
  풀 전체 불변식을 만족. (이 frac=0.6이 만든 term→다음 start 간격이 뒤의 O에서 dwell 한도를
  19프레임으로 묶는다 — 모든 게 연결돼 있다.)
- **검출**: `datagen_info_pool.py`의 overlap assertion 위치를 읽고 거리 임계로는 보장
  불가임을 확인 → 결정적 유도로 전환.

---

# 2단계 — 생성 디버깅 (데모가 안 쌓이는 문제)

어노테이션이 26/28로 끝나 생성을 돌렸더니, cuRobo는 계획을 하는데 **만들어진 데모가 큐브를
하나도 안 쌓았다.** 아래가 그 원인을 하나씩 벗겨낸 과정이다.

## K. cuRobo가 모든 전이에서 IK_FAIL — 목표를 베이스 프레임으로 안 줌 ★뿌리

- **문제**: 생성을 돌리자 cuRobo가 `MotionGenStatus.IK_FAIL`을 **2285회** 쏟아내고 전이
  계획이 거의 다 실패. (IK_FAIL = 그 손끝 목표 자세에 대한 역기구학 해가 없음.)
- **원인**: `CuroboPlanner.plan_motion`은 목표 포즈를 **그대로** cuRobo에 넘긴다(베이스 빼기
  없음). cuRobo는 로봇 베이스가 자기 프레임 **원점**에 있다고 가정하므로, env 프레임의 목표는
  우리 FR3 기준 **~0.79 m 떨어지고 180° 뒤집힌** 자리 → IK가 안 풀림. 정품 Franka는 원점에
  있어 우연히 맞던 코드.
- **수정**: `skillgen_register.py`에서 `plan_motion`을 래핑해 목표를 베이스 프레임으로 변환:
  `T_base = inv(T_env_base) @ T_env`. `T_env_base`는 그 환경의 FR3 루트 포즈
  (`root_pos_w − env_origin`, `root_quat_w`)로 매 호출 구성(`_base_T`).
- **곁가지 (start-state 충돌)**: 동시에 `INVALID_START_STATE_WORLD_COLLISION`도 떴다. cuRobo가
  씬을 스캔할 때 **풀 높이 책상 메시**(z≈0..0.74)를 장애물로 잡아 그 안에 박힌 베이스 스피어가
  충돌로 인식됨. → `world_ignore_substrings`에 `/Table`, `/WorkSurface` 추가로 두 표면을
  cuRobo 월드에서 제외. 단 빈 월드는 그래프 warmup이 **무한 정지**해서, get_world_config로
  무해한 바닥 큐보이드 하나를 줘 warmup을 통과시킴.
- **맞춰야 한 것**: cuRobo는 "베이스=원점" 프레임. **모든 목표 입력을 베이스 프레임으로.**
- **검출**: 생성 로그의 `IK_FAIL` 카운트 — 2285 → **0**, 전이가 96-waypoint 궤적으로 성공.

## L. 전이가 표면 25 cm 아래로 파고듦 — 계획 출력도 베이스 프레임이었음 ★진짜 원인

- **문제**: K로 IK_FAIL=0이 됐는데도 데모가 안 쌓임. 영상: 팔이 자기 베이스 쪽으로 접히며
  작업대 표면(z≈0.72) **25 cm 아래(z=0.47)** 로 파고들었다 돌아옴(데모 중간 U자 dip).
- **헛다리 한 번**: "표면 장애물이 없어서 옵티마이저가 표면 밑으로 경로를 짠다"고 보고
  cuRobo 월드에 **작업대 슬랩**을 넣음. 그런데 데모가 **바이트 단위로 동일** → 효과 0.
  DEBUG 덤프로 슬랩이 실제 계획 월드에 들어간 것까지 확인했는데도 dip 그대로 → **dip은 실제
  지오메트리를 통과한 게 아니라, 전이가 프레임이 어긋난 채 실행된 것**임을 역으로 증명.
- **원인**: `get_planned_poses()`가 계획 궤적의 EE 포즈를 `motion_gen.compute_kinematics()`,
  즉 **베이스 프레임 FK**로 뽑아 그대로 반환한다(docstring의 "world coordinates"는 정품
  Franka에서만 참). SkillGen은 이 포즈를 **env 프레임 waypoint 목표로 실행**하므로, FR3
  off-origin이라 전이가 0.79 m 어긋나 표면 아래로 간다. (스킬 구간은 env 프레임이라 정상 →
  스킬은 큐브로, 전이는 엉뚱한 데로 = U자.)
- **수정**: `get_planned_poses`를 래핑해 계획 포즈를 **base→env로 역변환**:
  `T_env = T_env_base @ T_base`. 즉 K(목표 입력 env→base)의 **정확한 역**. 짝을 맞춘 셈.
  (잠재적 footgun인 `get_next_waypoint_ee_pose`도 같은 변환으로 방어 래핑 — 현재 생성
  경로엔 안 쓰이지만 미래 소비자가 dip을 재유발하지 못하게.)
- **맞춰야 한 것**: cuRobo 경계는 **양방향**이다. 목표를 베이스로 넣었으면(K) 계획 포즈는
  env로 빼야(L) 한다.
- **검출**: 기록된 `eef_z` 최저값 **0.47 → 0.86**(표면 위), placement 정상화, **DGR 0% →
  50%**(첫 SkillGen 큐브 쌓기 성공).

## M. 엄한(목표 외) 큐브를 치고 다님 + attach가 0.79 m 떠다님 — 동적 장애물 프레임

- **문제**: 팔이 지금 다루는 큐브 말고 **다른 큐브를 자꾸 건드림.**
- **원인**: 같은 베이스-원점 가정의 세 번째 누수. `_initialize_static_world`는 장애물을
  `reference_prim_path=robot`(=베이스 프레임)로 스캔하는데, **동적 큐브 동기화**
  (`_sync_object_poses_with_isaaclab`)는 `root_pos_w − env_origin`(=env 프레임)으로 갱신한다.
  → cuRobo는 **유령 큐브**(env 좌표를 베이스 좌표로 오해한 위치)를 피하느라 진짜 큐브를
  관통. 같은 이유로 `attach_objects_to_robot`가 잡은 큐브의 충돌구를 **베이스 프레임 EE FK ↔
  env 프레임 큐브 포즈**로 맞춰 **그리퍼에서 ~0.79 m 떨어진 허공**에 붙인다.
- **수정**: 동적 큐브 동기화를 **베이스 프레임으로**(`_sync_object_poses_base_frame`,
  `inv(T_env_base) @ T_cube_env`) 바꾸고 **기본 ON**(`LAB_SKILLGEN_OBS_BASEFRAME`).
  world_model **과** collision-checker 둘 다 갱신(둘 중 하나만 빠지면 attach가 또 어긋남).
- **하지 말아야 할 것 (적대적 검증으로 기각)**: "구간마다 목표 큐브를 장애물에서 빼자"는
  안은 **유해**. `_plan_to_contact`에서 goal 단계가 `contact=True`와 짝지어져 손가락
  충돌구가 이미 비활성화되므로 목표 큐브가 goal을 막지 않는다 → 마스킹하면 접근 중 실제
  장애물을 지워 부딪힘이 **재발**한다. 그래서 "모든 큐브를 장애물로 두되, 잡은 큐브 한 개만
  attach 시 비활성"이 정답.
- **맞춰야 한 것**: 정적/동적 장애물 프레임을 **둘 다 베이스로** 통일.
- **검출**: 영상에서 큐브 침범 관찰 → ultracode 다중 에이전트 재검토로 동적 동기화 프레임
  불일치 + attach 누수를 코드로 확정.

## O. 스택이 형성됐다 막판에 무너짐 — vanilla settle/dwell 복원 ★가장 미묘

- **문제**: 프레임(K·L·M)을 다 고친 뒤에도 클린 DGR이 ~0. 궤적을 보면 스택이 **제대로
  형성됐다가**(클린 13~33프레임 유지) **막판에 cube_2·cube_3가 동시에 바닥으로 추락.**
  팔이 거의 안 움직이는데 무너짐 = 후퇴가 친 게 아니라 **불안정하게 놓여 스스로 토플.**
- **원인 (정량 격리)**: 생성된 placement는 받침 큐브 중심에서 **2.9~3.9 cm** 빗나간다. 큐브
  반쪽이 2.35 cm니 COM이 가장자리 밖 → 무조건 토플. **소스(사람) 데모는 ~1.0 cm**(안정).
  파지 순간 손목-큐브 오프셋은 소스와 비슷(정상) → 오차는 **놓기에서** 생긴다.
  진짜 원인: **NVIDIA SkillGen 베이스 cfg가 `subtask_term_offset_range`를 전부 `(0,0)`으로
  꺼놨다**(파일에 vanilla 값이 `# (10,20)`로 주석처리돼 남아 있다). 이 오프셋은 각 subtask의
  끝을 소스 데모에서 N프레임 더 끌어와 **"놓고 잠깐 누르고 있는" 정착 프레임**을 포함시키는데,
  (0,0)이면 stack 종료 신호가 뜨는 **즉시**(큐브가 아직 내려가는 중, 노이즈 미정착) release →
  불안정. vanilla MimicGen은 subtask 0~2에 `(10,20)`을 줘서 이 문제가 없다.
- **수정**: `lab_skillgen_cfg.py.__post_init__`에서 vanilla dwell 복원
  (`subtask_term_offset_range`, 마지막 subtask는 datagen이 (0,0)을 강제하므로 0~2만,
  `num_interpolation_steps=5`). 단 우리 시드는 vanilla보다 촘촘해서 **term→다음 start 최소
  간격이 19프레임**(J의 frac=0.6 산물) → (10,20)을 그대로 쓰면 다시 overlap assertion. 그래서
  간격 안에 드는 **(8,15)**로 설정(`LAB_SKILLGEN_DWELL_LO/HI`).
- **맞춰야 한 것**: NVIDIA가 SkillGen 기본값으로 **정착 dwell을 꺼놨다는 사실** + 시드 경계
  간격. dwell은 vanilla 값을 복원하되 시드의 term→start 간격 안으로 클램프.
- **검출**: 기록 궤적에서 placement xy 오프셋 정량화(**2.9 → 0.9 cm**, 소스 품질 회복),
  클린 DGR **0 → 50%**. 5가설(파지 타이밍 / 전이 잔차 / ee-offset / object-centric / dwell)을
  ultracode 워크플로로 병렬 검증 → dwell이 1차 원인으로 확정, 나머지 4개 기각.

## P. 성공 판정 강화 — "올라갔다 떨어져도 성공"을 reject (사용자 지적)

- **문제**: 큐브가 막판에 올라갔다 다시 떨어지는데도 데모가 **성공으로 export**됨.
- **원인**: env의 `cubes_stacked` 종료항 + MimicGen의 sticky success
  (`success = success or step_success`)가 **순간 상태**만 본다. 게다가 기하 허용
  `XY_THRESHOLD=0.04`(4 cm)가 **물리 안정 한계 2.35 cm보다 헐거워** 2.9 cm짜리 토플 직전
  스택도 "clean"으로 통과. (실제로 "성공" 2개가 `ever_clean=True`지만 최종 z가 셋 다 0.745 =
  바닥 = 토플 false-positive였다.)
- **수정**: `clean_success_hook.py`(`LAB_SKILLGEN_FULL=1`로 주입)가 **최종 상태**의 클린
  스택만 success로 인정 — export 에피소드와 generation_guarantee 카운터 **양쪽**을 게이팅.
  XY 허용을 **0.04 → 0.02**(반쪽 2.35 cm 아래)로 조임. 토플 데모는 최종이 평탄(z 0.745)이라
  z-order 검사에서 자동 reject.
- **맞춰야 한 것**: 성공의 정의 = **정착된** 클린 스택(순간 아님). 기하 허용은 물리 안정 한계
  아래로.
- **검출**: 최종 상태 분석 스크립트로 "성공" 데모의 최종 큐브 z·드리프트를 떠 토플을 확인.
  vanilla 때 이미 잡았던 함정으로, SkillGen 생성에도 같은 훅을 강제.

## Q. 처리량 — cuRobo가 CPU 병목, GPU는 90% 논다

- **문제**: 위까지 고치니 데모는 제대로 나오는데, 2000개에 **~40시간** 추정.
- **원인**: 라이브 계측 — `nvidia-smi` GPU util **11%**, 메모리 3.5/46 GB, CPU load 1.1.
  즉 **GPU가 거의 노는데** 병목은 **cuRobo의 graph search(자유공간 충돌없는 로드맵 탐색)가
  CPU에서 단일 스레드로** 도는 것 + 이 박스가 **4코어뿐**.
- **수정 (두 레버)**:
  1. **FAST 모드**(`LAB_SKILLGEN_FAST=1`): `enable_graph=False`, trajopt seed 12→6. 우리
     전이는 테이블 위 단순 자유공간 hop이라 무거운 graph 플래너가 과함 — trajopt만으로 풂.
  2. **num_envs 상향**: env 스텝핑은 GPU 병렬이라 환경을 늘리면 처리량이 오른다(ne=1→8에서
     trial 처리량 4.8배). DGR은 num_envs에 따라 변함(ne=8≈27%, **ne=16≈42~50%**) → 클린
     처리량 최적점은 **ne=16**(peak GPU 28.7 GB로 jake에 여유 남김, errs=0).
- **맞춰야 한 것**: 플래너 비용 ↔ 전이 난이도(자유공간 hop엔 graph 불필요), num_envs ↔ GPU
  메모리·DGR. 절대 멀티프로세스로 공유 GPU를 독점하지 말 것(jake 공유).
- **검출**: GPU util/메모리·CPU load 라이브 샘플링으로 CPU 병목 확정 → num_envs 스윕
  (1/8/16/24)으로 ne=16을 최적점으로. **2000개 ETA 40h → ~8h.**

---

## 맞춰야 했던 것 총정리 (alignment checklist)

SkillGen을 새 로봇·새 환경에 올릴 때 "서로 버전·좌표·이름·정착을 맞춰야" 작동하는 지점들:

1. **cuRobo 버전 ↔ 로봇 config 스키마** — 설치된 cuRobo의 franka.yml을 베이스로. (B)
2. **warp 버전 ↔ cuRobo의 `warp.torch` 기대** — shim으로 별칭. (D)
3. **로봇 조인트/링크 이름 ↔ cuRobo config + `gripper_joint_names`** — `fr3_*`로 통일. (B,E)
4. **ee_frame 오프셋(SkillGen=0, 손목) ↔ grasp/start 거리 임계** — 프레임 옮긴 만큼 임계도
   키운다. (H, I)
5. **cuRobo "베이스=원점" 프레임 ↔ env 프레임** — off-origin 로봇은 **세 곳 모두** 변환해야
   한다: 목표 입력 env→base(K), **계획 출력 base→env(L)**, 동적 장애물 env→base(M).
   ★이번 작업의 핵심. 한 곳만 빠져도 전이가 ~0.79 m 어긋난다.
6. **NVIDIA SkillGen 기본 cfg가 vanilla 정착 dwell을 (0,0)으로 꺼놨음** — 복원하되 시드의
   term→start 간격 안으로 클램프. (O)
7. **성공 판정 = 정착된 클린 스택** — sticky/순간 성공을 최종상태 훅으로 게이팅,
   기하 허용은 물리 안정 한계 아래로. (P)
8. **컨테이너 경로 ↔ 에셋(테이블 USD, URDF)** — 컨테이너로 복사 + env 변수. (F)
9. **플래너 비용 ↔ 전이 난이도, num_envs ↔ GPU/DGR** — 자유공간엔 graph off, ne=16. (Q)

---

## 검출에 쓴 테스트/관찰 요약

| 단계 | 테스트/관찰 | 무엇을 잡았나 |
|---|---|---|
| cuRobo 빌드 | import 스모크 | nested API 사용 가능 여부 (A) |
| FR3 config | `test_fr3_curobo.py` | 스키마 적합, 운동학, warp shim, DOF, **plan 성공** (B~E) |
| 환경 생성 | annotate 부팅 로그 | 테이블 USD 누락 (F) |
| 어노테이트 | annotate auto 로그 + 씬 cfg probe | start TypeError, grasp/start 미검출, **offset-0** (G~I) |
| 생성 부팅 | datagen assertion | subtask 경계 overlap (J) |
| 생성 계획 | 로그 `IK_FAIL` 카운트 | 목표 프레임 미변환 2285→0 (K) |
| 생성 실행 | 기록 궤적 `eef_z`, 영상 프레임 | 표면 아래 dip = 계획출력 프레임 (L) |
| 생성 실행 | 영상 + ultracode 재검토 | 큐브 침범 = 동적 장애물 프레임 + attach (M) |
| 토플 | placement xy 정량화 + 5가설 워크플로 | dwell 누락 2.9→0.9 cm (O) |
| 성공 판정 | 최종 상태 분석 | 순간성공 false-positive (P) |
| 처리량 | nvidia-smi util/load 라이브 | CPU 병목, ne 스윕 (Q) |

---

## 결과 (런 후 확정)

| 항목 | 값 |
|---|---|
| 소스(시드) 데모 | 28 (random-spawn → canonicalize → `..._sg_fixed.hdf5`) |
| 성공 어노테이션 | **26 / 28** (APPROACH_DIST 0.25 + grasp 임계 0.13) |
| placement 정밀도 (수정 후) | xy 오프셋 med **~0.9 cm** (소스 ~1.0, 수정 전 ~2.9) |
| 클린 DGR (XY<2cm, 정착) | ~**42–50%** (ne=16, FAST, dwell 복원) |
| 처리량 | 2000개 ETA **40h → ~8h** (FAST graph-off + ne=16) |
| 최종 설정 | `LAB_SKILLGEN_FAST=1 LAB_RUN_NE=16`, FULL(clean_success_hook + provenance), guarantee=True |
| 산출물 | `datasets/stack2000.hdf5` (+ `.provenance.json`) → jake 폴더 전달 예정 |
