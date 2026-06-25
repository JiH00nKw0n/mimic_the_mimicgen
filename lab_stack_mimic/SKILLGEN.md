# SkillGen v2 — 핵심 수정/확장 문서

MimicGen 파이프라인(v1, vanilla)을 **SkillGen**으로 확장해 FR3 3-큐브 스태킹 합성 데이터셋
v2(2000 demos)를 생성하면서 수정·확장한 핵심을 정리한다. "무엇을 왜 어떻게 바꿨는지"가 목적.

> 출처: NVIDIA Isaac Lab SkillGen
> (https://isaac-sim.github.io/IsaacLab/main/source/overview/imitation-learning/skillgen.html)

---

## 0. SkillGen이 MimicGen과 다른 점 (한 줄)

- **MimicGen**: 소스 데모의 object-centric 구간을 새 물체 포즈로 **변환(transform)** 한 뒤,
  구간 사이를 **직선 보간(linear interpolation)** 으로 잇는다. 자유공간 이동도 보간이라
  장애물·큰 분포에서 충돌/실패가 난다.
- **SkillGen**: 접촉이 있는 **스킬 구간**은 MimicGen처럼 변환·재생(replay)하되, 구간 사이
  **자유공간 전이(transition)** 는 **cuRobo GPU 모션플래너로 충돌 없이 계획**한다.
  → 더 큰 spawn 분포/장애물에서도 전이가 견고해지고, DGR(생성 성공률)이 올라가는 게 핵심 동기.

우리 v1 한계 분석(METHODOLOGY §10, research/)과 직접 연결된다: MimicGen의 보간 전이가
분포가 커질수록 깨지는 문제를, SkillGen은 전이를 "계획"으로 바꿔 완화한다.

---

## 1. 환경 격리 — Docker + 핀된 NVIDIA cuRobo (가장 큰 작업)

SkillGen은 cuRobo 모션플래너가 필수다. 그런데:

- 동료(jake) `env_uwlab`에 설치된 cuRobo는 **FLAT-API 포크**
  (`curobo.motion_planner.MotionPlanner`)라서, Isaac Lab 번들 SkillGen 플래너가 기대하는
  **NESTED API**(`curobo.wrap.reacher.motion_gen.MotionGen`,
  `curobo.types.base.TensorDeviceType`, `curobo.cuda_robot_model`)와 호환되지 않는다.
- 공유 환경을 건드리지 않기로 했으므로(제약), **Docker 컨테이너에 정식 NVIDIA cuRobo를
  핀 커밋으로 별도 빌드**했다.

| 항목 | 값 |
|---|---|
| 컨테이너 | `isaac-lab-base` → 스냅샷 `isaac-lab-skillgen:latest` |
| cuRobo 핀 | commit `ebb71702f3f70e767f40fd8e050674af0288abe8` (Isaac Lab 테스트 버전) |
| 빌드 결과 | `nvidia-curobo 0.7.7.post1.dev5`, nested API |
| 빌드 환경 | cuda-toolkit-12-8(apt), `TORCH_CUDA_ARCH_LIST=8.9`(L40S sm_89), torch 2.7.0+cu128 |
| python | 컨테이너는 `/workspace/isaaclab/_isaac_sim/python.sh` (`python` 아님) |

이로써 jake env / cgcg 컨테이너를 전혀 건드리지 않고 nested-API cuRobo를 확보했다.

---

## 2. FR3 cuRobo 로봇 설정 — `fr3_curobo.yml` (두 번째로 큰 작업)

cuRobo는 **Franka Panda 설정만 기본 제공**한다. 우리 로봇은 FR3.
다행히 FR3 ≈ Panda: 링크/조인트 네이밍이 동일 패턴(`panda_*` ↔ `fr3_*`), 운동학 거의 동일.

핵심 결정 — **설치된 cuRobo가 들고 있는 그 버전의 `franka.yml`을 베이스로 삼는다**:

- 처음에 jake의 `franka.yml`을 panda→fr3 치환해 썼더니
  `format_version`, `grasp_contact_link_names` 등 **스키마 불일치**로
  `MotionGenConfig.load_from_robot_config`가 거부.
- 원인: 그 franka.yml은 다른 cuRobo 버전 스키마. → **컨테이너에 설치된 cuRobo 0.7.7이
  번들한 `.../curobo/content/configs/robot/franka.yml`**(버전 일치)을 베이스로 사용.

`make_fr3_curobo.py`가 하는 일:
1. 설치된 cuRobo의 `franka.yml` + 그 `spheres/franka.yml`(충돌 스피어) 로드
2. 전체를 재귀적으로 `panda_` → `fr3_` 치환 (FR3 URDF의 링크/조인트명과 일치)
3. USD 운동학 필드 제거, `urdf_path`를 **메시 없는 FR3 URDF**로 교체
   (충돌은 스피어로 처리 → 메시 불필요. FR3 URDF가 참조하던 `./meshes/fr3/`가 없어서
   각 메시를 작은 박스로 치환한 `fr3_nomesh.urdf` 사용)
4. (panda→fr3로 rename된) 충돌 스피어를 **inline**으로 박아 파일 의존성 제거

**검증**(`test_fr3_curobo.py`, Isaac 없이 단독 실행):

```
[test] OK: MotionGen built + warmed up for FR3
[test] active planning joints (7): ['fr3_joint1'...'fr3_joint7']
[test] FK home ee pos = [[0.1106, ~0, 0.5907]]
[test] plan_single success=True  steps=torch.Size([31, 7])
```

→ FR3 운동학 파싱 + 충돌 스피어 부착 + 모션플랜 성공까지 end-to-end 확인.
이게 SkillGen 확장의 가장 큰 리스크였고, 통과했다.

---

## 3. warp 1.14 호환 shim — `warp_torch_shim.py`

cuRobo 0.7.7은 충돌 체커 생성 시 `wp.torch.device_from_torch(...)`를 호출한다.
그런데 컨테이너의 **warp 1.14**는 torch interop을 `warp._src.torch`로 옮기고 공개
`warp.torch` 네임스페이스를 없앴다 → `AttributeError: module 'warp' has no attribute 'torch'`.

해결: cuRobo 플래너가 빌드되기 전에 **`warp.torch`를 `warp._src.torch`로 alias** 등록.
공유 cuRobo/warp 설치를 수정하지 않고 프로세스 내 별칭만 건다. `skillgen_register.py`가
cuRobo import 전에 이 모듈을 import해서 적용한다.

---

## 4. 환경 확장 — `lab_skillgen_env.py`

SkillGen은 env에서 MimicGen보다 **두 가지를 더** 요구한다.

### 4-1. 모든 N개 subtask 종료신호 (MimicGen은 N-1개)

`get_subtask_term_signals`가 마지막 stack_2까지 4개 모두 반환:
`grasp_1, stack_1, grasp_2, stack_2`. MimicGen은 마지막 경계를 데모 끝으로 보고 생략하지만,
SkillGen은 마지막 스킬을 분리하려면 N개 모두 필요.

### 4-2. subtask START 신호 — `get_subtask_start_signals` (auto annotation의 핵심)

각 subtask를 **[cuRobo가 계획하는 자유공간 전이] + [재생되는 접촉 스킬]** 로 쪼개는 신호.
**어떤 stock Isaac env도 이걸 구현하지 않아서**, 공식 SkillGen 경로는 **키보드 수동
어노테이션**이다. 우리는 이걸 구현해 **`--auto` 자동 어노테이션**을 가능케 했다.

정의(상태 없는 깔끔한 0→1 edge):

```
start_i = (subtask i-1이 종료됨  OR  i == 0)  AND  (EE가 object_ref_i 에 APPROACH_DIST 이내)
```

- grasped/stacked "done" 조건은 유지(잡은 큐브는 놓을 때까지 잡힘, 쌓인 건 쌓인 채)되므로
  "현재의 직전 종료신호" 값만 보면 충분 — 히스토리/상태 불필요.
- 4개 object_ref는 정규 forward 스택 순서: `cube_2(grasp) → cube_1(밑) → cube_3(grasp) →
  cube_2(밑)`. 손이 home→cube_2→cube_1→cube_3→cube_2로 이동하며 edge가 순서대로 발화,
  각 직전 자유공간 hop이 cuRobo 전이가 된다.
- `APPROACH_DIST`(기본 0.15m)는 `LAB_SKILL_APPROACH_DIST`로 조정.

> base-frame IK 보정은 v1(`LabFR3CubeStackIKRelMimicEnv`)에서 상속 — SkillGen env는 그
> Mimic env를 상속하므로 IK 수정·임계값 보정·home 텔레포트가 그대로 적용된다.

---

## 5. 설정 확장 — `lab_skillgen_cfg.py`

베이스를 **MimicGen cfg가 아니라 `FrankaCubeStackIKRelSkillgenEnvCfg`**(공식 SkillGen cfg)로
잡는다. 이게 SkillGen에 필요한 두 가지를 들고 오기 때문:
- `subtask_term_signal="stack_2"` 인 4번째 subtask config (마지막 스킬 끝)
- `stack_2` term을 포함하는 `subtask_terms` 관측 그룹

그 위에 **v1과 동일한 lab override**(`_apply_lab_overrides` + `_apply_threshold_fixes`)를
적용 → FR3 로봇, lab 책상, FR3 finger 조인트 기반 grasp/stack 체크, home 텔레포트 리셋,
소스에 맞춘 cube spawn 범위. 즉 씬/로봇은 vanilla 런과 동일하고 **데이터 생성 방식(전이를
계획)만 다르다.** 공식 SkillGen subtask 순서가 이미 우리 정규 forward 스택이라 reverse
변형 불필요(시드를 forward로 canonicalize 후 어노테이트 — vanilla와 동일).

---

## 6. 등록 + cuRobo 라우팅 — `skillgen_register.py`

공유 Isaac Lab/cuRobo 소스를 건드리지 않고:
1. `Isaac-Stack-Cube-LabFR3-Skillgen-IK-Rel-v0` 태스크를 gym에 등록.
2. `CuroboPlannerCfg.from_task_name`을 **몽키패치**해서 태스크명에 `labfr3`가 들어가면
   FR3 플래너 설정(`fr3_stack_cube_config`)을 반환. 공식 `franka_stack_cube_config`를
   미러링하되 FR3 로봇/그리퍼 조인트(`fr3_finger_joint1/2`)/핸드 링크로 교체.
3. warp shim import(§3).

---

## 7. 실행 스크립트 (컨테이너용)

vanilla 런 스크립트는 jake venv를 쓰므로, SkillGen은 **컨테이너 `isaaclab.sh`** 를 쓰는
새 스크립트로 분리. 공식 스크립트를 `/tmp`로 복사 후 `import isaaclab_mimic.envs` 다음 줄에
`import skillgen_register`(+full런은 clean_success_hook/provenance_hooks) 한 줄 주입.

- `run_skillgen_annotate.sh <in> <out> [device]`
  → `--auto --annotate_subtask_start_signals` 로 term+start 신호 자동 기록.
  - 컨테이너엔 jake 호스트의 lab 테이블 USD가 없어서 `assets/table_scene.usdc`로 복사하고
    `LAB_TABLE_USD`를 그 경로로 설정.
  - **업스트림 버그 우회**: 공식 `annotate_demos.py`의 auto 모드 start-신호 체크가
    `torch.any()`를 raw 파이썬 리스트에 호출(term 루프엔 있는 텐서 변환을 start 루프에선
    빠뜨림). start-신호 auto 어노테이션이 거의 안 쓰여 안 잡힌 버그 → /tmp 복사본에 한 줄
    (`signal_flags = torch.tensor(...)`) sed 주입으로 우회(공유 소스 무수정).
- `run_skillgen_generate.sh <annotated> <out> [trials] [envs] [device]`
  → `--use_skillgen` 으로 cuRobo 전이 + 스킬 재생 생성.
  `LAB_SKILLGEN_FULL=1` 이면 clean-success 게이팅 + provenance 기록(2000 full 런용).

---

## 8. 생성 디버깅 요약 (어노테이트 후 → 실제로 쌓이기까지)

어노테이션(26/28)을 끝내고 생성을 돌리니 **데모가 하나도 안 쌓였다.** 거의 모든 버그가
**cuRobo의 "베이스=원점" 가정**에서 나왔다(정품 Franka는 원점, 우리 FR3는 책상 위
[0.72,0.138,0.722] yaw180 → 경계마다 ~0.79 m·180° 틀어짐). 자세한 문제→원인→수정→검출은
[SKILLGEN_CHANGELOG.md](SKILLGEN_CHANGELOG.md) K~Q. 한눈 요약:

| 증상 | 뿌리 | 수정 |
|---|---|---|
| cuRobo IK 전부 실패(2285) | 목표가 env 프레임 그대로 | 목표 env→base 변환 (K) |
| 팔이 표면 25cm 아래로 dip, 안 쌓임 | **계획 출력 포즈가 베이스 프레임** | 포즈 base→env 역변환 (L) ★ |
| 엄한 큐브 침범 + attach 0.79m 뜸 | 동적 큐브 동기화가 env 프레임 | 큐브 base 동기화 기본 ON (M) |
| 다 쌓았다 막판에 토플 | NVIDIA가 vanilla 정착 dwell을 (0,0)로 꺼둠 | dwell (8,15) 복원 (O) |
| 순간 쌓임도 성공 판정 | sticky/순간 + XY 4cm > 안정한계 2.35cm | clean_success_hook + XY 0.02 (P) |
| 2000개 ~40h | cuRobo graph가 CPU 병목(GPU 90% idle, 4코어) | FAST(graph off) + num_envs=16 (Q) |

## 8-1. 결과

| 항목 | 값 |
|---|---|
| 소스(시드) 데모 | 28개 random-spawn → canonicalize → `..._sg_fixed.hdf5` |
| 어노테이션 | **26 / 28** (term 4 + start 4, APPROACH_DIST 0.25 + grasp 임계 0.13) |
| placement 정밀도 (수정 후) | xy 오프셋 med **~0.9 cm** (소스 ~1.0, 수정 전 ~2.9 → 토플) |
| 클린 DGR (XY<2cm, 정착 최종) | 스모크 ~42–50%, 라이브 ne=16 ~25% (gross 실패가 잔여 병목) |
| full 2000 처리량 | ETA **~8–10 h** (FAST graph-off + ne=16, GPU 28.7GB) |
| 최종 런 설정 | `LAB_SKILLGEN_FAST=1 LAB_RUN_NE=16` + FULL(clean_success_hook+provenance) + guarantee=True |
| 산출물 | `datasets/stack2000.hdf5` (+ `.provenance.json`) → jake 폴더 전달 예정 |

---

## 9. 파일 목록 (이번 확장에서 추가)

| 파일 | 역할 |
|---|---|
| `lab_skillgen_env.py` | 4 term + auto start 신호 (IK 보정은 v1 상속) |
| `lab_skillgen_cfg.py` | 공식 SkillGen cfg + lab override |
| `skillgen_register.py` | 태스크 등록 + cuRobo from_task_name FR3 라우팅 + warp shim |
| `warp_torch_shim.py` | warp 1.14에서 `warp.torch` 재노출 |
| `make_fr3_curobo.py` | 설치된 franka.yml → FR3 cuRobo 설정 생성기 |
| `fr3_curobo.yml` | FR3 cuRobo 로봇 설정 (생성물) |
| `test_fr3_curobo.py` | cuRobo FR3 빌드+플랜 단독 검증 |
| `run_skillgen_annotate.sh` / `run_skillgen_generate.sh` | 컨테이너 실행 |
| `fix_start_signals.py` | start 신호를 term에서 결정적 유도(frac=0.6) → overlap 회피 (J) |
| `clean_success_hook.py` | 성공을 **정착된 최종 클린 스택**으로 게이팅, XY 0.02 (P) |
| `provenance_hooks.py` | 생성 데모의 소스 사용·성공 메타데이터 기록 (full 런) |

### `skillgen_register.py` 안의 핵심 패치 (생성 디버깅)
| 패치 | 역할 (env var) |
|---|---|
| `plan_motion` 래핑 | 목표 env→base: `inv(T_env_base) @ T_env` (K) |
| `get_planned_poses` 래핑 | 계획 포즈 base→env: `T_env_base @ T_base` (L) ★dip 수정 |
| `_sync_object_poses_base_frame` | 동적 큐브 base 동기화 (`LAB_SKILLGEN_OBS_BASEFRAME`, 기본 ON) (M) |
| `_isw_with_slab` | 작업대 슬랩 (`LAB_SKILLGEN_SURFACE_TOP`, 무해·플래그) (L) |
| cuRobo FAST 옵션 | `enable_graph=False`+seed6 (`LAB_SKILLGEN_FAST`) (Q) |

### `lab_skillgen_cfg.py` 안의 핵심 패치
| 패치 | 역할 (env var) |
|---|---|
| dwell 복원 | `subtask_term_offset_range=(8,15)` + interp 5 (`LAB_SKILLGEN_DWELL`) (O) |
| grasp 임계 | offset-0 손목 프레임 보정 0.13 (`LAB_SKILLGEN_GRASP_DIFF`) (H) |
