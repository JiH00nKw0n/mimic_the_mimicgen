# Task × Variant 매트릭스 — E1/E2 대상 태스크 선정과 커스텀 변형 정의

MimicGen 공개 자산(소스 데모·annotation·D변형)을 코드 레벨에서 전수 조사한 결과와, 그에 따른 우리 변형 사다리(variant ladder) 정의. 수치는 전부 `mimicgen v1.0.1` 코드의 `_get_initial_placement_bounds()`에서 직접 추출했다 (파일·라인은 각 절에 표기).

**요지**: 12개 robosuite 태스크 전부 HF에 10개 source demo가 있고 config 템플릿이 있다. 문제는 D변형의 기하가 태스크마다 다르다는 것 — Square·ThreePieceAssembly는 순수 **확장(expansion)** 사다리인 반면, **Threading·Coffee의 D2는 물체를 반대편으로 옮기는 재배치(relocation/mirror)** 라서 transform 축이 "거리"와 "영역 이동"으로 오염돼 있다. E1은 이 두 태스크에 확장형 D2E를 새로 정의하고, D0/D1만 있는 태스크에는 D2E를 추가해 태스크 풀을 넓힌다.

---

## 1. 전체 자산 매트릭스

출처: HF `amandlek/mimicgen_datasets` (API로 파일 목록 확인), `mimicgen/envs/robosuite/*.py`, `mimicgen/exps/templates/robosuite/*.json`. DGR은 MimicGen 논문 값(A-datagen_results). "기하"는 D변형 간 bounds의 집합 관계를 수치로 판정한 것.

| task | 공개 변형 | 기하 (수치 판정) | 논문 DGR (%) | source demo | 우리 계획 |
|---|---|---|---|---|---|
| **Square** | D0/D1/D2 | 확장 (D0⊂D1·D0⊂D2 성립; D2 vs D1은 nut-y 5mm·peg-x 상한 50mm 절단, 하한은 −0.1→−0.25 확장) | 73.7 / 48.9 / 31.8 | square.hdf5 16MB | **재사용** (Phase-0 완료) |
| **Threading** | D0/D1/D2 | D1=확장, **D2=재배치** (needle·tripod y축 미러) | 51.0 / 39.2 / 21.6 | threading.hdf5 19MB | D0/D1 재사용 + **D2E 신규** |
| **Coffee** | D0/D1/D2 | D1=혼합(machine이 D0 지점 이탈), **D2=재배치** (양 물체 y 미러) | 78.2 / 63.5 / 27.7 | coffee.hdf5 20MB | D0 재사용 + **D1E·D2E 신규** |
| **ThreePieceAssembly** | D0/D1/D2 | 순수 확장 (xy 동일 0.44×0.44, D1이 base 해방, D2가 회전 해방) | 35.6 / 35.5 / 31.3 | three_piece_assembly.hdf5 34MB | **재사용** (Phase-0 완료) |
| **Stack** | D0/D1 | 순수 확장 (±0.08 → ±0.20) | 94.3 / 90.0 | stack.hdf5 12MB | D0/D1 재사용 + **D2E 신규** |
| **StackThree** | D0/D1 | 순수 확장 (±0.10 → ±0.20) | 71.3 / 68.9 | stack_three.hdf5 25MB | D0/D1 재사용 + **D2E 신규** |
| **MugCleanup** | D0/D1 (+O1/O2) | 혼합 — mug y가 shift: D1 (-0.3,-0.15)는 D0 (-0.25,-0.1)의 (-0.15,-0.1) 구간을 잃음 | 29.5 / 17.0 | mug_cleanup.hdf5 34MB | Tier-2: **D1E**(y 합집합) + D2E |
| **HammerCleanup** | D0/D1 | 확장 (단, hammer 회전축이 D0 z축 ±0.1rad → D1 y축 2π로 바뀜 — 회전 거리 비교 불가) | 47.6 / 20.4 | hammer_cleanup.hdf5 33MB | Tier-2 (robosuite_task_zoo 필요) |
| **CoffeePreparation** | D0/D1 | 확장 (mug 확대 + machine 고정→소영역) | 53.2 / 36.1 | coffee_preparation.hdf5 75MB | Tier-2 (5 subtask, horizon 800) |
| **Kitchen** | D0/D1 | 혼합 — bread x가 relocation (D0 [0.05,0.08] vs D1 (-0.2,0.0) 서로소) | 100.0 / 42.7 | kitchen.hdf5 76MB | 보류 (7 subtask + task_zoo + relocation) |
| **NutAssembly** | D0만 | — | 50.0 | nut_assembly.hdf5 36MB | 보류 (변형 축 없음) |
| **PickPlace** | D0만 | — (BinsArena, 테이블 태스크 아님) | 32.7 | pick_place.hdf5 92MB | 제외 (bounds 메커니즘 자체가 다름) |

참고: Factory 계열(Gear 46.9/8.2/7.1, NutBolt, Frame)은 순수 확장 사다리에 DGR 하락이 가장 극적이지만 Isaac Gym 기반이라 robosuite 파이프라인과 분리된다. Gear는 [[mimicgen-failure-audit]]의 850/1000 사례가 나온 태스크로, 별도 트랙으로만 남겨둔다.

**주의 (annotation)**: HF의 source hdf5는 순수 teleop 데이터라 **사전 annotation이 안 되어 있다**. 태스크마다 `prepare_src_dataset.py --env_interface MG_<Task> --env_interface_type robosuite` 1회 실행 필요 (10 demo라 태스크당 수 분; `prepare_all_src_datasets.sh`에 전 태스크 커맨드 존재). Phase-0에서 square/threading/coffee/three_piece는 이미 처리됨 (aidas 서버).

---

## 2. 공개 bounds 원본 수치 (커스텀 변형의 기준)

레퍼런스는 각 env의 table_offset. Square/NutAssembly는 (0,0,0.82), Threading·Coffee·ThreePiece·MugCleanup·Stack·StackThree는 (0,0,0.8), HammerCleanup·Kitchen은 (-0.2,0,0.90). 테이블은 전부 0.8×0.8m (Kitchen만 1.0×0.8) — 중심 기준 반폭 ±0.4m.

### Threading (`envs/robosuite/threading.py` D0 L493, D1 L498, D2 L540)

| 변형 | needle x | needle y | needle z_rot | tripod x | tripod y | tripod z_rot |
|---|---|---|---|---|---|---|
| D0 | (-0.20, -0.05) | (0.15, 0.25) | (-120°, -60°) | 고정 0 | 고정 -0.15 | 고정 90° |
| D1 | (-0.20, 0.05) | (0.15, 0.25) | (-210°, 30°) | (-0.10, 0.15) | (-0.20, -0.10) | (30°, 150°) |
| D2 (공개) | (-0.20, 0.05) | **(-0.25, -0.15)** ← 미러 | (-210°, 30°) | (-0.10, 0.15) | **(0.10, 0.20)** ← 미러 | (-150°, -30°) |

### Coffee (`envs/robosuite/coffee.py` D0 L771, D1 L776, D2 L812)

| 변형 | machine x | machine y | machine z_rot | pod x | pod y |
|---|---|---|---|---|---|
| D0 | 고정 0 | 고정 -0.10 | 고정 -30° | (-0.13, -0.07) | (0.17, 0.23) |
| D1 | (0.05, 0.15) ← D0 점 이탈 | (-0.20, -0.10) | (-30°, 60°) | (-0.20, 0.05) | (0.17, 0.30) |
| D2 (공개) | (-0.05, 0.05) | **(0.10, 0.20)** ← 미러 | (120°, 210°) | (-0.20, 0.05) | **(-0.30, -0.17)** ← 미러 |

### Stack / StackThree (`envs/robosuite/stack.py`)

| 변형 | cube 공통 x,y | z_rot |
|---|---|---|
| Stack D0 | (±0.08) | (0, 2π) |
| Stack D1 | (±0.20) | (0, 2π) |
| StackThree D0 | (±0.10) | (0, 2π) |
| StackThree D1 | (±0.20) | (0, 2π) |

### MugCleanup (`envs/robosuite/mug_cleanup.py` D0 L616, D1 L621)

| 변형 | drawer x | drawer y | drawer z_rot | mug x | mug y | mug z_rot |
|---|---|---|---|---|---|---|
| D0 | 고정 0 | 고정 0.30 | 고정 0 | (-0.15, 0.15) | (-0.25, -0.10) | (0, 2π) |
| D1 | (-0.15, 0.05) | (0.25, 0.35) | (-30°, 30°) | (-0.25, 0.15) | (-0.30, **-0.15**) ← (-0.15,-0.10) 상실 | (0, 2π) |

(Square·ThreePieceAssembly는 순수 확장이 수치로 확인되어 생략 — MOTIVATION_INITIAL_CONDITION.md §3.2와 동일.)

---

## 3. 커스텀 변형 제안 (E-시리즈)

원칙 세 가지.

1. **엄밀한 superset**: 이전 단계 bounds가 새 bounds의 부분집합이어야 한다 (경계 포함 허용). 재배치 금지 — 물체의 좌우 배역(needle은 +y, tripod는 -y 등)을 유지한 채 박스만 키운다.
2. **도달 가능 영역 안에서**: 공개 변형의 실측 상한은 x 최대 +0.30(Square D1 peg), y 최대 0.35(MugCleanup D1 drawer), 박스형 최대 ±0.25(Square D2)다. 신규 bounds는 x는 ±0.25(공개 D2 관례), y는 공개 선례에 준해 태스크별 최대 0.38까지 허용하되, **아래 표의 물체별 초안 수치가 그대로 관문 대상** — 커밋 전 PLAN.md §1.5의 IK 스캔 + 50-attempt 프로브(+image 태스크는 렌더 커버리지)를 통과해야 동결된다 (수치 변경은 L_m 정규화를 바꾸므로 동결 전에만).
3. **회전은 보수적으로**: 위치 확장이 주 축이고, 회전 범위는 이전 단계 값을 포함하는 선에서 약간만 넓힌다 (회전이 transform 거리 정의의 부록 축이기 때문).

### 제안 수치 (reachability 검증 전 초안)

**Threading_D2E** (D1 ⊂ D2E, 미러 없이 확장):

| 물체 | x | y | z_rot |
|---|---|---|---|
| needle | (-0.25, 0.10) | (0.10, 0.30) | (-210°, 30°) 유지 |
| tripod | (-0.15, 0.20) | (-0.25, -0.05) | (15°, 165°) |

**Coffee_D1E** (D1의 machine shift 교정 — D0 점 (0, -0.10, -30°)을 내부/경계에 포함):

| 물체 | x | y | z_rot |
|---|---|---|---|
| machine | (0.00, 0.15) | (-0.20, -0.10) | (-30°, 60°) |
| pod | (-0.20, 0.05) | (0.17, 0.30) | 고정 0 |

**Coffee_D2E** (D1E ⊂ D2E; 1차 프로브에서 중앙 쪽 안쪽 모서리 2곳 도달 미달 → 해당 모서리 양 축 0.02m 축소 반영):

| 물체 | x | y | z_rot |
|---|---|---|---|
| machine | (-0.08, 0.20) | (-0.25, -0.07) | (-60°, 90°) |
| pod | (-0.25, 0.08) | (0.14, 0.33) | 고정 0 |

**Stack_D2E**: 전 cube x (-0.25, 0.23), y (-0.25, 0.23), z_rot (0, 2π). 원안 ±0.25에서 1차 프로브 결과 최원거리 모서리(0.25, 0.25 — base에서 ~0.85m, Panda 도달 한계)가 관문 미달(도달률 0.66 vs 내부 0.92)이라 상한 두 축을 0.02 축소. **StackThree_D2E**: 전 cube x,y = (±0.25) 유지 (그룹 고정 박스가 더 커서 실질 샘플이 안쪽으로 당겨지는 덕에 전 위치 통과).

**MugCleanup_D1E** (Tier-2): D1에서 mug y만 (-0.30, -0.10)로 (합집합). **MugCleanup_D2E**: drawer x (-0.20, 0.10), y (0.22, 0.38), z_rot (-45°, 45°); mug x (-0.25, 0.20), y (-0.32, -0.08).

### 구현 방법 (검증된 확장 지점)

변형 하나 추가 = env 서브클래스 1개 + import 1줄 + config에서 `--task_name` 지정. 예:

```python
# motivation/genaudit/envs/robosuite_variants.py
class Threading_D2E(Threading_D1):
    """D1의 엄밀한 superset. 미러 없이 needle(+y)·tripod(-y) 박스만 확장."""
    def _get_initial_placement_bounds(self):
        return { "needle": dict(x=(-0.25, 0.10), y=(0.10, 0.30), ...),
                 "tripod": dict(x=(-0.15, 0.20), y=(-0.25, -0.05), ...) }
```

- robosuite는 env 서브클래스를 정의 시점에 자동 등록하므로, 생성 스크립트 실행 전에 우리 모듈이 import되기만 하면 된다 (mimicgen 소스 수정 불필요 — wrapper 스크립트에서 `import genaudit.envs.robosuite_variants` 후 `mimicgen.scripts.generate_dataset.main()` 호출).
- 기존 config 템플릿 재사용: `--config templates/robosuite/threading.json --task_name Threading_D2E --source <prepared src>`. 생성 env는 source hdf5의 env_meta에서 이름만 바꿔 만들어지므로 env interface(MG_Threading)는 그대로 동작.
- 카메라: `agentview_full` 스왑은 **Stack_D1/StackThree_D1/Square_D2에만 있고 Threading·Coffee의 D변형에는 없다** (base가 agentview_full 카메라를 추가만 하고 기본으로 쓰지 않음). 따라서 Threading_D2E(Threading_D1) 파생으로는 자동 해결되지 않는다 — §1.5 렌더 커버리지 프로브에서 극단 배치가 잘리면 D2E 클래스에 스왑을 명시 구현한다 (image E2에 Threading이 포함되므로 필수 점검).
- Square D1처럼 특수 기계장치(peg XML 재작성, 충돌 회피 리샘플)가 있는 태스크는 해당 D 클래스를 부모로 삼으면 공짜로 상속된다.
- 샘플링은 `ensure_valid_placement=True` + 5000회 재시도라, 지나치게 넓은 bounds는 reset에서 `RandomizationError`로 시끄럽게 실패한다 (조용한 왜곡 없음).

---

## 4. E1/E2 대상 최종 구성

| Tier | task | 사다리 | 용도 |
|---|---|---|---|
| 1 | Square | D0 → D1 → D2 (공개) | E1 (Phase-0 재사용) + **E2** |
| 1 | Threading | D0 → D1 → **D2E** (+공개 D2를 대조군으로 유지) | E1 + **E2** |
| 1 | Coffee | D0 → **D1E** → **D2E** (+공개 D1·D2 대조) | E1 + **E2** |
| 1 | ThreePieceAssembly | D0 → D1 → D2 (공개) | E1 (Phase-0 재사용) |
| 1 | Stack | D0 → D1 → **D2E** | E1 + E2 고DGR 대조군 |
| 1 | StackThree | D0 → D1 → **D2E** | E1 |
| 2 | MugCleanup | D0 → **D1E** → **D2E** | E1 확장 |
| 2 | HammerCleanup | D0 → D1 (→ D2E 검토) | E1 확장 |
| 2 | CoffeePreparation | D0 → D1 | E1 확장 (long-horizon 대표) |

Tier 1만으로 "확장 사다리 6개 태스크 × 3단계"가 확보되고, 그중 재배치 confound가 제거된 Threading·Coffee에서 **공개 D2(재배치) vs D2E(확장)를 같은 attempted-거리 축 위에서 비교**할 수 있다 — transform 거리가 충분통계인지(= 영역 이동 자체는 추가 효과가 없는지) 가리는 E1의 부차 검증 (PLAN.md §1.4).
