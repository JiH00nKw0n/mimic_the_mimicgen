# Motivation 확장 실험 계획 — E1 (태스크 일반화) & E2 (policy 효과)

Phase-0(../MOTIVATION_INITIAL_CONDITION.md)의 후속. 조사 근거와 태스크별 수치는 TASKS.md, 소프트웨어 설계는 ARCHITECTURE.md 참조.

## 0. 무엇을 검증하나

Phase-0에서 확인한 것: source와 합성 장면 사이 **transform 거리가 커질수록 DGR이 떨어지고**, 남는 데이터는 초기조건 위치가 아니라 **source ancestry로 쏠린다** (Square 75/46/33%, Threading 51/37/22%, Coffee 79/66/28%). 이번 확장의 두 갈래:

- **E1 — 법칙의 태스크 일반화**: Threading·Coffee의 공개 D2는 영역 확장이 아니라 **반대편 재배치(대칭이동)** 라서 transform 축이 "거리"와 "영역 이동"으로 오염돼 있다. 두 태스크에 확장형 변형(D2E)을 새로 정의하고, source demo + annotation + D0가 있는 태스크로 풀을 넓혀 "transform↑ → DGR↓ + ancestry skew↑"를 태스크 일반 법칙으로 검증한다.
- **E2 — retention 편중의 policy 영향**: 최광 변형(D2/D2E)에서 생성한 데이터로 학습해 같은 분포에서 평가. 비교군 = 표준 파이프라인의 500개, 대조군 = transform 구간별 거의 uniform 500개. 구간화 규칙은 전 태스크 동일, uniformity는 metric으로 인증 — 논문에 그대로 실을 수 있는 셋업.

## 1. E1 — 확장 사다리로 태스크 일반화

### 1.1 태스크 구성 (수치 근거: TASKS.md)

| Tier | task | 사다리 | 비고 |
|---|---|---|---|
| 1 | Square | D0→D1→D2 (공개 재사용) | Phase-0 pool 재사용 |
| 1 | Threading | D0→D1→**D2E 신규** | 공개 D2(미러)는 대조군 유지 |
| 1 | Coffee | D0→**D1E**→**D2E 신규** | 공개 D1·D2 대조군 유지 |
| 1 | ThreePieceAssembly | D0→D1→D2 (공개 재사용) | Phase-0 pool 재사용 |
| 1 | Stack | D0→D1→**D2E(±0.25)** | 고DGR 대조 태스크 |
| 1 | StackThree | D0→D1→**D2E(±0.25)** | |
| 2 | MugCleanup / HammerCleanup / CoffeePreparation | D1E·D2E 등 | 여유 시 |

### 1.2 생성 프로토콜 (전 태스크 동일)

- 변형당 **500 attempts** 고정: `guarantee=False`, `keep_failed=True`, `max_num_failures=null`(무제한 — Phase-0의 500 cap은 500 attempts라 무손실이었지만 그 이상에선 실패 ancestry가 유실되므로 필수), 공식 `generate_dataset.py`.
- **source 선택 통일: `random` + `select_src_per_subtask=False`** — episode당 source 1개여야 transform 거리·ancestry가 단일값. Square·Stack류의 공식 NN 선택은 attempted 분포를 근거리로 편향시켜 far-bin을 못 채운다. 근거: MimicGen 논문 no_nn ablation — 선택 전략은 DGR만 바꾸고 policy SR은 거의 불변. Phase-0 공식-config 결과는 참조값으로 병기.
- 모든 attempt(성공·실패)를 AttemptRecord로 추출: `{task, variant, attempt_id, source_demo_id, per-object Δxy·Δyaw, d_raw, d_norm, success, ep_len}`.

### 1.3 Transform 거리 — 절대적·task-정의 기반 정의 (최종)

**대상 물체**: `M` = 최광 변형(D2/D2E)에서 배치 영역 넓이가 0이 아닌(=움직이는) 물체 집합. 고정 물체는 거리에서 제외.

**정규화 상수 (source 무관, task 정의만으로 결정)**:

```
L_m = diag(V_m) = √(w_x,m² + w_y,m²)
```

V_m = 물체 m의 **최광 변형 xy 배치 박스**, w = 박스 변 길이. source demo·annotation·데이터와 완전히 독립 — task 정의(bounds)만으로 확정되고, 사다리의 전 변형(D0/D1/D2E)이 같은 축을 공유한다. 예: Square D2 nut·peg 0.5×0.5 → L=0.707m; Threading D2E needle 0.35×0.20 → L=0.403m; Coffee D2E machine 0.30×0.20 → L=0.361m.

**위치 거리 (1차 정의, binning·E2 샘플링에 사용)**:

```
d_pos = (1/M) Σ_m ‖xy_new(m) − xy_src(m)‖ / L_m
```

**확장 사다리 pool에서는** 양 끝점(source ∈ D0 ⊂ V_m, new ∈ V_m)이 V_m 안에 있으므로 각 항 ≤ 1, 따라서 **d_pos ∈ [0,1]이고 d=1은 "물체가 자기 배치 영역의 대각선만큼 이동" = 그 영역에서 기하학적으로 가능한 최대 transform**. source가 어디 있었는지는 분모에 전혀 안 들어간다. 단, **재배치 대조군(공개 D2 미러) pool은 new가 V_m 밖이라 d_pos가 1을 넘는다** (Threading 미러 needle: 물체별 항 ~0.74–1.39) — 대조군 분석의 축 처리는 §1.4에서 별도 정의하고, 1차 사다리 축은 여기 정의 그대로 쓴다.

**회전 거리 (병행 축, 스칼라 혼합하지 않음)**:

```
d_rot = (1/M_rot) Σ_m |wrap_{n_m}(θ_new − θ_src)| / (π/n_m)
```

n_m = 물체의 회전 대칭 차수(cube=4, 비대칭 물체=1; task config에 명시), wrap_n = 2π/n 주기로 감아 [0, π/n]에 놓은 각차. d_rot ∈ [0,1], 1 = "대칭을 고려한 최대 회전". 위치·회전을 하나의 스칼라로 합치려면 임의 가중치 λ가 필요해 공격 지점이 되므로, **1차 정의는 d_pos, d_rot은 별도 축으로 병행 보고**하고 combined (d_pos+d_rot)/2는 부록 robustness로만.

**Equal weighting 방어**: 물체별 가중치를 두는 순간 "가중치는 어떻게 정했나"가 공격 지점이 된다. 무가중 평균을 1차로 하되, (a) **per-object marginal** — 물체별 정규화 변위 각각의 DGR 예측력(point-biserial; Phase-0의 nut_off/peg_off 분석의 일반화)을 항상 병기해 "어떤 물체가 중요한가"를 실증으로 답하고, (b) **max-aggregation** (`max_m` 버전)을 robustness로 병기한다.

**d_raw 병행 기록 (경향 보존 체크)**: Phase-0 정의 `d_raw = Σ_m ‖Δxy(m)‖`(미터)를 모든 record에 함께 기록. 다물체 태스크에선 정규화가 물체 간 가중을 바꿔 raw 합의 경향을 약화시킬 수 있으므로, B0에서 Phase-0 pool로 per-bin DGR 단조성·Spearman·point-biserial을 두 정의로 나란히 산출하고 **정규화판이 경향을 약화시키면 태스크 내 1차 정의는 d_raw로 확정** (d_pos는 크로스태스크 오버레이용). 구간화가 quantile 기반이라 어느 정의든 파이프라인은 동일하게 동작한다.

### 1.4 측정과 예상 결과

- 변형별 DGR + **연속 DGR-vs-d 곡선** + per-bin DGR + ancestry skew(top-3 share, n_eff)-vs-d.
- **DGR 분모 주의**: mimicgen은 예외로 죽은 attempt를 num_problematic으로 세고 재시도하며 어느 hdf5에도 기록하지 않는다 → 우리의 DGR 분모에서 빠진다. 넓은 D2E일수록 예외가 몰릴 수 있으므로 **run의 important_stats.json에서 num_problematic을 함께 보고**하고 (extract가 출력) 논문에 제외 사실을 명시한다.
- **부차 검증 (재배치 vs 확장)**: Threading·Coffee에서 공개 D2(미러)와 D2E(확장)를 비교한다. 미러 pool은 d_pos가 1을 넘고 두 pool의 d 지지집합 겹침이 얇을 수 있으므로 그대로 겹쳐 그리면 공허한 비교가 된다. 처리를 사전 등록: (a) **대조 전용 축** — 이 분석에 한해 정규화 박스를 V_m ∪ (미러 박스)의 bounding box로 바꾼 d′를 쓴다 (축이 다시 [0,1]로 유계), (b) **비교는 두 pool의 공통 지지구간으로 제한**하고 각 pool의 질량 중 그 구간에 들어오는 비율(overlap fraction)을 보고, (c) 겹침이 얇으면(<20% 급) 점별 비교 대신 pool별 DGR(d′) 회귀 기울기의 연속성만 주장한다. DGR(d′) 곡선이 공통 구간에서 일치하면 "transform 거리가 충분통계(영역 이동 자체는 무관)", 불일치하면 그 자체가 발견.
- 반증 조건: 새 태스크들에서 DGR-vs-d 단조 하락이 재현되지 않으면 법칙이 아니라 태스크 특수성.

### 1.5 Reachability — 확장 최대 영역의 정의와 관문

"그립퍼 닿는 영역"을 계산된 객체로: **R_reach = 테이블 위 점 중 태스크의 grasp 자세 패밀리(top-down ± 허용 tilt)로 IK 해가 존재하고 여유가 확보되는 집합** (Panda 마운트·테이블 기하로 결정, 오프라인 IK 스캔 1회). D2E 박스는 R_reach의 부분집합이어야 하며 TASKS.md의 ±0.25 초안은 스캔 결과로 확정한다. displacement 자체는 도달성 제약이 아님(양 끝점이 R_reach 안이면 d=1 변위도 유효).

참고로 공개 변형의 실측 상한은 "±0.25"가 아니다 — x는 +0.30(Square D1 peg), y는 0.35(MugCleanup D1 drawer)까지 쓰인다. 우리 D2E 초안(TASKS.md §3 = bounds.py 등록값)은 x ±0.25, y는 태스크별로 최대 0.38까지 가며, **일괄 ±0.25 클램프가 아니라 등록된 물체별 초안 수치를 프로브가 확정**한다 (수치를 바꾸면 L_m 정규화가 전부 바뀌므로 동결 전에만 조정).

실측 관문: bounds 동결 전 극단 서브영역(코너 4 + 변 중앙)에 물체를 고정한 50-attempt 프로브 — (a) reset 성공(RandomizationError 없음), (b) 첫 subtask 도달률이 내부 영역과 동급(IK 가능 ≠ OSC 추종 가능), (c) **렌더 커버리지**: image 모달리티 태스크(Square·Threading)는 극단 배치 8곳에서 agentview 프레임에 물체가 온전히 들어오는지 렌더로 확인 — Threading·Coffee에는 Stack_D1류의 agentview_full 스왑이 없어서 자동으로 해결되지 않으며, 잘리면 D2E 클래스에 카메라 스왑을 명시 구현한다 (far-bin 전용 화면 잘림은 image E2의 조용한 교란변수). 실패 시 0.02m 단위 축소.

## 2. E2 — transform-uniform 리샘플링의 policy 효과

### 2.1 구간화 규칙 (전 태스크 동일 "규칙", 계산은 태스크별)

**태스크·변형마다 자기 attempted d 분포의 K=5 quantile(등-attempt-mass) 구간을 그 pool 전체에서 1회 계산 후 동결.** edges 수치는 태스크마다 다르지만 **규칙은 한 문장으로 동일** — 태스크마다 d 분포가 완전히 다르므로 pooled quantile이 아니라 per-task quantile이 맞다 (태스크 간 비교는 정규화된 d_pos 축의 오버레이로). 근거: (a) 규칙이 태스크 불변·단위 불변, (b) attempted 분포는 필터링 이전에 결정 — 순환성 없음, (c) 등-mass 구간에선 무편향 retention ⇒ retained 히스토그램 uniform이므로 **편차 자체가 survivor bias의 크기**, (d) attempts 산정이 깔끔. 부록에서 절대 edges {0,.2,.4,.6,.8,1.0}(d_pos)로 구간화 불변성 재확인.

### 2.2 Uniformity 인증

**TV(p̂, uniform) ≤ 0.02 그리고 min bin ≥ 90/100** (K=5, 500개 기준). TV = "구간을 옮겨야 하는 demo 비율"로 직독. 처치군은 quota 추출로 TV=0이 기본 — metric의 실무 역할은 (i) 희소 bin 미달 허용 오차, (ii) **baseline의 skew를 같은 자로 정량화** (예상 TV 0.10–0.30, 그 자체가 헤드라인 통계).

### 2.3 Arms — 동일 pool에서 post-hoc 추출 (생성 과정 비트단위 동일, 차이는 추출 규칙뿐)

| arm | 구성 |
|---|---|
| A baseline | retained 무작위 500 — **random-selection 파이프라인의 표준 산출과 교환가능** (attempts i.i.d.이므로 guarantee 모드의 first-500과 동치). 단 공식 Square/Stack 설정은 NN·per-subtask 선택이라 A의 skew가 "공식 배포판의 skew"와 같지는 않다 — 이 주장은 하지 않고, Square에서는 Phase-0의 공식-config pool TV·ancestry를 나란히 보고해 실제 배포판 수치를 앵커한다 |
| B transform-uniform | bin당 정확히 100개 층화 추출 |
| C ancestry-balanced | source당 50개 (transform marginal 자유) — B의 효과가 transform 균형인지 ancestry 균형인지 분해 |
| (D raking) | transform×ancestry 이중 균형 — pool cell 점유(5×10 cell) 확인 후 조건부 |

전 arm에서 ancestry 히스토그램·n_eff 의무 보고. research/proposal.md의 "condition D(그냥 coverage) 반박 장치" 프레이밍과 일치.

**transition 교란 통제**: far-transform 에피소드는 체계적으로 더 길어서 B는 demo 수(500)는 같아도 **총 transition 수가 A보다 많다**. BC-RNN이 고정 gradient step이라 업데이트 수는 같지만, 리뷰어는 "데이터가 더 많아서"라고 반박할 수 있다. 대응(사전 등록): (a) arm별 총 transition 수·ep_len 분포 의무 보고 (AttemptRecord의 ep_len 사용), (b) robustness arm **B′ = B를 A의 transition 예산에 맞춰 bin 내 추가 추출로 매칭**한 버전 1개 (bin 균형은 TV ≤ 0.02 안에서 유지되는 한도로) — 주장이 B′에서도 유지되는지 확인.

### 2.4 Pool 크기

기본식 `N = 500 / p_min` (K 무관; p_min = 최희소 bin DGR). z=2 버퍼 포함 권장 **N ≈ 6,250/task** (가정 프로파일 55/45/33/22/10%; N은 pool seed들에 걸친 **총합** — seed당 N/len(seeds)씩 생성 후 병합, gen-config가 자동 분할). 커밋 전 **Phase-0 500-attempt pool로 per-bin DGR의 Wilson 95% 하한**을 추정해 태스크별 N 확정. Wilson-LB 기준 N > 10,000이면 K=4 전환(fallback_k, curate가 사전 등록대로 자동 적용) 또는 top bin을 attempted 90th percentile에서 절단(양 arm 동일 적용이라 공정). 사전 등록.

**arm C용 두 번째 제약**: C는 source당 50개가 필요하므로 bin 제약과 별도로 `N ≥ 50 × n_src / p_worst-src` (worst-source 성공률의 Wilson 하한, Phase-0 pool로 추정; Square D2 최저 source 8% → N ≈ 6,250에서 기대 50±7로 아슬아슬). **미달 시 사전 등록된 fallback**: 그 source를 C에서 제외하고 남은 source들로 등-quota 재구성(500 나누어떨어지는 최대 quota), 제외 목록과 C의 n_eff를 명시 보고. 샘플러는 조용한 재분배 대신 시끄럽게 실패하도록 구현돼 있어(quota 미달 시 인증 실패) 이 fallback은 수동 승인 경로다.

### 2.5 확정 스코프

- **태스크 4개**: Square(D2), Threading(D2E), Coffee(D2E) + **Stack(D2E)** 고DGR positive control("효과가 retention 압력에 비례" 검증).
- **모달리티**: low-dim 4태스크 전체(1차 결과) + **image는 Square·Threading**에서 재현 확인 (2-arm × 3 seed = 12 run, eval rate 완화).
- **arm**: A/B/C 3개 (+D는 조건부).
- 학습 run: low-dim 4×3×3=36 + Square 분산분해(3 dataset × 3 train) 6 + image 12 = **54 run** ≈ L40S 동시 3–4개로 4–6일.

### 2.6 학습·평가

- **학습**: mimicgen 공식 `generate_core_training_configs.py` 레시피 그대로 (BC-RNN GMM; low-dim 2000 epoch·LR 1e-3, image 600 epoch·LR 1e-4·84×84 2캠). 한 hdf5에 arm별 `hdf5_filter_key` — config는 동일, filter_key만 다름.
- **평가**: (1) 학습 중 50-rollout eval(논문 프로토콜) + (2) **frozen eval set** — 시드된 env에서 미리 뽑은 200개 초기상태를 `reset_to`로 주입 (전 arm·seed 공유 → paired, McNemar 가능). 평가 축 **d_eval = min_src d(에피소드 IC, source IC)** (동결 정규화 동일), eval set 자체 quantile로 층화. 보조 축: D0 중심 offset (Spearman 상관 보고).
- **1차 endpoint와 추론 (사전 등록)**: 1차 = **far 2-bin에서의 B−A SR 격차**, 태스크별. 1차 추론은 **per-episode paired 분석** — frozen 200 에피소드에서 seed-pair별 McNemar + seed·에피소드를 random effect로 둔 mixed-effects logistic(success ~ arm + (1|episode) + (1|run))으로 종합. **run-level Wilcoxon은 기술 통계로만** (n=3 pair는 최소 양측 p=0.25라 유의 판정이 산술적으로 불가능 — 검정으로 쓰지 않는다). 다중비교: 태스크 4 × 대조 2(B−A, C−A)에 Holm 보정, 기울기·aggregate 등 나머지는 exploratory로 명시. aggregate SR 의무 보고 (Coffee ceiling 77% 유의).
- **seeds**: 태스크·arm당 3 (dataset 추출+학습 seed 묶음; arm별 RNG 스트림 독립), Square만 3×3 분산 분해.
- **해석**: B>A(far bin) = 주장 성립 / B≈A = transform 편중 무해 — condition-D 대조로서 여전히 가치 / C가 B를 설명 = ancestry가 진짜 축 (Phase-0 결론과 정합).

## 3. Isaac Lab arm (generality, 본선 후)

robosuite 태스크 포팅 없음. **Isaac-Stack-Cube-Franka-IK-Rel-Mimic-v0** (NVIDIA 사전 annotation 10 demo, arpa 파이프라인 검증 완료)에 widened reset 변형을 정의하고 lab_stack_mimic의 `provenance_hooks.py`를 AttemptRecord 스키마로 확장해 E1 + 축소판 E2(state BC, run당 ~30분) 재현 — 발견이 robosuite 전용이 아님을 보이는 용도.

## 4. 실행 단계

| 단계 | 내용 | 자원 |
|---|---|---|
| A | 문서 + genaudit 스켈레톤 + 변형 클래스 + curation 단위 테스트 + configs | 로컬 (완료 대상) |
| B0 | Phase-0 pool 확인(aidas 3.151.29.145)→arpa 동기화, 신규 태스크 source 다운로드+`prepare_src_dataset.py`, Wilson-LB→태스크별 N 확정, **d_raw vs d_pos 경향 보존 체크** | 서버 CPU |
| B1 | reachability 스캔·프로브 → D2E bounds 동결 → E1 생성 (신규 변형 × 500) | 서버 CPU |
| B2 | E2 pool 생성 (4태스크 × N≈6,250, state-only; 선택 500개만 사후 obs 렌더) | 서버 CPU |
| C | E1 분석·그림 → arm 추출·인증 → BC-RNN 54 run + frozen eval | L40S |
| D | Isaac Lab stack arm | L40S |
| E | 결과 정리, MOTIVATION 문서 갱신 | 로컬 |

## 5. 검증 계획

- 단위: `pytest motivation/tests/` (binning·TV·sampler·L_m·스키마 — 시뮬레이터 불필요).
- 생성 스모크: 태스크당 `--debug` → 신규 변형 reset·생성·추출 왕복.
- E1 재현성: Square 재실행이 Phase-0 DGR(75/46/33) ±2pp (random-selection 전환분은 논문 no_nn 수치와 대조).
- E2 파이프라인 검증: 공개 core square_d2로 1 run → 논문 SR(low-dim 58.7±1.9) 대조 후 본 실험.

## 6. 리스크

- **D2E far bin 미충족** → Wilson-LB 사전 산정 + K=4 fallback + top-bin 절단 (사전 등록).
- **정규화가 경향 약화** → d_raw/d_pos 병기 + B0 경향 보존 체크로 1차 정의 확정 (§1.3).
- **selection 전략 변경으로 Phase-0 불연속** → no_nn ablation 인용 + Square 양쪽 병기.
- **aidas/arpa 이원화** → B0에서 arpa로 단일화.
- **Coffee ceiling** → per-bin endpoint 1차 + Square·Threading 포함으로 헤지.

## 7. 관련 연구 (novelty 근거)

Re-Mix(arXiv:2408.14037), Data Quality in IL(2306.02437), CUPID(2506.19121), SCIZOR(2505.22626), MimicLabs(2506.13536), DemoGen(2502.16932), Balanced BC(2508.06319), 오프라인 RL 리밸런싱(2210.09241), Gao et al.(2403.05110), Data Scaling Laws(2410.18647) — 전부 generic quality/diversity/influence 큐레이션. **생성기의 success-filter가 만드는, 메커니즘이 규명된 skew를 그 축만 토글해 policy 효과를 인과적으로 보인 것은 없음.** MimicGen 논문 스스로 Appendix R에서 retained 데이터의 coverage bias를 인정하고 future work로 미룸 — 직접 인용 지점.
