# Motivation 통제 실험 결과 (motivation_controlled)

2026-07-30 완료 (학습 39/39 완료 14:33 UTC, 평가 완료 17:07 UTC). 분석 2026-07-31.

## 0. 왜 이 실험을 했나

E2(6시드)의 두 결론 — "transform을 균등하게 해도 정책 성능이 안 변한다", "ancestry 균등화가 해로운 건 저품질 소스를 끌어올리기 때문이다" — 은 관찰 비교라서 교란변수 반론이 남아 있었다. 특히 "transform을 균일하게 뽑다 보니 나쁜 데모가 더 많이 섞였고, 그래서 이득이 상쇄된 것 아니냐"는 우려. 이번 실험은 소스 정체성, 소스 품질(DGR), 데이터 크기, transform 거리 분포를 각각 얼려 놓고 한 번에 한 축만 흔들어 이 결론들을 인과 수준에서 재검정했다.

공통 셋업: motivation_new N2 생성 풀에서 filter key로 arm 추출, BC-RNN-GMM low-dim 2000 epoch, frozen 200 reset 평가(paired), 시드 301–306(EXP-A만 6시드, 나머지 3시드). 거리 bin은 태스크별 retained d_pos 5분위(edges — stack 0.205/0.253/0.293/0.340, threading 0.201/0.250/0.293/0.345). near=bin0–1, mid=bin2, far=bin3–4. 통계는 per-episode paired McNemar(pair = 시드×reset), 등가성은 TOST ±5pp 기준.

소스별 DGR(생성 성공률): stack s4=0.95, s7=0.94, s3=0.74, s1=0.64 / threading s2=0.63, s8=0.61, s1=0.60, s0=0.53, s6=0.44, s4=0.30, s5=0.26.

| 실험 | 태스크 | arm | 크기 | 흔든 축 | 얼린 축 |
|---|---|---|---|---|---|
| A | stack | A_nearheavy(375near+125far) vs A_farheavy(125near+375far) | 500×6시드 | 거리 밀도 | 소스{4,7}, 품질, 크기 |
| B | stack | B_far(near150+mid100+far150) vs B_nearpad(near300+mid100, far 0개) | 400×3시드 | far 커버리지 유무 | 소스{4,7}, 품질, 크기 |
| C2 | stack | C2_hi{4,7} vs C2_mid{3,1} | 70/bin×5=350×3시드 | 소스 품질 | 거리 분포(5분위 완전 매칭), 크기 |
| C | threading | C_hi{2,8} / C_mid{0,6} / C_lo{4,5} | 50/bin×5=250×3시드 | 소스 품질(3단계) | 거리 분포, 크기 |
| D | threading | D_1{2,0}(평균DGR 0.58) vs D_2{8,1}(평균DGR 0.61) | 50/bin×5=250×3시드 | 소스 정체성 | DGR(거의 매칭), 거리 분포, 크기 |

## 1. EXP-A — far 밀도를 3배로 늘려도 아무 일도 없다 (transform-null 확정)

같은 소스({4,7}), 같은 500개, near:far 비율만 3:1 vs 1:3으로 뒤집었다. 6시드, 1200 pairs.

| 슬라이스 | n | A_nearheavy | A_farheavy | 차이 | p | 90% CI |
|---|---|---|---|---|---|---|
| 전체 | 1200 | 0.961 | 0.967 | −0.6pp | 0.39 | [−1.5, +0.4]pp |
| near(bin0–1) | 570 | 0.989 | 0.991 | −0.2pp | 1.0 | [−1.0, +0.7]pp |
| mid(bin2) | 252 | 0.948 | 0.948 | 0.0pp | 1.0 | [−2.4, +2.4]pp |
| far(bin3) | 222 | 0.968 | 0.977 | −0.9pp | 0.73 | [−3.0, +1.2]pp |
| far(bin4) | 156 | 0.865 | 0.891 | −2.6pp | 0.48 | [−7.0, +1.9]pp |

전체 90% CI가 ±5pp 등가 구간 안에 완전히 들어온다(TOST 통과). far 데이터를 3배 넣어도 far 평가에서조차 이득이 없다. 소스·품질·크기가 비트 단위로 통제된 상태이므로, E2 transform_uniform의 null이 "나쁜 데모 혼입" 탓이라는 가설은 기각된다. 밀도 재배분 자체가 무효다.

## 2. EXP-B — far 학습 데이터 0개여도 far 평가가 안 무너진다

같은 소스, 같은 400개. B_nearpad는 far bin 데모가 하나도 없다.

| 슬라이스 | n | B_far | B_nearpad | 차이 | p |
|---|---|---|---|---|---|
| 전체 | 600 | 0.955 | 0.955 | 0.0pp | 1.0 |
| far(bin3) | 111 | 0.937 | 0.901 | +3.6pp | 0.39 |
| far(bin4) | 78 | 0.833 | 0.872 | −3.8pp | 0.61 |

far 커버리지를 통째로 빼도 far 평가 무붕괴. 이전 exp2-B에서 nearest-40%만 남긴 near_only(square·coffee)는 mid/far에서 8~14pp 무너졌는데, 이번 B_nearpad는 mid(하위 60%)까지는 있다. 종합하면 커버리지 요구는 "전 구간"이 아니라 중간 어딘가까지이고, 그 너머는 일반화가 메운다 — 문턱은 nearest-40%와 nearest-60% 사이(또는 태스크 의존; near_only는 square·coffee, 이번은 stack이라 교차 확인은 남음). MimicGen 필터가 만드는 near 편중이 정책에 무해한 이유가 이걸로 설명된다.

## 3. EXP-C2 / EXP-C — 거리를 완전히 매칭해도 소스 품질 효과는 그대로 (인과 확정)

거리 5분위마다 같은 수를 뽑아 두 arm의 거리 분포를 동일하게 만들었다. 남는 차이는 소스뿐.

stack: C2_hi{4,7} 0.963 vs C2_mid{3,1} 0.765 → **+19.8pp, p=1.9e-26**, 90% CI [+16.8, +22.8]pp. 시드 간 겹침 전무(0.955–0.97 vs 0.76–0.77).

거리별 프로파일(자기 소스 기준 거리): C2_hi는 near 1.000 → far 0.899로 완만하게, C2_mid는 near 0.882 → far 0.612로 가파르게 떨어진다. 저품질 소스 데이터는 멀수록 더 나쁘다.

threading(3단계): C_hi 0.635 / C_mid 0.535 / C_lo 0.530.
- hi vs mid: +10.0pp, p=1.4e-4
- hi vs lo: +10.5pp, p=7.9e-5
- mid vs lo: +0.5pp, p=0.90 (무차이)

즉 선형 gradient가 아니라 상위 소스 프리미엄에 가깝다 — 좋은 소스(DGR 0.6대)와 그 아래(0.5 이하)의 격차가 크고, 중간과 하위는 구분이 안 된다.

exp2-A의 hidgr/lodgr +51pp가 "거리·커버리지 artifact"였을 가능성은 이걸로 닫혔다. 소스 품질이 진짜 축이다.

## 4. EXP-D — DGR은 충분통계가 아니다

평균 DGR을 거의 맞춘 두 소스쌍(D_1{2,0} 0.58 vs D_2{8,1} 0.61), 거리 분포도 매칭. DGR이 충분통계라면 무차이여야 한다.

결과: D_1 0.597 vs D_2 0.505 → **+9.2pp, p=8.4e-4**. 오히려 평균 DGR이 약간 높은 쪽이 진다.

범인 분해(C_hi{2,8}와의 교차 비교): 소스 8→0 교체(C_hi vs D_1)는 +3.8pp(p=0.14, 무의미)인데, 소스 2→1 교체(C_hi vs D_2)는 +13.0pp(p=1.2e-6). threading 소스 1은 DGR 0.60으로 상위권인데 정책 학습에는 해로운 데이터를 만든다. "생성이 잘 되는 소스"와 "학습에 좋은 데이터를 만드는 소스"는 강하게 겹치지만 동일하지 않다.

## 5. 종합

1. **transform/밀도 축은 완전 통제 하에서도 null** (EXP-A 등가성 통과, EXP-B far 커버리지 제거도 무해). E2의 transform_uniform null은 교란이 아니라 실제. 큐레이션에서 이 축을 만질 이유가 없다.
2. **소스 품질 축은 거리 매칭 후에도 크고 유의** (stack +20pp, threading +10pp). ancestry 균등화가 해로운 메커니즘(저품질 소스 상향)이 인과로 확정됐고, 필터의 hi-DGR 과대표집 편향은 오히려 이롭다.
3. **DGR은 좋은 1차 프록시지만 불완전** (EXP-D +9.2pp). 소스별 정책 학습 가치를 직접 추정하는 지표가 다음 과제 — attempt-vs-survivor 감사 프레임워크에 "per-source training value" 지표를 추가할 근거.

## 6. 한계

- EXP-A만 6시드, 나머지는 3시드. far 슬라이스는 에피소드 수가 작아(bin4 78 pairs, 90% CI 폭 ±8~12pp) "붕괴 없음" 이상의 정밀 주장은 못 한다.
- stack은 천장(0.96) 근처라 커버리지·밀도 효과가 압축돼 보일 수 있다. EXP-B의 교차 태스크 확인(near_only가 무너졌던 square·coffee에서 nearpad 재현)은 미실행.
- C 계열 arm은 소스 2개 묶음이라 "DGR"과 소스의 다른 속성이 완전히는 안 갈린다 — 그 간극을 정면으로 잰 것이 EXP-D이고, 실제로 정체성 효과가 나왔다.
- bin별 추출이라 arm 안에서 소스 2개의 비율은 정확히 50:50이 아니다(빈별 생존 풀 구성에 따름).
- P0 게이트 기준 stack{4,7} far 평가 상태의 약 15%는 학습 d_pos 지지 밖이며 bin4에 포함돼 있다.
- arm 크기가 실험 패밀리마다 다르므로(500/400/350/250) 패밀리 간 절대 SR 비교는 하지 말 것. 패밀리 안에서는 크기 동일.

## 7. ctrl2 — 다태스크 보강 결과 (2026-08-02)

같은 설계를 4~5개 태스크로 확장했다(시드 301–303, run-level paired t 기준, 소스쌍은 태스크별 feasibility 게이트로 선정). threading의 A/B arm 11런은 서버 사고(디스크 풀 → 학습 실패 → 오케스트레이터 행)로 8런만 늦게 완주해 B 대조가 평가에 못 들어갔다 — 아래 표의 공백. 전체 수치는 `motivation/scripts/ctrl2_analyze.py` + `motivation/data/ctrl2_eval/`로 재현.

| EXP | task | Δ (arm1−arm2) | 시드별 | p(run) | 판정 |
|---|---|---|---|---|---|
| C2 품질 | stack | +19.8 | +20.0/+21.0/+18.5 | 0.0013 | 확증 |
| C2 품질 | stack_three | +21.5 | +22.5/+27.0/+15.0 | 0.026 | 확증 |
| C2 품질 | square | +13.2 | +7.5/+14.0/+18.0 | 0.050 | 확증(경계) |
| C2 품질 | coffee | +12.0 | +15.5/+9.0/+11.5 | 0.024 | 확증 |
| C2 품질 | three_piece | +6.8 | +8.5/+9.0/+3.0 | 0.071 | 방향 일치(바닥 SR 0.13) |
| A 밀도 | stack | −1.5 | −3.0/0.0/−1.5 | 0.23 | null |
| A 밀도 | square | **+6.7 (near우세)** | +7.5/+7.0/+5.5 | **0.008** | 유의 |
| A 밀도 | coffee | **−6.3 (far우세)** | −8.0/−8.0/−3.0 | 0.063 | 경계 |
| A 밀도 | threading | **−7.7 (far우세)** | −7.0/−12.0/−4.0 | 0.082 | 경계 |
| B 커버리지 | stack | 0.0 | +1.0/0.0/−1.0 | 1.0 | null |
| B 커버리지 | square | +1.8 | −7.0/+1.5/+11.0 | 0.76 | null(시드 요동) |
| B 커버리지 | coffee | **+9.7 (far필요)** | +9.0/+7.0/+13.0 | **0.032** | 유의 |
| D DGR충분성 | threading | +9.2 | +12.5/+9.0/+6.0 | 0.040 | 재확인 |
| D DGR충분성 | stack | 0.0 | −2.0/0.0/+2.0 | 1.0 | null |
| D DGR충분성 | square | −1.0 | +4.0/+4.5/−11.5 | 0.87 | null |
| D DGR충분성 | coffee | +1.7 | +0.5/+0.5/+4.0 | 0.29 | null |

**해석.**

1. **소스 품질 인과는 태스크 일반이다 — 이 프로그램의 최강 결과.** 거리 5분위 완전 매칭 하에 소스쌍 교체만으로 5/5 태스크 동방향(+7~+22pp; 부호검정 p=1/16), 3시드 run-level로도 4/5 유의·경계. 1태스크 한정이라는 유일한 약점이 닫혔다.
2. **밀도는 "보편 null"이 아니라 "일관성 없는 태스크 의존"이다.** stack만 null이고 square는 near-heavy가 +6.7(유의), coffee·threading은 반대로 far-heavy가 +6~8(경계). 방향이 갈리므로 큐레이션 규칙은 못 만든다 — "밀도 축을 만질 이유가 없다"는 결론은 유지되지만 근거가 바뀐다: 효과가 없어서가 아니라 **방향을 예측할 수 없어서**다. E2의 8태스크 재균등화 null과도 정합적이다(완만한 재배분은 무효, 3:1 극단 반전에서만 태스크별 ±7pp).
3. **커버리지 문턱은 태스크 의존으로 확정 — "40~60%면 충분" 가설 기각.** coffee에서 far 데이터 제거(B_nearpad)가 −9.7pp(run-level p=0.032)를 만들고, 격차가 자기 소스 기준 far 구간에 집중된다(near +7.5 / mid +5.0 / **far +16.7pp**) — 순수한 커버리지 상실 시그니처. stack은 여전히 무해, square는 시드 요동 null. 즉 far 커버리지가 필요한 태스크(coffee)와 불필요한 태스크(stack)가 공존하고, 문턱은 상수가 아니다.
4. **DGR 불충분성은 threading 한정의 존재 증명으로 확정.** threading +9.2가 재확인됐지만 stack·square·coffee의 DGR-매칭 쌍은 전부 등가 — "DGR이 같으면 대개 비슷하고, 가끔(threading s1 같은) 예외 소스가 있다"가 정확한 형태. DGR은 대부분의 경우 좋은 프록시이고, 예외 탐지가 per-source 지표의 역할이다.

## 8. 재현 경로

- 서버(aidas-l40s): 데이터 `~/mimicgen_jihoonkwon/experiments/motivation_controlled/` (arms/<task>/train.hdf5 + filter key, results/ 39 run, gates/ deval matrix·P0), 오케스트레이터 로그 `~/ctrl_run.log`
- per-episode 평가: `motivation_controlled/arms/<task>_N2/eval/e2_<task>_<arm>_seed<s>.jsonl` — 로컬 사본 `motivation/data/ctrl_eval/`
- arm 구성: `motivation/scripts/ctrl_build_arms.py`, 실행 `motivation/scripts/ctrl_run.sh`
- 이 문서의 모든 수치: `motivation/scripts/ctrl_analyze.py` (CTRL_EVAL_DIR=motivation/data/ctrl_eval 로 실행)
