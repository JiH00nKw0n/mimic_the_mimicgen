# Motivation E2 — 재균등화 리샘플링이 정책 성능에 주는 영향

> 등방(isotropic)·무회전 재설계(motivation_new) 위에서 진행한 E2 정책 실험의 결과 정리.
> 8개 태스크, 저차원 BC-RNN-GMM, 고정 초기상태 200개 위 paired 평가. **6 seed 완전판**
> (dataset seed 101–106). 2 seed 시점과 어떻게 달라졌는지도 §5에 함께 기록한다.

---

## 1. 무엇을 물었나

E1에서 확인한 것: source에서 scene까지의 transform 거리가 커질수록 생성 성공률(DGR)이 단조로
떨어지고, 성공만 남기는 필터를 거치면 남는 데이터가 특정 source demo 쪽으로 쏠린다. 그러면
자연스러운 다음 질문은 — **이 쏠림을 학습 전에 되돌려 놓으면 정책이 더 좋아지는가.**

같은 생성 풀에서 500개를 뽑되 뽑는 규칙만 바꾼 세 벌(arm)을 만들어 비교했다.

| arm | 규칙 |
|---|---|
| **baseline** | retained에서 무작위 500 (표준 파이프라인이 내놓는 그대로) |
| **transform_uniform** | transform 거리 5구간 각 100개씩 (거리 축을 평평하게) |
| **ancestry_balanced** | source demo별로 균등 (쏠린 source 축을 평평하게) |

세 arm은 생성 과정·시뮬레이터·source demo가 전부 동일하고 **뽑는 규칙만 다르다.** 학습 설정도
같고, 학습 중 rollout은 끄고 2000 epoch까지 돌린 뒤, 시드를 고정해 만든 **동일한 200개 초기상태**
위에서 각 arm·시드를 굴렸다. 같은 장면을 모든 arm이 공유하므로 에피소드 단위로 짝지어 비교
(McNemar)할 수 있다. paired 표본은 태스크당 200장면 × 6시드 = **1,200쌍**.

**d_eval**: 각 평가 장면이 가장 가까운 source demo에서 얼마나 떨어져 있는지를, 학습 때 쓴 것과
똑같은 거리 정의(d_pos)로 매긴 값. 이 값으로 200개 장면을 near / mid / far 3등분해서, 효과가
어느 구간에서 나오는지 본다.

---

## 2. 전체 성공률 (6 seed 평균)

| task | baseline | transform_uniform | ancestry_balanced | Δ transform | Δ ancestry |
|---|---|---|---|---|---|
| square | .677 | .683 | .695 | +0.7 | +1.8 |
| coffee | .560 | .573 | .560 | +1.2 | +0.0 |
| three_piece | .189 | .216 | .186 | +2.7 | −0.3 |
| stack | .838 | .834 | .763 | −0.4 | **−7.5** |
| hammer | .870 | .848 | .858 | −2.2 | −1.2 |
| stack_three | .626 | .628 | .623 | +0.2 | −0.3 |
| threading | .518 | .523 | .484 | +0.5 | −3.3 |
| mug | .272 | .259 | — | −1.3 | — |

- **transform_uniform**은 대부분 ±1%p 안쪽으로 baseline과 사실상 동률이다. three_piece(+2.7)만
  조금 높고, hammer(−2.2)에서 오히려 낮다.
- **ancestry_balanced**는 합쳐서 마이너스 쪽. stack −7.5가 두드러진다.

---

## 3. 구간별 성공률 — 두 재샘플링 vs baseline (near / mid / far)

각 칸은 (arm SR) − (baseline SR), 단위 %p. 오른쪽은 far 구간 및 전체의 (장면, 시드) 단위 paired
McNemar p값.

**transform_uniform − baseline**

| task | near | mid | **far** | p (far) | p (전체) |
|---|---|---|---|---|---|
| square | +2.0 | 0.0 | **0.0** | 1.00 | 0.73 |
| coffee | +1.2 | 0.0 | **+2.5** | 0.40 | 0.46 |
| three_piece | +2.0 | +4.5 | **+1.5** | 0.64 | 0.10 |
| stack | +1.7 | −2.8 | **−0.3** | 1.00 | 0.81 |
| hammer | −2.5 | −2.8 | **−1.5** | 0.43 | **0.0073** |
| stack_three | +1.5 | −2.0 | **+1.2** | 0.76 | 0.92 |
| threading | +2.0 | −2.8 | **+2.2** | 0.53 | 0.83 |
| mug | +2.7 | −5.3 | **−1.2** | 0.72 | 0.49 |

**ancestry_balanced − baseline**

| task | near | mid | **far** | p (far) | p (전체) |
|---|---|---|---|---|---|
| square | +0.5 | +0.2 | **+4.7** | 0.15 | 0.31 |
| coffee | +0.5 | +1.2 | **−1.8** | 0.57 | 1.00 |
| three_piece | +2.3 | −0.8 | **−2.5** | 0.34 | 0.87 |
| stack | −8.0 | −8.1 | **−6.5** | **0.022** | **<0.001** |
| hammer | −1.5 | −1.2 | **−1.0** | 0.64 | 0.16 |
| stack_three | +1.2 | −2.0 | **−0.3** | 1.00 | 0.89 |
| threading | −1.9 | −5.8 | **−2.3** | 0.55 | 0.083 |

(mug은 생존 풀이 너무 얇아 ancestry arm을 만들 수 없어 제외.)

- **transform vs baseline: 어느 태스크·어느 구간에서도 유의한 이득이 없다.** 전체 검정에서 유일하게
  유의한 건 hammer의 **−2.2%p (p=0.0073) — transform이 오히려 나쁨**(천장 태스크).
- **ancestry: stack에서 near·mid·far 세 구간 모두 유의하게 −6~8%p** 깎는다(전체 p<0.001). 나머지는
  유의하지 않다.
- baseline SR은 거의 모든 태스크에서 near > mid > far로 떨어진다 → **d_eval은 난이도 축으로 유효**
  하지만, 어느 처치도 그 far 구간을 특별히 되살리거나 무너뜨리지 않는다(효과는 구간에 몰리지 않음).

---

## 4. 거리 특화(specialization) 검정 — baseline이 가까운 구간을 더 잘하나?

**동기.** baseline의 학습셋은 필터가 남긴 쏠림 탓에 near-source demo 쪽으로 편중돼 있다. "각 arm은
자기 학습 데이터가 몰린 거리에서 더 잘한다"는 특화 논리대로라면, baseline은 near에서 우세해야 한다.
각 태스크의 d_eval을 4분위로 잘라 최근접 Q1을 본다.

**최근접 25%(Q1)에서 arm별 SR과 baseline 대비 격차(transform Δt, ancestry Δa):**

| task | base | trans | Δt | anc | Δa |
|---|---|---|---|---|---|
| square | .770 | .783 | +1.3 | .770 | 0.0 |
| stack | .860 | .877 | +1.7 | .797 | **−6.3** (p=0.037) |
| three_piece | .180 | .187 | +0.7 | .200 | +2.0 |
| coffee | .683 | .707 | +2.3 | .683 | 0.0 |
| threading | .657 | .693 | +3.7 | .607 | −5.0 |
| stack_three | .713 | .727 | +1.3 | .710 | −0.3 |
| hammer | .897 | .880 | −1.7 | .890 | −0.7 |
| mug | .310 | .327 | +1.7 | — | — |

**특화는 확인되지 않는다.** 최근접 Q1에서 baseline이 transform을 이기는 건 hammer뿐(−1.7, 무유의)
이고, 나머지는 transform이 근소하게 앞서거나 동률이다 — 전부 무유의(p>0.37). baseline이 near-source
데이터에 편중돼 있는데도 near를 특별히 더 잘하지는 않는다. ancestry는 여기서도 stack만 유의하게
깎는다(Q1 −6.3%p, p=0.037). "한쪽은 near, 한쪽은 far"라는 깔끔한 거리 교차는 어느 태스크에서도
나타나지 않는다.

---

## 5. 정직한 결론 — 그리고 2 seed에서 무엇이 달라졌나

### (1) transform_uniform은 정책을 되살리지 못한다 — 2 seed의 우세는 소표본 착시였다

2 seed 시점에는 transform이 8개 중 5개에서 앞서고 square가 +4.8%p까지 벌어져, "거리 균등이 조금
낫다"는 인상을 줬다. **6 seed로 늘리자 그 격차가 0으로 수렴했다.**

| task | Δ transform (2 seed) | Δ transform (6 seed) |
|---|---|---|
| square | +4.8 | **+0.7** |
| coffee | +2.5 | +1.2 |
| three_piece | +3.8 | +2.7 |
| stack | +1.3 | −0.4 |

어느 태스크도 유의하지 않고, 오히려 hammer에서 유의하게 나쁘다(−2.2%p, p=0.0073). 즉 2 seed의
우세는 짝지은 표본이 태스크당 400쌍뿐이라 생긴 노이즈였고, 1,200쌍에서는 사라진다.

### (2) 유의한 처치 효과는 ancestry의 stack 손해 하나로 좁혀진다

전-태스크 McNemar에서 6 seed 기준 유의하게 남는 건 **stack의 ancestry 손해(−7.5%p, p<0.001,
near·mid·far 모두 유의)** 뿐이다. 2 seed에서 함께 유의했던 **three_piece의 ancestry 손해는
사라졌다**(−7.0%p·p=0.006 → −0.3%p·p=0.87). threading은 경계선(−3.3%p, p=0.083)으로 남는다.

| 비교 | 2 seed | 6 seed |
|---|---|---|
| stack: ancestry | c68/b36, p=0.002 | c210/b120, **p<0.001** (유지·강화) |
| three_piece: ancestry | c62/b34, p=0.006 | c159/b155, p=0.87 (**사라짐**) |

### 그래서 무슨 이야기가 되나

> 생성 필터가 만드는 쏠림은 E1에서 실재한다(거리 ↑ → DGR ↓, source 편중). 하지만 남은 데이터를
> 소박하게 다시 균등화하는 것은 정책 성능을 되살리지 못한다. **transform 축 균등화는 효과가 없고**
> (2 seed에서 보였던 이득은 표본을 늘리자 소멸), **source 축 균등화(ancestry)는 정책을 흔들지만 그
> 방향이 대체로 해로우며 stack에서는 유의하게 성능을 깎는다.**

편향은 분명히 있는데 뻔한 처방은 듣지 않고, source 축을 건드리면 오히려 위험하다 — 큐레이션이
공짜가 아니라는 걸 데이터로 보여준다. 6 seed 확장이 한 일은 이 메시지를 굳힌 것이다: 검정력을 4배로
늘려 **transform의 가짜 이득을 걷어내고 ancestry 손해를 진짜인 곳(stack)으로 좁혔다.**

---

## 6. 남은 한계

- **저성공·천장 태스크의 정보량이 적다.** three_piece(SR 0.19)는 바닥이라 처치 간 차이를 재기 어렵고,
  hammer(0.87)·stack(0.84)은 천장이라 개선 여지가 좁다. 신호는 중간 SR 태스크에서 가장 또렷하다.
- **효과는 여전히 작다.** 유의하게 남은 stack ancestry 손해 외에는 대부분 ±2%p 안쪽이다. "무해에
  가깝다"는 결론 자체가 결과이지만, 더 큰 스킥(원래 방향성 D2 변형처럼 TV가 큰 셋업)에서는 그림이
  달라질 수 있다(부록 C).
- **저차원 low-dim 한정.** image 관측 정책에서 같은지는 별도 확인이 필요하다(PLAN §2.5).

---

### 부록 A — 구간별 원자료 (SR, 6 seed)

near / mid / far 각 구간의 arm별 성공률.

| task | base near/mid/far | transform near/mid/far | ancestry near/mid/far |
|---|---|---|---|
| square | .749 / .715 / .567 | .769 / .715 / .567 | .754 / .717 / .614 |
| coffee | .672 / .581 / .428 | .684 / .581 / .453 | .677 / .593 / .410 |
| three_piece | .179 / .197 / .192 | .199 / .242 / .207 | .202 / .189 / .167 |
| stack | .871 / .846 / .799 | .888 / .818 / .796 | .791 / .765 / .734 |
| hammer | .896 / .866 / .848 | .871 / .838 / .833 | .881 / .854 / .838 |
| threading | .619 / .485 / .448 | .639 / .457 / .470 | .600 / .427 / .425 |
| stack_three | .687 / .641 / .550 | .702 / .621 / .562 | .699 / .621 / .547 |
| mug | .294 / .285 / .236 | .321 / .232 / .224 | — |

### 부록 B — 재현 경로

- d_eval 추출: `mnew_deval.py` — 각 고정 장면을 평가 env에 reset하고, 생성 때와 동일한 mimicgen
  env interface(`get_object_poses`)로 물체 위치를 읽어 `nearest_source_distance`로 계산. arm·시드와
  무관하므로 태스크당 1회.
- far-bin + paired: `mnew_farbin.py` — d_eval 3등분, arm별 구간 SR, (장면, 시드) 단위 McNemar 정확검정.
- 거리 분해(4분위): `mnew_quantile.py` — d_eval 4분위별 baseline vs transform / vs ancestry SR + McNemar.
- arm별 학습셋 거리 프로파일: `mnew_armdist.py` — 각 arm이 고른 데모의 d_pos 분포(부록 C).
- 시드 확장(2→6): `mnew_addseeds.py`(filter key 추가) + `mnew_seeds.sh`(학습·평가·분석 오케스트레이터).
- 전수 슬라이싱: `mnew_slice.py` — transform vs baseline을 데실·소스별·연속·효율·풀링·다중검정으로 모두 쪼갬(부록 D).
- 산출물: 태스크별 `e2_arms/<task>_N2/eval/{eval_summary,farbin_summary}.json`, `e2_arms/slice_results.json`.

### 부록 C — ancestry는 transform 거리 분포를 바꾸지 않는다 (효과는 순수 source 구성)

"source별 균등(ancestry)이 near-source 생존자를 끌어올려 학습셋을 가까운 쪽으로 쏠리게 한다"는
기대를 직접 검증했다. 각 arm이 고른 학습 데모의 transform 거리(d_pos) 분포다 — 평균과 균등 대비
TV(=구간을 옮겨야 하는 데모 비율), 6 seed 풀.

| task | baseline 평균 · TV | ancestry 평균 · TV | transform 평균 · TV |
|---|---|---|---|
| square | .259 · 0.074 | .259 · 0.077 | .271 · 0.000 |
| stack | .275 · 0.055 | .278 · 0.044 | .285 · 0.000 |
| three_piece | .353 · 0.153 | .353 · 0.144 | .385 · 0.000 |
| coffee | .258 · 0.086 | .256 · 0.092 | .271 · 0.000 |
| threading | .272 · 0.043 | .272 · 0.042 | .279 · 0.000 |
| stack_three | .280 · 0.068 | .285 · 0.047 | .292 · 0.000 |
| hammer | .265 · 0.040 | .261 · 0.072 | .271 · 0.000 |

**ancestry의 거리 분포는 baseline과 사실상 동일하다** — 평균 d_pos 차이가 셋째 자리(≤0.005)이고
TV도 baseline과 비슷하다. 거리를 실제로 옮기는 arm은 transform_uniform뿐이고(TV≈0, 즉 균등) 그것은
far 쪽 이동이다. 따라서 **ancestry의 정책 효과(특히 stack 손해)는 거리 재배분이 아니라 어떤 source를
얼마나 담느냐(순수 source 구성)에서 온다.**

또한 이 TV(baseline, 균등)가 곧 **transform 축 survivor skew의 크기**인데, 등방·소형 박스 재설계에서는
0.04~0.15로 얕다(원래 방향성 D2 변형은 계획상 0.10~0.30 예상). 처치가 건드릴 skew 자체가 작은 것이,
정책 효과가 작게 나오는 근본 원인이다.

### 부록 D — 전수 슬라이싱: transform_uniform은 어느 컷에서도 baseline과 다르지 않다

§3–4의 tercile·quartile보다 촘촘하게, transform_uniform과 baseline을 가능한 모든 축으로 쪼개
차이가 숨어 있는지 확인했다(`mnew_slice.py`, 6 seed, paired unit = (task, reset, seed)).

**최대검정력 앵커.** 8태스크를 모두 풀링하면 9,600 페어 — base 0.5686 vs treat 0.5704,
b1449/c1432, **McNemar p=0.766.** 이 크기면 ~1.5%p 실효과도 잡히는데 잡히지 않는다.

**모든 컷:**
- **d_eval 데실(태스크 × 10구간 = 80셀):** p<0.05가 8개 나오나 우연 기대 4.0개의 2배에 불과하고
  방향이 갈린다(transform 우세 3 / baseline 우세 5). 가족 최소 p(stack d0, 0.0117)를 Bonferroni
  보정하면 **0.936** — 전멸. "far일수록 유리" 같은 단조 구배도 없다(인접 구간에서 부호가 뒤집힘).
- **가장 가까운 source demo별(≈86층):** raw p들이 보정 문턱을 못 넘는다.
- **연속(불일치쌍 d_eval Mann-Whitney):** 전 태스크 무의 — transform이 이기는 장면과 baseline이
  이기는 장면의 거리가 다르지 않다.
- **효율(공동 성공 에피소드의 스텝, Wilcoxon):** 전 태스크 무의 — 성공률뿐 아니라 도달 속도도 같다.
- **시드 방향:** 대부분 갈린다(3+/3−, 4+/2−).

**유일하게 유의한 태스크 — 그것도 반대 방향.** hammer는 transform이 **더 나쁘다**(b34/c61,
p=0.0073, 6시드 중 5시드). 시드 방향은 일관되나 낙폭 2.25%p로 얕고, 8태스크 최소 p라
Bonferroni(0.0073 × 8 = 0.058)를 못 넘는다. "transform이 돕는다"의 증거가 아니라 "특정 태스크에서
살짝 해로울 수 있다"는 반대쪽 경계 신호다.

**추가 시드 값어치가 있는 유일한 후보.** three_piece는 transform 쪽 6시드 중 5시드 일관(+2.7%p,
overall p=0.10), 저성공이라 천장효과가 없다. 지금은 무의지만 8태스크 중 방향 일관성이 가장 깨끗해,
확정하려면 시드를 더 늘려야 한다.

**적대적 검증.** 6개 검사(prosecutor) 에이전트가 각 각도(풀링·태스크별·near/far 데실·소스·효율)에서
"차이를 찾아내라"고 후보를 끌어내고, 회의적 판정자가 다중검정·시드안정성·풀링모순으로 걸렀다
(총 13 에이전트, 만장일치). **진짜 차이 판정 = 0개.** 앵커 p=0.766과 함께, **소박한 transform
재균등화는 어느 구간에서도 정책을 바꾸지 않는다**가 이 데이터에서 확립된다.
