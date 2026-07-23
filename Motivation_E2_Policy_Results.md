# Motivation E2 — 재균등화 리샘플링이 정책 성능에 주는 영향

> 등방(isotropic)·무회전 재설계(motivation_new) 위에서 진행한 E2 정책 실험의 결과 정리.
> 8개 태스크, 저차원 BC-RNN-GMM, 고정 초기상태 200개 위 paired 평가. **2 seed 시점 결과이며,
> 6 seed까지 확장(103–106 학습)이 별도로 돌고 있어 완주 후 검정력 부분을 보강한다.**

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
(McNemar)할 수 있다.

**d_eval**: 각 평가 장면이 가장 가까운 source demo에서 얼마나 떨어져 있는지를, 학습 때 쓴 것과
똑같은 거리 정의(d_pos)로 매긴 값. 이 값으로 200개 장면을 near / mid / far 3등분해서, 효과가
어느 구간에서 나오는지 본다.

---

## 2. 전체 성공률 (2 seed 평균)

| task | baseline | transform_uniform | ancestry_balanced | Δ transform | Δ ancestry |
|---|---|---|---|---|---|
| square | .665 | **.713** | .698 | **+4.8** | +3.3 |
| coffee | .540 | **.565** | .585 | +2.5 | +4.5 |
| three_piece | .180 | **.218** | .110 | +3.8 | **−7.0** |
| stack | .820 | **.833** | .740 | +1.3 | **−8.0** |
| hammer | .865 | **.868** | .843 | +0.3 | −2.3 |
| stack_three | .630 | .628 | .618 | −0.3 | −1.3 |
| threading | .518 | .510 | .478 | −0.8 | −4.0 |
| mug | .265 | .263 | — | −0.3 | — |

- **transform_uniform**은 8개 중 5개에서 baseline 이상(square·coffee·three_piece·stack·hammer),
  크기는 대체로 1–5%p. 나머지 3개는 −1%p 안쪽으로 사실상 동률.
- **ancestry_balanced**는 합쳐서 마이너스 쪽. stack −8.0, three_piece −7.0처럼 크게 깎이는 경우가
  있다.

전체 성공률만 보면 방향은 있으나 폭이 얕다. 효과가 far 구간에 몰려 있다면 3등분했을 때 그 구간에서
격차가 벌어져야 한다 — 그걸 확인한다.

---

## 3. 구간별 성공률 — 두 재샘플링 vs baseline (near / mid / far)

각 칸은 (arm SR) − (baseline SR), 단위 %p. 오른쪽은 far 구간 및 전체의 (장면, 시드) 단위 paired
McNemar p값.

**transform_uniform − baseline**

| task | near | mid | **far** | p (far) | p (전체) |
|---|---|---|---|---|---|
| square | +7.5 | +5.3 | **+1.5** | 0.89 | 0.13 |
| coffee | 0.0 | +3.8 | **+3.7** | 0.50 | 0.41 |
| three_piece | +1.5 | +3.8 | **+6.0** | 0.26 | 0.20 |
| stack | +4.5 | −3.8 | **+3.0** | 0.61 | 0.68 |
| hammer | +0.7 | −1.5 | **+1.5** | 0.75 | 1.00 |
| threading | +0.8 | +1.5 | **−4.5** | 0.52 | 0.88 |
| stack_three | +2.2 | 0.0 | **−3.0** | 0.70 | 1.00 |
| mug | +3.0 | +1.5 | **−5.2** | 0.35 | 1.00 |

**ancestry_balanced − baseline**

| task | near | mid | **far** | p (far) | p (전체) |
|---|---|---|---|---|---|
| square | +5.3 | +6.1 | **−1.5** | 0.89 | 0.32 |
| coffee | +5.2 | +5.3 | **+3.0** | 0.64 | 0.12 |
| three_piece | −7.5 | −6.0 | **−7.5** | 0.09 | **0.006** |
| stack | −10.5 | −9.1 | **−4.5** | 0.39 | **0.002** |
| hammer | −0.7 | −3.8 | **−2.3** | 0.61 | 0.18 |
| threading | −2.2 | −5.3 | **−4.5** | 0.52 | 0.26 |
| stack_three | −1.5 | −5.3 | **+3.0** | 0.68 | 0.75 |

(mug은 생존 풀이 너무 얇아 ancestry arm을 만들 수 없어 제외.)

ancestry는 **부호가 태스크마다 뚜렷이 갈린다** — square·coffee는 near·mid에서 +5~6%p 돕고,
stack·three_piece는 전 구간에서 −5~10%p 깎는다(둘 다 전체 p<0.01). 두 방향 모두 **차이가 far가
아니라 near·mid에서 가장 크다** — transform과 같은 자리다. (4분위로 더 잘게 본 건 §4.)

참고로 baseline 자체는 거의 모든 태스크에서 near > mid > far로 성공률이 떨어진다 (예: square
.716 → .689 → .590). **d_eval은 난이도 축으로 제대로 작동한다** — source에서 멀리 떨어진 장면일수록
어느 정책에게든 어렵다. 축은 맞다.

---

## 4. 거리 특화(specialization) 검정 — baseline이 가까운 구간을 더 잘하나?

**동기.** baseline의 학습셋은 필터가 남긴 쏠림 탓에 near-source demo 쪽으로 편중돼 있다. "각 arm은
자기 학습 데이터가 몰린 거리에서 더 잘한다"는 특화 논리대로라면, baseline은 near에서,
transform_uniform은 far에서 상대적으로 우세해야 한다. 각 태스크의 d_eval을 4분위로 잘라 이를 직접
확인했다.

**최근접 25%(Q1)에서 arm별 SR과 baseline 대비 격차(transform Δt, ancestry Δa):**

| task | base | trans | Δt | anc | Δa |
|---|---|---|---|---|---|
| threading | **.730** | .660 | −0.070 | .650 | −0.080 |
| three_piece | **.210** | .190 | −0.020 | .120 | −0.090 |
| coffee | **.670** | .660 | −0.010 | .650 | −0.020 |
| hammer | .900 | .900 | 0.000 | .890 | −0.010 |
| stack_three | .740 | .750 | +0.010 | .690 | −0.050 |
| stack | .870 | .890 | +0.020 | .770 | −0.100 |
| square | .730 | .780 | +0.050 | .800 | +0.070 |
| mug | .270 | .320 | +0.050 | — | — |

**transform 특화는 확인되지 않는다.** 최근접 Q1에서 baseline이 transform보다 근소하게 앞선 건
threading(−7%p)·three_piece(−2)·coffee(−1) 세 태스크뿐이고, 나머지 다섯(square +5, mug +5,
stack +2, stack_three +1, hammer 0)은 오히려 transform이 앞서거나 동률이다. 여덟 중 어느 것도
유의하지 않다(전부 p>0.3). **baseline이 near-source 데이터에 편중돼 있는데도 near를 특별히 더
잘하지는 않는다.**

**게다가 transform 우세가 앉는 구간이 태스크마다 다르다.** square는 Q1~Q3에 걸쳐 +5~6%p로 넓게
퍼지고 최원거리 Q4에서만 사그라들지만, three_piece는 반대로 far 쪽(Q1 −2 → Q4 +8), threading은
일관된 방향이 없다. **"한쪽은 near, 한쪽은 far"라는 깔끔한 거리 교차는 어느 태스크에서도 나타나지
않는다.** 태스크마다 우세 위치가 흩어져 평균에서 상쇄되는 것이, §2 aggregate가 얕고 §3 far 격차의
부호가 갈린 근본 이유다.

**demo 균등(ancestry)도 far가 아니라 near·mid에서 작동한다.** 최근접 Q1만 보면 ancestry는 대부분
baseline보다 낮고(stack −10, three_piece −9, threading −8%p) square만 +7이다. 더 중요한 건
ancestry가 크게 깎는 stack·three_piece에서 그 손실이 far가 아니라 near~mid에 걸쳐 있다는 점이다 —
stack은 Q1~Q3에서 각각 −10·−10·−11%p(각 p≈0.06)로 꾸준히 깎이고 최원거리 Q4에서만 무승부(−1)다.
반대로 ancestry가 돕는 coffee는 그 이득이 Q2(near-mid)에 몰려 +16%p(p=0.009, 단 여러 셀 중 하나라
다중검정 보정 후엔 약함)다. 즉 **ancestry의 부호는 태스크마다 갈리지만(coffee 도움 /
stack·three_piece 손해), 작동 구간은 transform과 똑같이 near·mid이고 far는 어느 arm에게든
무승부다.**

**단, 이 자름은 2 seed에서 검정력이 없다.** Q1은 장면 50개 × 2 seed = 표본 100(불일치쌍 ~20–35개)
이라 ±7%p를 유의하게 가려낼 수 없다(threading의 −7%p도 p=0.34). 특화 가설은 점추정상 지지되지
않으나, 2 seed로는 확정도 반증도 못 한다 — 6 seed 확장과 **박스 기하 기반 절대 D0 밴드**(진짜 D0
안에 든 상태만 추림)로 매듭짓는다.

---

## 5. 정직한 결론 두 가지

### (1) "효과가 far 구간에 몰린다"는 예상은 확증되지 않았다

- far 구간 격차의 **부호가 태스크마다 갈린다** (양 5개 / 음 3개). 크기도 작고, transform_uniform 대
  baseline은 **어느 태스크·어느 구간에서도 통계적으로 유의하지 않다** (McNemar p 최솟값이 0.13).
- 가설이 그린 모양(near < mid < far로 점점 벌어짐)에 맞는 건 three_piece(+1.5 → +3.8 → +6.0)와
  coffee 정도다. square는 오히려 near·mid에서 더 벌어지고 far에서 사라진다.
- 즉 transform 거리를 평평하게 만든 재배분이, 정작 먼 구간의 성능을 특별히 되살리지는 못한다.

### (2) 유의한 처치 효과는 ancestry에서만 나오고, 그 방향은 태스크마다 갈린다

전-태스크(whole-task) 단위로 짝지어 본 McNemar에서 통계적으로 유의하게 나온 건 ancestry의 두 개뿐이다.
transform은 어느 태스크에서도 whole-task 유의가 없다.

| 비교 | 판정 | 방향 |
|---|---|---|
| three_piece: baseline vs ancestry | b 34 / **c 62, p = 0.006** | baseline 승 (ancestry가 나쁨) |
| stack: baseline vs ancestry | b 36 / **c 68, p = 0.002** | baseline 승 (ancestry가 나쁨) |

(b = ancestry가 이긴 에피소드 수, c = baseline이 이긴 수.) 다만 방향이 한쪽만은 아니다 — §4에서 봤듯
coffee는 ancestry가 오히려 돕는다(전체 +4.5%p, near-mid에서 +16%p). 정리하면 **source 축
재균등화(ancestry)는 다른 어떤 arm보다 정책을 크게 흔들되, 그 부호가 태스크에 따라 갈려(coffee 도움 /
stack·three_piece 유의한 손해) 안정적인 처방이 못 된다.** net은 음수다. transform은 흔드는 폭 자체가
작아 어느 쪽으로도 유의하지 않다. 이 ancestry 효과는 **거리 이동이 아니라 순수 source 구성에서
온다** — ancestry의 transform 거리 분포는 baseline과 사실상 동일하다(부록 C).

---

## 6. 그래서 무슨 이야기가 되나

깨끗한 승리 서사 — "transform 균등 리샘플링이 정책을, 특히 먼 구간에서 살린다" — 는 **이 데이터와
2 seed로는 성립하지 않는다.** 대신 성립하는 건 이렇다.

> 생성 필터가 만드는 쏠림은 E1에서 실재한다(거리 ↑ → DGR ↓, source 편중). 하지만 남은 데이터를
> 소박하게 다시 균등화하는 것 — transform 축이든 source 축이든 — 은 정책 성능을 안정적으로 되살리지
> 못한다. transform 축 균등화는 흔드는 폭이 작아 어느 쪽으로도 유의하지 않고, **source 축
> 균등화(ancestry)는 정책을 크게 흔들지만 그 부호가 태스크마다 갈려 되레 해로울 때가 많다.**

이건 오히려 더 정직하고 반증 가능한 메시지다. "편향은 분명히 있는데, 뻔한 처방은 듣지 않고
심지어 역효과가 난다"는 결과는, 큐레이션이 공짜가 아니라는 걸 데이터로 보여준다.

---

## 7. 한계와 진행 중인 보강

- **2 seed다.** transform 효과 폭이 1–5%p로 얕은데, 짝지은 표본이 태스크당 400쌍(200장면 × 2시드)
  이라 이 크기 효과를 유의하게 잡기엔 검정력이 모자란다. 예로 square 전체가 b 82 / c 63 (p = 0.13)인데,
  같은 비율이 유지된 채 시드를 6개로 늘리면 p ≈ 0.006까지 내려갈 수 있다.
- **천장에 걸린 태스크**(hammer .87, stack .82)는 애초에 개선 여지가 적어, 여기서 효과가 안 보이는 게
  이상하지 않다.
- **지금 6 seed까지 확장(시드 103–106) 학습이 돌고 있다.** 완주하면 §3–4의 far-bin·paired 표를 6 seed로
  다시 채워, transform 효과가 유의해지는지 / 확실한 노이즈인지 못을 박는다. 이 문서의 해당 절을
  그 결과로 교체·보강할 예정.

---

### 부록 A — 구간별 원자료 (SR)

near / mid / far 각 구간의 arm별 성공률.

| task | base near/mid/far | transform near/mid/far | ancestry near/mid/far |
|---|---|---|---|
| square | .716 / .689 / .590 | .791 / .742 / .605 | .769 / .750 / .575 |
| coffee | .627 / .568 / .425 | .627 / .606 / .463 | .679 / .621 / .455 |
| three_piece | .194 / .174 / .172 | .209 / .212 / .231 | .119 / .114 / .097 |
| stack | .866 / .841 / .754 | .910 / .803 / .784 | .761 / .750 / .709 |
| hammer | .888 / .871 / .836 | .896 / .856 / .851 | .881 / .833 / .813 |
| threading | .634 / .462 / .455 | .642 / .477 / .410 | .612 / .409 / .410 |
| stack_three | .702 / .629 / .560 | .724 / .629 / .530 | .687 / .576 / .590 |
| mug | .284 / .242 / .269 | .313 / .258 / .216 | — |

### 부록 B — 재현 경로

- d_eval 추출: `mnew_deval.py` — 각 고정 장면을 평가 env에 reset하고, 생성 때와 동일한 mimicgen
  env interface(`get_object_poses`)로 물체 위치를 읽어 `nearest_source_distance`로 계산. arm·시드와
  무관하므로 태스크당 1회.
- far-bin + paired: `mnew_farbin.py` — d_eval 3등분, arm별 구간 SR, (장면, 시드) 단위 McNemar 정확검정.
- 거리 분해(4분위): `mnew_quantile.py` — d_eval 4분위별 baseline vs transform / baseline vs ancestry SR + McNemar.
- arm별 학습셋 거리 프로파일: `mnew_armdist.py` — 각 arm이 고른 데모의 d_pos 분포(부록 C).
- 산출물: 태스크별 `e2_arms/<task>_N2/eval/farbin_summary.json`.

### 부록 C — ancestry는 transform 거리 분포를 바꾸지 않는다 (효과는 순수 source 구성)

"source별 균등(ancestry)이 near-source 생존자를 끌어올려 학습셋을 가까운 쪽으로 쏠리게 한다"는
기대를 직접 검증했다. 각 arm이 고른 학습 데모의 transform 거리(d_pos) 분포다 — near% = 최근접
quantile bin(attempted 기준 20%)에 든 비율, 2 seed 풀.

| task | baseline 평균(near%) | ancestry 평균(near%) | transform 평균(near%) |
|---|---|---|---|
| square | .260 (27.3%) | .257 (28.9%) | .270 (20%) |
| stack | .275 (24.7%) | .274 (23.3%) | .285 (20%) |
| three_piece | .352 (28.7%) | .354 (30.4%) | .386 (20%) |
| coffee | .257 (27.1%) | .255 (26.0%) | .270 (20%) |
| threading | .272 (24.1%) | .271 (22.9%) | .279 (20%) |
| stack_three | .283 (24.2%) | .286 (21.5%) | .292 (20%) |
| hammer | .263 (23.5%) | .262 (23.4%) | .271 (20%) |

**ancestry의 거리 분포는 baseline과 사실상 동일하다** — 평균 d_pos 차이가 셋째 자리(≤0.003)이고
near% 차이도 ±2%p 안이며 방향도 일정하지 않다(square·three_piece는 근소하게 더 가깝지만
stack·stack_three·threading은 오히려 더 멀다). 거리를 실제로 옮기는 arm은 transform_uniform뿐이고
(near% 정확히 20%, 평균도 더 큼) 그것은 far 쪽 이동이다.

기대가 빗나간 이유: "생존자가 적은 source는 가까운 데서만 성공한다"는 전제는 source들의 거리
프로파일이 서로 크게 다를 때만 성립하는데, 이 등방·소형 박스에서는 source demo 10개가 비슷한 거리
범위에서 성공해 source를 재배분해도 거리 분포가 거의 움직이지 않는다. 특히 ancestry가 가장 크게 깎는
stack(전체 p=0.002)조차 near%가 baseline보다 **낮다**(23.3 < 24.7) — near-shift로는 손해가 설명되지
않는다. 따라서 **ancestry의 정책 효과는 거리 재배분이 아니라 어떤 source를 얼마나 담느냐(순수 source
구성)에서 온다.** §5-(2)의 "source 축 재균등화가 정책을 흔든다"는, transform 축을 고정한 채 source
축만 토글한 대조로서 성립한다.
