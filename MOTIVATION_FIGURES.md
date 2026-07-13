# Bias in Synthetic Data Generation - Motivation 실험

## Recall

### Backgrounds

- **다양한 상황에 맞는 동작을 학습하기 위해 많은 데이터가 필요하다** — 모방학습은 물체 자세·배치·주변 클러터에 따라 필요한 동작이 달라지므로, 소수의 데모만으로는 deploy(배포) 시 마주하는 다양한 상황과 그에 따른 다양한 동작을 충분히 학습하기 어렵고, 따라서 조건별 변화를 담은 크고 다양한 데모 데이터셋이 필요하다.
- **많은 데이터를 만들기 위해 합성 데이터를 활용한다** — 실제 로봇으로 이런 데이터를 직접 모으는 것은 비용과 시간이 많이 들어, 소수의 human 데모를 transform·stitch·retain 파이프라인으로 재활용해 다양한 조건의 합성 데모를 자동 생성하는 연구가 활발하다.
- **합성 데이터도 충분히 다양한 상황을 학습할 수 있어야 한다** — 이때 학습에는 task에 성공해 retain된 데모만 쓰이므로, retained 데이터셋이 조건별 다양성을 충분히 보존해야 한다.
- **그러나 실제로 합성 데이터가 충분히 다양한 상황을 다루는가?** — 그러나 기존 방법이 이 다양성을 보존하는지는 불분명하며, 널리 쓰이는 data generation rate(생성 rollout 중 retain된 비율)는 "얼마나 자주" 남는지만 잴 뿐 "무엇이" 남는지는 보지 못한다.
- **성공한 데이터만 남기는 과정은 최종 데이터셋 분포에 편향을 만들 수 있다** — 그 결과 data generation rate가 비슷해도 retained 분포와 downstream 정책 성능이 크게 달라질 수 있어, retention이 retained 분포를 어떻게 형성하는지 이해하는 것이 정책 성능 설명에 필수적이다.
#### 연구 질문

> 효과적인 downstream 모방학습에 필요한 retained 분포의 성질은 무엇이며, task 성공 기준으로 생성 rollout을 필터링하는 것이 그 성질을 보존하는가?

#### 가설

다음과 같은 가설을 세웠다.

- 성공한 rollout만 남기는 retention이, 생성기가 다양한 물체 조건과 source 선택으로 rollout을 시도하더라도 task 성공 가능성이 높은 조건의 데모를 과대표집하여 원래 탐색한 다양성을 보존하지 못한다.
- 이 분포 변화가 downstream 정책 성능 차이로 이어진다.

### MimicGen Pipeline ([Mandlekar et al. 2023](https://arxiv.org/abs/2310.17596))

<img src="figures/report/system_v5.png" width="100%">

**Figure 2: MimicGen System Pipeline.** (left) MimicGen first parses the demos from the source dataset into segments, where each segment corresponds to an object-centric subtask. (right) Then, to generate new demonstrations for a new scene, MimicGen generates and follows a sequence of end-effector target poses for each subtask by (1) choosing a segment from a source demonstration (chosen segments shown with blue border in figure above), (2) transforming it for the new scene, and (3) executing it.

이 파이프라인은 세 가지 핵심 단계로 구성되어 있다.

- **Transform**: source segment를 새로운 object configuration에 맞춘다.
- **Stitch**: transformed subtask segments를 interpolation과 execution으로 연결한다.
- **Retain**: task success 이후에만 trajectory를 dataset에 추가한다.

## 금주 진행 상황

이번 주에는 가설의 첫 부분, 즉 "task 성공 가능성이 높은 조건의 데모를 과대표집하여 원래 탐색한 다양성을 보존하지 못한다"를 살펴봤다. 이때 다양성을 이루는 요소(factor) 중 하나로 초기조건(initial condition), 즉 장면에서 각 물체가 놓이는 처음 위치·자세를 탐구했다. 다시 말해, "성공해서 남는 합성 데모가 정말 '쉬운' 초기조건 쪽으로 쏠리는가?"를 실험해본 것이다.

여기서 '쉬운' 초기조건이란 합성 장면에서의 물체 배치(initial condition)가 원본 source 데모의 물체 배치와 비슷한 경우를 말한다. 원본과 비슷할수록 원본 궤적을 조금만 변형(transform)하면 되니 생성이 성공하기 쉬울 것이라는 생각이다.

이를 확인하려고 MimicGen이 데모와 주석을 잘 제공하는 task 4개(Square, Three Piece Assembly, Threading, Coffee Prep)를 골라, 각 task의 D0/D1/D2 초기조건 분포에서 물체 배치를 샘플링해 합성 데모를 생성하고 그 과정을 전부 기록했다.

구체적으로 던진 질문은 이것이다.

> source의 물체 위치와 합성 데이터의 물체 위치 차이(transform)가 작을수록 데이터 생성 성공률(DGR)이 높고, 차이가 클수록 DGR이 낮은가?

실험 결과, 대체로 그런 경향이 나타났다. transform이 작은(원본에 가까운) 분포일수록 DGR이 높고, transform이 커질수록 DGR이 낮아졌다. 아래에서는 이 실험의 대상 task와 초기조건 분포 설정, 그리고 transform에 따른 DGR 변화와 성공·실패 분포를 순서대로 보여준다.

## Task Overview

### Square
정사각형 너트를 집어 고정된 정사각형 페그(기둥)에 끼운다.

<img src="figures/report/task_Square.png" width="70%">

### Three Piece Assembly
베이스 위에 두 조각을 순서대로 끼워 구조물을 조립한다.

<img src="figures/report/task_ThreePieceAssembly.png" width="70%">

### Threading
바늘을 집어 삼각대(tripod)의 구멍에 통과시킨다.

<img src="figures/report/task_Threading.png" width="70%">

### Coffee Prep
커피 팟을 집어 커피 머신에 넣고 뚜껑을 닫는다.

<img src="figures/report/task_CoffeePrep.png" width="70%">

## Distribution Overview

- 데모(source demo)는 10개 세팅입니다.
- 데모는 D0 분포에서 샘플링되었습니다.
- 합성 데이터는 D0 / D1 / D2 분포에서 초기조건을 샘플링해 생성합니다.

### Square

| | D0 | D1 | D2 |
|---|---|---|---|
| nut x | [-0.11, -0.11] | [-0.11, 0.11] | [-0.25, 0.25] |
| nut y | [0.11, 0.23] | [-0.25, 0.25] | [-0.25, 0.25] |
| nut yaw | 360° | 360° | 360° |
| peg x | 0.23 (고정) | [-0.10, 0.30] | [-0.25, 0.25] |
| peg y | 0.10 (고정) | [-0.20, 0.20] | [-0.25, 0.25] |
| peg yaw | 고정 | 고정 | 90° |

<img src="figures/report/overlay_square.png" width="55%">

### Three Piece Assembly

| | D0 | D1 | D2 |
|---|---|---|---|
| base x | 0.00 (고정) | [-0.22, 0.22] | [-0.22, 0.22] |
| base y | 0.00 (고정) | [-0.22, 0.22] | [-0.22, 0.22] |
| base yaw | 고정 | 고정 | 90° |
| piece_1 x | [-0.22, 0.22] | [-0.22, 0.22] | [-0.22, 0.22] |
| piece_1 y | [-0.22, 0.22] | [-0.22, 0.22] | [-0.22, 0.22] |
| piece_1 yaw | 고정 | 고정 | 180° |
| piece_2 x | [-0.22, 0.22] | [-0.22, 0.22] | [-0.22, 0.22] |
| piece_2 y | [-0.22, 0.22] | [-0.22, 0.22] | [-0.22, 0.22] |
| piece_2 yaw | 고정 | 고정 | 180° |

<img src="figures/report/overlay_three_piece_assembly.png" width="55%">

### Threading

| | D0 | D1 | D2 |
|---|---|---|---|
| needle x | [-0.20, -0.05] | [-0.20, 0.05] | [-0.20, 0.05] |
| needle y | [0.15, 0.25] | [0.15, 0.25] | [-0.25, -0.15] |
| needle yaw | 60° | 239° | 239° |
| tripod x | 0.00 (고정) | [-0.10, 0.15] | [-0.10, 0.15] |
| tripod y | -0.15 (고정) | [-0.20, -0.10] | [0.10, 0.20] |
| tripod yaw | 고정 | 120° | 120° |

<img src="figures/report/overlay_threading.png" width="55%">

### Coffee Prep

| | D0 | D1 | D2 |
|---|---|---|---|
| machine x | 0.00 (고정) | [0.05, 0.15] | [-0.05, 0.05] |
| machine y | -0.10 (고정) | [-0.20, -0.10] | [0.10, 0.20] |
| machine yaw | 고정 | 89° | 89° |
| pod x | [-0.13, -0.07] | [-0.20, 0.05] | [-0.20, 0.05] |
| pod y | [0.17, 0.23] | [0.17, 0.30] | [-0.30, -0.17] |
| pod yaw | 고정 | 고정 | 고정 |

<img src="figures/report/overlay_coffee.png" width="55%">

## Data Generation Rate vs Transform

<img src="figures/report/line_square_square_nut.png" width="32%"> <img src="figures/report/line_square_square_peg.png" width="32%"> <img src="figures/report/line_square_SUM.png" width="32%">

<img src="figures/report/line_three_piece_assembly_base.png" width="24%"> <img src="figures/report/line_three_piece_assembly_piece_1.png" width="24%"> <img src="figures/report/line_three_piece_assembly_piece_2.png" width="24%"> <img src="figures/report/line_three_piece_assembly_SUM.png" width="24%">

<img src="figures/report/line_threading_needle.png" width="32%"> <img src="figures/report/line_threading_tripod.png" width="32%"> <img src="figures/report/line_threading_SUM.png" width="32%">

<img src="figures/report/line_coffee_coffee_machine.png" width="32%"> <img src="figures/report/line_coffee_coffee_pod.png" width="32%"> <img src="figures/report/line_coffee_SUM.png" width="32%">

## Success by Transform Offset (D2)

<img src="figures/report/conc_square_square_nut.png" width="45%"> <img src="figures/report/conc_square_square_peg.png" width="45%">

<img src="figures/report/conc_three_piece_assembly_base.png" width="32%"> <img src="figures/report/conc_three_piece_assembly_piece_1.png" width="32%"> <img src="figures/report/conc_three_piece_assembly_piece_2.png" width="32%">

<img src="figures/report/conc_threading_needle.png" width="45%"> <img src="figures/report/conc_threading_tripod.png" width="45%">

<img src="figures/report/conc_coffee_coffee_machine.png" width="45%"> <img src="figures/report/conc_coffee_coffee_pod.png" width="45%">

## Source-Demo Ancestry Bias

- 위의 실험을 통해 transform이 커지면 데이터 생성 비율(DGR)이 낮아진다는 것을 확인했다. 이는 학습에 쓰이는 합성 데이터가 source demo와 유사한 초기 조건 쪽으로 편향될 수 있음을 시사한다.
- 또 하나의 흥미로운 발견은, transform이 커질수록 성공해서 남는 합성 데이터가 특정 source demo로부터 주로 생성된다는 점이다(Ancestry Bias).
- MimicGen류 방법론은 source의 움직임을 rule-based로 변환해 사용한다; 따라서 이 Ancestry Bias는 초기 조건(initial condition)의 편향이 행동 궤적(action trajectory)의 편향으로 이어질 수 있음을 의미한다.

이를 확인하기 위해, 합성 데이터 분포(D0/D1/D2)별로 각 source demo가 합성 시도에 활용된 횟수와 그중 성공한 횟수·비율을 집계했다.

### Square

**D1**

| src | attempted | retained | succ% | ret share% |
|---|---|---|---|---|
| s5 | 67 | 53 | 79% | 23% |
| s2 | 50 | 33 | 66% | 14% |
| s0 | 56 | 35 | 62% | 15% |
| s6 | 46 | 24 | 52% | 10% |
| s4 | 67 | 33 | 49% | 14% |
| s1 | 20 | 8 | 40% | 3% |
| s9 | 47 | 17 | 36% | 7% |
| s7 | 35 | 10 | 29% | 4% |
| s3 | 41 | 10 | 24% | 4% |
| s8 | 71 | 7 | 10% | 3% |

- retained의 top-3 source 비중: **53%** (attempted 41%)

**D2**

| src | attempted | retained | succ% | ret share% |
|---|---|---|---|---|
| s6 | 46 | 28 | 61% | 17% |
| s1 | 42 | 21 | 50% | 13% |
| s2 | 55 | 25 | 45% | 15% |
| s5 | 50 | 22 | 44% | 13% |
| s0 | 55 | 19 | 35% | 12% |
| s9 | 38 | 13 | 34% | 8% |
| s3 | 64 | 14 | 22% | 9% |
| s7 | 36 | 7 | 19% | 4% |
| s4 | 62 | 10 | 16% | 6% |
| s8 | 52 | 4 | 8% | 2% |

- retained의 top-3 source 비중: **46%** (attempted 36%)

<img src="figures/report/anc_square.png" width="90%">

### Three Piece Assembly

**D1**

| src | attempted | retained | succ% | ret share% |
|---|---|---|---|---|
| s7 | 57 | 23 | 40% | 15% |
| s4 | 43 | 16 | 37% | 10% |
| s8 | 55 | 20 | 36% | 13% |
| s3 | 43 | 15 | 35% | 9% |
| s9 | 49 | 17 | 35% | 11% |
| s0 | 54 | 18 | 33% | 11% |
| s2 | 62 | 19 | 31% | 12% |
| s6 | 47 | 13 | 28% | 8% |
| s5 | 44 | 10 | 23% | 6% |
| s1 | 46 | 7 | 15% | 4% |

- retained의 top-3 source 비중: **39%** (attempted 35%)

**D2**

| src | attempted | retained | succ% | ret share% |
|---|---|---|---|---|
| s3 | 43 | 17 | 40% | 12% |
| s7 | 57 | 22 | 39% | 16% |
| s8 | 55 | 21 | 38% | 15% |
| s0 | 54 | 19 | 35% | 13% |
| s4 | 43 | 15 | 35% | 11% |
| s5 | 44 | 14 | 32% | 10% |
| s2 | 62 | 13 | 21% | 9% |
| s9 | 49 | 9 | 18% | 6% |
| s6 | 47 | 6 | 13% | 4% |
| s1 | 46 | 5 | 11% | 4% |

- retained의 top-3 source 비중: **44%** (attempted 35%)

<img src="figures/report/anc_three_piece_assembly.png" width="90%">
