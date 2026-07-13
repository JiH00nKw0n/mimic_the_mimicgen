# Motivation Figures

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

<img src="figures/report/anc_three_piece_assembly.png" width="90%">
