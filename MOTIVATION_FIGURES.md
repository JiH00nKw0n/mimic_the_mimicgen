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
