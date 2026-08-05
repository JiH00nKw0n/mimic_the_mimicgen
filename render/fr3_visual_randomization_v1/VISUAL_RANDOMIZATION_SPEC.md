# Visual Randomization Specification

## 1. 목표

목표는 무작위로 보이는 simulation을 만드는 것이 아니라, 실제 연구실 관측 분포 주변을 충분히 덮으면서 task identity와 geometry cue를 보존하는 것입니다.

분포는 다음 세 층으로 구성합니다.

| Profile | 비율 | 의미 | 기준색 scale | Indoor HDRI | Light intensity |
|---|---:|---|---|---|---:|
| `nominal_lab` | 50% | 현재 연구실과 가장 가까운 좁은 분포 | 0.965 / 1.0 / 1.035 | `studio_small_04_1k.hdr` | 1100 |
| `lab_variation` | 40% | 같은 연구실의 조명·표면 편차 | 0.90 / 1.0 / 1.10 | `studio_small_07_1k.hdr` | 1350 |
| `stress_tail` | 10% | identity를 유지하는 강한 indoor tail | 0.82 / 1.0 / 1.18 | `empty_workshop_1k.hdr` | 1650 |

Profile은 episode별 label로 저장합니다. DomeLight/HDRI는 global prim이므로 하나의 tiled simulation process 안에서 섞지 않고 profile별 process/shard를 분리합니다.

## 2. 고정하는 것

- 연구실 table geometry와 placement
- 회색 카펫 floor의 의미와 texture family
- FR3 공식 articulation/controller/physics
- FR3 full-arm 외형
- cube identity: cube 1 red, cube 2 blue, cube 3 near-black
- 카메라 parent semantic과 nominal extrinsic/intrinsic
- 카메라 3개: third-person 2개 + wrist 1개
- 16:9 시야와 동기화된 10 Hz observation/action contract

오른쪽의 사용하지 않는 빈 robot pedestal는 RGB table wrapper에서 숨깁니다. 충돌 및 task geometry는 state task와 동일하게 유지합니다.

## 3. episode마다 randomize하는 것

### 색상과 재질

기준 RGB:

- gripper: `(0.93, 0.93, 0.91)`
- cube 1: `(0.72, 0.06, 0.04)`
- cube 2: `(0.03, 0.17, 0.55)`
- cube 3: `(0.04, 0.04, 0.04)`
- table: `(0.84, 0.83, 0.77)`
- side/back curtains: `(0.48, 0.49, 0.48)`
- front curtain: `(0.88, 0.87, 0.83)`

각 profile의 scale 세 값 중 하나를 기준색에 곱하고 `[0,1]`로 clamp합니다. 임의 RGB 전체 범위나 arbitrary texture를 사용하지 않습니다.

공통 재질 범위:

- gripper/cubes: roughness 0.25–0.75, metallic 0–0.03, specular 0.18–0.50
- table: roughness 0.45–0.88, metallic 0–0.05, specular 0.18–0.50
- side/back curtains: roughness 0.72–1.0, metallic 0–0.02, specular 0.08–0.25
- front curtain: roughness 0.65–0.96, metallic 0–0.01, specular 0.08–0.30

FR3 arm 전체를 recolor하지 않습니다. 실제 replacement hand/finger facade에 해당하는 gripper mesh만 위 범위로 randomize합니다.

### 카메라

카메라 pose와 wrist focal은 `config/camera_nominal_measured_ranges.yaml`을 단일 source of truth로 사용합니다.

- position: nominal translation 중심의 uniform ball
- rotation: nominal rotation 중심의 rotation-vector ball
- focal: nominal focal length × measured uniform scale
- parent frame: calibration YAML의 semantic을 유지

모든 profile에서 동일한 measured camera range를 사용합니다. Profile 차이는 카메라가 아니라 appearance/light coverage입니다.

## 4. process 시작 시 한 번 randomize하는 것

Floor는 tiled env 밖의 global prim이므로 process startup에서 한 번만 설정합니다.

- texture: `Carpet_Gray_BaseColor.png`
- tint: 0.72–0.92 grayscale
- texture scale: 1.5–2.5
- roughness: 0.88–1.0
- metallic: 0–0.01
- specular: 0.05–0.18

HDRI와 DomeLight intensity도 profile process마다 고정합니다. episode reset에서 바꾸면 다른 env의 진행 중 observation이 갑자기 변하므로 금지합니다.

## 5. 데이터 분포 생성

80,000 successful episodes 기준:

- nominal_lab: 40,000
- lab_variation: 32,000
- stress_tail: 8,000

현재 저장 포맷은 3-camera H.264, 320×180, 10 fps입니다. 성공은 5 consecutive success confirmation 후 즉시 종료하며 확인 이후 padding frame은 저장하지 않습니다.

## 6. 필수 runtime gate

- 모든 카메라·모든 env에서 foreign robot/object pixel 최대 0
- instance-id mapping이 매 step 완전함
- RGB standard deviation 최솟값 10 이상
- camera shape 정확히 `(180, 320, 3)`
- 모든 저장 video가 H.264, yuv420p, 320×180, 10 fps
- profile episode 수가 정확히 50/40/10
- episode 안에서 material/HDRI가 불연속적으로 변하지 않음
- 물체별 색 identity가 사람이 보아 구분 가능함

## 7. 새 task로 일반화할 때

그대로 유지할 것은 profile 계층, global/per-episode scope 분리, measured camera range, identity-preserving 원칙, quantitative isolation gate입니다.

교체할 것은 task asset geometry, object nominal colors, camera parent prim mapping, 작업 공간을 반영한 lab reference image입니다. 새 task의 물체 identity나 성공 cue를 파괴하는 randomization은 추가하지 마십시오.

