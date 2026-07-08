# synthgen — SART · CP-Gen 을 우리 Isaac Lab + cuRobo 합성데이터 파이프라인에 통합

`robot_data workspace — augmentation_methods/INTEGRATION_PLAN.md` 의 설계를 **실행 가능한 스켈레톤**으로 구현한 것.
SART 와 CP-Gen 을 **하나의 SkillGen-family 생성기의 교체 가능한 부품**으로 둔다.

> 📌 **실제로 돌려서 검증한 결과물은 [`mimicgen_substrate/`](mimicgen_substrate/README.md) 참고.**
> robosuite MimicGen 위에서 SART(DGR 55%, 접근 다양성)와 CP-Gen(DGR 35%, 크기 일반화)을 실제로 붙여
> 합성 데이터를 뽑고 falsifiable 지표로 검증했다. 아래 `synthgen/` 은 Isaac Lab/cuRobo 를 겨냥한 설계 스켈레톤.

- 순수 파이썬 코어(알고리즘)는 **지금 바로 실행/테스트**된다(mock 백엔드).
- Isaac Lab / cuRobo 연동부는 **가드된 스텁 + TODO** 로 두었고, 실제 실행 시 이 스텁만 채우면 된다.
  알고리즘 코드(SART/CP-Gen)는 백엔드가 뭐든 **그대로**다.

## 지금 바로 돌려보기 (설치 불필요, numpy 만 있으면 됨)
```bash
cd integration
export PYTHONPATH=.
python tests/test_smoke.py          # 7 smoke tests
python examples/run_sart_augment.py # SART 국소 증강 (mock)
python examples/run_cpgen_generate.py # CP-Gen transform + SART boost (mock)
```

## 아키텍처 — 한 파이프라인, 두 부품

```
                 ┌─────────────────── SkillGenPipeline.generate ───────────────────┐
 새 scene 샘플 →  │ (2) segment transform      (3) transit stitch     (4) replay+filter │
                 │   rigid  = SE(3)            cuRobo MotionPlanner   execute_trajectory │
                 │   keypoint = CP-Gen(+scale)                        (IK→collision→step) │
                 │                                                                        │
                 │   insert 스킬 → SART 국소 증강(SartAugmentor): 구 샘플→수렴→IK→충돌→replay │
                 └────────────────────────────────────────────────────────────────────────┘
                                   ↓ success 만 기록
                            DataWriter (HDF5 / Isaac Lab Mimic)
```

- **CP-Gen** = `transform_mode='keypoint'` : object-centric 변환에 **geometry(scale) 샘플**을 더해
  gear/peg 의 크기 다른 인스턴스까지 커버. 삽입/정렬 구간은 자동으로 **좁은 범위 + scale 고정**.
- **SART** = insert 스킬에만 붙는 **국소 정밀 증강**. 원래는 사람 주석 구였던 안전장치를
  **cuRobo 충돌체크(hand-camera point cloud)** 로 대체 가능 → 원본보다 강함. 우리 success filter 통과.

## 인터페이스(백엔드 교체 지점) — `synthgen/interfaces.py`
| Protocol | mock (지금) | 실제 (우리 스택) |
|---|---|---|
| `SimEnv` | `MockEnv` | `IsaacLabEnv` (isaaclab_env.py) |
| `IKSolver` | `MockIK` | `CuroboIK` (curobo_backend.py) |
| `MotionPlanner` | `MockPlanner` | `CuroboPlanner` |
| `CollisionChecker` | `MockCollision` | `CuroboCollision` (+point cloud) |
| `DataWriter` | `InMemoryDataWriter` | `HDF5DataWriter` (data_schema.py) |

실제 실행 = 위 오른쪽 4개 클래스의 `NotImplementedError` 를 채우는 것. 알고리즘 파일
(`sart_augmentor.py`, `cpgen_transform.py`, `pipeline.py`)은 건드리지 않는다.

## 데모 영상 (source ↔ synthetic)
두 방법론의 원본 데모와 합성 데이터를 나란히 본다: **[media/README.md](media/README.md)**
- CP-Gen: `media/cpgen/{source,synthetic}/` — peg-in-hole / assembly / threading
- SART:   `media/sart/{source,synthetic}/` — teleop insert / self-augmented insert

## 파일 맵
```
synthgen/
  math_utils.py       SE(3)/quaternion (numpy 전용)
  skills.py           Demo/Waypoint/SkillSegment/SkillType, insert 판별, from_skillgen()
  interfaces.py       SimEnv/IKSolver/MotionPlanner/CollisionChecker/DataWriter (Protocol)
  runtime.py          execute_trajectory: IK→충돌→step→기록→success (SART/CP-Gen 공용)
  sart_augmentor.py   SART 재구현 (RMB/pinocchio 의존 제거, 우리 인터페이스 위)
  cpgen_transform.py  CP-Gen keypoint/geometry transform (+ rigid baseline)
  pipeline.py         SkillGenPipeline: transform→stitch→replay→filter (+SART boost)
  mocks.py            Mock 백엔드 + toy 데모 (examples/tests 용)
  data_schema.py      HDF5DataWriter (robomimic/Isaac Lab Mimic 레이아웃)
  curobo_backend.py   CuroboIK/Planner/Collision  (가드된 스텁)
  isaaclab_env.py     IsaacLabEnv                (가드된 스텁)
configs/              sart_peg_in_hole.yaml, cpgen_gear_assembly.yaml
examples/             run_sart_augment.py, run_cpgen_generate.py
tests/                test_smoke.py
```

## 실제 실행으로 가는 순서
1. `isaaclab_env.py` — peg/gear 태스크용 ManagerBasedRLEnv 연결, (8,) EEF 액션 매핑, reward≥1=success.
2. `curobo_backend.py` — 우리 cuRobo 로 IK/Planner/Collision 채우기. point cloud 를 world 로 로드.
3. `data_schema.py` — action/obs 키를 대상 정책 트레이너에 맞추기.
4. `examples/` 의 Mock* 를 위 실제 클래스로 교체 → 동일 코드로 대량 생성.

관련 문서: `robot_data workspace — augmentation_methods/INTEGRATION_PLAN.md`,
`robot_data workspace — augmentation_methods/sart_robomanipaug/adapters/ISAAC_LAB_CUROBO_COMPAT.md`
</content>
