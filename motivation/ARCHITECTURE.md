# genaudit — Motivation 실험 프레임워크 설계

목표: **실험 실행은 config로, 코드는 재사용 가능한 추상화로.** 지금은 initial-condition 축의 E1/E2지만, 이후 (a) factor 축 추가(proprioception, skill action, motion action), (b) 태스크 추가, (c) 생성 백엔드 추가(Isaac Lab Mimic), (d) policy 학습 확장(diffusion 등)이 코드 수정 최소로 가능해야 한다. Clean Code 레퍼런스(`linq-meeting-agent/.clean-code-reference`) 기준 적용.

## 1. 디렉토리 레이아웃

```
motivation/
  PLAN.md  TASKS.md  ARCHITECTURE.md  README.md
  pyproject.toml
  configs/
    tasks/<task>.yaml            # 태스크 기하·사다리·물체·대칭차수·소스 경로
    experiments/<exp>.yaml       # 실험 1개 = config 1개 (e1_sweep, e2_square, ...)
  genaudit/
    records/    schema.py  extract.py
    factors/    base.py  initial_condition.py
    envs/       bounds.py  robosuite_variants.py
    generation/ mimicgen_backend.py  run_mimicgen.py
    curation/   binning.py  uniformity.py  samplers.py  filter_keys.py
    training/   robomimic_config.py
    evaluation/ frozen_resets.py  stratify.py
    analysis/   dgr.py  ancestry.py
    config.py   cli.py
  tests/
  scripts/                       # 서버 launch 셸
```

## 2. 핵심 계약: AttemptRecord

모든 모듈이 공유하는 유일한 데이터 계약. 생성 백엔드가 무엇이든(robosuite mimicgen, Isaac Lab mimic) **추출기가 이 스키마로 변환하는 순간부터 하류(비닝·샘플링·분석)는 백엔드를 모른다.**

```python
AttemptRecord(
  task, variant, attempt_id,      # attempt_id = "demo_3@demo_failed.hdf5"
  source_demo_id, success, episode_length,
  displacements=(ObjectDisplacement(name, dxy_m, dyaw_rad), ...),
  d_raw, d_pos, d_rot,            # PLAN §1.3의 세 정의를 전부 병기
  extras={...},                   # factor 축 확장 컬럼 (proprio_*, action_* 등)
)
```

직렬화는 JSONL(사람이 읽고 diff 가능, 의존성 없음). pandas 변환 헬퍼 제공.

## 3. 모듈과 책임 (SRP)

| 모듈 | 책임 | 경계(외부 의존) |
|---|---|---|
| `records/schema` | AttemptRecord 정의·JSONL 왕복 | 없음 |
| `records/extract` | mimicgen hdf5(성공+실패) → AttemptRecord | h5py (guarded) |
| `factors/base` | FactorAxis 프로토콜: record에 컬럼 제공 + 스칼라 축 | 없음 |
| `factors/initial_condition` | L_m(대각선)·d_raw/d_pos/d_rot·nearest-source d_eval | numpy |
| `envs/bounds` | 공개+E시리즈 bounds **순수 데이터 레지스트리** + superset/대각선 유틸 | 없음 |
| `envs/robosuite_variants` | 레지스트리에서 env 서브클래스 동적 생성·등록 | mimicgen (guarded) |
| `generation/mimicgen_backend` | 템플릿 → 생성 config 패치(500 attempts, keep_failed, random selection) | 없음(JSON 조작) |
| `generation/run_mimicgen` | 변형 등록 후 공식 generate_dataset 실행 (서버 진입점) | mimicgen |
| `curation/binning` | per-task quantile edges 계산·동결(JSON artifact)·bin 할당 | numpy |
| `curation/uniformity` | TV distance + 인증(임계값·min-bin) | numpy |
| `curation/samplers` | baseline / transform-uniform / ancestry-balanced 추출 (+부족분 정책) | numpy |
| `curation/filter_keys` | robomimic `mask/<key>` 기록 | h5py (guarded) |
| `training/robomimic_config` | BC-RNN config 패치(filter_key·seed·경로) + 실행 커맨드 생성 | 없음(JSON 조작) |
| `training/run_train` | 변형 등록 후 robomimic train 위임 (E-변형 env_meta 해석용 서버 진입점) | robomimic |
| `evaluation/frozen_resets` | 고정 초기상태 세트 생성·paired 평가 | robomimic (guarded) |
| `evaluation/stratify` | d_eval 층화 SR·slope 통계 | numpy |
| `analysis/dgr`·`ancestry` | DGR-vs-d 곡선, per-bin DGR, ancestry skew·n_eff, 정의 비교(d_raw vs d_pos) | numpy |
| `config` | YAML → 타입 있는 dataclass (TaskSpec, ExperimentSpec) | pyyaml |
| `cli` | `python -m genaudit <cmd> --config ...` | 위 모듈 조립만 |

**bounds가 순수 데이터인 이유**: superset 검증·대각선 계산·문서 표 생성이 시뮬레이터 없이 로컬에서 테스트되고, env 클래스는 이 데이터를 소비하는 파생물이 된다 (단일 진실 원천).

## 4. Config 스키마

### tasks/threading.yaml (발췌)

```yaml
task: threading
objects:
  needle: {symmetry_order: 1}
  tripod: {symmetry_order: 1}
source_dataset: datasets/source/threading.hdf5
env_interface: MG_Threading
generation_template: exps/templates/robosuite/threading.json
rollout_horizon: 400
ladder: [D0, D1, D2, D2E]        # bounds는 envs/bounds.py 레지스트리가 진실
widest_variant: D2E              # L_m·d 정규화 기준
contrast_variants: [D2]          # 재배치 대조군 (E1 부차 검증)
```

### experiments/e2_threading.yaml (발췌)

```yaml
experiment: e2
task: threading
variant: D2E
distance: {primary: d_pos}        # B0 경향 체크 후 확정값 기입 (d_raw fallback)
pool: {num_attempts: 6250, seeds: [1, 2]}   # 총합 — seed당 3125 생성 후 병합
binning: {k: 5, fallback_k: 4}    # fallback은 curate가 인증 실패 시 자동 적용
certification: {tv_threshold: 0.02, min_bin_fraction: 0.9}
arms:
  baseline: {size: 500}
  transform_uniform: {size: 500, quota_per_stratum: 100}   # 500/100 = K
  ancestry_balanced: {size: 500, quota_per_stratum: 50}    # 500/50 = source 수
dataset_seeds: [101, 102, 103]    # arm별 RNG 스트림은 (seed, arm) 독립
```

실험 재현 = config 파일 1개 재실행. 결과 폴더에 config 사본·bin edges·인증 결과가 함께 저장돼 사후 감사가 가능하다.

## 5. 확장 시나리오 (설계 검증용)

1. **Proprioception 축 추가**: `factors/proprioception.py`에 FactorAxis 구현 1개(예: grasp 시점 관절각과 source의 차) → extract가 extras 컬럼으로 기록 → binning·samplers·stratify는 **수정 없음** (스칼라 축이면 그대로 소비).
2. **Isaac Lab 백엔드**: `generation/isaaclab_backend.py` + lab_stack_mimic `provenance_hooks.py`를 AttemptRecord로 변환하는 extractor 1개. 하류 전체 재사용.
3. **Diffusion policy**: `training/diffusion_config.py` 추가. filter_key·frozen eval은 그대로.

## 6. Clean Code 적용 규칙

- **이름(Ch2)**: 도메인 어휘 그대로 — attempt/retained/ancestry/quota/certification. 축약 금지(`d_pos`는 논문 표기라 예외적 허용, docstring에 정의 명시).
- **함수(Ch3)**: 한 함수 한 일. I/O와 계산 분리(계산 함수는 배열 입출력의 순수 함수).
- **경계(Ch8)**: mimicgen·robomimic·h5py는 모듈 경계 안에서만 import(guarded, 함수 내부). 실패 시 설치 안내를 담은 명시적 예외.
- **에러(Ch7)**: 조용한 보정 금지 — bounds superset 위반, bin 부족분 임계 초과, source가 V_m 밖 등은 전부 구체적 메시지의 예외로 시끄럽게 실패.
- **테스트(Ch9)**: 시뮬레이터 없는 결정적 단위 테스트 — binning 결정성, TV 수치, sampler quota·부족분 재분배, L_m·d의 성질(∈[0,1], 대칭 wrap), superset 레지스트리 검증, JSONL 왕복.
- **시스템(Ch11)**: 조립(config → 객체)과 사용(계산) 분리 — cli/config만 조립을 안다.

## 7. 서버 실행 형태

```bash
# 생성 config 만들기 (E1 스윕 또는 E2 pool; E2는 pool seed별로 자동 분할)
python -m genaudit gen-config --task-config configs/tasks/threading.yaml \
  --experiment-config configs/experiments/e2_threading.yaml --variant D2E
# 생성 (서버, CPU): 변형 등록 + 공식 스크립트 위임
python -m genaudit.generation.run_mimicgen --config .../mg_D2E_seed1.json
# 추출 → 비닝·arm 추출 → robomimic mask 기록 → 분석
python -m genaudit extract     --task-config ... --experiment-config ...
python -m genaudit curate      --task-config ... --experiment-config ...
python -m genaudit filter-keys --experiment-config ...
python -m genaudit analyze     --experiment-config ...
# 학습 (서버, GPU): E-변형 env_meta를 해석하려면 반드시 wrapper로
python -m genaudit.training.run_train --config bc_rnn_low_dim_....json
```
