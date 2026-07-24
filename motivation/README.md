# motivation/ — 성공 필터링 편향 확장 실험 (E1 + E2)

Phase-0(../MOTIVATION_INITIAL_CONDITION.md)의 후속 실험 패키지. 문서와 코드가 함께 산다.

| 문서 | 내용 |
|---|---|
| [PLAN.md](PLAN.md) | 실험 설계 전문 — E1(태스크 일반화 + 대칭이동 confound 제거), E2(transform-uniform 리샘플링의 policy 효과), transform 거리 정의, 구간화·uniformity 인증, 컴퓨트 |
| [TASKS.md](TASKS.md) | 태스크 × 변형 자산 매트릭스, 공개 bounds 수치, 커스텀 E-시리즈 변형 제안 |
| [ARCHITECTURE.md](ARCHITECTURE.md) | `genaudit` 패키지 설계 — config 중심 실행, AttemptRecord 계약, 확장 시나리오 |

## 빠른 시작

```bash
pip install -e .            # 로컬: numpy/pyyaml만으로 코어 동작 (sim 의존은 guarded)
pytest tests/               # 시뮬레이터 없이 전체 단위 테스트
```

서버(생성·학습)는 ARCHITECTURE.md §7과 scripts/ 참조.

## motivation_new — 실제 실행 파이프라인 (E2 정책 실험, `scripts/mnew_*`)

PLAN.md의 방향성 D2 변형 대신 **등방·무회전 재설계**로 다시 돌린 실행본이다. N0/N1/N2를
source 위치 중심의 정사각형으로 정의하고(회전 없음, `genaudit/envs/bounds_new.py`), genaudit
코어는 그대로 쓰되 **단일 N2 풀에서 사후 추출**하는 구체 스크립트를 `scripts/mnew_*`로 얹었다.
전 과정은 서버(aidas-l40s, g6e.2xlarge · 8 vCPU · L40S)에서 돈다. 결과 리포트는
[`../Motivation_E2_Policy_Results.md`](../Motivation_E2_Policy_Results.md).

### 생성 → 학습 데이터 준비

| 단계 | 스크립트 | 하는 일 | 산출물 |
|---|---|---|---|
| 1 생성 | `mnew_generate.py` | 8태스크 × N0/N1/N2 attempts를 500개 청크로 병렬 생성(느린 것 먼저) | `gen/<task>_N2/demo{,_failed}.hdf5` 청크 |
| 2 병합 | `mnew_merge.py` | 청크를 태스크별 단일 풀로 합침(demo 그룹 전역 리네임, env_args 보존) | `gen/<task>_N2/demo.hdf5` 병합본 |
| 3 추출 | `mnew_extract.py` | 병합 풀 → AttemptRecord jsonl (d_pos·source·success) | `records/…jsonl` |
| 4 arm | `mnew_arms.py` | 단일 N2 풀에서 baseline / transform_uniform / ancestry_balanced × 2시드 샘플 → robomimic filter key | `e2_arms/<task>_N2/{train.hdf5, attempts.jsonl, arms_manifest.json, bin_edges.json}` |
| 4b 시드확장 | `mnew_addseeds.py` | train.hdf5 재빌드 없이 provenance로 시드 103–106 filter key 추가(기존 시드 보존) | 위 train.hdf5에 key 추가 |

### 학습 → 평가 (오케스트레이터)

| 단계 | 스크립트 | 하는 일 |
|---|---|---|
| 5 학습 | `c_make_train_configs.py` → `c_train_all.py` | arm별 BC-RNN low-dim config 생성(**학습 중 rollout OFF**) → 고정 동시성 학습(최종 ckpt 있으면 skip=재개 가능) |
| 6 고정장면 | `c_make_frozen_resets.py` | 태스크별 200개 초기상태를 시드로 뽑아 저장 → 모든 arm·시드가 **같은 장면** 공유(paired) |
| 7 평가 | `mnew_eval.py` | 각 (arm,시드) ckpt를 200 고정장면에 굴려 per-episode 성공 기록 → arm별 SR + paired McNemar |
| 묶음 | `mnew_seeds.sh` | 4b→5→7을 한 번에 도는 **6시드 확장 오케스트레이터**. `mnew_finish.sh`는 2시드판 마무리본 |

### 거리 분석 (리포트 근거)

| 스크립트 | 하는 일 | 리포트 절 |
|---|---|---|
| `mnew_deval.py` | 각 고정장면의 d_eval(가장 가까운 source까지 d_pos)을 **생성과 동일한 mimicgen env interface**(`get_object_poses`)로 추출 → 좌표계 일치 | §1, §3–4 축 |
| `mnew_farbin.py` | d_eval 3등분(near/mid/far) arm SR + (장면, 시드) 단위 McNemar 정확검정 | §3 |
| `mnew_quantile.py` | d_eval 4분위 baseline vs transform / vs ancestry (특화 검정) | §4 |
| `mnew_armdist.py` | arm별 학습셋 d_pos 분포(평균·near%·균등 대비 TV) | 부록 C |

### 지금 실행 상태 (2026-07-24)

- **2시드 결과 완료** → 리포트에 반영. (E1: 거리↑ → DGR↓ 전 태스크 재현 / E2: transform 얕게 우세·무유의, ancestry 태스크 의존, far 아닌 near·mid에서 작동.)
- **6시드 확장 진행 중**: `mnew_seeds.sh`가 시드 103–106을 추가 학습(약 92런). low-dim BC-RNN은 GPU를 안 쓰는 **CPU 학습**이라 `OMP_NUM_THREADS=1 × concurrency 8`로 8코어를 독립 런으로 꽉 채워 돌린다(util ~91%, ~5.8 run/h). 완주 시 `SEEDS_DONE` 마커 → 6시드로 farbin/quantile/armdist 자동 갱신 → 리포트 §3–4·부록 C 교체 예정.
- 서버 경로: `~/mimicgen_jihoonkwon/experiments/motivation_new/{gen,records,e2_arms,e2_results,e2_train_cfgs}`, venv `~/mimicgen_jihoonkwon/robosuite_mimicgen/venv`, `PYTHONPATH=<repo>/motivation`.

> **주의**: `mnew_*`는 서버 절대경로(`/home/ubuntu/...`)가 파일 상단 상수로 하드코딩된 **실제 실행본**이다. 다른 환경에서 재현하려면 그 상수만 바꾸면 된다.

## 상태

- Phase A (문서 + 코어 + 테스트): 이 폴더
- Phase B0 이후 (서버 생성·학습): PLAN.md §4 실행 단계 참조
- **motivation_new (E2 정책)**: 2시드 완료·리포트 작성, 6시드 확장 학습 중 — 위 "실제 실행 파이프라인" 참조
