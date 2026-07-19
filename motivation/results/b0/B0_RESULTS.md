# B0 결과 — Phase-0 pool 감사 (2026-07-19, aidas)

PLAN.md §4 B0 완료 기록. 원본 수치는 `b0_report.json` (attempt 레코드 JSONL 12개는 aidas `~/mimicgen_jihoonkwon/experiments/motivation_ic/b0/`). 실행: `scripts/b0_audit_phase0_pools.py`, Phase-0 12개 pool (square/threading/coffee/three_piece × D0/D1/D2, 각 500 attempts).

## 1. 거리 정의 확정 — d_pos를 1차로

d_raw(미터 합)와 d_pos(대각선 정규화 평균)의 경향 강도가 **12개 pool 전부에서 사실상 동일** (point-biserial·Spearman 차이 ≤ 0.003, per-bin 단조성 판정 12/12 일치). 물체별 정규화가 raw 합의 경향을 약화시키지 않음이 실측으로 확인됨 → **PLAN §1.3의 1차 정의 = d_pos 확정** (d_raw는 계속 병기).

수치 요점 (point-biserial r, d_pos 기준):

| pool | DGR | r | 비고 |
|---|---|---|---|
| square D0/D1/D2 | .75/.46/.33 | −.06/−.02/−.19 | 확장 사다리 — D2에서 음의 경향 뚜렷 |
| threading D0/D1 | .51/.37 | −.11/−.09 | 확장 구간 — 단조 하락(per-bin) |
| threading **D2(미러)** | .22 | **+.07** | 재배치 pool에서 경향 반전 |
| coffee D0/D2 | .79/.28 | −.04/−.03 | |
| coffee **D1(machine shift)** | .66 | **+.12** | 혼합 변형에서 경향 반전 |
| three_piece D0/D1/D2 | .34/.32/.28 | −.30/−.22/−.22 | 가장 강한 within-pool 경향 |

주목: **양(+)의 상관은 정확히 재배치/shift 변형(threading D2, coffee D1)에서만** 나타난다 — "공개 변형의 영역 이동이 transform 축을 오염시킨다"는 E1의 전제를 Phase-0 데이터가 이미 지지. within-pool 상관이 전반적으로 약한 것(|r|≤0.3)은 Phase-0 결론(변형 간 격차가 주효과) 그대로.

## 2. E2 pool 크기 — N=6,250 총합으로 확정

scarcest-bin DGR의 Wilson 95% 하한 기준 필요 attempts (K=5, bin당 100 retained 목표):

| task (최광 변형) | N (d_pos) | scarcest bin DGR |
|---|---|---|
| square D2 | 4,590 | 0.17 |
| threading D2(미러 기준 추정) | 5,865 | 0.14 |
| coffee D2 | 2,996 | 0.24 |
| three_piece D2 | 3,996 | 0.19 |

전부 6,250 이하 → **e2 config의 N=6,250(총합)이 전 태스크에 충분** (threading D2E는 미러보다 온건한 확장이라 상한 추정으로 안전). Stack D2E는 pilot이 없으므로 E1의 500-attempt pool로 같은 산정을 한 뒤 확정.

## 3. Arm C(ancestry-balanced) — fallback 필요 확정

worst-source 성공률 Wilson 하한으로 "50/source 전원 충족"에 필요한 N:

- coffee D2 ≈ 6,614 / three_piece D2 ≈ 10,563 / square D2 ≈ 16,491 / threading D2 ≈ 25,234, **threading D1은 생존 0 source 존재(불가능)**.

→ N=6,250에서는 PLAN §2.4의 **사전 등록 fallback이 실제로 발동**된다: worst source 1–2개를 C에서 제외하고 남은 source 등-quota 재구성, 제외 목록·n_eff 보고. (bin 제약이 아니라 source 제약이 바인딩이라는 것이 B0의 새 정보.)

## 4. 인프라 검증

- **Threading_D2E end-to-end 스모크 통과** (aidas, `run_mimicgen --debug`): 변형 등록 → env 이름 해석 → reset → 생성 2 attempts (1 성공/1 실패, 예외 0). 초기 배치가 공개 D1 상한 밖(tripod x=0.166>0.15, needle x=0.085>0.05)임을 확인 — **bounds override 실작동**. (종료 시 EGL teardown 에러는 headless 렌더러 정리 노이즈.)
- **Stack·StackThree source 다운로드(HF) + `prepare_src_dataset.py` annotation 완료** (aidas `robosuite_mimicgen/mimicgen/datasets/source/`).
- b0 소스 심링크: `experiments/motivation_ic/b0_sources/` (square는 `square_prepared.hdf5`).

## 5. 다음 (B1)

1. reachability 관문: IK 스캔 + 극단 배치 50-attempt 프로브 + (image 태스크) 렌더 커버리지 → D2E bounds 동결.
2. E1 생성: threading D2E / coffee D1E·D2E / stack D0·D1·D2E / stack_three D0·D1·D2E / square D2(random-selection 재실행) × 500 attempts.
3. E2 pool 생성 (B2): 4태스크 × 6,250 (seed 2개 분할).
