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

## 상태

- Phase A (문서 + 코어 + 테스트): 이 폴더
- Phase B0 이후 (서버 생성·학습): PLAN.md §4 실행 단계 참조
