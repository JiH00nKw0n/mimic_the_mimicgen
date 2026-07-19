# C단계 사전 검증 — 학습 파이프라인 (2026-07-20 새벽, aidas L40S)

공개 배포된 square D2 생성 데이터셋(1,000 demo)으로 논문 레시피(BC-RNN low-dim,
2000 epoch) 학습이 aidas에서 끝까지 돌아가는지, 성공률이 논문과 정합하는지 확인.

- 학습: 2,000 epoch 완주, 크래시 0 (검증된 런처 조건 필수 — scripts/c_train_launch.sh)
- 평가: 최종(epoch 2000) 체크포인트, 50 rollout → **성공률 46%**
- 논문 대조: 58.7±1.9%는 "학습 중 40회 평가의 최대값 × 3 seed" 프로토콜.
  최종 체크포인트 단일 평가는 통상 5~15%p 낮게 나오므로 46%는 정합 범위.
  본 실험(C단계)은 논문 프로토콜(주기 평가 + 최대값)을 그대로 쓰므로 문제없음.
- 인프라 교훈: 평가·학습 프로세스 모두 mimicgen env 등록 wrapper 필수
  (genaudit.training.run_train / genaudit.evaluation.run_eval).
