# Visual Randomization Adoption Checklist

다른 팀은 비고란에 경로뿐 아니라 실제 측정 숫자를 기록해야 합니다.

| 항목 | O/X | 정량 증거 | 비고 |
|---|---|---|---|
| package checksum 통과 |  | mismatches = 0 | `python3 tools/verify_bundle.py` |
| 대상 scene semantic mapping 완료 |  | unmapped prims = 0 | robot/table/floor/camera parent |
| FR3 visual-only composition |  | articulation/actuator diff = 0 | collision/physics 불변 |
| camera calibration 연결 |  | 3/3 cameras | nominal + measured range |
| 저장 camera 구성 |  | 3 cameras | third_person_0/1 + wrist |
| 해상도와 fps |  | 320×180, 10 fps | H.264 yuv420p |
| profile 분리 |  | 50/40/10 | episode label 포함 |
| identity-preserving 색상 |  | 3/3 objects distinguishable | arbitrary RGB 금지 |
| episode 내부 appearance 안정성 |  | jumps = 0 | reset 경계에서만 변경 |
| global HDRI 안전성 |  | cross-env changes = 0 | process-level profile |
| foreign pixel isolation |  | max = 0 pixels | 모든 camera/env/step |
| instance-id mapping |  | 100% complete | fail-closed |
| RGB corruption gate |  | min std ≥ 10 | camera별 수치 기록 |
| profile별 gallery 검토 |  | 최소 8 samples/profile | 연구실 reference 비교 |
| unresolved USD reference |  | count = 0 | asset resolver 검사 |
| short collection smoke |  | 성공 episode 수 기록 | video/meta/parquet load |
| profile count audit |  | exact requested counts | 중복/누락 seed 포함 |

