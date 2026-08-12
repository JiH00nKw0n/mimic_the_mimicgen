FR3 큐브 쌓기 80K 생성 파이프라인 - 자산 묶음
================================================

도커 이미지를 빌드할 때 필요한 자산 세 가지입니다. 코드는 별도 저장소
(mimic_the_mimicgen/hf80k/)에 있고, 이 묶음은 코드에 포함하지 않은 바이너리입니다.

1) fwd_annotated.hdf5  (3.4 MB)
   사람이 로봇을 직접 조종해 녹화한 큐브 쌓기 시연 13개입니다. MimicGen이 이것을
   입력으로 받아 새 배치의 궤적을 만들어 냅니다. 구간 경계 신호까지 붙어 있는
   상태(annotated)라 그대로 쓰면 됩니다.
   놓을 위치: hf80k/assets/fwd_annotated.hdf5

2) fr3_cube_system_calibration_bundle_v1/  (12 MB)
   SysID팀이 실물 FR3와 큐브로 측정한 물리값입니다. 마찰, 큐브 질량, 관절 마찰과
   관성, 그리퍼 힘 배율, 손목 카메라 무게가 들어 있습니다. 생성할 때 시도마다 이
   범위에서 값을 하나씩 뽑아 시뮬레이터의 물리 설정에 넣습니다. 그래서 8만 편이
   하나의 고정된 물리가 아니라 실측 범위 안의 여러 물리에서 나옵니다.
   놓을 위치: hf80k/assets/fr3_cube_system_calibration_bundle_v1/

3) fr3_visual_randomization_v1/  (74 MB)
   RL팀이 2026-08-04에 전달한 시각 랜덤화 패키지입니다. 조명용 파노라마 사진 3장
   (resources/hdri/*.hdr), 실험실 책상 3D 모델(assets/table/table_scene.usdc),
   물체별 색과 재질 범위(config/), 카메라 실측 캘리브레이션
   (config/camera_nominal_measured_ranges.yaml)이 들어 있습니다. RL팀이 만든
   것이므로 이미 가지고 계실 수 있는데, 판본이 어긋나면 결과가 달라지므로
   이 묶음의 것을 쓰시기를 권합니다.
   놓을 위치: hf80k/assets/fr3_visual_randomization_v1/
   무결성 확인: 안에 MANIFEST.sha256이 들어 있습니다.

실행 순서
---------

압축을 푼 뒤 세 항목을 hf80k/assets/ 아래로 옮깁니다. 그 다음은 hf80k/ 안에서
다음 네 줄입니다.

   cp .env.example .env     # 허깅페이스 토큰과 저장소 이름을 채웁니다
   make check               # 자산과 .env가 갖춰졌는지 확인합니다
   make build               # 이미지를 굽습니다. 처음에는 40분쯤 걸립니다
   make smoke               # 100편으로 동작을 확인합니다

확인이 끝나면 make run으로 본 실행에 들어갑니다. GPU 4장짜리 박스라면
make run-4gpu가 2만 편씩 네 개로 나눠 띄웁니다. 자세한 설명은 hf80k/README.md에
있고, 환경 변수 하나하나의 뜻은 hf80k/.env.example에 주석으로 적혀 있습니다.

토큰은 .env 파일에만 넣습니다. 이미지 안에는 들어가지 않고, .env는 저장소에서
제외돼 있습니다.
