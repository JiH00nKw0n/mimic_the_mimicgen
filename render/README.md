# render — 실제 카메라 4대 시점으로 데모 재렌더링

주상님이 보내준 실기 카메라 calibration 번들(`fr3_camera_overlay_v1`,
schema `stage2.fr3_camera_overlay.v1`)의 4개 카메라 시점 — third-person D435
3대(로봇 베이스 기준 고정) + wrist D405(핸드 TCP 강결합) — 으로, 이미 생성해 둔
3-cube-stack 데모들(hdf5, 상태 기록 포함)을 다시 렌더링해서 **이미지 + action**
robomimic 스타일 데이터셋을 만든다.

방식: `../lab_stack_mimic/record_video.py --mode states`와 동일하게 기록된
state를 그대로 sim에 써 넣으면서 (물리 재실행 없음 = 데모와 100% 동일한 장면),
Isaac Lab Camera 센서 4대로 매 스텝 RGB를 캡처한다.

## 파일

- `fr3_camera_overlay_v1/` — 주상님 번들 사본 (`overlay.yaml`이 canonical)
- `overlay_cameras.py` — overlay 파싱 → Isaac Lab `CameraCfg` 4개 생성
- `lab_env.py` — lab FR3 + 책상 scene (lab_stack_mimic/record_video.py와 동일) +
  카메라 부착. 테이블 USD가 없는 서버에서는 대체 책상 슬래브로 자동 fallback
- `probe_tcp_binding.py` / `run_probe.sh` — **1회 선행 필수.** overlay의
  Franka-Home 기준 FK(`base_T_hand_tcp`)와 sim FR3의 `fr3_hand` FK를 대조해
  `fr3_hand_T_fr3v2_hand_tcp` 어댑터를 계측 → `fr3_binding.yaml` 기록.
  (번들 정책: prim 이름이 비슷하다고 바로 binding하지 말 것 → 계측으로 판정)
- `render_viewpoints.py` / `run_render.sh` — 본 렌더러
- `FR3_CAMERA_OVERLAY_notion.md` — 랩 Notion Archive용 정리 초안

## arpa에서 실행 순서

```bash
# 0) 서버에서 리포를 최신으로 (push 후)
ssh arpa-l40s 'cd /home/ubuntu/mimicgen_jihoonkwon/mimic_the_mimicgen && git pull'

# 1) 바인딩 계측 (1회, ~1분)
bash run_probe.sh                      # -> fr3_binding.yaml, ready_to_apply=true 확인

# 2) 소규모 테스트 렌더 + 프리뷰 영상 확인
bash run_render.sh /home/ubuntu/jake/aidas/3cube_stack/datasets/random_generated_2000_FINAL.hdf5 \
    --count 3 --preview_video 3
# -> *_fr3cams.hdf5 옆의 *_preview.mp4 (2x2: tp0|tp1 / tp2|wrist) 를 눈으로 확인

# 3) 본 렌더 (원하는 규모로)
bash run_render.sh .../random_generated_2000_FINAL.hdf5 --count 100 --append
bash run_render.sh .../skillgen_stack2000.hdf5 --count 100 --append
```

`--append`는 중단된 런 재개용(이미 렌더된 demo는 건너뜀). 스팟 인스턴스에서 유용.

## 다른 태스크 렌더링 (peg in hole 등)

씬은 데이터셋에서 자동 발견된다 (rigid object 이름들, 예: peg). 외부 등록 태스크는
`--task`와 `--register`로 지정하면 환경 설정을 그대로 가져와 리셋 이벤트만 벗겨 쓴다.
aidas의 peg 예시 (isaac-lab-skillgen 도커, peg 코드와 asset 마운트 필요):

```bash
docker run --rm --gpus all -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y -e OMNI_KIT_ACCEPT_EULA=YES \
  -e PYTHONPATH=/peg -e LAB_PEG_ENV_USD=/work/assets/peg_hole_env.usd -e LAB_PEG_HOLE_USD=/work/assets/hole_01.usd \
  -v /home/ubuntu/fr3_render_smoke:/work -v /home/ubuntu/peg_work:/peg_work:ro \
  -v /home/ubuntu/peg_work/lab_peg_mimic:/peg:ro -v /home/ubuntu/peg_work/peg_in_hole/assets:/work/assets:ro \
  -v /home/ubuntu/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
  -v /home/ubuntu/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
  --entrypoint bash isaac-lab-skillgen:latest -lc "cd /work/render && \
    /workspace/isaaclab/isaaclab.sh -p render_viewpoints.py \
      --dataset /peg_work/gen_skillgen.hdf5 --output /work/out/peg_fr3cams.hdf5 \
      --task Isaac-PegInsert-LabFR3-IK-Rel-Mimic-v0 --register peg_register \
      --count 5 --preview_video 2 --device cpu"
```

주의: peg 환경의 기록된 obs eef_pos는 TCP가 아니라 fr3_hand 프레임 기준이다.
`[align]` 진단이 두 규약을 모두 보고하고 어느 쪽이 맞는지 알려준다 (peg는 hand@t가
mm 수준이면 정상). 검증 결과: 생성 stack(fwd_gen_s05) 1.1mm, 생성 peg(gen_skillgen)
hand 기준 서브밀리미터, 프리뷰 4개 뷰 정상 (peg 파지·삽입 과정 wrist 시점 포함).

## 출력 포맷

```
data/demo_i/
    actions                  (T,7)  원본 그대로
    obs/third_person_{0,1,2}_image, obs/wrist_image   (T,H,W,3) uint8
    obs/<원본 low-dim obs 전부 복사>
data.attrs: env_args, fr3_camera_overlay(JSON: K행렬·extrinsic·serial·calibration id), source_*
demo attrs: num_samples, replay_success_any_order, stack_order
```

이미지[t]는 actions[t] **직전** 상태의 렌더다 (표준 BC 페어링). Isaac Lab recorder는
states[t]를 actions[t] 적용 **후** 상태로 기록하므로, 기본값(`--state_offset pre`)은
t=0은 initial_state, t>0은 states[t-1]을 렌더한다. 첫 데모에서 기록된 obs/eef_pos와
렌더 시점 TCP를 비교한 정렬 진단이 출력되니 데이터셋마다 한 번 확인할 것
(`[align]` 줄에서 @t 오차가 작아야 정상, @t+1이 작으면 반대 offset으로 재실행).

## 용량/속도 감각

기본 640×360(원본 1280×720의 1/4, 16:9 유지 — FOV 동일). 데모당 ~500스텝이면
4캠 gzip 후 대략 0.3~0.7GB/demo. 2000개 전부는 수백 GB + L40S에서 수십 시간이므로
먼저 25~100개로 시작 권장. 조절 손잡이: `--count`, `--every 2`(시간 절반),
`--width 320 --height 180`, `--no_compress`(빠름·큼).

## 검증 상태 (2026-07-22, aidas isaac-lab 3.0 도커 스모크)

fwd_annotated.hdf5(랩 FR3 데모 13개) 중 2개를 전체 파이프라인으로 렌더해 확인 완료.
probe는 base 어댑터 identity, hand_T_tcp (0, 0, 0.1034) 회전 0도로 교과서적으로 계측됐고,
리플레이 TCP와 기록된 obs eef_pos의 평균 오차 0.6mm ([align] 진단), 프리뷰 영상에서
4개 뷰 모두 정상 (wrist는 그리퍼 시점에서 파지한 큐브가 손가락 사이로 보임).

Isaac Lab 3.0에서 얻은 핵심 교훈 (코드에 자동 감지로 반영됨):
- 3.0은 쿼터니언 규약이 API마다 다름. cfg와 write_*_to_sim은 (w,x,y,z) 유지,
  데이터 읽기(body_link_quat_w 등)와 카메라 API(OffsetCfg, set_world_poses)는 (x,y,z,w).
  `camera_quat_order()`가 감지해 읽기와 카메라 쪽만 변환. 데이터셋 pose는 raw로 통과.
- fabric 파이프라인은 physx를 USD로 되쓰지 않아 링크에 붙인 카메라가 움직임을 못 따라감.
  `use_fabric=False` + `--device cpu`(기본, run 스크립트에 포함)로 고전 USD 파이프라인 사용.
- states만 쓰고 렌더하면 화면이 갱신 안 됨. 매 프레임 상태 기록 후 물리 1스텝으로 동기화
  (PD가 해당 자세를 잡고 다음 프레임에 상태를 다시 쓰므로 리플레이 왜곡 없음).
- `env.reset_to()`는 카메라 뷰를 무력화하므로 initial_state를 직접 기록.

디버깅용 스크립트: `probe_camera_pose.py`(카메라 pose 왕복 검증),
`diag_quat_orders.py`(쿼터니언 인코딩 판별), `diag_loop_poses.py`(루프 재현) — 평소엔 불필요.

## 한계 / 주의 (overlay 번들 명시사항)

- wrist D405는 **테이프 마운트 임시 calibration** (`provisional_wrist_tape_mount`).
  마운트 재제작 후 새 번들이 오면 `fr3_camera_overlay_v1/`만 교체하고 probe부터 재실행.
  (binding에 번들 revision이 박혀 있어서, 번들만 바꾸고 probe를 안 돌리면 렌더러가 거부함)
- RealSense 왜곡계수는 metadata로만 보존 — 렌더는 무왜곡 pinhole.
- Omniverse 렌더러는 USD aperture offset(주점 오프셋)과 vertical aperture를 무시함
  (OM-42611). 실제 렌더 픽셀은 fx 기준 정사각 픽셀 + 중앙 주점이라, calibrated K로
  재투영하면 원본 해상도 기준 최대 ~13px 편차. 출력 hdf5 메타데이터에 실렌더용
  `K_effective_render`와 실카메라용 `K_calibrated_at_render_size`를 둘 다 기록해 둠.
  (이건 주상님 번들의 자체 Isaac 적용 도구도 동일하게 겪는 렌더러 한계)
- 실기 테이블 top은 베이스 기준 +19mm, sim 책상은 -2mm (≈21mm 차이). 카메라
  pose는 베이스 기준이라 그대로 정확하고, 화면 속 책상/큐브 높이만 그만큼 다르게 보임.
- sim 큐브 작업영역과 실기 table-task 원점(베이스 기준 x0.56,y0.29)이 완전히
  겹치지는 않음 — 프리뷰 영상으로 프레이밍 확인 후 필요하면 논의.
