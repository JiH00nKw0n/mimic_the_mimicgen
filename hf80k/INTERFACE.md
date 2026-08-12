# hf80k 내부 인터페이스 규격

이 파일은 구성 요소들이 서로 맞물리는 지점을 고정한다. 코드를 쓰기 전에 여기를 먼저 읽고,
여기 적힌 이름과 경로와 형식을 그대로 쓴다.

## 1. 환경 변수 (`.env`로 주입, 전부 여기서만 정의)

필수는 두 개다. 나머지는 기본값이 있다.

| 변수 | 기본값 | 뜻 |
|---|---|---|
| `HF_TOKEN` | 없음 (필수) | 허깅페이스 쓰기 토큰 |
| `HF_REPO_ID` | 없음 (필수) | 올릴 저장소. 예: `myorg/fr3-cube-stack-80k` |
| `HF_PRIVATE` | `1` | 1이면 비공개 저장소로 만든다 |
| `TARGET_EPISODES` | `80000` | 만들 성공 에피소드 총 개수 |
| `PROFILE_SPLIT` | `nominal_lab:0.50,lab_variation:0.40,stress_tail:0.10` | 시각 프로파일 배분 |
| `CHUNK_SIZE` | `500` | 청크 하나가 담는 에피소드 수 |
| `NUM_ENVS` | `16` | MimicGen 생성 시 동시 환경 수 |
| `GEN_PROCS` | `1` | 동시에 띄울 생성 프로세스 수 |
| `RENDER_PROCS` | `2` | 동시에 띄울 렌더 프로세스 수 |
| `CUDA_VISIBLE_DEVICES` | `0` | 이 컨테이너가 쓸 GPU. GPU마다 컨테이너를 따로 띄운다 |
| `IMAGE_WIDTH` | `320` | 렌더 가로 픽셀 |
| `IMAGE_HEIGHT` | `180` | 렌더 세로 픽셀 |
| `PHYSICS_PROFILE` | `robust_stochastic` | `nominal`, `posterior_stochastic`, `robust_stochastic`, `off` |
| `SOURCE_DEMO_FILTER` | `exclude_zero_yield` | `all`이면 전체, `exclude_zero_yield`면 수율 0인 소스 제외, `0,1,2`처럼 직접 지정도 된다 |
| `SUBTASK_OFFSETS` | `10,20` | MimicGen 구간 경계 오프셋. 비우면 기본값 유지 |
| `WORK_DIR` | `/work` | 중간 파일과 청크가 쌓이는 곳. 컨테이너에 마운트한다 |
| `KEEP_INTERMEDIATE` | `0` | 1이면 청크 후 중간 HDF5를 지우지 않는다. 디버그용 |
| `UPLOAD_EACH_CHUNK` | `1` | 1이면 청크마다 바로 올린다 |
| `RESUME` | `1` | 1이면 완료된 청크를 건너뛰고 이어서 한다 |
| `SEED_BASE` | `42000` | 청크마다 `SEED_BASE + chunk_index`를 시드로 쓴다 |
| `LOG_LEVEL` | `INFO` | |

### 1b. 내부 조절 변수

위 표는 운영자가 만지는 값이다. 아래는 코드가 읽지만 보통 기본값 그대로 두는 값이다.
바꿔야 할 일이 생기면 `.env`에 적으면 똑같이 전달된다.

| 변수 | 기본값 | 뜻 |
|---|---|---|
| `LAB_ROBOT_SPAWN_ROT` | `0,0,1,0` | 로봇 받침의 스폰 회전. **이 값을 바꾸면 안 된다.** 기본값이 실기와 같은 방향이고, `0,0,0,1`로 두면 로봇이 180° 반대를 보며 수율이 절반이 되고 카메라가 약 1 m 빗나간다 |
| `LEROBOT_SITE` | 이미지가 정함 | lerobot을 격리 설치한 경로. 기록과 업로드 단계만 쓴다 |
| `LAB_SYSID_BUNDLE_ROOT` | `assets/fr3_cube_system_calibration_bundle_v1` | 물리 캘리브레이션 번들 위치 |
| `LAB_SYSID_SEED_OFFSET` | `73000` | 물리값 표본 추출 시드에 더하는 값 |
| `LAB_SYSID_LOG_SAMPLES` | `0` | 1이면 환경마다 뽑힌 물리값을 로그에 찍는다 |
| `LAB_TABLE_USD` | 이미지가 정함 | 실험실 책상 3D 모델 경로. 없는 경로면 회색 슬랩으로 대체된다 |
| `LAB_ARM_SCALE` | `0.5` | 생성 시 팔 동작 명령의 배율. 1.0이면 손목이 튀어 수율이 떨어진다 |
| `LAB_ARM_JITTER` | `0.02` | 리셋 때 관절에 주는 흔들림, 라디안 |
| `LAB_GEN_YAW` | `0.0` | 생성 시 큐브를 수직축으로 돌리는 범위 |
| `LAB_KEEP_FAILED` | `0` | 1이면 실패한 시도도 별도 파일로 남긴다. 대량 실행에서는 용량만 먹는다 |
| `LAB_EPISODE_LENGTH_S` | 비움 | 에피소드 시간 상한. 비우면 기본값을 쓴다 |
| `LAB_GEN_SEED` | 청크마다 자동 | 생성 난수 시드. 오케스트레이터가 넣는다 |
| `LAB_SOURCE_YIELD_JSON` | `assets/source_yield.json` | 소스별 수율 표 |
| `LAB_PROVENANCE_OUT` | 청크마다 자동 | 소스 사용 기록을 쓸 경로 |

## 2. 작업 디렉터리 배치

`WORK_DIR` 아래에 전부 만든다. 컨테이너 밖에서 마운트하므로 컨테이너를 지워도 남는다.

```
$WORK_DIR/
  chunks/
    chunk_00000/
      gen.hdf5                 생성 산출물 (성공만)
      gen.provenance.json      소스별 사용 기록
      contract.hdf5            계약 형식 변환 결과
      rgb.hdf5                 렌더 결과
      vrand_log.json           에피소드별 적용된 시각 랜덤화 값
      contract_report.json     계약 변환 검사 결과
      lerobot/                 LeRobot v3 데이터셋 (이 청크만)
      MANIFEST.json            완료 표시 겸 요약
    chunk_00001/
      ...
  merged/                      마지막에 aggregate_datasets로 합친 최종본
  source_filtered.hdf5         소스 필터를 적용한 사본. 원본은 고치지 않는다
  source_filtered.hdf5.filter.json  무엇을 남기고 무엇을 뺐는지
  generate_lab.py              Isaac Lab 생성 스크립트에 import 한 줄을 끼운 사본
  logs/
    orchestrate.log            전체 실행 로그
    chunk_00000.log            청크별 로그
  state.json                   전체 진행 상황
```

## 3. `MANIFEST.json` (청크 완료 표시)

이 파일이 있고 `status`가 `"done"`이면 그 청크는 끝난 것으로 보고 건너뛴다.

```json
{
  "schema_version": "fr3_cube.hf80k.chunk.v1",
  "chunk_index": 0,
  "status": "done",
  "profile": "nominal_lab",
  "episodes": 500,
  "frames": 88000,
  "attempts": 3289,
  "yield": 0.152,
  "seed": 42000,
  "physics_profile": "robust_stochastic",
  "image_size": [320, 180],
  "cameras": ["third_person_0", "third_person_1", "wrist"],
  "uploaded": true,
  "started_at": "2026-08-12T00:00:00Z",
  "finished_at": "2026-08-12T01:00:00Z",
  "durations_s": {"generate": 1200, "convert": 60, "render": 4000, "lerobot": 300, "upload": 120}
}
```

## 4. LeRobot 특성 규격

재익님 수집기 `collect_demos_lerobot.py`와 이름을 맞춘다.

아래 이름은 재익님 수집기와 글자까지 같아야 한다. 하나라도 다르면 두 데이터셋을 이어
붙이거나 같은 학습 코드로 읽을 때 이름이 어긋난다. 대조 위치는
`collect_demos_lerobot.py`의 47행(task 문자열), 100행(상태 이름), 193행(이미지 names),
272행(robot_type)이다.

```python
features = {
    "observation.images.third_person_0": {"dtype": "video", "shape": (H, W, 3),
                                          "names": ["height", "width", "channels"]},
    "observation.images.third_person_1": {"dtype": "video", "shape": (H, W, 3), "names": [...]},
    "observation.images.wrist":          {"dtype": "video", "shape": (H, W, 3), "names": [...]},
    "observation.state": {"dtype": "float32", "shape": (23,),
                          "names": [f"joint_pos.{i}" for i in range(9)]
                                 + [f"end_effector_pose.{i}" for i in range(7)]
                                 + [f"prev_actions.{i}" for i in range(7)]},
    "action":            {"dtype": "float32", "shape": (7,),
                          "names": [f"osc_action.{i}" for i in range(7)]},
    "visual.profile_id": {"dtype": "int64", "shape": (1,), "names": ["profile_id"]},
}
```

`observation.state`는 계약 HDF5에서 이어 붙인다. 순서는 `joint_position`(9), `actual_ee_pose`
(7, x y z qw qx qy qz), 직전 스텝의 `actions`(7)다. 첫 스텝의 직전 행동은 0으로 채운다.
프로파일 번호는 `nominal_lab=0`, `lab_variation=1`, `stress_tail=2`다.

`fps`는 10, `robot_type`은 `"franka_fr3_osc"`, `use_videos`는 `True`다. `add_frame`에 넘기는
사전에는 `task` 키가 반드시 있어야 하고, 값은
`"Stack three cubes into a three-level tower"`로 고정한다. 둘 다 재익님 수집기와 같은 값이다.

## 5. 단계별 입출력

| 단계 | 실행 파일 | 입력 | 출력 |
|---|---|---|---|
| 생성 | `src/env` + Isaac Lab `generate_dataset.py` | `assets/fwd_annotated.hdf5` | `gen.hdf5`, `gen.provenance.json` |
| 변환 | `src/convert/convert_demo.py` | `gen.hdf5` | `contract.hdf5` |
| 렌더 | `src/render/render_viewpoints.py` | `gen.hdf5` | `rgb.hdf5`, `vrand_log.json` |
| 기록 | `src/lerobot_writer.py` | `contract.hdf5`, `rgb.hdf5`, `vrand_log.json` | `lerobot/` |
| 업로드 | `src/hf_upload.py` | `lerobot/` | 허깅페이스 |

렌더는 생성 산출물을 직접 읽는다. 계약 변환 결과가 아니라 생성 결과를 읽는 이유는
렌더러가 관절 상태를 되돌려 재생하는 방식이고 그 상태가 `gen.hdf5`에만 있기 때문이다.

## 6. 영상과 행동의 시각 맞춤

생성 데이터는 초당 20스텝, 계약 행동은 초당 10개다. 렌더를 `--every 2`로 돌려 처음부터
초당 10장만 만든다. 두 길이가 1 이내로 다를 수 있으므로 프레임 번호가 아니라 **시간으로**
맞춘다. 계약 HDF5의 `timestamps`와 렌더 프레임의 시각(`frame_index / 10.0`)을 비교해 가장
가까운 프레임을 고른다. 길이가 다르면 짧은 쪽에 맞춰 자른다. 기존
`contract/join_rgb_contract.py`의 정수배 검사는 쓰지 않는다. 그 방식은 비가 정확히
정수가 아니면 에피소드를 통째로 버린다.

## 7. 소스 시연 필터

`SOURCE_DEMO_FILTER`가 `exclude_zero_yield`면 `assets/source_yield.json`에 적힌 수율 0인
소스를 제외한다.

거르는 방법은 **원본에서 쓸 시연만 담은 사본을 물리적으로 만드는 것**이다. Isaac Lab Mimic은
필터 키를 읽지 않는다. `DataGenInfoPool.load_from_dataset_file()`이 HDF5의 `data` 그룹 키를
전부 훑어 그대로 불러오고, 설정에서 고를 수 있는 것은 파일 하나뿐이다. 그래서 `mask/train`만
써 넣으면 아무것도 걸러지지 않는다. 사본에 `mask/train`도 함께 쓰지만 그것은 기록용이고,
실제로 효력을 내는 것은 사본에 담긴 시연 목록이다. 원본 `assets/fwd_annotated.hdf5`는 고치지
않는다.
