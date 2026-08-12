# FR3 세 개 큐브 쌓기, 80,000 에피소드 생성 컨테이너

## 이 컨테이너가 만들어 내는 것

컨테이너를 한 번 띄우면 시뮬레이션 안에서 FR3 로봇팔이 큐브 세 개를 쌓는 성공 시연을
80,000개 만들고, 그 결과를 LeRobot 데이터셋 v3 형식으로 정리해서 여러분이 지정한
허깅페이스 저장소에 올린다. 사람이 중간에 손댈 일은 없다. 끊기면 이어서 돌리면 된다.

산출물의 내용은 다음과 같다.

- 카메라 영상 3종이 들어간다. 이름은 `third_person_0`, `third_person_1`, `wrist`다.
- 영상 해상도는 320x180이고 초당 10프레임이며 H.264로 압축된다.
- 행동은 계약 형식 7차원이고 초당 10개다. 계약 형식이 무엇인지는 아래 용어 절에 적었다.
- 상태는 23차원이다. 관절 위치 9개, 손끝 자세 7개, 직전 스텝의 행동 7개를 이어 붙인 값이다.
- 에피소드마다 시각 프로파일 번호가 한 개 붙는다. `nominal_lab`은 0, `lab_variation`은 1,
  `stress_tail`은 2다.
- 과제 문자열은 모든 프레임에서 `Stack three cubes`로 같다.

여러분이 준비할 것은 두 가지뿐이다. 허깅페이스 쓰기 토큰 한 개와 올릴 저장소 이름 한 개다.
둘 다 `.env` 파일에 적는다.

걸리는 시간은 GPU 4장 기준으로 보수적으로 2.9일, 낙관적으로 1.3일이다. 두 숫자 모두 추정치이며
어떤 가정에서 나왔는지는 "걸리는 시간" 절에 그대로 적어 두었다.

## 처음 보는 용어

| 용어 | 뜻 |
|---|---|
| Isaac Lab Mimic | 사람이 만든 시연 몇 개를 새 물체 배치로 옮겨 붙여 새 시연을 만드는 도구다. MimicGen 계열이다. |
| 시도와 수율 | 시연을 한 번 옮겨 붙여 보는 것이 시도이고, 그중 과제를 성공한 비율이 수율이다. 실패한 시도는 버린다. |
| 계약 형식 | 강화학습 팀이 고정해 둔 행동 규격이다. 숫자 7개를 초당 10번 보낸다. 앞의 6개는 현재 손끝 자세 기준 상대 이동량이고 마지막 1개는 그리퍼 신호다. |
| 계약 단위 | 위치는 1단위가 0.02 m, 회전은 1단위가 0.02 rad이다. 그리퍼는 값이 양수면 열기, 0 이하면 닫기다. 기준 좌표계는 로봇 받침이다. |
| LeRobot 데이터셋 v3 | 허깅페이스 LeRobot 라이브러리가 쓰는 저장 형식이다. 영상은 mp4 파일로, 나머지 수치는 parquet 표로 저장한다. |
| 시각 프로파일 | 조명, 재질, 카메라 자세를 얼마나 흔들지 정한 설정 묶음이다. 세 단계가 있고 기본 배분은 50 대 40 대 10이다. |
| 청크 | 에피소드 500개 묶음이다. 한 청크가 끝날 때마다 결과를 저장하고 올린 뒤 중간 파일을 지운다. |

## 파이프라인이 청크 하나에서 하는 일

| 순서 | 하는 일 | 남기는 파일 |
|---|---|---|
| 1 | Isaac Lab Mimic으로 성공 에피소드를 만든다. | `gen.hdf5`, `gen.provenance.json` |
| 2 | 행동을 계약 형식으로 바꾸고 초당 10개로 다시 뽑는다. | `contract.hdf5` |
| 3 | Isaac Sim RTX로 카메라 3대 영상을 렌더링한다. | `rgb.hdf5`, `vrand_log.json` |
| 4 | 영상과 행동을 시각으로 맞춰 LeRobot 형식으로 기록한다. | `lerobot/` |
| 5 | 허깅페이스에 올리고 완료 표시를 남긴다. | `MANIFEST.json` |

3번이 2번 결과가 아니라 1번 결과를 읽는 이유는, 렌더러가 관절 상태를 되돌려 재생하는 방식이고
그 상태 값이 `gen.hdf5`에만 들어 있기 때문이다.

## 여러분이 준비할 것

### 1. `.env` 파일

`hf80k/` 안에 `.env`라는 이름으로 만든다. 아래 내용을 그대로 복사한 뒤 위의 두 줄만 자기 값으로
바꾸면 된다.

```
# 필수 두 개
HF_TOKEN=hf_여기에_쓰기_권한_토큰
HF_REPO_ID=myorg/fr3-cube-stack-80k

# 아래는 기본값이다. 그대로 두어도 돌아간다.
HF_PRIVATE=1
TARGET_EPISODES=80000
PROFILE_SPLIT=nominal_lab:0.50,lab_variation:0.40,stress_tail:0.10
CHUNK_SIZE=500
NUM_ENVS=16
GEN_PROCS=1
RENDER_PROCS=2
IMAGE_WIDTH=320
IMAGE_HEIGHT=180
PHYSICS_PROFILE=robust_stochastic
SOURCE_DEMO_FILTER=exclude_zero_yield
SUBTASK_OFFSETS=10,20
WORK_DIR=/work
KEEP_INTERMEDIATE=0
UPLOAD_EACH_CHUNK=1
RESUME=1
SEED_BASE=42000
LOG_LEVEL=INFO
```

토큰은 허깅페이스 웹의 Settings 화면 Access Tokens에서 만든다. 권한은 write여야 한다. read
토큰으로는 업로드 단계에서 401 오류가 난다. 저장소는 미리 만들어 두지 않아도 된다. 없으면
`HF_PRIVATE=1` 설정에 따라 비공개로 새로 만든다.

`.env`에는 토큰이 평문으로 들어 있다. 만든 뒤 `chmod 600 .env`로 권한을 좁히고, 형상 관리에는
올리지 않는다.

변수 전체 목록과 각 변수의 정확한 뜻은 `INTERFACE.md` 1절에 있다. 위 표는 그중 실제로 손댈 만한
것만 뽑은 것이다.

### 2. 입력 자산

컨테이너 이미지 안에 함께 들어가는 파일들이다. 빌드하기 전에 `hf80k/assets/` 아래에 있어야 한다.

| 파일 | 내용 | 크기 |
|---|---|---|
| `assets/fwd_annotated.hdf5` | 구간 경계가 표시된 사람 시연 원본이다. Isaac Lab Mimic의 입력이다. | 약 3.5 MB |
| `assets/source_yield.json` | 소스 시연별 수율 기록이다. 수율 0인 소스를 걸러내는 데 쓴다. | 수 KB |
| `assets/fr3_binding_v2.yaml` | 손끝 좌표계와 로봇 뼈대 이름을 잇는 설정이다. | 수 KB |
| `assets/fr3_camera_overlay_v2/overlay.yaml` | 실측으로 잡은 카메라 3대의 기준 자세다. | 수 KB |
| `assets/vrand/` | 시각 랜덤화 꾸러미다. HDRI 조명 파일과 카펫 텍스처, 프로파일 YAML이 들어 있다. | 약 1.6 MB |

`fwd_annotated.hdf5`와 `source_yield.json`이 빠져 있으면 빌드는 되지만 첫 청크의 생성 단계에서
바로 멈춘다. 빌드 전에 다섯 항목이 다 있는지 확인한다.

`assets/vrand/`의 위치는 `INTERFACE.md`가 정해 두지 않아서 이 문서에서 정했다. 렌더 스크립트의
`--vrand_root` 기본값과 맞물리는 경로다.

### 3. 장비

| 항목 | 필요한 값 |
|---|---|
| GPU | RTX 렌더링이 되는 NVIDIA GPU가 최소 1장 필요하다. 우리 측정은 L40S 48GB에서 했다. |
| CPU | GPU 1장당 코어 8개를 기준으로 잡았다. |
| 디스크 | GPU 1장으로 전부 만들 때는 300 GB, 4장으로 나눌 때는 장당 100 GB를 비워 둔다. |
| 스왑 | 메모리가 모자랄 때 디스크를 대신 쓰는 공간이다. 최소 16 GB를 권한다. 스왑이 0인 장비에서 렌더 프로세스가 강제 종료된 적이 있다. |
| 네트워크 | 청크마다 업로드가 일어난다. 업로드가 막히면 그 청크는 완료로 표시되지 않는다. |

## 빌드

기반 이미지는 `nvcr.io/nvidia/isaac-lab:3.0.0-beta2-post1`이다. NGC에 로그인이 필요하면
`docker login nvcr.io`를 먼저 한다.

```bash
cd /path/to/mimic_the_mimicgen
docker build -f hf80k/docker/Dockerfile -t fr3-hf80k:1 .
```

빌드 컨텍스트가 `hf80k`가 아니라 그 위 저장소 루트다. Dockerfile이 `hf80k/src`와
`hf80k/assets`를 복사하기 때문이다. `hf80k/scripts/build.sh`를 쓰면 이 경로를 알아서
잡아 준다.

빌드는 기반 이미지 내려받기를 빼면 몇 분이면 끝난다. 파이썬 코드와 자산만 이미지에 복사하기
때문이다.

이 문서는 이미지 이름을 `fr3-hf80k:1`로 적는다. 이름은 정해진 것이 아니므로 바꿔도 된다.
컨테이너 진입점은 `docker/entrypoint.sh`이고, 그것이 `src/orchestrate.py`를 실행한다.

## 실행

### GPU 1장으로 돌리기

```bash
mkdir -p /data/hf80k/gpu0 /data/hf80k/cache/kit /data/hf80k/cache/ov

docker run -d --name fr3-hf80k-gpu0 \
  --gpus '"device=0"' --network host --shm-size=8g \
  --env-file /path/to/mimic_the_mimicgen/hf80k/.env \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y -e OMNI_KIT_ACCEPT_EULA=YES \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e WORK_DIR=/work \
  -v /data/hf80k/gpu0:/work \
  -v /data/hf80k/cache/kit:/isaac-sim/kit/cache \
  -v /data/hf80k/cache/ov:/root/.cache/ov \
  fr3-hf80k:1
```

명령의 각 줄이 왜 있는지는 다음과 같다.

- `--gpus '"device=0"'`는 0번 GPU만 컨테이너에 붙인다. 따옴표 두 겹은 도커 문법이라 그대로 써야
  한다.
- `-e CUDA_VISIBLE_DEVICES=0`은 컨테이너 안에서의 번호다. 도커가 붙여 준 GPU는 안에서 항상 0번이
  되므로, 몇 번 GPU를 쓰든 이 값은 0으로 둔다.
- `--shm-size=8g`는 공유 메모리 부족으로 Isaac Sim이 죽는 것을 막는다.
- `-v /data/hf80k/gpu0:/work`가 결과물이 쌓이는 곳이다. 컨테이너를 지워도 이 디렉터리는 남는다.
- 캐시 두 개를 마운트하면 두 번째 실행부터 Isaac Sim 시작 시간이 크게 줄어든다.
- `ACCEPT_EULA` 계열 세 개는 NVIDIA 이미지가 요구하는 사용 동의다.

### GPU 4장으로 돌리기

GPU마다 컨테이너를 따로 띄운다. 한 컨테이너가 여러 GPU를 쓰지 않는다.

```bash
for g in 0 1 2 3; do
  mkdir -p /data/hf80k/gpu$g
  docker run -d --name fr3-hf80k-gpu$g \
    --gpus "\"device=$g\"" --network host --shm-size=8g \
    --env-file /path/to/mimic_the_mimicgen/hf80k/.env \
    -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y -e OMNI_KIT_ACCEPT_EULA=YES \
    -e CUDA_VISIBLE_DEVICES=0 \
    -e WORK_DIR=/work \
    -e TARGET_EPISODES=20000 \
    -e SEED_BASE=$((42000 + g * 100000)) \
    -e HF_REPO_ID=myorg/fr3-cube-stack-80k-p$g \
    -v /data/hf80k/gpu$g:/work \
    -v /data/hf80k/cache/kit:/isaac-sim/kit/cache \
    -v /data/hf80k/cache/ov:/root/.cache/ov \
    fr3-hf80k:1
done
```

4장으로 나눌 때 반드시 지켜야 할 것이 세 가지다.

- `TARGET_EPISODES`를 20,000으로 낮춘다. 네 컨테이너가 각각 80,000개를 만들면 320,000개가 된다.
- `SEED_BASE`를 컨테이너마다 다르게 준다. 같은 시드를 쓰면 네 컨테이너가 똑같은 장면을 만든다.
  위 예시의 간격 100,000은 청크 번호가 겹치지 않을 만큼 넉넉하다.
- 저장소를 컨테이너마다 따로 준다. 네 컨테이너가 같은 저장소에 동시에 올리면 에피소드 번호와
  메타데이터가 충돌한다. 위 예시는 저장소 이름 뒤에 조각 번호 `-p0`부터 `-p3`까지 붙였다.

작업 디렉터리도 `gpu0`부터 `gpu3`까지 따로 준다. 캐시 디렉터리는 읽기가 대부분이라 네
컨테이너가 같이 써도 된다.

### 조각 저장소 네 개를 하나로 합치기

학습 쪽에서 저장소 네 개를 그대로 같이 읽어도 된다. 하나로 합치고 싶으면 LeRobot의
`aggregate_datasets` 함수를 쓴다. 인자 이름은 이미지에 들어 있는 LeRobot 판본에 따라 다를 수
있으므로 먼저 아래 명령으로 확인한다.

```bash
docker run --rm fr3-hf80k:1 \
  python3 -c "from lerobot.datasets.aggregate import aggregate_datasets; \
help(aggregate_datasets)"
```

이미지 안에서 `python3`을 찾지 못하면 대신 `/workspace/isaaclab/isaaclab.sh -p`를 쓴다. 아래에
나오는 다른 임시 명령들도 마찬가지다.

확인한 이름대로 아래처럼 부른다.

```bash
docker run --rm --network host --env-file .env fr3-hf80k:1 \
  python3 -c "
from lerobot.datasets.aggregate import aggregate_datasets
aggregate_datasets(
    repo_ids=['myorg/fr3-cube-stack-80k-p%d' % i for i in range(4)],
    aggr_repo_id='myorg/fr3-cube-stack-80k',
)
"
```

## 걸리는 시간

### 측정한 값

아래 네 개는 실제로 재서 얻은 값이다.

| 항목 | 값 | 조건 |
|---|---|---|
| 생성 수율 | 15.2% | 수율이 0으로 나온 소스 시연을 제외한 상태에서 잰 값이다. |
| 시도 1회 생성 시간 | 3.17초 | 동시 환경 수 `NUM_ENVS=4`에서 잰 값이다. |
| 카메라 1장 렌더 시간 | 15.1밀리초 | 320x180 RTX 렌더에서 잰 값이다. |
| 에피소드 1개 길이 | 계약 기준 176스텝 | 초당 10스텝 기준이다. 원본은 초당 20스텝으로 352스텝이다. |

여기서 바로 따라 나오는 값이 두 개 있다. 성공 1개를 얻는 데 드는 시도는 평균 6.6회다. 성공
에피소드 1개의 렌더 작업량은 176프레임 곱하기 카메라 3대 곱하기 15.1밀리초로 약 8.0초다.

### 추정한 값

아래 표는 위 측정값에 실행 조건을 얹어 계산한 추정치다. 80,000개를 끝까지 돌려 본 적은 아직 없다.

| 장비 | 보수 추정 | 낙관 추정 |
|---|---|---|
| L40S 1장, CPU 코어 8개 | 11.5일 | 5.1일 |
| L40S 4장, CPU 코어 32개 | 2.9일 | 1.3일 |

가정은 다음과 같다.

- 보수 추정은 동시 환경 수를 4로 두고, 렌더 프로세스 3개가 생성과 겹쳐 도는 조건에서 계산한
  값이다. 위 측정값 네 개만 손으로 곱해서는 같은 숫자가 나오지 않는다. Isaac Sim 시작 시간,
  청크 마무리, 업로드 같은 부대 시간이 계산에 함께 들어가 있기 때문이다.
- 낙관 추정은 동시 환경 수를 4에서 16으로 올리면 생성 처리량이 3배가 된다고 가정한 값이다.
- 그 3배는 SkillGen 실행에서 측정한 수치다. MimicGen에서는 확인하지 않았다. 확인되지 않은
  가정이므로, 일정을 약속할 때는 보수 추정 쪽을 쓰는 것이 안전하다.
- GPU 4장 값은 1장 값을 단순히 4로 나눈 것이다. 업로드 대역폭 경합과 디스크 입출력 경합은 넣지
  않았다.
- 기본 설정 `NUM_ENVS=16`은 이미 낙관 쪽 조건이다. 보수 추정 조건으로 돌려 보려면 `NUM_ENVS=4`로
  낮춘다.

### 첫 청크로 어느 쪽인지 확인하기

첫 청크가 끝나면 `MANIFEST.json`의 `durations_s` 항목에 단계별 소요 시간이 초 단위로 적힌다. 청크
하나가 500개이므로, 전체 예상 시간은 그 값에 `TARGET_EPISODES / 500`을 곱하면 된다. 이 계산이
위 표보다 정확하다. 반나절 뒤에 한 번 확인하는 것을 권한다.

## 디스크

- 중간 파일은 청크가 끝날 때마다 지운다. `gen.hdf5`, `contract.hdf5`, `rgb.hdf5`가 지워지는
  대상이다. 디버깅 때문에 남기고 싶으면 `KEEP_INTERMEDIATE=1`로 둔다. 대신 청크마다 수십 GB가
  계속 쌓인다.
- 렌더 프로세스를 3개 동시에 돌릴 때 최대 사용량은 약 43 GB다. 이 값은 청크 하나분 중간 파일
  크기에서 계산한 것이고, 장시간 실측한 값은 아니다. 기본값인 2개에서는 이보다 적다.
- 완성된 데이터셋은 약 100 GB에서 200 GB 사이로 예상한다. 영상 압축률에 따라 달라진다. 이 값도
  추정치다.
- 최종본을 로컬에 합치면 그만큼의 공간이 더 필요하다. 작업 디렉터리에는 최종본 크기에 중간 파일
  최대치를 더한 만큼을 비워 둔다.
- GPU 4장으로 나눠 돌리면 작업 디렉터리 하나가 담는 양은 4분의 1이 된다. 합계는 같다.

## 진행 확인

컨테이너가 살아 있는지, 어디까지 갔는지 확인하는 명령이다.

```bash
# 컨테이너 상태와 최근 로그
docker ps --filter name=fr3-hf80k
docker logs -f fr3-hf80k-gpu0

# 지금 돌고 있는 청크의 상세 로그
ls -t /data/hf80k/gpu0/logs | head -1
tail -f /data/hf80k/gpu0/logs/chunk_00012.log

# 전체 진행 상황 요약
cat /data/hf80k/gpu0/state.json

# 끝난 청크 하나의 요약
cat /data/hf80k/gpu0/chunks/chunk_00003/MANIFEST.json

# GPU 사용률
nvidia-smi
```

지금까지 확보한 성공 에피소드 총 개수는 다음처럼 센다. 작업 디렉터리 네 개를 한꺼번에 센다.

```bash
python3 - <<'PY'
import glob, json
total, done, uploaded = 0, 0, 0
for path in sorted(glob.glob("/data/hf80k/gpu*/chunks/chunk_*/MANIFEST.json")):
    with open(path) as handle:
        manifest = json.load(handle)
    if manifest.get("status") != "done":
        continue
    done += 1
    total += int(manifest.get("episodes", 0))
    uploaded += 1 if manifest.get("uploaded") else 0
print(f"완료 청크 {done}개, 에피소드 {total}개, 업로드 끝난 청크 {uploaded}개")
PY
```

`MANIFEST.json`의 `yield` 값도 같이 보면 좋다. 15.2%에서 크게 떨어져 있으면 소스 시연 필터나
물리 프로파일 설정이 의도와 다르게 들어갔을 가능성이 있다.

## 중단한 뒤 이어하기

`RESUME=1`이 기본값이다. `MANIFEST.json`이 있고 그 안의 `status`가 `done`인 청크는 건너뛴다.
따라서 처음 띄울 때 쓴 `docker run` 명령을 그대로 다시 실행하면 이어서 진행한다.

한 가지만 손으로 정리해 주어야 한다. 중단된 순간에 만들던 청크는 파일이 반쯤 쓰인 상태로 남는다.
이 디렉터리는 `MANIFEST.json`이 없으므로 아래 명령으로 찾아서 지운다.

```bash
for d in /data/hf80k/gpu0/chunks/chunk_*; do
  [ -f "$d/MANIFEST.json" ] || { echo "지움 $d"; rm -rf "$d"; }
done
```

지워도 손실은 그 청크 하나뿐이다. 시드는 `SEED_BASE`에 청크 번호를 더한 값으로 정해지므로, 다시
만들면 같은 장면이 다시 나온다.

컨테이너를 멈출 때는 `docker stop fr3-hf80k-gpu0`을 쓴다. `docker kill`은 쓰지 않는 것이 좋다.
쓰기 도중에 강제로 죽이면 아래 문제 해결 절의 두 번째 항목에 나오는 손상된 HDF5가 생긴다.

## 알면서 적용하지 않은 것

시각 랜덤화 규격에는 커튼과 벽의 색과 재질 범위가 정해져 있다. 항목 이름은 `curtain_side_back`과
`curtain_front`다. 그런데 우리 생성 장면에는 벽과 천장 같은 방 구조물의 3D 모델이 없다. 책상과
로봇과 큐브와 바닥만 있다. 색을 칠할 대상 자체가 없으므로 이 항목들은 적용하지 않는다.

적용하지 않았다는 사실은 숨기지 않고 기록한다. 각 청크의 `vrand_log.json` 안에 `skipped` 목록이
있고, 거기에 건너뛴 항목 이름이 그대로 들어간다. 같은 목록에는 장면에 대응하는 물체가 없어서
건너뛴 다른 항목도 함께 쌓인다. 데이터를 받은 뒤 이 목록을 한 번 열어 보기를 권한다.

이 차이가 학습에 영향을 주는 지점은 배경이다. 조명과 바닥과 책상 재질은 규격대로 흔들리지만,
벽과 커튼은 장면에 없다. 배경 변화에 강한 정책을 목표로 한다면 이 점을 감안해야 한다.

## 문제가 생겼을 때

여기 적은 두 가지는 우리가 실제로 겪은 실패다.

### 렌더 프로세스가 메모리 부족으로 강제 종료된다

- 증상은 렌더 단계 로그가 문장 중간에서 끊기고, 프로세스 종료 코드가 137로 찍히는 것이다.
  파이썬 예외는 남지 않는다.
- 확인은 `dmesg -T | grep -i "killed process"`로 한다. 커널의 메모리 부족 강제 종료 장치가
  어떤 프로세스를 죽였는지 시각과 함께 나온다.
- 원인은 스왑 공간이 없는 장비에서 렌더 프로세스 여러 개가 동시에 최대 메모리를 쓰는 것이다.
  스왑이 0이면 커널이 메모리 내용을 디스크로 옮겨 둘 수 없어서 프로세스를 강제 종료한다.
- 조치 하나는 `RENDER_PROCS`를 2 또는 1로 낮추는 것이다. 렌더 시간이 늘어나는 대신 최대 메모리
  사용량이 줄어든다.
- 조치 둘은 스왑을 만드는 것이다. `sudo fallocate -l 32G /swapfile && sudo chmod 600 /swapfile
  && sudo mkswap /swapfile && sudo swapon /swapfile`로 만든다.
- 조치 셋은 죽은 청크 디렉터리를 지우고 다시 실행하는 것이다. 위 "중단한 뒤 이어하기"의 정리
  명령을 그대로 쓰면 된다.

### 중간에 끊긴 HDF5 파일이 깨져 있다

- 증상은 변환 단계나 렌더 단계가 시작하자마자 `OSError: Unable to open file (file signature not
  found)` 또는 파일이 잘렸다는 메시지로 실패하는 것이다.
- 원인은 HDF5 파일이 쓰기 도중에 프로세스가 죽으면 파일 머리말이 완성되지 않는 것이다. 위의 강제
  종료나 `docker kill` 뒤에 나온다.
- 확인은 아래 명령으로 한다. 정상이면 에피소드 개수가 찍히고, 깨졌으면 위 오류가 그대로 난다.

```bash
docker run --rm -v /data/hf80k/gpu0:/work fr3-hf80k:1 \
  python3 -c "import h5py; f=h5py.File('/work/chunks/chunk_00012/gen.hdf5','r'); \
print(len(f['data']))"
```

- 조치는 그 청크 디렉터리를 통째로 지우고 다시 실행하는 것이다. 부분 복구는 시도하지 않는다.
  들이는 시간에 비해 얻는 것이 적고, 어차피 같은 시드로 같은 내용을 다시 만들 수 있다.
- 이미 완료 표시가 붙은 청크는 안전하다. `MANIFEST.json`은 그 청크의 모든 파일을 다 쓴 뒤 마지막에
  기록하기 때문이다.

## 더 알아야 할 때 볼 파일

- `INTERFACE.md`는 환경 변수 전체 목록, 디렉터리 배치, `MANIFEST.json` 형식, LeRobot 특성 이름을
  정확히 적어 둔 규격 문서다.
- `src/convert/convert_demo.py`의 첫 주석은 행동을 계약 형식으로 바꾸는 절차를 단계별로 설명한다.
- `src/render/visual_randomization.py`의 첫 주석은 시각 랜덤화 규격을 우리 렌더 루프에 옮기면서
  무엇을 그대로 따랐고 무엇을 바꿨는지 적어 두었다.
