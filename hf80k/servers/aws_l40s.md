# AWS L40S 인스턴스

접속 별칭은 각자의 `~/.ssh/config`에 적어 둔다. 주소는 인스턴스를 멈추고 다시 켜면
바뀌므로 여기에 적지 않는다.

## 가장 먼저 알아야 할 것

**`/opt/dlami/nvme`는 인스턴스를 멈출 때 지워진다.** 이 기계의 큰 디스크가 거기라서
에셋과 생성 결과를 전부 그 아래에 두게 되는데, 멈추면 다 사라진다. 2026-10-08에
핀 꽂기와 큐브의 합친 데이터셋을 이렇게 잃었다.

그래서 이 기계에서 일을 끝내면 멈추기 전에 반드시 다른 곳으로 옮긴다. 옮길 때는 파일
개수와 해시를 양쪽에서 맞춰 보고, 정렬 순서가 기계마다 다르므로 `LC_ALL=C`로 고정한다.

```bash
find <폴더> -type f -print0 | sort -z | xargs -0 sha256sum | LC_ALL=C sort | sha256sum
```

도커 이미지는 `/var/lib/docker`에 있고 그쪽은 루트 디스크라 멈춰도 남는다. 이미지를
다시 받을 걱정은 없다.

## 기계

- GPU는 NVIDIA L40S 한 장이고 46 GB다.
- 스레드 16개, 메모리 124 GiB다. 연구실 RTX PRO 6000 서버의 128스레드에 비하면 작아서
  생성 프로세스를 여러 개 띄우기 어렵다.
- 루트 디스크가 290 GB인데 2026-10-22 기준 268 GB를 써서 93% 찼다. 큰 것을 받기 전에
  이것부터 확인한다.
- 지워지는 임시 디스크 `/opt/dlami/nvme`는 549 GB이고 넉넉하다.

## 기어 파이프라인을 띄우는 법

컨테이너 이미지는 `fr3-hf80k:16`이다. 이미지의 기본 시작 스크립트가 `/work/.env`를 읽어
파이프라인 전체를 돌린다.

```bash
docker run -d --name <실행이름> --gpus all --network host --shm-size=8g \
  -e CUDA_VISIBLE_DEVICES=0 \
  -v /opt/dlami/nvme/gear_assets:/opt/hf80k/assets:ro \
  -v <홈>/hf80k_live/src:/opt/hf80k/src:ro \
  -v <홈>/hf80k_live/tools:/opt/hf80k/tools:ro \
  -v /opt/dlami/nvme/cache/kit:/isaac-sim/kit/cache \
  -v /opt/dlami/nvme/cache/ov:/root/.cache/ov \
  -v <작업폴더>:/work fr3-hf80k:16
```

소스 코드를 홈 디렉터리에 두는 이유가 있다. 홈은 멈춰도 남으므로 코드를 매번 다시
보내지 않아도 된다. 에셋과 결과만 지워지는 디스크에 둔다.

진단 도구 하나만 돌릴 때는 시작 스크립트를 덮어쓴다. 필요한 환경변수는 시작 스크립트가
넣어 주던 것이라서 직접 줘야 한다.

```bash
docker run --rm --gpus all --network host --shm-size=8g \
  -e CUDA_VISIBLE_DEVICES=0 -e HEADLESS=1 \
  -e PYTHONPATH=/opt/hf80k/src/env_gear:/opt/hf80k/src/env:/opt/hf80k/src \
  -e LAB_GEAR_TABLE_USD=/opt/hf80k/assets/legacy_multi_gear/assets/lab/table_scene.usdc \
  -e LAB_GEAR_ASSET_ROOT=/opt/hf80k/assets/legacy_multi_gear/assets/gears \
  -e LAB_ARM_SCALE=1.0 \
  <위와 같은 마운트들> \
  --entrypoint bash fr3-hf80k:16 -lc \
  "/workspace/isaaclab/isaaclab.sh -p /opt/hf80k/tools/<도구>.py ..."
```

## 생성 속도

2026-10-22에 기어 조립 통짜 합성을 환경 8개로 돌려 쟀다. 시도 134번에 4,341초가 걸려
시도 하나에 32초다. 성공한 에피소드 48편을 72분에 만들었다. 생성은 일부러 `--device cpu`로
돈다. 이유는 `hf80k/src/orchestrate.py`에 적어 두었다.

## 호스트에서 데이터 파일을 열 때

호스트의 `python3`에 h5py가 들어 있다. 다만 `-I`(격리 모드)로 띄우면 불러오지 못한다.
`-I` 없이 돌려야 한다. 신뢰할 수 없는 파일을 다루는 것이 아니라면 그렇게 쓴다.
