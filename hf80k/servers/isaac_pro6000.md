# 연구실 RTX PRO 6000 서버

접속 별칭은 각자의 `~/.ssh/config`에 적어 둔다. 2026-10-21과 2026-10-22에 직접 재서 확인한 값만 적는다.

## 왜 이 기계를 쓰나

지워지는 임시 디스크가 없다. AWS에서는 인스턴스를 멈출 때마다 작업물이 사라져서 매번
옮겨야 했는데 여기서는 두어도 된다. 그리고 스레드가 128개라 생성 프로세스를 여러 개
띄우는 데 유리하다.

## 기계

- GPU는 NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition 네 장이고 장당 약 96 GB다.
- 0번과 1번이 한 CPU 소켓에, 2번과 3번이 다른 소켓에 붙어 있다. 네 장을 한꺼번에 쓰는
  학습은 이 배치를 맞춰야 한다. 장마다 독립으로 돌리는 hf80k 생성에는 영향이 없다.
- CPU는 두 소켓에 물리 코어 64개, 스레드 128개다. 메모리는 251 GiB다.
- 디스크는 단일 7.0 TB이고 2026-10-22 기준 5.6 TB가 비어 있다.

## 공용 계정이라 생기는 제약

계정 하나를 여러 사람이 쓴다. 작업 폴더가 `/home/<공용계정>/workspace/<사람>` 아래에
사람별로 있고, 컨테이너 안에서는 자기 작업 폴더가 `/workspace/project`로 보인다.

**GPU 예약은 실제 메모리 사용량이 아니라 컨테이너에 붙은 표딱지로 판단한다.** 표딱지
이름은 `org.local.isaac.gpus`다. 메모리가 0 MiB여도 표딱지가 있으면 `isaac-run`이
`GPU N is reserved/busy`로 거절한다. 반대로 표딱지가 없어도 메모리를 64 MiB 넘게 쓰면
외부 사용으로 보고 막는다. 대기열이 없으므로 잡혀 있으면 쓰는 사람에게 말해야 한다.
현황은 `isaac-status`로 본다. 남의 컨테이너를 내릴 수 있지만 그 사람 작업이 날아가므로
하지 않는다.

## 컨테이너를 띄우는 두 가지 길

### 공용 이미지로는 기어 태스크가 뜨지 않는다

`isaac-run`이 띄우는 공용 이미지는 Isaac Lab 3.0.0-rc1이고 Isaac Sim 6.1.0-rc.26이다.
2026-10-22에 이 이미지로 기어 재생 도구를 돌려 보니 이렇게 멈춘다.

```
ModuleNotFoundError: No module named 'isaaclab_tasks.manager_based'
```

우리 기어 태스크를 gym에 등록하는 `hf80k/src/env_gear/gear_register.py`가 공식 큐브
쌓기 환경 설정을 `isaaclab_tasks.manager_based` 아래에서 가져오는데, 3.0.0-rc1 빌드에는
그 경로가 없다. 그래서 공용 이미지로는 기어 쪽 어떤 실행도 되지 않는다. 큐브나 핀 꽂기는
확인하지 않았다.

`isaac-run`에는 이미지를 바꾸는 선택지가 없다. 받는 인자가 `--workspace`, `--gpu`,
`--detach`, `--cpus`, `--memory`뿐이다.

### 우리 이미지를 `docker run`으로 직접 띄운다

`fr3-hf80k:16`이 이 기계의 도커 이미지 저장소에 이미 있다. 약 33.6 GB다. 이쪽은 Isaac
Lab 3.0.0에 Isaac Sim 6.0.1-rc.7이고 기어 태스크가 정상으로 뜬다. 2026-10-22에 시연
20편 재생과 명령 좌표계 측정을 이 길로 돌렸다.

**이 길은 GPU 예약 장부를 우회한다.** 그래서 띄우기 전에 `isaac-status`로 장부를 보고,
내 표딱지가 붙은 장만 `CUDA_VISIBLE_DEVICES`로 지정해야 한다. 그러지 않으면 같은 장을
두 사람이 모르고 나눠 쓰다 메모리가 터진다. 우리 이미지를 장부에 올려 쓸 수 있는지는
관리하는 쪽에 물어봐야 하고 아직 묻지 않았다.

띄우는 명령은 `hf80k/servers/run_gear.sh`에 들어 있다. 직접 치려면 이렇게 한다.
`<내작업폴더>`는 `/home/<공용계정>/workspace/<사람>`이다.

```bash
docker run --rm --gpus all --network host --shm-size=8g \
  -e CUDA_VISIBLE_DEVICES=0 -e HEADLESS=1 \
  -e PYTHONPATH=/opt/hf80k/src/env_gear:/opt/hf80k/src/env:/opt/hf80k/src \
  -e LAB_GEAR_TABLE_USD=/opt/hf80k/assets/legacy_multi_gear/assets/lab/table_scene.usdc \
  -e LAB_GEAR_ASSET_ROOT=/opt/hf80k/assets/legacy_multi_gear/assets/gears \
  -e LAB_ARM_SCALE=1.0 \
  -v <내작업폴더>/gear_assets:/opt/hf80k/assets:ro \
  -v <내작업폴더>/hf80k_live/src:/opt/hf80k/src:ro \
  -v <내작업폴더>/hf80k_live/tools:/opt/hf80k/tools:ro \
  -v <내작업폴더>/cache/kit:/isaac-sim/kit/cache \
  -v <내작업폴더>/cache/ov:/root/.cache/ov \
  -v <내작업폴더>/<실행폴더>:/work \
  --entrypoint bash fr3-hf80k:16 -lc \
  "/workspace/isaaclab/isaaclab.sh -p /opt/hf80k/tools/gear_replay_probe.py ..."
```

`--entrypoint bash`를 빼면 이미지의 기본 시작 스크립트가 돌면서 `/work/.env`를 읽고
파이프라인 전체를 실행한다. 진단 도구 하나만 돌릴 때는 위처럼 `--entrypoint bash`로
덮어쓴다.

`HEADLESS=1`이 필요한 이유가 따로 있다. Isaac Sim 6.1.0의 `annotate_demos.py`와
`generate_dataset.py`는 `--headless` 인자를 거부하고 `--visualizer`를 받는다. 6.0.1은
`--headless`를 받는다. 스크립트를 들여다봐도 구분되지 않는데, AppLauncher가 인자를
나중에 더하기 때문이다. `hf80k/src/annotate.py`와 `hf80k/src/orchestrate.py`는 그래서
`--headless`로 한 번 시도하고 거부당하면 그 인자를 떼고 `HEADLESS=1`을 주어 다시 띄운다.

## 이 기계에 있는 기어 작업물

| 경로 | 무엇인가 |
|---|---|
| `<내작업폴더>/hf80k_live/{src,tools}` | 이 저장소의 `hf80k/src`와 `hf80k/tools`를 rsync로 보내 둔 것. 컨테이너가 읽기 전용으로 붙인다 |
| `<내작업폴더>/gear_assets` | 기어와 받침 에셋, 그리고 소스 시연 파일 |
| `<내작업폴더>/cache/{kit,ov}` | Kit 셰이더와 에셋 캐시 |
| `<내작업폴더>/aws_nvme_20261022` | AWS를 끄기 전에 옮겨 둔 기어 작업물 338 MB. 파일 536개의 해시를 양쪽에서 맞춰 확인했다 |

`aws_nvme_20261022/multigear/refcode/teleop_multigear_lab_v3.py`는 사람이 기어 시연을
녹화할 때 쓴 스크립트인데 이 저장소에 없다. 물리 설정을 녹화 당시와 맞추어 비교할 때
이 파일이 유일한 근거다.

## AWS와 자료를 주고받는 방향

이 서버에서 AWS의 22번 포트로는 나간다. **AWS에서 이 서버로는 들어오지 못한다.** 학교 쪽
방화벽이 막아서 `Connection reset by peer`로 끊긴다. 2026-10-22에 다시 확인했다. ICMP도
막혀 있어 ping으로는 판단할 수 없다.

그래서 길이 둘이다.

1. 이 서버에서 AWS로 rsync를 걸어 당겨온다. 한 연결로는 1.52 MiB/s밖에 안 나오고,
   4흐름 6.71 MiB/s, 8흐름 15.44 MiB/s에서 포화한다. 16흐름은 오히려 느리고 AWS 쪽
   동시 접속 제한에 걸려 몇 개가 끊긴다. 붙는 순서를 1초씩 띄워야 그 제한을 피한다.
2. 양쪽에 ssh로 붙을 수 있는 제3의 기계를 거쳐 흘려보낸다. 2026-10-22에 338 MB를 이
   방식으로 옮겼다. 중간 디스크를 쓰지 않는다.

```bash
ssh <AWS별칭> 'tar -cf - -C <보낼폴더의부모> <폴더이름>' \
  | ssh <이서버별칭> 'tar -xf - -C <받을폴더>'
```

옮긴 뒤에는 파일 개수와 해시를 맞춰 본다. 파일 이름 정렬 순서가 기계마다 다르므로
`LC_ALL=C`로 정렬을 고정하지 않으면 내용이 같은데도 다르게 나온다.

```bash
find <폴더> -type f -print0 | sort -z | xargs -0 sha256sum | LC_ALL=C sort | sha256sum
```
