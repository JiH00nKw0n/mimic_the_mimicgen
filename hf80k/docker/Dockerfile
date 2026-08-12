# hf80k image: Isaac Lab 3.0 plus the FR3 three-cube pipeline
# (generate -> convert -> render -> LeRobot -> Hugging Face).
#
# Everything the pipeline needs is baked in EXCEPT state and secrets: /work is a
# bind mount (chunks, logs, state.json survive the container) and HF_TOKEN
# arrives through the environment at run time. Nothing here reads it.
#
# Build from the repository root so the COPY paths below resolve:
#   DOCKER_BUILDKIT=1 docker build -f hf80k/docker/Dockerfile -t fr3-hf80k:latest .
# scripts/build.sh does exactly that. docker/Dockerfile.dockerignore keeps the
# context down to hf80k/ so the 90 MB of assets is all that travels.
FROM nvcr.io/nvidia/isaac-lab:3.0.0-beta2-post1

# Kit refuses to start without these, and it refuses interactively — a missing
# EULA variable looks like a hang, not an error. Set once for both build-time
# RUN steps and the entrypoint.
ENV ACCEPT_EULA=Y \
    PRIVACY_CONSENT=Y \
    OMNI_KIT_ACCEPT_EULA=YES \
    PYTHONUNBUFFERED=1

# 기반 이미지는 uid 1000(ubuntu)으로 돈다. apt와 /opt 쓰기에는 root가 필요하므로
# 빌드 동안만 root로 올라갔다가 마지막에 되돌린다. 실행까지 root로 두면 /work 바인드
# 마운트에 쌓이는 파일이 호스트에서 root 소유가 되어 운영자가 지울 수 없다.
USER root

# LeRobot v3 encodes the three camera streams to H.264 by shelling out to
# ffmpeg. Some Isaac Lab tags ship it and some do not, so ask first.
RUN if ! command -v ffmpeg > /dev/null 2>&1; then \
      apt-get update \
      && apt-get install -y --no-install-recommends ffmpeg \
      && rm -rf /var/lib/apt/lists/* ; \
    else echo "[build] ffmpeg already present: $(ffmpeg -version | head -n 1)" ; fi

# Isaac Lab hides its interpreter behind isaaclab.sh. Ask the wrapper once, here,
# for the real interpreter path and the two variables it exports, and record
# them. The entrypoint then `exec`s that binary directly instead of the wrapper:
# bash does not forward SIGTERM to a foreground child, so execing the wrapper
# would make `docker stop` a 10-second wait followed by SIGKILL mid-chunk.
# /vrand는 파일시스템 최상위라 root만 만들 수 있다. 가리키는 대상은 나중에 COPY되지만
# 심볼릭 링크는 대상이 없어도 만들어진다.
RUN ln -sfn /opt/hf80k/assets/fr3_visual_randomization_v1 /vrand

# 아래 모든 쓰기를 실행 사용자로 하기 위해 디렉터리 하나만 미리 넘긴다. 이렇게 하면
# 마지막에 chown -R을 걸 필요가 없다. chown -R은 오버레이 파일시스템에서 아래층 파일을
# 전부 위층으로 복사하게 만들어, 5 GB짜리 레이어와 20분을 그냥 더한다.
RUN mkdir -p /opt/hf80k && chown 1000:1000 /opt/hf80k
USER ubuntu

RUN mkdir -p /opt/hf80k \
 && printf '%s\n' \
      'import os, sys' \
      'e = os.environ' \
      'print("HF80K_PYTHON=" + sys.executable)' \
      'print("HF80K_LD_LIBRARY_PATH=" + e.get("LD_LIBRARY_PATH", ""))' \
      'print("HF80K_PYTHONPATH=" + e.get("PYTHONPATH", ""))' \
      > /opt/hf80k/_probe.py \
 && { /workspace/isaaclab/isaaclab.sh -p /opt/hf80k/_probe.py || true ; } \
      | grep '^HF80K_' > /opt/hf80k/isaac_python.env \
 || echo "HF80K_PYTHON=$(command -v python3)" > /opt/hf80k/isaac_python.env
RUN cat /opt/hf80k/isaac_python.env && rm -f /opt/hf80k/_probe.py

# lerobot + huggingface_hub live in their own directory, never in the Isaac
# site-packages. Isaac Sim pins its own torch/numpy build; letting a resolver
# touch that tree is how you get a simulator that imports but renders nothing.
# The RL team keeps the same split on the box
# (/home/ubuntu/jake/aidas/deps/lerobot_0_4_4_py311); this is that layout moved
# into the image. `--target` was chosen over a venv because the Isaac
# interpreter is an embedded Kit python with no guaranteed ensurepip, while
# --target only needs pip itself. pip >= 20 no longer implies --ignore-installed
# for --target, so requirements the Isaac interpreter already satisfies (torch,
# numpy, opencv) are skipped and nothing heavy is duplicated.
#
# 판본 조합은 실측으로 정했다. lerobot[dataset]이 datasets를 끌어오고, datasets가
# huggingface-hub<1.0을 요구한다. 상한을 안 걸면 hub 1.27이 깔려 datasets가
# "huggingface-hub>=0.34.0,<1.0 is required"로 죽는다. 상한을 걸면 lerobot 0.4.4,
# hub 0.35.3, datasets 4.8.5로 풀리고 LeRobotDataset import가 통과한다. 0.4.4는
# 재익님 쪽이 쓰는 판본과 같아서(lerobot_0_4_4_py311) 데이터 호환에도 유리하다.
#
# HF80K_LEROBOT_PATH goes on PYTHONPATH ONLY for the writer and upload steps
# (src/lerobot_writer.py, src/hf_upload.py). The generate and render steps run
# inside Isaac and must not see it.
ENV HF80K_LEROBOT_PATH=/opt/hf80k/deps/lerobot_0_4_py311
# 기록·업로드 스크립트가 찾는 이름. 둘을 같은 값으로 두어 어긋나지 않게 한다.
ENV LEROBOT_SITE=/opt/hf80k/deps/lerobot_0_4_py311
RUN . /opt/hf80k/isaac_python.env \
 && "$HF80K_PYTHON" -m pip install --no-cache-dir --target "$HF80K_LEROBOT_PATH" \
      "lerobot[dataset]>=0.4,<0.5" "huggingface_hub>=0.34,<1.0" \
 && test -d "$HF80K_LEROBOT_PATH/lerobot" \
 && test -d "$HF80K_LEROBOT_PATH/huggingface_hub"

# Pipeline code and assets. src/ is small; assets/ is the 90 MB that
# scripts/fetch_assets.sh puts there before the build.
COPY --chown=1000:1000 hf80k/src /opt/hf80k/src
COPY --chown=1000:1000 hf80k/assets /opt/hf80k/assets
COPY --chown=1000:1000 hf80k/docker/entrypoint.sh /opt/hf80k/entrypoint.sh
RUN chmod +x /opt/hf80k/entrypoint.sh

# The lab table USD ships inside the visual randomization package, but its file
# name is the package's business, not ours. Resolve it at build time and expose
# one stable path. If it is absent the symlink is absent too, and
# lab_mimic_cfg.py falls back to its desk slab with a warning instead of dying.
RUN T=/opt/hf80k/assets/fr3_visual_randomization_v1/assets/table ; \
    F="$(ls "$T"/table_scene.usd* 2>/dev/null | head -n 1)" ; \
    [ -n "$F" ] || F="$(ls "$T"/*.usd* 2>/dev/null | head -n 1)" ; \
    if [ -n "$F" ]; then ln -sfn "$F" /opt/hf80k/assets/table_scene.usdc ; \
      echo "[build] table USD -> $F" ; \
    else echo "[build] WARNING: no table USD under $T (slab fallback)" ; fi
ENV LAB_TABLE_USD=/opt/hf80k/assets/table_scene.usdc

# Defaults that match INTERFACE.md; .env or `docker run -e` override them.
ENV HF80K_SRC=/opt/hf80k/src \
    HF80K_ASSETS=/opt/hf80k/assets \
    WORK_DIR=/work

WORKDIR /opt/hf80k/src
# 기반 이미지의 헬스체크는 Isaac이 컨테이너 안에서 계속 떠 있다고 보고 Kit 로그의
# AppReady를 찾는다. 우리는 오케스트레이터가 단계마다 Isaac을 자식 프로세스로 띄웠다
# 내리므로 그 로그가 없어 항상 unhealthy가 된다. 무해해 보이지만 compose나 쿠버네티스가
# 헬스 조건을 걸면 멀쩡한 컨테이너를 계속 재시작한다. 우리 기준으로 바꾼다.
# 판정 기준은 하나다. 오케스트레이터가 살아 있는가.
HEALTHCHECK --interval=60s --timeout=10s --start-period=300s --retries=3 \
  CMD pgrep -f "orchestrate.py" > /dev/null 2>&1 || exit 1

ENTRYPOINT ["/opt/hf80k/entrypoint.sh"]
