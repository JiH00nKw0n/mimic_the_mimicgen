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
# HF80K_LEROBOT_PATH goes on PYTHONPATH ONLY for the writer and upload steps
# (src/lerobot_writer.py, src/hf_upload.py). The generate and render steps run
# inside Isaac and must not see it.
ENV HF80K_LEROBOT_PATH=/opt/hf80k/deps/lerobot_0_4_py311
# 기록·업로드 스크립트가 찾는 이름. 둘을 같은 값으로 두어 어긋나지 않게 한다.
ENV LEROBOT_SITE=/opt/hf80k/deps/lerobot_0_4_py311
RUN . /opt/hf80k/isaac_python.env \
 && "$HF80K_PYTHON" -m pip install --no-cache-dir --target "$HF80K_LEROBOT_PATH" \
      "lerobot>=0.4" "huggingface_hub>=0.34" \
 && test -d "$HF80K_LEROBOT_PATH/lerobot" \
 && test -d "$HF80K_LEROBOT_PATH/huggingface_hub"

# Pipeline code and assets. src/ is small; assets/ is the 90 MB that
# scripts/fetch_assets.sh puts there before the build.
COPY hf80k/src /opt/hf80k/src
COPY hf80k/assets /opt/hf80k/assets
COPY hf80k/docker/entrypoint.sh /opt/hf80k/entrypoint.sh
RUN chmod +x /opt/hf80k/entrypoint.sh

# render_viewpoints.py defaults --vrand_root/--vrand_config to /vrand; keep that
# default working so a debug run inside the container needs no extra flags.
RUN ln -sfn /opt/hf80k/assets/fr3_visual_randomization_v1 /vrand

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

# 빌드 산출물을 실행 사용자에게 넘기고 root에서 내려온다.
RUN chown -R 1000:1000 /opt/hf80k
USER ubuntu

# Defaults that match INTERFACE.md; .env or `docker run -e` override them.
ENV HF80K_SRC=/opt/hf80k/src \
    HF80K_ASSETS=/opt/hf80k/assets \
    WORK_DIR=/work

WORKDIR /opt/hf80k/src
ENTRYPOINT ["/opt/hf80k/entrypoint.sh"]
