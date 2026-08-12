#!/usr/bin/env bash
# Container entrypoint: load /work/.env, run the preflight, then hand the process
# over to the orchestrator.
#
# Two rules drive the shape of this script.
#
# 1) Real environment variables beat the file. The 4-GPU launch keeps ONE shared
#    .env on disk and gives each container its own CUDA_VISIBLE_DEVICES,
#    TARGET_EPISODES and SEED_BASE with `docker run -e`. If the file won, that
#    loop would silently produce four identical runs.
# 2) SIGTERM must reach python. We `exec` the interpreter so it replaces this
#    shell; the interpreter path was resolved at build time (see the Dockerfile)
#    precisely so we can exec a real binary instead of the isaaclab.sh wrapper,
#    which would swallow the signal. Note that a process running as PID 1 only
#    sees signals it has a handler for, so orchestrate.py installs one — and
#    scripts/run.sh passes --init so tini is PID 1 and reaps the Kit children.
#
# Debug escape hatch: `docker run ... fr3-hf80k:latest shell` drops to bash with
# the same environment instead of starting a run.
set -euo pipefail

SRC="${HF80K_SRC:-/opt/hf80k/src}"
ENV_FILE="${HF80K_ENV_FILE:-/work/.env}"

# --- .env loading -----------------------------------------------------------
# A deliberately small parser: KEY=VALUE, optional `export ` prefix, one layer of
# surrounding quotes, `#` comments (whole-line, or trailing on unquoted values).
# Anything fancier belongs in the file's author's shell, not here.
load_env_file() {
  local file="$1" line key value
  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line%$'\r'}"                          # tolerate CRLF
    line="${line#"${line%%[![:space:]]*}"}"       # drop leading whitespace
    [[ -z "$line" || "$line" == \#* ]] && continue
    [[ "$line" == export\ * ]] && line="${line#export }"
    [[ "$line" == *=* ]] || continue
    key="${line%%=*}"
    value="${line#*=}"
    [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
    if [[ "$value" == \"*\" || "$value" == \'*\' ]]; then
      value="${value:1:${#value}-2}"
    else
      value="${value%%[[:space:]]#*}"             # trailing comment
      value="${value%"${value##*[![:space:]]}"}"  # right-trim
    fi
    # Unset OR empty in the real environment means the file may fill it in;
    # `docker run -e HF_TOKEN` with an empty host value should not blank it out.
    if [[ -z "${!key:-}" ]]; then
      export "$key=$value"
    fi
  done < "$file"
}

if [[ -f "$ENV_FILE" ]]; then
  echo "[entrypoint] loading $ENV_FILE (real environment wins)"
  load_env_file "$ENV_FILE"
else
  echo "[entrypoint] no $ENV_FILE; using the real environment only"
fi

export WORK_DIR="${WORK_DIR:-/work}"
mkdir -p "$WORK_DIR/logs"

# --- interpreter ------------------------------------------------------------
# isaac_python.env is written during the build and holds the interpreter path
# plus the two variables isaaclab.sh exports. Prepend rather than replace, so a
# caller-supplied PYTHONPATH still works, and let an explicit HF80K_PYTHON from
# the environment beat the recorded one (debugging with another interpreter).
PY_OVERRIDE="${HF80K_PYTHON:-}"
if [[ -f /opt/hf80k/isaac_python.env ]]; then
  # shellcheck disable=SC1091
  . /opt/hf80k/isaac_python.env
fi
PY="${PY_OVERRIDE:-${HF80K_PYTHON:-$(command -v python3)}}"
[[ -x "$PY" ]] || { echo "[entrypoint] no usable python ($PY)" >&2; exit 1; }

prepend_path() {  # prepend_path VAR VALUE — skips empties so no ":" entries appear
  local var="$1" value="$2" current="${!1:-}"
  [[ -z "$value" ]] && return 0
  if [[ -z "$current" ]]; then export "$var=$value"; else export "$var=$value:$current"; fi
}
prepend_path LD_LIBRARY_PATH "${HF80K_LD_LIBRARY_PATH:-}"
prepend_path PYTHONPATH "${HF80K_PYTHONPATH:-}"
# lerobot 설치 경로. 기록·업로드 스크립트가 LEROBOT_SITE로 찾는다. 이미지에서 이미
# 정해 두지만, 밖에서 다른 경로를 주면 그쪽이 이긴다.
export LEROBOT_SITE="${LEROBOT_SITE:-${HF80K_LEROBOT_PATH:-}}"

if [[ "${1:-}" == "shell" ]]; then
  shift
  echo "[entrypoint] debug shell (python: $PY)"
  exec bash "$@"
fi

echo "[entrypoint] gpu=${CUDA_VISIBLE_DEVICES:-unset} work_dir=$WORK_DIR" \
     "target=${TARGET_EPISODES:-80000} chunk=${CHUNK_SIZE:-500} python=$PY"

# --- preflight --------------------------------------------------------------
# Fail here, loudly, rather than three hours into a chunk: preflight checks the
# assets, the writable mount, the GPU and the Hugging Face credentials.
# orchestrate.py도 시작할 때 같은 검사를 하므로 여기서는 돌리지 않는다. 두 번 돌면
# 허깅페이스 whoami 왕복이 두 번 생기고, 재시작 정책과 겹치면 계속 두 번씩 돈다.
: "preflight는 orchestrate.py가 수행한다"

echo "[entrypoint] starting orchestrator"
exec "$PY" "$SRC/orchestrate.py" "$@"
