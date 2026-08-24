#!/usr/bin/env python3
"""Chunk orchestrator — the single entry point of the hf80k container.

WHY this file exists: 80k successful episodes cannot come out of one long
process. At the measured rates (2.97 s per generation attempt, 14.2% yield with
the zero-yield source excluded, 31.9 s of wall clock per finished episode on one
L40S) a full run is weeks of wall clock, and anything
that runs for days gets interrupted — a spot reclaim, an OOM kill, a driver
hiccup, someone restarting the box. So the run is cut into chunks of
CHUNK_SIZE episodes; each chunk walks the whole pipeline on its own
(generate -> convert -> render -> lerobot -> upload) and leaves exactly one
durable marker behind, MANIFEST.json. Everything else inside a chunk directory
is disposable. Resuming is then "skip the chunks that already have a manifest",
which is cheap to check and impossible to half-trust, because the manifest is
written atomically (temp file + os.replace) only after the last stage returned 0.

Two more things this file is responsible for, both learned the hard way:

  * memory. The box has no swap and an Isaac Sim render process has been
    OOM-killed here before. Every extra subprocess is therefore gated on
    /proc/meminfo MemAvailable, and launches are staggered, so the second
    process starts after the first one is past its allocation burst.
  * not dying on one bad chunk. A failed chunk is retried once, then recorded
    in state.json as failed and the run moves on. Losing 500 episodes out of
    80000 is a rounding error; losing three days of queue because chunk 47 hit
    a USD hiccup is not.

Configuration comes ONLY from the environment variables in INTERFACE.md §1.
The command line carries operational switches (--dry-run, --skip-preflight)
that do not change what gets produced.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time

# ----------------------------------------------------------------- constants
PROFILE_NAMES = ("nominal_lab", "lab_variation", "stress_tail")
PROFILE_IDS = {"nominal_lab": 0, "lab_variation": 1, "stress_tail": 2}
CHUNK_SCHEMA = "fr3_cube.hf80k.chunk.v1"
STATE_SCHEMA = "fr3_cube.hf80k.state.v1"
# INTERFACE.md §4: the dataset carries these three views. Passed to the renderer
# explicitly so a change of its default cannot quietly add a fourth camera.
CAMERAS = ["third_person_0", "third_person_1", "wrist"]   # 프로필이 아래에서 덮어쓴다

# In-container Isaac Lab install (we are already inside the isaac-lab image, so
# no nested docker — this is the difference from contract/run_lab_generate_docker.sh).
ISAACLAB_SH = "/workspace/isaaclab/isaaclab.sh"
GEN_DATASET_SRC = ("/workspace/isaaclab/scripts/imitation_learning/isaaclab_mimic/"
                   "generate_dataset.py")
# 태스크마다 다른 값은 프로필 한 장에서 온다. TASK_PROFILE 환경변수로 고르고, 없으면
# 큐브 프로필이 기본이다. 로더는 예외를 던지지 않으므로 여기서 죽지 않는다.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import task_profile as _task_profile          # noqa: E402
PROFILE = _task_profile.load()
TASK_ID = PROFILE.get("generate.task_id",
                      "Isaac-Stack-Cube-LabFR3-HF80K-Fwd-IK-Rel-Mimic-v0")
# Chosen (spec silent): the physics device of every working run so far is cpu —
# run_generate.sh and run_render_aidas.sh both pass --device cpu. RTX rendering
# still uses the GPU selected by CUDA_VISIBLE_DEVICES.
PHYSICS_DEVICE = "cpu"
# 실험실 책상 USD. Dockerfile이 시각 랜덤화 패키지에서 꺼내 이 경로에 둔다.
# 예전 도커 스크립트는 일부러 없는 경로를 넣어 회색 슬랩으로 대체했는데, 그건
# 계약 변환처럼 책상이 안 보여도 되는 단계용이었다. 렌더는 실물 책상이어야 하고
# assets/fr3_binding_v2.yaml의 카메라 바인딩도 실물 책상 기준으로 측정돼 있다.
TABLE_USD = os.environ.get("LAB_TABLE_USD", "/opt/hf80k/assets/table_scene.usdc")

PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PKG_ROOT, "src")
ASSETS_DIR = os.path.join(PKG_ROOT, "assets")
ENV_DIR = os.path.join(SRC_DIR, "env")
RENDER_DIR = os.path.join(SRC_DIR, "render")
CONVERT_DIR = os.path.join(SRC_DIR, "convert")
RENDER_SCRIPT = os.path.join(RENDER_DIR, "render_viewpoints.py")
CONVERT_SCRIPT = os.path.join(CONVERT_DIR, "convert_demo.py")
LEROBOT_WRITER = os.path.join(SRC_DIR, "lerobot_writer.py")
# SART 증강 단계. 태스크 프로필이 켜 준 태스크에서만 돈다.
SART_DIR = os.path.join(SRC_DIR, "sart")
SART_SCRIPT = os.path.join(SART_DIR, "sart_augment.py")
HF_UPLOAD = os.path.join(SRC_DIR, "hf_upload.py")
SOURCE_HDF5 = os.path.join(ASSETS_DIR, "fwd_annotated.hdf5")
SOURCE_YIELD_JSON = os.path.join(ASSETS_DIR, "source_yield.json")
OVERLAY_YAML = os.path.join(ASSETS_DIR, "fr3_camera_overlay_v2/overlay.yaml")
BINDING_YAML = os.path.join(ASSETS_DIR, "fr3_binding_v2.yaml")
# The RL team's visual-randomization handoff package. Preferred location is
# inside our assets; /vrand is the mount the existing render script assumes.
VRAND_ROOTS = (os.path.join(ASSETS_DIR, "fr3_visual_randomization_v1"), "/vrand")

# ---- 여기서부터 태스크 프로필이 이깁니다 -------------------------------------
# 위의 값들은 프로필을 읽지 못했을 때 쓰는 기본값이다. 프로필이 정상이면 아래가 덮는다.
CAMERAS = list(PROFILE.get("render.cameras", CAMERAS))
ENV_DIR = os.path.join(SRC_DIR, PROFILE.get("generate.module_dir", "env"))
SOURCE_HDF5 = os.path.join(ASSETS_DIR, PROFILE.get("generate.source_hdf5",
                                                   "fwd_annotated.hdf5"))
_yield_name = PROFILE.get("generate.source_yield_json", "source_yield.json")
SOURCE_YIELD_JSON = os.path.join(ASSETS_DIR, _yield_name) if _yield_name else ""
OVERLAY_YAML = os.path.join(ASSETS_DIR, PROFILE.get("render.overlay_yaml",
                                                    "fr3_camera_overlay_v2/overlay.yaml"))
BINDING_YAML = os.path.join(ASSETS_DIR, PROFILE.get("render.binding_yaml",
                                                    "fr3_binding_v2.yaml"))
VRAND_ROOTS = (os.path.join(ASSETS_DIR, PROFILE.get("visual.package_dir",
                                                    "fr3_visual_randomization_v1")), "/vrand")
CHUNK_SCHEMA = PROFILE.get("dataset.schema_prefix", "fr3_cube.hf80k") + ".chunk.v1"
STATE_SCHEMA = PROFILE.get("dataset.schema_prefix", "fr3_cube.hf80k") + ".state.v1"
# 생성 스크립트에 끼워 넣을 모듈. 큐브는 clean_success_hook을 쓰고 peg는 peg 판정 훅을 쓴다.
REGISTER_MODULES = list(PROFILE.get("generate.register_modules",
                                    ["lab_register", "clean_success_hook", "provenance_hooks"]))

# 생성기 조정값. 태스크마다 달라서 프로필에 적고, 환경변수를 주면 그쪽이 이긴다.
#
#   arm_scale                 IK 상대 제어에서 한 스텝에 허용하는 이동량의 배율.
#                             큐브는 0.5이고 핀 꽂기는 1.0이다. 핀은 집은 자리에서
#                             구멍까지 최대 25 cm를 옮겨야 해서 0.5로는 도달하지 못한다.
#   subtask_offsets           구간 경계를 원래 자리보다 몇 스텝 뒤로 옮길지 정하는
#                             "최소,최대"다. 매번 그 범위에서 하나를 무작위로 뽑는다.
#   action_noise              매 스텝 행동에 더하는 잡음의 크기. 0이면 넣지 않는다.
#                             핀과 구멍의 틈이 0.7 mm라 잡음이 크면 테두리에 걸린다.
#   num_interpolation_steps   구간이 바뀔 때 지금 손 위치에서 다음 구간 시작 자세까지
#                             몇 스텝에 걸쳐 옮길지 정한다. 짧으면 도달하지 못한 채
#                             다음 구간이 시작되고, 그 어긋남이 끝까지 남는다.
GEN_ARM_SCALE = str(PROFILE.get("generate.arm_scale", "0.5"))
GEN_SUBTASK_OFFSETS = str(PROFILE.get("generate.subtask_offsets", "10,20"))
GEN_ACTION_NOISE = PROFILE.get("generate.action_noise", None)
GEN_NUM_INTERP = PROFILE.get("generate.num_interpolation_steps", None)

# 계약 형식으로 바꿀 때 자세를 기록할 강체 이름. 큐브 쌓기는 큐브 3개, 핀 삽입은 핀 하나다.
# 시연 파일의 states/rigid_object/<이름>/root_pose에서 읽는다. 예전에는 이 이름이 변환
# 코드에 박혀 있어서 큐브가 아닌 태스크에서 `component not found`로 죽었다.
CONVERT_OBJECTS = list(PROFILE.get("convert.object_states",
                                   ["cube_1", "cube_2", "cube_3"]))

# 렌더가 재생하면서 매기는 성공 판정. 태스크마다 기준이 달라 모듈과 함수 이름을 프로필이
# 정한다. 기록 단계는 여기서 적은 속성 이름을 읽어 성공한 에피소드만 남긴다. 이 배선이
# 없으면 렌더가 성공 표시를 남기지 않고, 기록 단계가 모든 에피소드를 버린다.
RENDER_SUCCESS_MODULE = PROFILE.get("render.success.module", "success_criteria")
RENDER_SUCCESS_FUNCTION = PROFILE.get("render.success.function", "replay_verdict")
RENDER_SUCCESS_ATTR = PROFILE.get("render.success.verdict_attr", "replay_success_any_order")


# SART 증강 설정. 생성과 변환 사이에 들어가는 선택 단계이고, 프로필에 generate.sart 절이
# 있는 태스크에서만 돈다. 큐브 쌓기 프로필에는 그 절이 없으므로 값이 전부 아래 기본값이
# 되고 enable이 거짓이라 단계가 즉시 0초를 돌려주고 끝난다.
#
# 키를 하나씩 제 이름으로 읽는다. 절 이름만 적어 통째로 읽으면 안 된다. 죽은 키 검사
# (src/tests/test_profile_keys_used.py)는 프로필의 잎 키를 점으로 이은 문자열이 소스
# 어딘가에 따옴표째 나타나는지를 보는데, 절 이름 하나만 적어 두면 그 한 문자열이 절
# 아래 모든 키를 대신 통과시켜서 오타 난 키가 조용히 무시된다.
def _sart_profile() -> dict:
    """프로필의 generate.sart 절을 읽어 사전 하나로 돌려준다."""
    return {
        "enable": bool(PROFILE.get("generate.sart.enable", False)),
        "samples_per_source": int(PROFILE.get("generate.sart.samples_per_source", 4)),
        "radius_m": float(PROFILE.get("generate.sart.radius_m", 0.05)),
        "rotation_deg": float(PROFILE.get("generate.sart.rotation_deg", 10.0)),
        "fix_position": bool(PROFILE.get("generate.sart.fix_position", False)),
        "divert_steps": int(PROFILE.get("generate.sart.divert_steps", 10)),
        "converge_steps": int(PROFILE.get("generate.sart.converge_steps", 20)),
        "settle_steps": int(PROFILE.get("generate.sart.settle_steps", 5)),
        "tail_steps": int(PROFILE.get("generate.sart.tail_steps", 25)),
        "converge_rule": str(PROFILE.get("generate.sart.converge_rule", "radial_gate")),
        "converge_object": str(PROFILE.get("generate.sart.converge_object", "")),
        "converge_target": str(PROFILE.get("generate.sart.converge_target", "")),
        "converge_radius_m": float(PROFILE.get("generate.sart.converge_radius_m", 0.016)),
        "floor_margin_m": float(PROFILE.get("generate.sart.floor_margin_m", 0.005)),
        "grip_closed_m": float(PROFILE.get("generate.sart.grip_closed_m", 0.035)),
        "source_frac": float(PROFILE.get("generate.sart.source_frac", 1.0)),
        "on_failure": str(PROFILE.get("generate.sart.on_failure", "continue")),
        "max_consecutive_failures": int(
            PROFILE.get("generate.sart.max_consecutive_failures", 3)),
        "seed_offset": int(PROFILE.get("generate.sart.seed_offset", 7717)),
    }


SART_PROFILE = _sart_profile()
# 프로필에 sart 절 아래로 적을 수 있는 키 전부. 실행 전 검사가 오타를 잡는 데 쓴다.
SART_KEYS = tuple(sorted(SART_PROFILE))

# 시각 규격의 물체 이름을 이 장면의 프림 경로에 잇는 표. 규격은 물체를 큐브 쌓기 장면의
# 이름으로 부르므로, 다른 태스크는 여기서 이어 줘야 색과 재질 랜덤화가 실제로 적용된다.
VRAND_OBJECT_PRIMS = dict(PROFILE.get("visual.object_prims", {}) or {})

# 물리 절을 환경변수로 옮긴다. Isaac 환경 모듈은 시뮬레이터가 뜬 뒤에 임포트되고 프로필
# 로더를 쓰지 않으므로, 여기서 이름을 붙여 넘긴다. 값이 비어 있으면 넘기지 않고 환경
# 모듈의 기본값을 쓴다.
def _physics_env() -> dict:
    """프로필의 physics 절을 환경 모듈이 읽는 이름으로 바꾼다."""
    import json as _json

    out = {}
    bundle = PROFILE.get("physics.bundle_dir", "")
    if bundle:
        out["LAB_SYSID_BUNDLE_ROOT"] = os.path.join(ASSETS_DIR, str(bundle))
    objects = PROFILE.get("physics.primary_objects", None)
    if objects:
        out["LAB_PHYS_OBJECTS"] = ",".join(str(n) for n in objects)
    for key, name in (("physics.surface", "LAB_PHYS_SURFACE"),
                      ("physics.arm_actuator", "LAB_PHYS_ARM_ACTUATOR"),
                      ("physics.gripper_actuator", "LAB_PHYS_GRIPPER_ACTUATOR"),
                      ("physics.object_size_m", "LAB_PHYS_OBJECT_SIZE")):
        value = PROFILE.get(key, None)
        if value not in (None, ""):
            out[name] = str(value)
    masses = PROFILE.get("physics.object_masses_kg", None)
    if masses:
        out["LAB_PHYS_OBJECT_MASSES"] = _json.dumps(masses)
    return out


PHYSICS_ENV = _physics_env()

# Memory guard. An Isaac Sim process (generation or RTX render) sits around
# 6-8 GB resident; we refuse to add another one below this much MemAvailable.
MEM_HEADROOM_MB = 9000
MEM_POLL_S = 15
MEM_WAIT_MAX_S = 1800          # after this, launch anyway rather than deadlock
LAUNCH_STAGGER_S = 25          # Isaac Sim allocates in bursts while starting

STOP = threading.Event()       # set by SIGTERM/SIGINT: stop launching new work


class StageError(RuntimeError):
    """A stage subprocess returned nonzero (or produced nothing usable)."""


# ------------------------------------------------------------- env var access
def env_str(name: str, default: str) -> str:
    v = os.environ.get(name)
    return default if v is None or v == "" else v


def env_int(name: str, default: int) -> int:
    v = env_str(name, str(default))
    try:
        return int(v)
    except ValueError:
        raise SystemExit(f"[orch] {name}={v!r} is not an integer")


def env_float(name: str, default: float) -> float:
    """소수 환경변수를 읽는다. 잘못된 값이면 어느 변수가 문제인지 한 줄로 말하고 멈춘다.

    맨 float()로 읽으면 소수점 대신 쉼표를 쓴 0,06 같은 흔한 오타에서 파이썬 역추적만
    쏟아지고 어느 변수가 잘못됐는지는 나오지 않는다.
    """
    v = env_str(name, str(default))
    try:
        return float(v)
    except ValueError:
        raise SystemExit(f"[orch] {name}={v!r}는 숫자가 아니다")


_FALSEY = ("0", "false", "no", "off")


def env_flag(name: str, default: str) -> bool:
    """0/false/no/off만 거짓으로 본다.

    예전에는 값이 정확히 "1"일 때만 참이었다. 그러면 HF_PRIVATE=true 로 적은 사람이
    공개 저장소를 얻는다. src/hf_upload.py의 판정과 같은 규칙을 쓴다.
    """
    value = env_str(name, default).strip().lower()
    if value == "":
        value = default.strip().lower()
    return value not in _FALSEY


def parse_profile_split(text: str) -> dict:
    """`nominal_lab:0.50,lab_variation:0.40,stress_tail:0.10` -> dict."""
    out = {}
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise SystemExit(f"[orch] PROFILE_SPLIT entry {part!r} is not name:weight")
        name, weight = part.split(":", 1)
        name = name.strip()
        if name not in PROFILE_NAMES:
            raise SystemExit(f"[orch] PROFILE_SPLIT has unknown profile {name!r}")
        out[name] = float(weight)
    for name in PROFILE_NAMES:
        out.setdefault(name, 0.0)
    if sum(out.values()) <= 0:
        raise SystemExit("[orch] PROFILE_SPLIT weights sum to zero")
    return out


def load_config() -> dict:
    """Every knob of the run, read once, from the environment only."""
    cfg = {
        "hf_token": os.environ.get("HF_TOKEN", ""),
        "hf_repo_id": os.environ.get("HF_REPO_ID", ""),
        "hf_private": env_flag("HF_PRIVATE", "1"),
        "target_episodes": env_int("TARGET_EPISODES", 80000),
        "profile_split": parse_profile_split(env_str(
            "PROFILE_SPLIT", "nominal_lab:0.50,lab_variation:0.40,stress_tail:0.10")),
        "chunk_size": env_int("CHUNK_SIZE", 500),
        "num_envs": env_int("NUM_ENVS", 16),
        "gen_procs": env_int("GEN_PROCS", 1),
        "render_procs": env_int("RENDER_PROCS", 2),
        "cuda_devices": env_str("CUDA_VISIBLE_DEVICES", "0"),
        "image_width": env_int("IMAGE_WIDTH", 320),
        "image_height": env_int("IMAGE_HEIGHT", 180),
        "physics_profile": env_str("PHYSICS_PROFILE", "robust_stochastic"),
        "source_demo_filter": env_str("SOURCE_DEMO_FILTER", "exclude_zero_yield"),
        # raw get, not env_str: SUBTASK_OFFSETS="" means "leave the env module's
        # own default alone" (INTERFACE §1), which is not the same as "10,20"
        "subtask_offsets": os.environ.get("SUBTASK_OFFSETS", GEN_SUBTASK_OFFSETS),
        "arm_scale": env_str("LAB_ARM_SCALE", GEN_ARM_SCALE),
        # 값이 없으면 빈 문자열로 두고, 아래에서 환경변수를 아예 넘기지 않는다.
        # 그러면 환경 모듈이 자기 기본값을 그대로 쓴다.
        "action_noise": os.environ.get(
            "LAB_ACTION_NOISE", "" if GEN_ACTION_NOISE is None else str(GEN_ACTION_NOISE)),
        "num_interpolation_steps": os.environ.get(
            "LAB_NUM_INTERP", "" if GEN_NUM_INTERP is None else str(GEN_NUM_INTERP)),
        "work_dir": env_str("WORK_DIR", "/work"),
        "keep_intermediate": env_flag("KEEP_INTERMEDIATE", "0"),
        "upload_each_chunk": env_flag("UPLOAD_EACH_CHUNK", "1"),
        # 0이면 허깅페이스에 아무것도 올리지 않고 결과를 작업 디렉터리에만 남긴다.
        # 남의 파이프라인 안에서 한 모듈로 돌 때는 저장소도 토큰도 없는 것이 보통이라,
        # 그때 이 값을 0으로 두면 자격증명 없이 끝까지 돈다.
        "hf_upload": env_flag("HF_UPLOAD", "1"),
        "resume": env_flag("RESUME", "1"),
        "seed_base": env_int("SEED_BASE", 42000),
        "log_level": env_str("LOG_LEVEL", "INFO"),
        # SART 증강. 기본값은 태스크 프로필이 정하고, 환경변수를 주면 그쪽이 이긴다.
        # SART_ENABLE=0이면 어떤 태스크에서도 꺼진다.
        "sart_enable": env_flag("SART_ENABLE", "1" if SART_PROFILE["enable"] else "0"),
        "sart_procs": env_int("SART_PROCS", 1),
        "sart_samples": env_int("SART_SAMPLES", int(SART_PROFILE["samples_per_source"])),
        "sart_radius_m": env_float("SART_RADIUS_M", float(SART_PROFILE["radius_m"])),
        "sart_source_frac": env_float("SART_SOURCE_FRAC", float(SART_PROFILE["source_frac"])),
    }
    for key in ("chunk_size", "num_envs", "gen_procs", "render_procs"):
        if cfg[key] < 1:
            raise SystemExit(f"[orch] {key.upper()} must be >= 1 (got {cfg[key]})")
    # SART 값 검사는 SART를 켠 실행에서만 한다.
    #
    # 이 검사를 늘 돌리면 쓰지도 않는 값 때문에 실행이 시작조차 못 한다. GPU 네 대로 돌릴 때
    # .env 한 장을 네 컨테이너가 함께 읽는데, 거기에 SART_SAMPLES=0을 적어 두면 SART를 쓰지
    # 않는 큐브 컨테이너까지 멈춘다. 쓰지 않는 값이 실행을 막아서는 안 된다.
    if cfg["sart_enable"]:
        if cfg["sart_procs"] < 1:
            raise SystemExit(f"[orch] SART_PROCS는 1 이상이어야 한다 (받은 값 {cfg['sart_procs']})")
        if cfg["sart_samples"] < 1:
            raise SystemExit(f"[orch] SART_SAMPLES는 1 이상이어야 한다 (받은 값 {cfg['sart_samples']})")
        if cfg["sart_radius_m"] <= 0:
            raise SystemExit(f"[orch] SART_RADIUS_M는 0보다 커야 한다 (받은 값 {cfg['sart_radius_m']})")
        if not 0.0 < cfg["sart_source_frac"] <= 1.0:
            raise SystemExit("[orch] SART_SOURCE_FRAC는 0보다 크고 1 이하여야 한다 "
                             f"(받은 값 {cfg['sart_source_frac']})")
    if cfg["target_episodes"] < 1:
        raise SystemExit("[orch] TARGET_EPISODES must be >= 1")
    cfg["chunks_dir"] = os.path.join(cfg["work_dir"], "chunks")
    cfg["logs_dir"] = os.path.join(cfg["work_dir"], "logs")
    cfg["merged_dir"] = os.path.join(cfg["work_dir"], "merged")
    cfg["state_path"] = os.path.join(cfg["work_dir"], "state.json")
    cfg["source_dataset"] = os.path.join(cfg["work_dir"], "source_filtered.hdf5")
    cfg["gen_script"] = os.path.join(cfg["work_dir"], "generate_lab.py")
    cfg["vrand_root"] = next((p for p in VRAND_ROOTS if os.path.isdir(p)), VRAND_ROOTS[0])
    cfg["vrand_config"] = os.path.join(cfg["vrand_root"], "config")
    return cfg


# ------------------------------------------------------------------- planning
def largest_remainder(total: int, weights) -> list:
    """Split `total` into integers proportional to `weights`, exactly.

    Same idea as render/visual_randomization.py:episode_profile_plan (floor
    everything, hand the leftover units to the largest fractional parts), minus
    numpy: the orchestrator should start even if the science stack is broken.
    """
    s = float(sum(weights))
    if s <= 0:
        raise ValueError("weights must sum to a positive number")
    raw = [total * (w / s) for w in weights]
    counts = [int(math.floor(v)) for v in raw]
    order = sorted(range(len(raw)), key=lambda i: (-(raw[i] - counts[i]), i))
    for k in range(total - sum(counts)):
        counts[order[k % len(counts)]] += 1
    return counts


def plan_chunks(cfg: dict) -> list:
    """Ordered chunk plan: profile quotas, split into CHUNK_SIZE pieces.

    The order interleaves the profiles proportionally instead of doing all of
    nominal_lab first. A run that gets killed at 30% then still holds roughly
    the 50/40/10 mixture, which makes partial output usable.
    """
    quotas = largest_remainder(cfg["target_episodes"],
                               [cfg["profile_split"][p] for p in PROFILE_NAMES])
    pieces = []
    for pi, (name, quota) in enumerate(zip(PROFILE_NAMES, quotas)):
        sizes, left = [], quota
        while left > 0:
            sizes.append(min(cfg["chunk_size"], left))
            left -= sizes[-1]
        for i, size in enumerate(sizes):
            key = (i + 0.5) / len(sizes)
            pieces.append((key, pi, name, size))
    pieces.sort(key=lambda p: (p[0], p[1]))
    plan = []
    for idx, (_, _, name, size) in enumerate(pieces):
        plan.append({
            "chunk_index": idx,
            "profile": name,
            "profile_id": PROFILE_IDS[name],
            "episodes": size,
            "seed": cfg["seed_base"] + idx,
            "dir": os.path.join(cfg["chunks_dir"], f"chunk_{idx:05d}"),
            "status": "pending",
        })
    return plan


# ---------------------------------------------------------------------- io
def utcnow() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def atomic_write_json(path: str, obj: dict) -> None:
    """Write JSON through a temp file + os.replace.

    A half-written MANIFEST.json is worse than no manifest: resume would trust
    it and skip a chunk that has no data.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=2, sort_keys=False)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def read_json(path: str):
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def natural_key(name: str):
    """demo_2 before demo_10 — same convention as render_viewpoints.natural_key."""
    m = re.search(r"(\d+)$", name)
    return (int(m.group(1)) if m else 1 << 30, name)


def make_logger(path: str):
    """One line per event, to stdout and to the chunk log."""
    def log(msg: str) -> None:
        line = f"[{utcnow()}] {msg}"
        print(line, flush=True)
        try:
            with open(path, "a") as fh:
                fh.write(line + "\n")
        except OSError:
            pass
    return log


# ------------------------------------------------------------ subprocess layer
def mem_available_mb():
    """MemAvailable in MiB, or None when /proc is not there (dev machines)."""
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def wait_for_memory(state: dict, lock: threading.Lock, need_mb: int, name: str, log) -> None:
    """Block until there is room for one more Isaac process.

    Backing off means waiting, which effectively drops the number of concurrent
    processes for as long as the pressure lasts. If nothing else is running we
    always go ahead — otherwise a wrong estimate would stall the whole run.
    """
    waited = 0
    while True:
        avail = mem_available_mb()
        if avail is None or avail >= need_mb:
            return
        with lock:
            if state["active"] == 0:
                log(f"mem: only {avail} MiB available but nothing is running, "
                    f"starting {name} anyway")
                return
        if waited >= MEM_WAIT_MAX_S:
            log(f"mem: waited {waited}s with {avail} MiB available, starting {name} anyway")
            return
        if waited == 0:
            log(f"mem: {avail} MiB available < {need_mb} MiB, holding {name} back")
        time.sleep(MEM_POLL_S)      # not STOP.wait: a set STOP must not turn this into a spin
        waited += MEM_POLL_S


def run_cmd(cmd: list, env: dict, log_path: str):
    """Run one stage subprocess, append its output to the chunk log.

    start_new_session so a Ctrl-C aimed at the orchestrator does not tear down
    a stage that is halfway through writing an HDF5.
    """
    t0 = time.time()
    with open(log_path, "ab", buffering=0) as fh:
        fh.write(("\n$ " + " ".join(cmd) + "\n").encode())
        proc = subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT,
                                start_new_session=True)
        rc = proc.wait()
    return rc, time.time() - t0


def run_parallel(jobs: list, max_procs: int, log, log_path: str) -> float:
    """Run jobs concurrently with a memory gate; raise StageError on any failure.

    `jobs` are dicts {name, cmd, env}. Returns the wall-clock seconds of the
    whole group (not the sum of the parts).
    """
    t0 = time.time()
    state = {"active": 0}
    lock = threading.Lock()
    failures = []

    def worker(index: int, job: dict):
        if index and LAUNCH_STAGGER_S:
            time.sleep(index * LAUNCH_STAGGER_S)     # Isaac Sim start-up is an allocation burst
        wait_for_memory(state, lock, MEM_HEADROOM_MB, job["name"], log)
        with lock:
            state["active"] += 1
        try:
            log(f"start {job['name']}")
            rc, secs = run_cmd(job["cmd"], job["env"], log_path)
        finally:
            with lock:
                state["active"] -= 1
        log(f"end   {job['name']} rc={rc} {secs:.1f}s")
        return rc

    with cf.ThreadPoolExecutor(max_workers=max(1, min(max_procs, len(jobs)))) as pool:
        futs = {pool.submit(worker, i, j): j for i, j in enumerate(jobs)}
        for fut in cf.as_completed(futs):
            job = futs[fut]
            try:
                rc = fut.result()
            except Exception as exc:                      # noqa: BLE001 - report, do not crash
                failures.append(f"{job['name']}: {exc}")
                continue
            if rc != 0:
                failures.append(f"{job['name']}: rc={rc}")
    if failures:
        raise StageError("; ".join(failures))
    return time.time() - t0


def base_env(cfg: dict, extra_pythonpath: list) -> dict:
    """Environment shared by every Isaac subprocess.

    The parent environment is inherited wholesale on purpose: PHYSICS_PROFILE
    and any LAB_* override the operator set outside the container reach the
    env modules without this file having to know about them.
    """
    env = os.environ.copy()
    env.update({
        "ACCEPT_EULA": "Y",
        "PRIVACY_CONSENT": "Y",
        "OMNI_KIT_ACCEPT_EULA": "YES",
        "PYTHONUNBUFFERED": "1",
        "CUDA_VISIBLE_DEVICES": cfg["cuda_devices"],
        "PHYSICS_PROFILE": cfg["physics_profile"],
        "LAB_TABLE_USD": TABLE_USD,
        # 받침 방향 교정. 이 값이 빠지면 로봇이 실기와 180도 반대를 보고, 수율이
        # 12.4%에서 6.5%로 떨어지며 실측 카메라가 작업면에서 약 1 m 빗나간다.
        # assets/source_yield.json의 수율은 이 교정이 켜진 상태에서 측정한 값이다.
        "LAB_ROBOT_SPAWN_ROT": os.environ.get("LAB_ROBOT_SPAWN_ROT", "0,0,1,0"),
        # 격리 설치한 lerobot을 기록·업로드 단계가 찾는 경로.
        "LEROBOT_SITE": os.environ.get("LEROBOT_SITE",
                                       os.environ.get("HF80K_LEROBOT_PATH", "")),
    })
    # 프로필의 물리 절. 번들 경로와 장면 물체의 역할 이름이 여기로 간다. 밖에서 같은
    # 이름을 주면 그쪽이 이긴다.
    for key, value in PHYSICS_ENV.items():
        env.setdefault(key, value)
    # 태스크 프로필이 정한 추가 환경변수. peg는 여기로 핀 구멍과 책상 USD 경로를 받는다.
    # 생성·변환·렌더가 모두 같은 장면을 만들어야 하므로 한 곳에서 넣는다. 예전에는
    # 어노테이션 단계에만 넣어서, 생성이 기본 경로 /work/assets/peg_hole_env.usd를 찾다가
    # FileNotFoundError로 25초 만에 죽었다.
    for key, value in (PROFILE.get("generate.extra_env", {}) or {}).items():
        env.setdefault(str(key), str(value))
    parts = [p for p in extra_pythonpath if p]
    # 태스크와 무관하게 함께 쓰는 모듈이 src/env에 있다. 실측 물리 항(calibrated_sysid)이
    # 그것이고, 큐브 환경과 핀 환경이 같은 파일을 부른다. 태스크 디렉터리 뒤에 붙이므로
    # 같은 이름의 파일이 태스크 쪽에 있으면 그쪽이 이긴다.
    shared = os.path.join(SRC_DIR, "env")
    if shared not in parts:
        parts.append(shared)
    # 단계와 무관하게 공유하는 모듈이 src 바로 아래에 있다. dataset_format이 그것이고,
    # 재생·증강·렌더가 모두 같은 사원수 규약 검사를 부른다.
    if SRC_DIR not in parts:
        parts.append(SRC_DIR)
    if env.get("PYTHONPATH"):
        parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = ":".join(parts)
    return env


# ---------------------------------------------------------------- source prep
def prepare_source_dataset(cfg: dict, log) -> str:
    """Resolve the source dataset MimicGen should read (INTERFACE.md §7).

    The work is done by src/env/source_filter.py, which physically copies the
    kept demos rather than only writing mask/train: isaaclab_mimic builds its
    source pool from `h5file["data"].keys()` and never looks at a filter key,
    so a mask alone would filter nothing. It returns the original path when the
    setting keeps every demo, and caches its copy across chunks and restarts.
    """
    if not os.path.isfile(SOURCE_HDF5):
        raise SystemExit(f"[orch] missing source dataset {SOURCE_HDF5}")
    if ENV_DIR not in sys.path:
        sys.path.insert(0, ENV_DIR)
    import source_filter

    path = source_filter.build_filtered_source(
        SOURCE_HDF5,
        setting=cfg["source_demo_filter"],
        out_path=cfg["source_dataset"],
        yield_path=SOURCE_YIELD_JSON if os.path.isfile(SOURCE_YIELD_JSON) else None)
    cfg["source_dataset"] = path          # may be the original, when nothing is dropped
    log(f"source: filter {cfg['source_demo_filter']!r} -> {path}")
    return path


def prepare_generate_script(cfg: dict, log) -> str:
    """Copy Isaac Lab's generate_dataset.py and inject our three modules.

    Straight port of the sed trick in run_generate.sh / run_lab_generate_docker.sh:
    the shared Isaac Lab source is never edited, and clean_success_hook must be
    imported before provenance_hooks so provenance counts the gated successes.
    """
    dst = cfg["gen_script"]
    if not os.path.isfile(GEN_DATASET_SRC):
        raise SystemExit(f"[orch] missing {GEN_DATASET_SRC} (not inside the isaac-lab image?)")
    with open(GEN_DATASET_SRC) as fh:
        text = fh.read()
    inject = "".join(f"\nimport {name}" for name in REGISTER_MODULES)
    out, done = [], False
    for line in text.splitlines():
        out.append(line)
        if not done and line.startswith("import isaaclab_mimic.envs"):
            out.append(inject.strip("\n"))
            done = True
    if not done:
        raise SystemExit("[orch] could not find 'import isaaclab_mimic.envs' to inject after")
    body = "\n".join(out) + "\n"
    with open(dst, "w") as fh:
        fh.write(body)
    log(f"generate: patched script -> {dst} (--seed "
        f"{'supported' if '--seed' in body else 'not supported, using LAB_GEN_SEED'})")
    return dst


def script_supports(path: str, flag: str) -> bool:
    """Does this script's argparse know `flag`? Isaac Lab versions differ."""
    try:
        with open(path) as fh:
            return f'"{flag}"' in fh.read()
    except OSError:
        return False


# ----------------------------------------------------------------- hdf5 merge
def _demo_names(path: str) -> list:
    import h5py
    with h5py.File(path, "r") as fh:
        return sorted(fh["data"].keys(), key=natural_key)


def copy_dataset_file_attrs(out, src_path: str, require_format_version: bool) -> None:
    """원본 HDF5의 파일 루트 속성을 병합본에 그대로 옮긴다.

    Isaac Lab의 HDF5DatasetFileHandler는 파일 루트에 format_version 속성이 없으면 그
    파일을 옛 형식으로 판단하고, 읽을 때 root_pose의 사원수를 WXYZ 순서에서 XYZW
    순서로 바꾼다. 우리가 기록하는 파일은 이미 XYZW다. 그래서 이 속성이 빠지면 로봇
    받침의 회전이 z축 180도에서 y축 180도로 바뀌고, 받침 링크에 매달린 카메라 세 대가
    전부 책상 밑을 보게 된다. 물리 재생은 멀쩡한데 영상만 망가지므로 수율 검사로는
    잡히지 않는다. 병합은 새 파일을 만드는 일이라 이 속성을 반드시 손으로 옮겨야 한다.

    require_format_version은 묶는 파일이 어떤 종류인지에 따라 다르다. 로봇의 관절과
    물체 자세가 들어 있는 생성 결과는 Isaac Lab의 적재기가 읽으므로 참을 준다. 카메라
    영상이 들어 있는 렌더 결과는 기록 단계가 h5py로 직접 읽고 자세를 해석하지 않으므로
    거짓을 준다. 렌더 결과에는 이 속성이 처음부터 없다.
    """
    import h5py

    with h5py.File(src_path, "r") as src:
        for key, value in src.attrs.items():
            out.attrs[key] = value
    if require_format_version and "format_version" not in out.attrs:
        raise StageError(
            f"{os.path.basename(src_path)}에 format_version 속성이 없다. 이 파일을 그대로 "
            f"읽으면 Isaac Lab이 옛 형식으로 보고 로봇 받침의 사원수를 뒤집는다. "
            f"생성 단계가 이 파일을 어떻게 만들었는지 먼저 확인해야 한다.")


def merge_hdf5_shards(shards: list, out_path: str, renumber: bool, log,
                      require_format_version: bool = True) -> int:
    """Stitch per-process shards into one file of external links.

    Real copies are out of the question: one chunk of rendered RGB is tens of
    gigabytes, and we would be reading and rewriting all of it for nothing.
    h5py resolves external links transparently, so downstream stages see one
    dataset. Absolute link targets, because nothing guarantees the reader's cwd.
    """
    import h5py

    if len(shards) == 1:
        os.replace(shards[0], out_path)
        return len(_demo_names(out_path))
    n = 0
    # 임시 이름에 쓰고 다 끝났을 때만 제자리로 옮긴다. 도중에 멈추면 반쯤 쓰인 파일이
    # 남는데, 그 파일이 다음 실행에서 완성본으로 오인될 여지를 아예 없앤다.
    tmp_path = f"{out_path}.tmp{os.getpid()}"
    try:
        with h5py.File(tmp_path, "w") as out:
            copy_dataset_file_attrs(out, shards[0], require_format_version)
            data = out.create_group("data")
            for si, shard in enumerate(shards):
                with h5py.File(shard, "r") as src:
                    if si == 0:
                        for k, v in src["data"].attrs.items():
                            data.attrs[k] = v
                    names = sorted(src["data"].keys(), key=natural_key)
                for name in names:
                    target = f"demo_{n}" if renumber else name
                    if target in data:
                        raise StageError(f"duplicate demo name {target} while merging shards")
                    data[target] = h5py.ExternalLink(os.path.abspath(shard), f"data/{name}")
                    n += 1
            total = 0
            for name in data:
                total += int(data[name].attrs.get("num_samples", 0))
            data.attrs["total"] = total
        os.replace(tmp_path, out_path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
    log(f"merge: {len(shards)} shards -> {os.path.basename(out_path)} ({n} demos)")
    return n


def merge_sart_into_gen(gen_path: str, shards: list, out_path: str, log) -> tuple:
    """생성 결과와 SART 조각들을 외부 링크 파일 하나로 묶는다.

    두 가지 규칙이 있다. 첫째, 생성 결과의 데모는 아무 검사 없이 전부 옮긴다. SART가
    MimicGen이 만든 편을 줄이는 일이 절대 없어야 하기 때문이다. 둘째, SART 조각의
    데모는 세 가지를 통과해야 들어간다. 열려야 하고, success 속성이 참이어야 하고,
    num_samples가 4 이상이어야 한다. 4 미만이면 초당 10개로 다시 뽑을 때 기록 단계가
    요구하는 2스텝이 나오지 않는다. 통과하지 못한 편은 세어만 두고 넘어간다.

    gen.hdf5 자체가 외부 링크 파일일 수 있다. 생성 프로세스를 둘 이상 띄우면 그렇게
    된다. 그때는 링크가 가리키는 곳을 그대로 옮겨 링크가 두 겹이 되지 않게 한다.

    (전체 데모 수, 받아들인 SART 편수)를 돌려준다.
    """
    import h5py

    accepted = 0
    tmp_path = f"{out_path}.tmp{os.getpid()}"
    try:
        with h5py.File(tmp_path, "w") as out:
            copy_dataset_file_attrs(out, gen_path, require_format_version=True)
            data = out.create_group("data")
            n = 0
            with h5py.File(gen_path, "r") as src:
                for k, v in src["data"].attrs.items():
                    data.attrs[k] = v
                names = sorted(src["data"].keys(), key=natural_key)
                links = {}
                for name in names:
                    link = src["data"].get(name, getlink=True)
                    if isinstance(link, h5py.ExternalLink):
                        links[name] = (link.filename, link.path)
                    else:
                        links[name] = (os.path.abspath(gen_path), f"data/{name}")
            for name in names:
                filename, path = links[name]
                data[f"demo_{n}"] = h5py.ExternalLink(filename, path)
                n += 1
            base_demos = n

            for shard in shards:
                try:
                    with h5py.File(shard, "r") as src:
                        shard_names = sorted(src["data"].keys(), key=natural_key)
                        keep = []
                        for name in shard_names:
                            grp = src["data"][name]
                            ok = bool(grp.attrs.get("success", False))
                            samples = int(grp.attrs.get("num_samples", 0))
                            if samples == 0 and "actions" in grp:
                                samples = int(grp["actions"].shape[0])
                            if ok and samples >= 4:
                                keep.append(name)
                except Exception as exc:                      # noqa: BLE001
                    log(f"sart: 조각 {os.path.basename(shard)}을 읽지 못해 통째로 뺀다 ({exc})")
                    continue
                for name in keep:
                    data[f"demo_{n}"] = h5py.ExternalLink(os.path.abspath(shard), f"data/{name}")
                    n += 1
                    accepted += 1

            total = 0
            for name in data:
                total += int(data[name].attrs.get("num_samples", 0))
            data.attrs["total"] = total
        os.replace(tmp_path, out_path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
    log(f"sart: 생성 {base_demos}편 + 증강 {accepted}편 -> "
        f"{os.path.basename(out_path)} ({n}편)")
    return n, accepted


def merge_provenance(paths: list, out_path: str) -> dict:
    """Sum attempts/successes across generation processes (yield needs both)."""
    merged = {"n_success": 0, "n_attempts": 0, "input_file": "",
              "counts_success": [], "counts_all": [], "per_demo": []}
    agg = {"counts_success": {}, "counts_all": {}}
    for path in paths:
        doc = read_json(path)
        if doc is None:
            continue
        merged["n_success"] += int(doc.get("n_success", 0))
        merged["n_attempts"] += int(doc.get("n_attempts", 0))
        merged["input_file"] = doc.get("input_file", merged["input_file"])
        merged["per_demo"].extend(doc.get("per_demo", []))
        for key in ("counts_success", "counts_all"):
            for row in doc.get(key, []):
                k = (row.get("eef"), row.get("subtask"), row.get("src_ind"))
                agg[key][k] = agg[key].get(k, 0) + int(row.get("count", 0))
    for key in ("counts_success", "counts_all"):
        merged[key] = [{"eef": k[0], "subtask": k[1], "src_ind": k[2], "count": c}
                       for k, c in sorted(agg[key].items(), key=lambda kv: str(kv[0]))]
    atomic_write_json(out_path, merged)
    return merged


def merge_vrand_logs(paths: list, out_path: str) -> None:
    merged = {}
    for path in paths:
        doc = read_json(path)
        if isinstance(doc, dict):
            merged.update(doc)
    atomic_write_json(out_path, merged)


def read_lerobot_summary(log_path: str):
    """Recover the writer's JSON summary line from the chunk log.

    lerobot_writer.py prints one JSON object as its last stdout line and may
    legitimately skip an episode (bad alignment), so the manifest counts what
    the dataset really holds instead of what we asked for.
    """
    try:
        with open(log_path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            fh.seek(max(0, fh.tell() - 262144))
            tail = fh.read().decode("utf-8", "replace")
    except OSError:
        return None
    for line in reversed(tail.splitlines()):
        line = line.strip()
        if not line.startswith("{") or "episodes_written" not in line:
            continue
        try:
            doc = json.loads(line)
        except ValueError:
            continue
        if isinstance(doc, dict) and "episodes_written" in doc:
            return doc
    return None


def count_frames(contract_path: str) -> int:
    """Contract frames in a chunk — the `frames` field of MANIFEST.json."""
    import h5py
    total = 0
    try:
        with h5py.File(contract_path, "r") as fh:
            for name in fh["data"]:
                grp = fh["data"][name]
                if "num_samples" in grp.attrs:
                    total += int(grp.attrs["num_samples"])
                elif "timestamps" in grp:
                    total += int(grp["timestamps"].shape[0])
                elif "actions" in grp:
                    total += int(grp["actions"].shape[0])
    except (OSError, KeyError):
        return 0
    return total


# --------------------------------------------------------------------- stages
def already_has_demos(path: str, want: int) -> bool:
    """이 파일이 이미 원하는 만큼의 데모를 담고 있으면 참.

    재시도할 때 앞 단계를 다시 돌리지 않기 위해 쓴다. 생성 17분, 렌더 8분을 마지막
    단계 실패 때문에 버리지 않게 하는 것이 목적이다. 열리지 않거나 개수가 모자라면
    거짓을 돌려 그 단계를 다시 하게 한다.
    """
    if not os.path.isfile(path):
        return False
    try:
        import h5py
        with h5py.File(path, "r") as fh:
            return len(fh.get("data", {})) >= want
    except Exception:
        return False


def sart_source_quota(cfg: dict, chunk: dict) -> int:
    """MimicGen에 요청할 성공 편수. SART가 꺼져 있으면 청크 할당량 그대로다.

    SART를 켜면 생성이 할당량의 일부만 만들고 나머지는 증강이 채운다. 비율은 프로필의
    generate.sart.source_frac이 정하고 SART_SOURCE_FRAC으로 덮을 수 있다.
    """
    if not cfg["sart_enable"]:
        return chunk["episodes"]
    return max(1, round(chunk["episodes"] * cfg["sart_source_frac"]))


def gen_dataset_path(chunk: dict) -> str:
    """convert와 render가 읽을 생성 결과 파일.

    SART가 성공했다고 보고서에 적혀 있을 때만 증강본을 가리킨다. 보고서가 없거나 ok가
    아니면 원래의 gen.hdf5를 그대로 돌려준다. 판단을 청크 딕셔너리가 아니라 디스크에서
    하므로, stage.py로 단계를 따로 돌려도 같은 답이 나온다.
    """
    cdir = chunk["dir"]
    merged = os.path.join(cdir, "gen_sart.hdf5")
    doc = read_json(os.path.join(cdir, "sart_report.json"))
    if isinstance(doc, dict) and doc.get("ok") and os.path.isfile(merged):
        return merged
    return os.path.join(cdir, "gen.hdf5")


def _last_traceback_hint(log_path: str, lines: int = 25) -> str:
    """청크 로그에서 마지막 파이썬 예외를 찾아 돌려준다.

    Isaac Sim 위에서 도는 단계는 파이썬이 예외로 죽어도 종료 코드가 0으로 끝나는 경우가
    있다. 생성과 변환 둘 다 그렇다. 오케스트레이터가 내는 메시지만으로는 원인을 알 수
    없으므로, 로그에서 마지막 Traceback 이후를 잘라 붙인다.
    """
    try:
        with open(log_path, errors="replace") as fh:
            body = fh.read()
    except OSError:
        return ""
    idx = body.rfind("Traceback (most recent call last)")
    if idx < 0:
        return ""
    tail = [ln for ln in body[idx:].splitlines() if ln.strip()][:lines]
    return "\n  로그에 남은 실제 예외:\n    " + "\n    ".join(tail)


def stage_generate(cfg: dict, chunk: dict, log, log_path: str) -> float:
    """MimicGen generation, GEN_PROCS processes over disjoint success quotas.

    --generation_num_trials is a *success* quota (the generator keeps attempting
    until it has that many), so splitting it is enough to split the work.
    """
    cdir = chunk["dir"]
    merged = os.path.join(cdir, "gen.hdf5")
    want = sart_source_quota(cfg, chunk)
    if already_has_demos(merged, want):
        # 파일에 실제로 몇 편이 들어 있는지 세서 넣는다. 요청한 개수를 그대로 넣으면
        # 안 된다. 뒤 단계가 이 값만큼만 처리하므로, 파일이 더 많이 담고 있을 때
        # 나머지가 조용히 버려진다.
        chunk["produced"] = len(_demo_names(merged))
        chunk["mimicgen_produced"] = chunk["produced"]
        log(f"generate: {merged} 이미 {chunk['produced']}개를 담고 있어 건너뛴다")
        return 0.0
    quotas = largest_remainder(want, [1.0] * cfg["gen_procs"])
    quotas = [q for q in quotas if q > 0]
    jobs, shards, provs = [], [], []
    has_seed = script_supports(cfg["gen_script"], "--seed")
    for i, quota in enumerate(quotas):
        shard = os.path.join(cdir, f"gen.part{i:02d}.hdf5")
        prov = os.path.join(cdir, f"gen.part{i:02d}.provenance.json")
        seed = chunk["seed"] * 100 + i
        cmd = [ISAACLAB_SH, "-p", cfg["gen_script"],
               "--task", TASK_ID, "--headless", "--device", PHYSICS_DEVICE,
               "--num_envs", str(cfg["num_envs"]),
               "--generation_num_trials", str(quota),
               "--input_file", cfg["source_dataset"],
               "--output_file", shard]
        if has_seed:
            cmd += ["--seed", str(seed)]
        env = base_env(cfg, [ENV_DIR, RENDER_DIR])
        env.update({
            # IK 상대 제어의 이동량 배율. 프로필에서 오고, 환경변수를 주면 그쪽이 이긴다.
            "LAB_ARM_SCALE": cfg["arm_scale"],
            "LAB_KEEP_FAILED": "0",          # INTERFACE §2: gen.hdf5 holds successes only
            "LAB_SUBTASK_OFFSETS": cfg["subtask_offsets"],
            "LAB_PROVENANCE_INPUT": cfg["source_dataset"],
            "LAB_PROVENANCE_OUT": prov,
            "LAB_GEN_SEED": str(seed),       # chosen name for env-side seeding hooks
        })
        # 빈 문자열은 "정하지 않았다"는 뜻이라 넘기지 않는다.
        if cfg["action_noise"] != "":
            env["LAB_ACTION_NOISE"] = cfg["action_noise"]
        if cfg["num_interpolation_steps"] != "":
            env["LAB_NUM_INTERP"] = cfg["num_interpolation_steps"]
        jobs.append({"name": f"generate[{i}] quota={quota}", "cmd": cmd, "env": env})
        shards.append(shard)
        provs.append(prov)
    secs = run_parallel(jobs, cfg["gen_procs"], log, log_path)
    # Isaac Lab의 generate_dataset.py는 환경 생성이 예외로 죽어도 종료 코드 0으로
    # 끝나는 경우가 있다. 종료 코드만 믿으면 빈 파일을 들고 다음 단계로 넘어가고,
    # 거기서 "truncated file" 같은 엉뚱한 오류가 나서 진짜 원인이 가려진다. 그래서
    # 산출물을 직접 열어 확인하고, 비어 있으면 로그에서 실제 예외를 뽑아 보고한다.
    missing = [s for s in shards if not os.path.isfile(s)]
    if missing:
        raise StageError(f"generation produced no file: {missing}"
                         + _last_traceback_hint(log_path))
    import h5py
    for s in shards:
        try:
            with h5py.File(s, "r") as h:
                if len(h.get("data", {})) == 0:
                    raise StageError(f"generation wrote no demos into {s}"
                                     + _last_traceback_hint(log_path))
        except StageError:
            raise
        except Exception as exc:
            raise StageError(f"generation output {s} is unreadable ({exc})"
                             + _last_traceback_hint(log_path)) from exc
    n = merge_hdf5_shards(shards, os.path.join(cdir, "gen.hdf5"), renumber=True, log=log)
    prov = merge_provenance(provs, os.path.join(cdir, "gen.provenance.json"))
    if n == 0:
        raise StageError("generation produced 0 episodes")
    chunk["produced"] = n
    # SART가 뒤에서 편수를 늘리므로, MimicGen이 만든 편수를 따로 남긴다. 매니페스트의
    # 수율은 이 값으로 계산해야 "생성이 몇 번 시도해 몇 번 성공했는가"라는 뜻을 지킨다.
    chunk["mimicgen_produced"] = n
    chunk["attempts"] = int(prov.get("n_attempts", 0))
    log(f"generate: {n} episodes from {chunk['attempts']} attempts "
        f"(yield {n / max(1, chunk['attempts']):.3f})")
    return secs


def stage_sart(cfg: dict, chunk: dict, log, log_path: str) -> float:
    """MimicGen 결과를 소스로 삼아 접근 구간만 다양화한 에피소드를 더 만든다.

    이 단계는 gen.hdf5를 읽기로만 연다. 새 파일만 쓰고, 마지막에 보고서를 통째로 바꿔
    쓴 뒤에야 뒤 단계가 증강본을 읽기 시작한다. 그래서 증강이 어떻게 실패하든 이미
    만들어 둔 MimicGen 에피소드는 한 편도 잃지 않는다.
    """
    # 이 한 줄이 큐브 쪽 보장의 전부다. 꺼져 있으면 파일도 경로도 건드리지 않는다.
    if not cfg["sart_enable"]:
        return 0.0

    cdir = chunk["dir"]
    gen_path = os.path.join(cdir, "gen.hdf5")
    merged_path = os.path.join(cdir, "gen_sart.hdf5")
    report_path = os.path.join(cdir, "sart_report.json")
    t0 = time.time()

    try:
        source_n = len(_demo_names(gen_path))
    except Exception as exc:                              # noqa: BLE001
        raise StageError(f"sart: {gen_path}를 읽지 못했다 ({exc})") from exc

    # 이미 해 둔 일이면 다시 하지 않는다. 실패로 끝난 기록도 그대로 존중한다. 그래야
    # 청크를 다시 시도할 때 방금 실패한 일에 GPU 시간을 또 쓰지 않는다. 살아 있는
    # 소스 편수와 비교하므로, 생성을 다시 돌려 편수가 달라졌으면 옛 보고서는 버린다.
    doc = read_json(report_path)
    if isinstance(doc, dict) and int(doc.get("source_demos", -1)) == source_n:
        if doc.get("ok") and already_has_demos(merged_path, int(doc.get("total_demos", 0))):
            chunk["produced"] = int(doc["total_demos"])
            chunk["mimicgen_produced"] = source_n
            chunk["sart_added"] = int(doc.get("added", 0))
            log(f"sart: 이미 {chunk['sart_added']}편을 더해 두었다. 건너뛴다")
            return 0.0
        if not doc.get("ok"):
            chunk["produced"] = source_n
            chunk["mimicgen_produced"] = source_n
            reason = doc.get("reason", "적혀 있지 않다")
            # on_failure=fail은 "증강이 실패하면 이 청크를 실패로 처리하라"는 뜻이다.
            # 기록된 실패를 그냥 넘기면 첫 시도에서 청크가 재시도로 넘어갔다가 두 번째
            # 시도에서 이 자리를 그대로 통과해 버린다. 그러면 증강이 하나도 안 붙은 청크가
            # 성공으로 표시되고, 설정은 지켜지지 않는다.
            if SART_PROFILE["on_failure"] == "fail":
                raise StageError(f"sart: 앞선 시도가 실패로 기록돼 있다 (이유: {reason}). "
                                 f"generate.sart.on_failure가 fail이라 청크를 실패로 둔다. "
                                 f"증강 없이 넘어가려면 프로필을 continue로 바꾸거나 "
                                 f"{report_path} 파일을 지운다")
            log(f"sart: 앞선 시도가 실패로 기록돼 있어 다시 하지 않는다 (이유: {reason}). "
                f"다시 시도하려면 {report_path} 파일을 지운다")
            return 0.0

    # 남아 있는 옛 산출물을 지운다. 이 단계는 덧붙이지 않고 항상 처음부터 다시 만든다.
    for name in sorted(os.listdir(cdir)):
        if name == "gen_sart.hdf5" or (name.startswith("sart_")
                                       and (name.endswith(".hdf5") or name.endswith(".json"))):
            try:
                os.remove(os.path.join(cdir, name))
            except OSError:
                pass

    def give_up(reason: str, secs: float) -> float:
        """증강을 포기하고 MimicGen 편수 그대로 뒤 단계로 넘긴다."""
        log(f"sart: {reason}")
        chunk["produced"] = source_n
        chunk["mimicgen_produced"] = source_n
        chunk["sart_added"] = 0
        atomic_write_json(report_path, {"ok": False, "reason": reason,
                                        "source_demos": source_n,
                                        "total_demos": source_n, "added": 0,
                                        "seconds": round(secs, 1)})
        if SART_PROFILE["on_failure"] == "fail":
            raise StageError(f"sart: {reason}")
        return secs

    # 청크 할당량을 넘기지 않게 할 때만 상한을 준다. source_frac이 1.0이면 상한이 없고
    # 증강분이 그대로 더해지므로 청크가 할당량보다 커진다.
    room = max(0, chunk["episodes"] - source_n)
    # 프로세스마다 소스를 이어지는 구간으로 나눠 맡는다. 남은 자리가 프로세스 수보다 적으면
    # 프로세스를 그만큼만 띄운다. 안 그러면 상한이 0인 프로세스가 생기는데, 그 프로세스는
    # 시뮬레이터를 다 띄우고 한 편도 만들지 않고 끝난다. 시작 비용만 60초에서 90초다.
    procs = max(1, min(cfg["sart_procs"], source_n))
    if cfg["sart_source_frac"] < 1.0 and room > 0:
        procs = max(1, min(procs, room))
    counts = largest_remainder(source_n, [1.0] * procs)
    caps = largest_remainder(room, [1.0] * procs) if cfg["sart_source_frac"] < 1.0 else None
    if caps is not None and room == 0:
        # 생성이 이미 청크 할당량을 채웠다. 더 만들면 넘치므로 시뮬레이터를 띄우지 않는다.
        # 보고서를 쓰지 않으므로 뒤 단계는 원래의 gen.hdf5를 읽고, 청크를 다시 시도하면
        # 이 확인만 다시 한다.
        log(f"sart: 생성이 이미 {source_n}편을 만들어 청크 할당량 {chunk['episodes']}편을 "
            f"채웠다. 증강을 돌리지 않는다")
        chunk["produced"] = source_n
        chunk["mimicgen_produced"] = source_n
        chunk["sart_added"] = 0
        return 0.0

    jobs, shards, reports, start = [], [], [], 0
    for i, count in enumerate(counts):
        if count == 0:
            continue
        if caps is not None and caps[i] == 0:
            # 만들 자리가 없는 몫이다. 시뮬레이터를 띄워도 한 편도 못 만든다.
            log(f"sart[{i}]: 남은 자리가 0이라 띄우지 않는다")
            start += count
            continue
        shard = os.path.join(cdir, f"sart_{i:02d}.hdf5")
        # 이름에 ".part"를 넣지 않는다. cleanup_intermediates가 ".part"가 든 파일을
        # 모두 지우므로, 보고서가 사라지고 gen_dataset_path가 증강본을 못 찾게 된다.
        report = os.path.join(cdir, f"sart_{i:02d}.json")
        seed = chunk["seed"] * 100 + i + SART_PROFILE["seed_offset"]
        cmd = [ISAACLAB_SH, "-p", SART_SCRIPT,
               "--task", TASK_ID, "--headless", "--device", PHYSICS_DEVICE,
               "--dataset", gen_path, "--output", shard, "--report", report,
               "--register", ",".join(REGISTER_MODULES),
               "--source-start", str(start), "--source-count", str(count),
               "--samples-per-source", str(cfg["sart_samples"]),
               "--radius-m", str(cfg["sart_radius_m"]),
               "--rotation-deg", str(SART_PROFILE["rotation_deg"]),
               "--divert-steps", str(SART_PROFILE["divert_steps"]),
               "--converge-steps", str(SART_PROFILE["converge_steps"]),
               "--settle-steps", str(SART_PROFILE["settle_steps"]),
               "--tail-steps", str(SART_PROFILE["tail_steps"]),
               "--floor-margin-m", str(SART_PROFILE["floor_margin_m"]),
               "--converge-rule", SART_PROFILE["converge_rule"],
               "--converge-object", SART_PROFILE["converge_object"],
               "--converge-target", SART_PROFILE["converge_target"],
               "--converge-radius-m", str(SART_PROFILE["converge_radius_m"]),
               "--grip-closed-m", str(SART_PROFILE["grip_closed_m"]),
               "--max-consecutive-failures", str(SART_PROFILE["max_consecutive_failures"]),
               "--seed", str(seed)]
        if SART_PROFILE["fix_position"]:
            cmd += ["--fix-position"]
        if caps is not None:
            cmd += ["--max-total-demos", str(caps[i])]
        env = base_env(cfg, [SART_DIR, ENV_DIR, CONVERT_DIR, RENDER_DIR])
        env.update({
            # 소스를 만들 때 쓴 값과 같아야 한다. 다르면 같은 명령이 다른 거리를 간다.
            "LAB_ARM_SCALE": cfg["arm_scale"],
            "LAB_KEEP_FAILED": "0",
            "LAB_GEN_SEED": str(chunk["seed"] * 100 + i),
        })
        jobs.append({"name": f"sart[{i}] sources {start}..{start + count - 1}",
                     "cmd": cmd, "env": env})
        shards.append(shard)
        reports.append(report)
        start += count

    try:
        secs = run_parallel(jobs, cfg["sart_procs"], log, log_path)
    except StageError as exc:
        return give_up(f"증강 프로세스가 실패했다 ({exc})", time.time() - t0)
    except Exception as exc:                              # noqa: BLE001
        return give_up(f"증강 프로세스가 죽었다 ({exc!r})", time.time() - t0)

    # 조각 하나가 죽어도 나머지는 살린다. 열리지 않거나 비어 있는 조각만 뺀다.
    import h5py
    usable = []
    for shard in shards:
        if not os.path.isfile(shard):
            log(f"sart: {os.path.basename(shard)}이 없다. 그 몫만 뺀다")
            continue
        try:
            with h5py.File(shard, "r") as fh:
                if len(fh.get("data", {})) == 0:
                    log(f"sart: {os.path.basename(shard)}에 편이 없다. 그 몫만 뺀다")
                    continue
        except Exception as exc:                          # noqa: BLE001
            log(f"sart: {os.path.basename(shard)}을 읽지 못했다 ({exc}). 그 몫만 뺀다")
            continue
        usable.append(shard)
    if not usable:
        return give_up("쓸 수 있는 증강 조각이 하나도 없다", secs)

    try:
        total, added = merge_sart_into_gen(gen_path, usable, merged_path, log)
    except Exception as exc:                              # noqa: BLE001
        return give_up(f"증강본을 묶지 못했다 ({exc!r})", secs)
    if added == 0:
        return give_up("성공 판정을 통과한 증강 편이 하나도 없다", secs)
    if total != source_n + added:
        raise StageError(f"sart: 묶은 파일이 {total}편인데 생성 {source_n}편에 증강 "
                         f"{added}편을 더하면 {source_n + added}편이어야 한다")

    chunk["produced"] = total
    chunk["mimicgen_produced"] = source_n
    chunk["sart_added"] = added

    merged_report = {"ok": True, "reason": "", "source_demos": source_n,
                     "total_demos": total, "added": added,
                     "seconds": round(secs, 1),
                     "processes": [read_json(p) for p in reports]}
    for key in ("attempts", "successes", "degenerate_offsets", "reset_pose_mismatch",
                "errors"):
        merged_report[key] = sum(int((read_json(p) or {}).get(key, 0)) for p in reports)
    merged_report["dgr_pct"] = round(
        100.0 * merged_report["successes"] / max(1, merged_report["attempts"]), 1)
    # 접근 다양성. 재지 못한 프로세스는 None을 돌려주므로 평균에서 빼고 센다.
    # 0.0과 "재지 못했다"를 같은 값으로 묶으면 증강이 원본 복사로 무너진 실행을 놓친다.
    # 접근 다양성. 접근 구간이 삽입 구간보다 몇 배 흩어져 있는지를 본다. 삽입 구간은
    # 원본을 그대로 재생하므로 거의 0이어야 하고, 접근 구간은 그보다 훨씬 커야 한다.
    # 재지 못한 프로세스는 None을 돌려주므로 빼고 센다. 0.0과 "재지 못했다"를 같은 값으로
    # 묶으면 증강이 원본 복사로 무너진 실행을 놓친다.
    peaks, tails, ratios = [], [], []
    for path in reports:
        doc = read_json(path) or {}
        if doc.get("approach_std_peak_m") is not None:
            peaks.append(float(doc["approach_std_peak_m"]))
        if doc.get("approach_std_tail_m") is not None:
            tails.append(float(doc["approach_std_tail_m"]))
        if doc.get("approach_std_peak_over_tail") is not None:
            ratios.append(float(doc["approach_std_peak_over_tail"]))
    merged_report["approach_std_peak_m"] = round(max(peaks), 6) if peaks else None
    merged_report["approach_std_tail_m"] = round(min(tails), 6) if tails else None
    merged_report["approach_std_peak_over_tail"] = round(max(ratios), 1) if ratios else None
    if peaks:
        diversity = (f"접근 구간 다양성 {merged_report['approach_std_peak_m'] * 1000:.1f} mm, "
                     f"삽입 구간 {(merged_report['approach_std_tail_m'] or 0.0) * 1000:.2f} mm")
        if merged_report["approach_std_peak_over_tail"] is not None:
            diversity += f", 대비 {merged_report['approach_std_peak_over_tail']}배"
            if merged_report["approach_std_peak_over_tail"] < 3.0:
                diversity += " (대비가 작다. 증강이 원본 복사로 무너졌는지 확인해야 한다)"
    else:
        diversity = ("접근 다양성은 재지 못했다. 같은 소스에서 두 편 이상 성공해야 잴 수 "
                     "있으므로 samples_per_source를 늘린다")
    log(f"sart: {merged_report['attempts']}번 시도해 {added}편을 더했다 "
        f"(성공률 {merged_report['dgr_pct']}%, {diversity})")
    # 보고서를 맨 마지막에 쓴다. 이 파일이 자리에 놓인 뒤에야 뒤 단계가 증강본을 읽는다.
    atomic_write_json(report_path, merged_report)
    return secs


def stage_convert(cfg: dict, chunk: dict, log, log_path: str) -> float:
    """gen.hdf5 -> contract.hdf5 (10 Hz control contract, one process)."""
    cdir = chunk["dir"]
    out = os.path.join(cdir, "contract.hdf5")
    if already_has_demos(out, chunk["produced"]):
        log(f"convert: {out} 이미 완성돼 있어 건너뛴다")
        return 0.0
    cmd = [ISAACLAB_SH, "-p", CONVERT_SCRIPT, "--device", PHYSICS_DEVICE,
           # SART가 성공했으면 증강본을, 아니면 원래 생성 결과를 읽는다.
           "--dataset", gen_dataset_path(chunk), "--output", out,
           "--count", str(chunk["produced"]),
           "--report", os.path.join(cdir, "contract_report.json"),
           "--table_usd", TABLE_USD,
           "--objects", ",".join(CONVERT_OBJECTS)]
    env = base_env(cfg, [CONVERT_DIR, RENDER_DIR, ENV_DIR])
    secs = run_parallel([{"name": "convert", "cmd": cmd, "env": env}], 1, log, log_path)
    # 생성 단계와 같은 함정이 여기에도 있다. 변환 스크립트는 Isaac Sim 위에서 도는데,
    # 파이썬이 예외로 죽어도 종료 코드가 0으로 나오는 경우가 있다. 종료 코드만 믿으면
    # 잘려 나간 파일을 들고 다음 단계로 넘어가고, 기록 단계에서 "truncated file"이라는
    # 엉뚱한 오류가 나서 진짜 원인이 가려진다. 그래서 산출물을 직접 열어 확인한다.
    if not os.path.isfile(out):
        raise StageError("convert produced no contract.hdf5" + _last_traceback_hint(log_path))
    try:
        names = _demo_names(out)
    except Exception as exc:
        raise StageError(f"convert output {out} is unreadable ({exc})"
                         + _last_traceback_hint(log_path)) from exc
    if len(names) < chunk["produced"]:
        raise StageError(
            f"convert wrote {len(names)} demos but generation produced "
            f"{chunk['produced']}" + _last_traceback_hint(log_path))
    return secs


def stage_render(cfg: dict, chunk: dict, log, log_path: str) -> float:
    """RTX render, RENDER_PROCS processes over disjoint episode ranges.

    --every 2 makes the renderer emit 10 fps from the 20 Hz source directly
    (INTERFACE §6), so no frame is rendered that the dataset will not use.
    """
    cdir = chunk["dir"]
    n = chunk["produced"]
    rgb_out = os.path.join(cdir, "rgb.hdf5")
    if already_has_demos(rgb_out, n):
        log(f"render: {rgb_out} 이미 완성돼 있어 건너뛴다")
        return 0.0
    counts = largest_remainder(n, [1.0] * cfg["render_procs"])
    jobs, shards, vlogs, start = [], [], [], 0
    for i, count in enumerate(counts):
        if count == 0:
            continue
        shard = os.path.join(cdir, f"rgb.part{i:02d}.hdf5")
        vlog = os.path.join(cdir, f"vrand_log.part{i:02d}.json")
        cmd = [ISAACLAB_SH, "-p", RENDER_SCRIPT, "--device", PHYSICS_DEVICE,
               # SART가 성공했으면 증강본을, 아니면 원래 생성 결과를 읽는다.
               "--dataset", gen_dataset_path(chunk), "--output", shard,
               "--overlay", OVERLAY_YAML, "--binding", BINDING_YAML,
               "--table_usd", TABLE_USD,
               "--start", str(start), "--count", str(count),
               "--width", str(cfg["image_width"]), "--height", str(cfg["image_height"]),
               "--every", "2", "--preview_video", "0",
               "--cameras", ",".join(CAMERAS),
               "--vrand", chunk["profile"],
               "--vrand_config", cfg["vrand_config"], "--vrand_root", cfg["vrand_root"],
               "--vrand_seed", str(chunk["seed"] * 100 + i),
               "--vrand_log", vlog,
               "--success-module", RENDER_SUCCESS_MODULE,
               "--success-function", RENDER_SUCCESS_FUNCTION,
               "--success-verdict-attr", RENDER_SUCCESS_ATTR]
        if VRAND_OBJECT_PRIMS:
            cmd += ["--vrand-object-prims", json.dumps(VRAND_OBJECT_PRIMS)]
        # 렌더는 기본 환경이 큐브 장면이다. 다른 태스크는 자기 환경을 명시해야 한다.
        # 렌더 스크립트에 --task와 --register가 이미 있는데 넘기지 않고 있었다.
        render_task = PROFILE.get("render.task_id", "")
        render_modules = PROFILE.get("render.register_modules", [])
        if render_task:
            cmd += ["--task", render_task]
        if render_modules:
            cmd += ["--register", ",".join(render_modules)]
        # 프로파일과 시드는 위에서 --vrand / --vrand_seed로 넘긴다. 예전 도커
        # 스크립트가 내보내던 VRAND_PROFILE / VRAND_SEED는 읽는 코드가 없어 뺐다.
        env = base_env(cfg, [RENDER_DIR, ENV_DIR])
        jobs.append({"name": f"render[{i}] demos {start}..{start + count - 1}",
                     "cmd": cmd, "env": env})
        shards.append(shard)
        vlogs.append(vlog)
        start += count
    secs = run_parallel(jobs, cfg["render_procs"], log, log_path)
    missing = [s for s in shards if not os.path.isfile(s)]
    if missing:
        raise StageError(f"render produced no file: {missing}")
    # 렌더 결과는 카메라 영상이다. 기록 단계가 h5py로 직접 읽고 로봇 자세를 해석하지
    # 않으므로 사원수 형식 표시가 필요 없고, 렌더가 애초에 그 표시를 적지 않는다.
    rendered = merge_hdf5_shards(shards, os.path.join(cdir, "rgb.hdf5"),
                                 renumber=False, log=log,
                                 require_format_version=False)
    merge_vrand_logs(vlogs, os.path.join(cdir, "vrand_log.json"))
    if rendered != n:
        raise StageError(f"render covered {rendered} of {n} episodes")
    return secs


def stage_lerobot(cfg: dict, chunk: dict, log, log_path: str) -> float:
    """contract.hdf5 + rgb.hdf5 + vrand_log.json -> lerobot/ (this chunk only)."""
    cdir = chunk["dir"]
    out = os.path.join(cdir, "lerobot")
    cmd = [ISAACLAB_SH, "-p", LEROBOT_WRITER,
           "--contract", os.path.join(cdir, "contract.hdf5"),
           "--rgb", os.path.join(cdir, "rgb.hdf5"),
           "--vrand_log", os.path.join(cdir, "vrand_log.json"),
           "--output", out,
           "--profile", chunk["profile"],
           "--cameras", ",".join(CAMERAS),
           "--task-string", PROFILE.get("dataset.task_string",
                                        "Stack three cubes into a three-level tower"),
           "--robot-type", PROFILE.get("dataset.robot_type", "franka_fr3_osc"),
           # 초당 프레임 수. 계약 형식이 초당 10개이고 렌더도 --every 2로 그 수를
           # 맞추므로 두 값이 어긋나면 영상과 행동의 시각이 맞지 않는다.
           "--fps", str(PROFILE.get("dataset.fps", 10)),
           # 걸러진 에피소드는 기록으로 남기고 청크는 살린다. 기본값 0.9로 두면
           # 10%만 걸려도 종료 코드가 1이 되고, 오케스트레이터가 청크를 통째로
           # 버려 이미 쓴 몇 시간의 GPU 시간을 날린다. 실제 개수는 MANIFEST.json의
           # episodes 항목이 말해 준다.
           "--min-write-rate", "0",
           "--overwrite"]        # a manifest-less lerobot/ is leftover, never data
    env = base_env(cfg, [SRC_DIR, CONVERT_DIR, RENDER_DIR])
    secs = run_parallel([{"name": "lerobot", "cmd": cmd, "env": env}], 1, log, log_path)
    if not os.path.isdir(out):
        raise StageError("lerobot writer produced no dataset directory")
    return secs


def stage_upload(cfg: dict, chunk: dict, log, log_path: str) -> float:
    cmd = [ISAACLAB_SH, "-p", HF_UPLOAD,
           "--dataset", os.path.join(chunk["dir"], "lerobot"),
           "--repo_id", cfg["hf_repo_id"],
           "--private", "1" if cfg["hf_private"] else "0",
           "--chunk_index", str(chunk["chunk_index"])]
    env = base_env(cfg, [SRC_DIR])
    return run_parallel([{"name": "upload", "cmd": cmd, "env": env}], 1, log, log_path)


def cleanup_intermediates(chunk: dict, log) -> None:
    """Drop the big HDF5s once the LeRobot copy exists.

    lerobot/ stays: it is the product, and `merged/` is aggregated from it at
    the end of the run. Small JSON (provenance, vrand log, report) stays too.
    """
    removed = 0
    for name in sorted(os.listdir(chunk["dir"])):
        if name.endswith(".hdf5") or ".part" in name:
            try:
                os.remove(os.path.join(chunk["dir"], name))
                removed += 1
            except OSError:
                pass
    log(f"cleanup: removed {removed} intermediate files")


# ------------------------------------------------------------------ chunk run
def manifest_path(chunk: dict) -> str:
    return os.path.join(chunk["dir"], "MANIFEST.json")


def manifest_done(chunk: dict) -> dict:
    doc = read_json(manifest_path(chunk))
    if isinstance(doc, dict) and doc.get("status") == "done":
        return doc
    return None


def run_chunk(cfg: dict, chunk: dict, log, log_path: str) -> dict:
    """One chunk, all stages. Returns the manifest it wrote."""
    started = utcnow()
    t0 = time.time()
    durations = {}
    durations["generate"] = round(stage_generate(cfg, chunk, log, log_path), 1)
    if STOP.is_set():
        raise StageError("interrupted after generate")
    # SART 증강. 꺼져 있으면 첫 줄에서 0.0을 돌려주고 끝난다. 그래도 항상 부르는 이유는
    # durations_s에 키가 빠지면 이 파일을 읽는 쪽이 KeyError로 죽기 때문이다. upload가
    # 같은 이유로 늘 0.0을 적는다.
    durations["sart"] = round(stage_sart(cfg, chunk, log, log_path), 1)
    if STOP.is_set():
        raise StageError("interrupted after sart")
    durations["convert"] = round(stage_convert(cfg, chunk, log, log_path), 1)
    if STOP.is_set():
        raise StageError("interrupted after convert")
    durations["render"] = round(stage_render(cfg, chunk, log, log_path), 1)
    if STOP.is_set():
        raise StageError("interrupted after render")
    frames = count_frames(os.path.join(chunk["dir"], "contract.hdf5"))
    durations["lerobot"] = round(stage_lerobot(cfg, chunk, log, log_path), 1)
    episodes = chunk["produced"]
    summary = read_lerobot_summary(log_path)
    if summary is not None:
        episodes = int(summary.get("episodes_written", episodes))
        frames = int(summary.get("frames_written", frames))
        if episodes != chunk["produced"]:
            log(f"lerobot: wrote {episodes} of {chunk['produced']} episodes "
                f"({summary.get('episodes_skipped_count', 0)} skipped)")
    if episodes <= 0:
        # 한 개도 못 썼는데 done으로 적으면 재시작 때 영원히 건너뛰고, 목표 개수에
        # 못 미친 채 조용히 끝난다.
        raise StageError("이 청크는 기록된 에피소드가 0개다")
    uploaded = False
    if cfg["upload_each_chunk"] and cfg["hf_upload"]:
        durations["upload"] = round(stage_upload(cfg, chunk, log, log_path), 1)
        uploaded = True
    else:
        # 규격(INTERFACE.md 3절)이 durations_s에 upload가 항상 있다고 적고 있다.
        # 없으면 이 파일을 읽는 쪽이 KeyError로 죽는다.
        durations["upload"] = 0.0
    if not cfg["keep_intermediate"]:
        cleanup_intermediates(chunk, log)
    attempts = int(chunk.get("attempts", 0))
    # 수율은 MimicGen이 몇 번 시도해 몇 번 성공했는지를 뜻한다. SART가 더한 편수를 넣으면
    # 그 뜻이 사라지고 1을 넘는 값이 나오므로, 생성이 만든 편수로만 계산한다.
    generated = int(chunk.get("mimicgen_produced", chunk["produced"]))
    manifest = {
        "schema_version": CHUNK_SCHEMA,
        "chunk_index": chunk["chunk_index"],
        "status": "done",
        "profile": chunk["profile"],
        "episodes": episodes,
        "frames": frames,
        "attempts": attempts,
        # generation yield: successes out of MimicGen attempts, the 0.152 figure
        "yield": round(generated / attempts, 4) if attempts else 0.0,
        "seed": chunk["seed"],
        "physics_profile": cfg["physics_profile"],
        "image_size": [cfg["image_width"], cfg["image_height"]],
        "cameras": CAMERAS,
        "uploaded": uploaded,
        "started_at": started,
        "finished_at": utcnow(),
        "durations_s": durations,
    }
    if cfg["sart_enable"]:
        # 증강을 켠 청크만 이 항목을 붙인다. 끄면 durations_s["sart"]가 0.0인 것 외에
        # 매니페스트에 아무 흔적도 남지 않는다.
        manifest["sart"] = read_json(os.path.join(chunk["dir"], "sart_report.json")) or {
            "ok": False, "reason": "보고서가 없다"}
    chunk["written"] = episodes
    atomic_write_json(manifest_path(chunk), manifest)
    log(f"chunk {chunk['chunk_index']:05d} done in {time.time() - t0:.0f}s: "
        f"{manifest['episodes']} episodes, {manifest['frames']} frames")
    if episodes < 0.9 * chunk["episodes"]:
        # 목표에 크게 못 미친 청크를 로그에서 바로 보이게 한다. SART를 켜면 생성에
        # 요청하는 편수가 줄어들기 때문에, 증강 성공률이 예상보다 낮으면 청크가
        # 조용히 작아진다. 그 상태를 마지막 합계에서야 알게 되면 늦다.
        log(f"chunk {chunk['chunk_index']:05d} 경고: 계획한 {chunk['episodes']}편 중 "
            f"{episodes}편만 기록했다")
    return manifest


def wipe_chunk_products(chunk: dict) -> None:
    """Before a retry: clear everything but the log, so nothing stale is reused."""
    for name in os.listdir(chunk["dir"]):
        path = os.path.join(chunk["dir"], name)
        try:
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
        except OSError:
            pass


# ------------------------------------------------------------------- run loop
def chunk_episodes(chunk: dict) -> int:
    """Episodes a chunk really contributed (written < produced if any were skipped)."""
    return int(chunk.get("written", chunk.get("produced", chunk["episodes"])))


def write_state(cfg: dict, plan: list, started_at: str) -> None:
    done = [c for c in plan if c["status"] == "done"]
    atomic_write_json(cfg["state_path"], {
        "schema_version": STATE_SCHEMA,
        "started_at": started_at,
        "updated_at": utcnow(),
        "target_episodes": cfg["target_episodes"],
        "chunk_size": cfg["chunk_size"],
        "profile_split": cfg["profile_split"],
        "profile_quota": {p: sum(c["episodes"] for c in plan if c["profile"] == p)
                          for p in PROFILE_NAMES},
        "physics_profile": cfg["physics_profile"],
        "image_size": [cfg["image_width"], cfg["image_height"]],
        "seed_base": cfg["seed_base"],
        "chunks_total": len(plan),
        "chunks_done": len(done),
        "episodes_done": sum(chunk_episodes(c) for c in done),
        "failed_chunks": [c["chunk_index"] for c in plan if c["status"] == "failed"],
        "chunks": [{k: c[k] for k in ("chunk_index", "profile", "episodes", "seed", "status")}
                   for c in plan],
    })


def progress_line(plan: list, cfg: dict, t_start: float) -> str:
    done = [c for c in plan if c["status"] == "done"]
    eps = sum(chunk_episodes(c) for c in done)
    elapsed = time.time() - t_start
    rate = eps / elapsed * 3600.0 if elapsed > 0 and eps else 0.0
    if rate > 0:
        remaining = max(0, cfg["target_episodes"] - eps) / rate * 3600.0
        eta = (dt.datetime.now(dt.timezone.utc)
               + dt.timedelta(seconds=remaining)).strftime("%Y-%m-%dT%H:%MZ")
    else:
        eta = "unknown"
    return (f"progress: chunks {len(done)}/{len(plan)} | episodes {eps}/{cfg['target_episodes']} "
            f"| elapsed {elapsed / 3600.0:.2f}h | {rate:.1f} ep/h | ETA {eta}")


def install_signal_handlers(log) -> None:
    def handler(signum, _frame):
        if not STOP.is_set():
            log(f"signal {signum} received: finishing the running stage, then stopping")
        STOP.set()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, handler)


def run_preflight() -> int:
    """Validate before touching anything (imported late: preflight imports us)."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import preflight
    return preflight.main([])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="hf80k chunk orchestrator "
                                             "(configuration comes from env vars)")
    ap.add_argument("--dry-run", action="store_true", help="print the chunk plan and exit")
    ap.add_argument("--skip-preflight", action="store_true",
                    help="do not run preflight.py first (debugging only)")
    args = ap.parse_args(argv)

    cfg = load_config()
    for d in (cfg["work_dir"], cfg["chunks_dir"], cfg["logs_dir"], cfg["merged_dir"]):
        os.makedirs(d, exist_ok=True)
    run_log = make_logger(os.path.join(cfg["logs_dir"], "orchestrate.log"))
    install_signal_handlers(run_log)

    plan = plan_chunks(cfg)
    quota = {p: sum(c["episodes"] for c in plan if c["profile"] == p) for p in PROFILE_NAMES}
    run_log(f"plan: {len(plan)} chunks of <= {cfg['chunk_size']} episodes, quota {quota}")
    if args.dry_run:
        for c in plan[:5] + (plan[-2:] if len(plan) > 7 else []):
            print(f"  chunk_{c['chunk_index']:05d} {c['profile']:14s} "
                  f"{c['episodes']:5d} ep seed={c['seed']}")
        return 0

    if not args.skip_preflight:
        rc = run_preflight()
        if rc != 0:
            run_log("preflight failed, refusing to start")
            return rc

    started_at = utcnow()
    for chunk in plan:
        os.makedirs(chunk["dir"], exist_ok=True)
        doc = manifest_done(chunk) if cfg["resume"] else None
        if doc is not None:
            chunk["status"] = "done"
            chunk["produced"] = int(doc.get("episodes", chunk["episodes"]))
            chunk["written"] = chunk["produced"]
    write_state(cfg, plan, started_at)

    prepare_source_dataset(cfg, run_log)
    prepare_generate_script(cfg, run_log)

    t_start = time.time()
    interrupted = False
    for chunk in plan:
        if chunk["status"] == "done":
            continue
        if STOP.is_set():
            interrupted = True
            break
        log_path = os.path.join(cfg["logs_dir"], f"chunk_{chunk['chunk_index']:05d}.log")
        log = make_logger(log_path)
        log(f"chunk {chunk['chunk_index']:05d} start "
            f"profile={chunk['profile']} episodes={chunk['episodes']} seed={chunk['seed']}")
        if os.listdir(chunk["dir"]):
            # leftovers from an interrupted attempt: no manifest means nothing here
            # is trustworthy, and the disk is better spent on the retry
            log(f"chunk {chunk['chunk_index']:05d} clearing leftovers from an earlier attempt")
            wipe_chunk_products(chunk)
        for attempt in (1, 2):
            try:
                run_chunk(cfg, chunk, log, log_path)
                chunk["status"] = "done"
                break
            except StageError as exc:
                if STOP.is_set():
                    log(f"chunk {chunk['chunk_index']:05d} stopped: {exc}")
                    interrupted = True
                    break
                log(f"chunk {chunk['chunk_index']:05d} attempt {attempt} failed: {exc}")
                if attempt == 1:
                    # 이미 만든 산출물은 남긴다. 각 단계가 자기 결과물이 있으면
                    # 건너뛰므로, 실패한 단계부터 이어서 한다.
                    log(f"chunk {chunk['chunk_index']:05d} 이미 만든 산출물은 두고 "
                        f"실패한 단계부터 다시 한다")
                else:
                    chunk["status"] = "failed"
                    chunk["error"] = str(exc)
            except Exception as exc:                     # noqa: BLE001 - keep the run alive
                log(f"chunk {chunk['chunk_index']:05d} attempt {attempt} crashed: {exc!r}")
                if attempt == 1:
                    log(f"chunk {chunk['chunk_index']:05d} 이미 만든 산출물은 두고 "
                        f"실패한 단계부터 다시 한다")
                else:
                    chunk["status"] = "failed"
                    chunk["error"] = repr(exc)
        write_state(cfg, plan, started_at)
        run_log(progress_line(plan, cfg, t_start))
        if interrupted:
            break

    write_state(cfg, plan, started_at)
    failed = [c["chunk_index"] for c in plan if c["status"] == "failed"]
    done = [c for c in plan if c["status"] == "done"]
    run_log(progress_line(plan, cfg, t_start))
    run_log(f"summary: {len(done)}/{len(plan)} chunks done, {len(failed)} failed"
            + (f", failed chunks: {failed}" if failed else ""))
    run_log(f"note: merged/ is a separate final step — "
            f"{HF_UPLOAD} --mode aggregate --work_dir {cfg['work_dir']}")
    if interrupted:
        run_log("stopped on signal; rerun with RESUME=1 to continue")
        return 0
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
