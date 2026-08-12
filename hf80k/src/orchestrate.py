#!/usr/bin/env python3
"""Chunk orchestrator — the single entry point of the hf80k container.

WHY this file exists: 80k successful episodes cannot come out of one long
process. At the measured rates (3.17 s per generation attempt at num_envs=4,
15.2% yield with the zero-yield source excluded, 15.1 ms per camera-frame,
176 contract steps per episode) a full run is days of wall clock, and anything
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
CAMERAS = ["third_person_0", "third_person_1", "wrist"]

# In-container Isaac Lab install (we are already inside the isaac-lab image, so
# no nested docker — this is the difference from contract/run_lab_generate_docker.sh).
ISAACLAB_SH = "/workspace/isaaclab/isaaclab.sh"
GEN_DATASET_SRC = ("/workspace/isaaclab/scripts/imitation_learning/isaaclab_mimic/"
                   "generate_dataset.py")
TASK_ID = "Isaac-Stack-Cube-LabFR3-HF80K-Fwd-IK-Rel-Mimic-v0"
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
HF_UPLOAD = os.path.join(SRC_DIR, "hf_upload.py")
SOURCE_HDF5 = os.path.join(ASSETS_DIR, "fwd_annotated.hdf5")
SOURCE_YIELD_JSON = os.path.join(ASSETS_DIR, "source_yield.json")
OVERLAY_YAML = os.path.join(ASSETS_DIR, "fr3_camera_overlay_v2/overlay.yaml")
BINDING_YAML = os.path.join(ASSETS_DIR, "fr3_binding_v2.yaml")
# The RL team's visual-randomization handoff package. Preferred location is
# inside our assets; /vrand is the mount the existing render script assumes.
VRAND_ROOTS = (os.path.join(ASSETS_DIR, "fr3_visual_randomization_v1"), "/vrand")

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
        "subtask_offsets": os.environ.get("SUBTASK_OFFSETS", "10,20"),
        "work_dir": env_str("WORK_DIR", "/work"),
        "keep_intermediate": env_flag("KEEP_INTERMEDIATE", "0"),
        "upload_each_chunk": env_flag("UPLOAD_EACH_CHUNK", "1"),
        "resume": env_flag("RESUME", "1"),
        "seed_base": env_int("SEED_BASE", 42000),
        "log_level": env_str("LOG_LEVEL", "INFO"),
    }
    for key in ("chunk_size", "num_envs", "gen_procs", "render_procs"):
        if cfg[key] < 1:
            raise SystemExit(f"[orch] {key.upper()} must be >= 1 (got {cfg[key]})")
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
    parts = [p for p in extra_pythonpath if p]
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
    inject = "\nimport lab_register\nimport clean_success_hook\nimport provenance_hooks"
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


def merge_hdf5_shards(shards: list, out_path: str, renumber: bool, log) -> int:
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
    with h5py.File(out_path, "w") as out:
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
    log(f"merge: {len(shards)} shards -> {os.path.basename(out_path)} ({n} demos)")
    return n


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


def _generate_failure_hint(log_path: str, lines: int = 25) -> str:
    """청크 로그에서 마지막 파이썬 예외를 찾아 돌려준다.

    생성이 종료 코드 0으로 끝나 놓고 아무것도 안 만드는 경우가 있어서, 오케스트레이터가
    내는 메시지만으로는 원인을 알 수 없다. 로그에서 마지막 Traceback 이후를 잘라 붙인다.
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
    if already_has_demos(merged, chunk["episodes"]):
        log(f"generate: {merged} 이미 {chunk['episodes']}개를 담고 있어 건너뛴다")
        chunk["produced"] = chunk["episodes"]
        return 0.0
    quotas = largest_remainder(chunk["episodes"], [1.0] * cfg["gen_procs"])
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
            # generation IK-rel scale 0.5 is the official setting (run_generate.sh)
            "LAB_ARM_SCALE": env.get("LAB_ARM_SCALE", "0.5"),
            "LAB_KEEP_FAILED": "0",          # INTERFACE §2: gen.hdf5 holds successes only
            "LAB_SUBTASK_OFFSETS": cfg["subtask_offsets"],
            "LAB_PROVENANCE_INPUT": cfg["source_dataset"],
            "LAB_PROVENANCE_OUT": prov,
            "LAB_GEN_SEED": str(seed),       # chosen name for env-side seeding hooks
        })
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
                         + _generate_failure_hint(log_path))
    import h5py
    for s in shards:
        try:
            with h5py.File(s, "r") as h:
                if len(h.get("data", {})) == 0:
                    raise StageError(f"generation wrote no demos into {s}"
                                     + _generate_failure_hint(log_path))
        except StageError:
            raise
        except Exception as exc:
            raise StageError(f"generation output {s} is unreadable ({exc})"
                             + _generate_failure_hint(log_path)) from exc
    n = merge_hdf5_shards(shards, os.path.join(cdir, "gen.hdf5"), renumber=True, log=log)
    prov = merge_provenance(provs, os.path.join(cdir, "gen.provenance.json"))
    if n == 0:
        raise StageError("generation produced 0 episodes")
    chunk["produced"] = n
    chunk["attempts"] = int(prov.get("n_attempts", 0))
    log(f"generate: {n} episodes from {chunk['attempts']} attempts "
        f"(yield {n / max(1, chunk['attempts']):.3f})")
    return secs


def stage_convert(cfg: dict, chunk: dict, log, log_path: str) -> float:
    """gen.hdf5 -> contract.hdf5 (10 Hz control contract, one process)."""
    cdir = chunk["dir"]
    out = os.path.join(cdir, "contract.hdf5")
    if already_has_demos(out, chunk["produced"]):
        log(f"convert: {out} 이미 완성돼 있어 건너뛴다")
        return 0.0
    cmd = [ISAACLAB_SH, "-p", CONVERT_SCRIPT, "--device", PHYSICS_DEVICE,
           "--dataset", os.path.join(cdir, "gen.hdf5"), "--output", out,
           "--count", str(chunk["produced"]),
           "--report", os.path.join(cdir, "contract_report.json"),
           "--table_usd", TABLE_USD]
    env = base_env(cfg, [CONVERT_DIR, RENDER_DIR, ENV_DIR])
    secs = run_parallel([{"name": "convert", "cmd": cmd, "env": env}], 1, log, log_path)
    if not os.path.isfile(out):
        raise StageError("convert produced no contract.hdf5")
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
               "--dataset", os.path.join(cdir, "gen.hdf5"), "--output", shard,
               "--overlay", OVERLAY_YAML, "--binding", BINDING_YAML,
               "--table_usd", TABLE_USD,
               "--start", str(start), "--count", str(count),
               "--width", str(cfg["image_width"]), "--height", str(cfg["image_height"]),
               "--every", "2", "--preview_video", "0",
               "--cameras", ",".join(CAMERAS),
               "--vrand", chunk["profile"],
               "--vrand_config", cfg["vrand_config"], "--vrand_root", cfg["vrand_root"],
               "--vrand_seed", str(chunk["seed"] * 100 + i),
               "--vrand_log", vlog]
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
    rendered = merge_hdf5_shards(shards, os.path.join(cdir, "rgb.hdf5"),
                                 renumber=False, log=log)
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
    if cfg["upload_each_chunk"]:
        durations["upload"] = round(stage_upload(cfg, chunk, log, log_path), 1)
        uploaded = True
    else:
        # 규격(INTERFACE.md 3절)이 durations_s에 upload가 항상 있다고 적고 있다.
        # 없으면 이 파일을 읽는 쪽이 KeyError로 죽는다.
        durations["upload"] = 0.0
    if not cfg["keep_intermediate"]:
        cleanup_intermediates(chunk, log)
    attempts = int(chunk.get("attempts", 0))
    manifest = {
        "schema_version": CHUNK_SCHEMA,
        "chunk_index": chunk["chunk_index"],
        "status": "done",
        "profile": chunk["profile"],
        "episodes": episodes,
        "frames": frames,
        "attempts": attempts,
        # generation yield: successes out of MimicGen attempts, the 0.152 figure
        "yield": round(chunk["produced"] / attempts, 4) if attempts else 0.0,
        "seed": chunk["seed"],
        "physics_profile": cfg["physics_profile"],
        "image_size": [cfg["image_width"], cfg["image_height"]],
        "cameras": CAMERAS,
        "uploaded": uploaded,
        "started_at": started,
        "finished_at": utcnow(),
        "durations_s": durations,
    }
    chunk["written"] = episodes
    atomic_write_json(manifest_path(chunk), manifest)
    log(f"chunk {chunk['chunk_index']:05d} done in {time.time() - t0:.0f}s: "
        f"{manifest['episodes']} episodes, {manifest['frames']} frames")
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
