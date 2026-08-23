#!/usr/bin/env python3
"""Preflight checks — refuse to start a multi-day run on a broken box.

WHY: every failure this file looks for is one we would otherwise discover
hours in, after burning GPU time. A missing HF token surfaces at the first
upload, a missing asset at the first generation, a full disk in the middle of
the first render, an old LeRobot at the first dataset write. All of them are
one-second questions to ask up front, so they get asked up front.

The rule for statuses: FAIL means the run cannot work and we exit nonzero;
WARN means it can work but somebody should look. Everything is printed as one
table so the answer to "why did it not start" is the first screen of the log.

Run:  isaaclab.sh -p src/preflight.py     (orchestrate.py runs it for you)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys

import orchestrate as orch          # env parsing + paths live there, single source

# Intermediate-size estimates per episode, from the measured facts: 176 contract
# steps at 10 Hz, three rendered cameras (render_viewpoints.py CONTRACT_CAMERAS),
# gzip-4 on RGB lands near 0.4 of raw. Used only to size the disk check.
RENDER_CAMERAS = 3
STEPS_PER_EPISODE = 176
RGB_COMPRESSION = 0.4
EST_GEN_MB = 8.0
EST_CONTRACT_MB = 1.0
EST_LEROBOT_MB = 3.0
DISK_SAFETY = 1.3

MIN_LEROBOT = (0, 4)


def _rgb_mb_per_episode(width: int, height: int) -> float:
    raw = STEPS_PER_EPISODE * RENDER_CAMERAS * width * height * 3
    return raw * RGB_COMPRESSION / 1e6


def _probe(code: str):
    """Run a snippet under the Isaac Lab python (used when our own import fails)."""
    if not os.path.isfile(orch.ISAACLAB_SH):
        return None
    try:
        res = subprocess.run([orch.ISAACLAB_SH, "-p", "-c", code],
                             capture_output=True, text=True, timeout=300)
    except (OSError, subprocess.SubprocessError):
        return None
    return res.stdout.strip() if res.returncode == 0 else None


def _version_tuple(text: str) -> tuple:
    parts = []
    for piece in str(text).lstrip("v").split("."):
        digits = ""
        for ch in piece:
            if ch.isdigit():
                digits += ch
            else:
                break
        if digits == "":
            break
        parts.append(int(digits))
    return tuple(parts) if parts else (0,)


# ------------------------------------------------------------------- checks
def check_task_profile(cfg):
    """어떤 태스크 프로필로 도는지, 요청한 것이 실제로 실려 있는지 본다.

    로더는 잘못된 프로필을 만나면 기본(큐브) 프로필로 되돌린다. 그 자체는 옳지만, 조용히
    되돌아가면 peg를 돌리려던 사람이 큐브 데이터를 8만 편 만들게 된다. 여기서 막는다.
    """
    profile = orch.PROFILE
    wanted = os.environ.get("TASK_PROFILE", "").strip()
    if profile.error:
        return "FAIL", f"프로필 {wanted or profile.name}: {profile.error}"
    if wanted and profile.name != wanted:
        return "FAIL", f"{wanted}를 요청했는데 {profile.name}이 실렸다"
    return "PASS", (f"{profile.name} / 태스크 {orch.TASK_ID} / "
                    f"카메라 {len(orch.CAMERAS)}대 / 소스 "
                    f"{os.path.basename(orch.SOURCE_HDF5)}")


def check_required_env(cfg):
    """허깅페이스에 올릴 때만 자격증명을 요구한다.

    예전에는 무조건 요구했다. 그러면 결과를 로컬에만 쌓고 싶은 사람도 토큰을 만들어야
    했고, 남의 파이프라인 안에 한 모듈로 끼워 넣을 때 특히 부담이었다. HF_UPLOAD=0이면
    업로드 단계를 아예 돌지 않으므로 자격증명이 필요 없다.
    """
    if not cfg.get("hf_upload", True):
        return "PASS", "HF_UPLOAD=0, 업로드를 하지 않으므로 자격증명을 보지 않는다"
    missing = [n for n in ("HF_TOKEN", "HF_REPO_ID") if not os.environ.get(n)]
    if missing:
        return "FAIL", (f"missing: {', '.join(missing)}. "
                        f"올리지 않을 것이면 HF_UPLOAD=0으로 둔다")
    return "PASS", f"repo {cfg['hf_repo_id']} private={int(cfg['hf_private'])}"


def check_config_sanity(cfg):
    problems = []
    total = sum(cfg["profile_split"].values())
    if abs(total - 1.0) > 1e-6:
        problems.append(f"PROFILE_SPLIT sums to {total:.4f}, not 1.0")
    if cfg["physics_profile"] not in ("nominal", "posterior_stochastic",
                                      "robust_stochastic", "off"):
        problems.append(f"PHYSICS_PROFILE={cfg['physics_profile']!r} unknown")
    spec = cfg["source_demo_filter"]
    if spec not in ("all", "exclude_zero_yield"):
        bad = [x for x in spec.replace(" ", "").split(",") if not x.isdigit()]
        if bad:
            problems.append(f"SOURCE_DEMO_FILTER={spec!r} is not all|exclude_zero_yield|indices")
    if cfg["chunk_size"] > cfg["target_episodes"]:
        problems.append("CHUNK_SIZE > TARGET_EPISODES")
    if problems:
        return "FAIL", "; ".join(problems)
    quota = orch.largest_remainder(cfg["target_episodes"],
                                   [cfg["profile_split"][p] for p in orch.PROFILE_NAMES])
    n_chunks = len(orch.plan_chunks(cfg))
    return "PASS", (f"{cfg['target_episodes']} eps -> {n_chunks} chunks, "
                    f"quota {dict(zip(orch.PROFILE_NAMES, quota))}")


def check_work_dir(cfg):
    path = cfg["work_dir"]
    try:
        os.makedirs(path, exist_ok=True)
        probe = os.path.join(path, ".preflight_write_test")
        with open(probe, "w") as fh:
            fh.write("ok")
        os.remove(probe)
    except OSError as exc:
        return "FAIL", f"{path} not writable: {exc}"
    return "PASS", path


def check_assets(cfg):
    needed = [orch.SOURCE_HDF5, orch.OVERLAY_YAML, orch.BINDING_YAML]
    if cfg["source_demo_filter"] == "exclude_zero_yield":
        needed.append(orch.SOURCE_YIELD_JSON)
    # 프로필이 적어 둔 자산 목록. 파일일 수도 폴더일 수도 있어서 따로 센다. 태스크를
    # 바꿀 때 빠뜨린 파일을 여기서 잡는 것이 이 목록의 존재 이유다.
    listed = [os.path.join(orch.ASSETS_DIR, str(name))
              for name in (orch.PROFILE.get("assets.required", []) or [])]
    # 물리 랜덤화를 켜면 환경 설정이 이 세 파일을 반드시 읽는다. 없으면 첫 생성
    # 프로세스 안에서야 RuntimeError가 나므로, 여기서 미리 잡는다. 번들 경로는
    # 프로필의 physics.bundle_dir에서 오고, 밖에서 환경변수를 주면 그쪽이 이긴다.
    if cfg["physics_profile"] != "off":
        bundle = os.environ.get(
            "LAB_SYSID_BUNDLE_ROOT",
            orch.PHYSICS_ENV.get(
                "LAB_SYSID_BUNDLE_ROOT",
                os.path.join(orch.ASSETS_DIR, "fr3_cube_system_calibration_bundle_v1")))
        needed += [
            os.path.join(bundle, "modules", "dynamics_controller",
                         "domain_randomization_samples.csv"),
            os.path.join(bundle, "modules", "contact", "posterior_samples.csv"),
            os.path.join(bundle, "parameters.json"),
        ]
    missing = [p for p in needed if not os.path.isfile(p)]
    missing += [p for p in listed if not os.path.exists(p)]
    if missing:
        return "FAIL", "없는 자산: " + ", ".join(missing)
    size_gb = os.path.getsize(orch.SOURCE_HDF5) / 1e9
    return "PASS", (f"{len(needed)}개 파일 + 프로필 목록 {len(listed)}개, "
                    f"소스 {size_gb:.2f} GB")


def check_source_filter(cfg):
    """The filter runs before the first generation; a bad yield table must fail here."""
    if orch.ENV_DIR not in sys.path:
        sys.path.insert(0, orch.ENV_DIR)
    try:
        import source_filter
    except Exception as exc:                      # noqa: BLE001
        return "FAIL", f"src/env/source_filter.py not importable ({exc.__class__.__name__})"
    setting = cfg["source_demo_filter"]
    if setting != "exclude_zero_yield":
        return "PASS", f"filter {setting!r}"
    try:
        dropped = source_filter.zero_yield_indices(
            source_filter.load_yield_table(orch.SOURCE_YIELD_JSON))
    except Exception as exc:                      # noqa: BLE001
        return "FAIL", f"source_yield.json unusable: {exc}"
    if not dropped:
        return "WARN", "no measured zero-yield source, the filter will drop nothing"
    return "PASS", f"drops source indices {dropped}"


def check_vrand_package(cfg):
    cfg_dir = cfg["vrand_config"]
    needed = [os.path.join(cfg_dir, "visual_randomization_profiles.yaml"),
              os.path.join(cfg_dir, "camera_nominal_measured_ranges.yaml")]
    missing = [p for p in needed if not os.path.isfile(p)]
    if missing:
        return "FAIL", "missing: " + ", ".join(missing)
    return "PASS", cfg["vrand_root"]


def check_stage_scripts(cfg):
    needed = [orch.RENDER_SCRIPT, orch.CONVERT_SCRIPT,
              orch.LEROBOT_WRITER, orch.HF_UPLOAD]
    if cfg.get("sart_enable"):
        # 증강 단계 스크립트가 없으면 생성에 GPU 시간을 다 쓰고 나서야 알게 된다.
        needed.append(orch.SART_SCRIPT)
    missing = [p for p in needed if not os.path.isfile(p)]
    if missing:
        return "FAIL", "missing: " + ", ".join(os.path.basename(p) for p in missing)
    return "PASS", "generate/convert/render/lerobot/upload entry points present"


def check_sart(cfg):
    """SART 증강 설정을 본다.

    프로필 로더는 절 안의 첫 단계 이름만 검사하므로, generate.sart 아래의 오타는
    로더를 그냥 통과한다. 그 오타는 값이 조용히 기본값으로 돌아가는 결과만 낳는다.
    여기서 이름 하나하나를 대조해 막는다.
    """
    if not cfg.get("sart_enable"):
        why = ("SART_ENABLE=0으로 껐다" if os.environ.get("SART_ENABLE", "") else
               "이 태스크 프로필에 generate.sart 절이 없거나 enable이 거짓이다")
        return "PASS", f"SART를 돌지 않는다 ({why})"

    # 절을 통째로 꺼낸다. 점으로 이은 경로 문자열을 여기 쓰면, 죽은 키 검사가 그 한
    # 문자열로 절 아래 모든 키를 통과시켜 버린다. 그래서 사전을 두 단계로 따라간다.
    block = (orch.PROFILE.doc.get("generate") or {}).get("sart")
    if not isinstance(block, dict):
        return "FAIL", ("SART를 켰는데 태스크 프로필에 generate.sart 절이 없다. "
                        "끄려면 SART_ENABLE=0으로 둔다")
    unknown = sorted(set(block) - set(orch.SART_KEYS))
    if unknown:
        return "FAIL", ("generate.sart 절에 모르는 키가 있다: " + ", ".join(unknown)
                        + ". 받을 수 있는 이름은 " + ", ".join(orch.SART_KEYS) + "이다")

    p = orch.SART_PROFILE
    problems = []
    if not p["converge_object"]:
        problems.append("converge_object가 비어 있다. 옮기는 물체 이름을 적는다")
    if not p["converge_target"]:
        problems.append("converge_target이 비어 있다. 고정 목표 좌표계 이름을 적는다")
    if cfg["sart_samples"] < 1:
        problems.append(f"소스당 시도 횟수가 {cfg['sart_samples']}회다. 1 이상이어야 한다")
    if cfg["sart_radius_m"] <= 0:
        problems.append(f"공의 반지름이 {cfg['sart_radius_m']} m다. 0보다 커야 한다")
    if p["converge_steps"] < 1:
        problems.append(f"되돌아오는 스텝이 {p['converge_steps']}이다. 1 이상이어야 한다")
    if p["converge_rule"] not in ("radial_gate", "descent_onset", "tail_offset"):
        problems.append(f"수렴 규칙 {p['converge_rule']!r}은 radial_gate, descent_onset, "
                        f"tail_offset 중 하나가 아니다")
    if p["on_failure"] not in ("continue", "fail"):
        problems.append(f"실패 처리 {p['on_failure']!r}는 continue나 fail이어야 한다")
    if not 0.0 < cfg["sart_source_frac"] <= 1.0:
        problems.append(f"생성 요청 비율이 {cfg['sart_source_frac']}이다. "
                        f"0보다 크고 1 이하여야 한다")
    if problems:
        return "FAIL", "; ".join(problems)

    asked = max(1, round(cfg["chunk_size"] * cfg["sart_source_frac"]))
    return "PASS", (f"규칙 {p['converge_rule']} / 소스당 {cfg['sart_samples']}회 / "
                    f"반지름 {cfg['sart_radius_m']} m / 청크 {cfg['chunk_size']}편 중 "
                    f"생성에 {asked}편을 요청하고 나머지를 증강이 채운다 / "
                    f"프로세스 {cfg['sart_procs']}개")


def check_isaaclab(cfg):
    if not os.path.isfile(orch.ISAACLAB_SH):
        return "FAIL", f"{orch.ISAACLAB_SH} not found (not inside the isaac-lab image?)"
    if not os.path.isfile(orch.GEN_DATASET_SRC):
        return "FAIL", f"{orch.GEN_DATASET_SRC} not found"
    with open(orch.GEN_DATASET_SRC) as fh:
        text = fh.read()
    if "import isaaclab_mimic.envs" not in text:
        return "FAIL", "generate_dataset.py has no 'import isaaclab_mimic.envs' anchor to patch"
    return "PASS", "isaaclab.sh + generate_dataset.py ready"


def check_gpu(cfg):
    want = cfg["cuda_devices"]
    try:
        import torch
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            return "PASS", f"CUDA_VISIBLE_DEVICES={want} -> {', '.join(names)}"
        detail = "torch.cuda.is_available() is False"
    except Exception as exc:                      # noqa: BLE001 - torch may be absent here
        detail = f"torch unusable ({exc.__class__.__name__})"
    try:
        res = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return "FAIL", f"{detail}; nvidia-smi not runnable"
    if res.returncode != 0 or not res.stdout.strip():
        return "FAIL", f"{detail}; nvidia-smi reported no GPU"
    return "WARN", f"{detail}; nvidia-smi sees {res.stdout.strip().splitlines()[0]}"


def check_disk(cfg):
    per_ep = (_rgb_mb_per_episode(cfg["image_width"], cfg["image_height"])
              + EST_GEN_MB + EST_CONTRACT_MB + EST_LEROBOT_MB)
    episodes = cfg["chunk_size"]
    if cfg.get("sart_enable") and cfg["sart_source_frac"] >= 1.0:
        # 생성에 할당량을 다 요청한 채로 증강을 얹으면 청크가 그만큼 커진다. 소스 한
        # 편이 자기 자신과 증강 시도 수만큼을 낳으므로 최대 (1 + 시도 수)배가 된다.
        # 비율이 1보다 작으면 증강이 할당량에서 멈추므로 청크 크기가 그대로다.
        episodes = cfg["chunk_size"] * (1 + cfg["sart_samples"])
    shard_eps = math.ceil(episodes / cfg["render_procs"])
    need_gb = cfg["render_procs"] * shard_eps * per_ep * DISK_SAFETY / 1000.0
    try:
        free_gb = shutil.disk_usage(cfg["work_dir"]).free / 1e9
    except OSError as exc:
        return "FAIL", f"cannot stat {cfg['work_dir']}: {exc}"
    detail = (f"{free_gb:.0f} GB free, one chunk needs ~{need_gb:.0f} GB "
              f"({cfg['render_procs']} shards x {shard_eps} ep x {per_ep:.0f} MB)")
    if free_gb < need_gb:
        return "FAIL", detail
    if free_gb < need_gb * 3:
        return "WARN", detail + " — under 3 chunks of headroom"
    return "PASS", detail


def check_memory(cfg):
    avail = orch.mem_available_mb()
    if avail is None:
        return "WARN", "/proc/meminfo unreadable, the launch memory guard is disabled"
    procs = max(cfg["gen_procs"], cfg["render_procs"])
    need = orch.MEM_HEADROOM_MB * procs
    detail = f"{avail} MiB available, {procs} concurrent Isaac processes want ~{need} MiB"
    if avail < orch.MEM_HEADROOM_MB:
        return "FAIL", detail
    if avail < need:
        return "WARN", detail + " — the guard will serialize them"
    return "PASS", detail


def _add_lerobot_site():
    """격리 설치한 lerobot 경로를 import 경로에 붙인다.

    Dockerfile이 lerobot과 huggingface_hub을 Isaac 파이썬의 site-packages가 아니라
    별도 디렉터리에 설치한다(LEROBOT_SITE). 여기서 그 경로를 붙이지 않으면 두 검사가
    모두 실패하고 컨테이너가 첫 청크도 시작하지 못한다.
    """
    site = os.environ.get("LEROBOT_SITE") or os.environ.get("UWLAB_LEROBOT_SITE", "")
    if site and os.path.isdir(site) and site not in sys.path:
        sys.path.insert(0, site)
    return site


def check_lerobot(cfg):
    _add_lerobot_site()
    version = None
    try:
        import lerobot
        version = getattr(lerobot, "__version__", None)
        if version is None:
            from importlib import metadata
            version = metadata.version("lerobot")
    except Exception:                             # noqa: BLE001 - fall back to isaac python
        version = _probe("import lerobot,importlib.metadata as m;"
                         "print(getattr(lerobot,'__version__',None) or m.version('lerobot'))")
    if not version:
        return "FAIL", "lerobot not importable"
    if _version_tuple(version)[:2] < MIN_LEROBOT:
        return "FAIL", f"lerobot {version} < {'.'.join(str(v) for v in MIN_LEROBOT)}"
    # 여기까지는 껍데기만 확인한 것이다. 실제로 쓰는 클래스를 불러 봐야 추가 의존성이
    # 빠진 경우를 잡는다. lerobot 0.6은 datasets 패키지가 없으면 이 지점에서만 죽는데,
    # 그걸 놓치면 생성에 20분을 쓰고 마지막 단계에서 실패한다.
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: F401
    except Exception as exc:                      # noqa: BLE001
        detail = _probe("from lerobot.datasets.lerobot_dataset import LeRobotDataset;"
                        "print('ok')")
        if detail != "ok":
            return "FAIL", f"lerobot {version}이지만 LeRobotDataset을 불러올 수 없다: {exc}"
    return "PASS", f"lerobot {version} (LeRobotDataset 확인)"


def check_h5py(cfg):
    try:
        import h5py
        return "PASS", f"h5py {h5py.__version__}"
    except Exception:                             # noqa: BLE001
        version = _probe("import h5py;print(h5py.__version__)")
        if version:
            return "WARN", f"h5py {version} only in the isaac python, not in {sys.executable}"
        return "FAIL", "h5py not importable (the orchestrator merges shards with it)"


def check_hf_token(cfg):
    if not cfg.get("hf_upload", True):
        return "PASS", "HF_UPLOAD=0, 토큰을 확인하지 않는다"
    _add_lerobot_site()          # huggingface_hub도 같은 격리 경로에 설치돼 있다
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        return "FAIL", "HF_TOKEN not set (올리지 않을 것이면 HF_UPLOAD=0)"
    try:
        from huggingface_hub import HfApi
    except Exception as exc:                      # noqa: BLE001
        return "FAIL", f"huggingface_hub not importable ({exc.__class__.__name__})"
    try:
        info = HfApi().whoami(token=token)
    except Exception as exc:                      # noqa: BLE001 - network/auth both land here
        return "FAIL", f"whoami failed: {exc.__class__.__name__}: {exc}"
    name = info.get("name", "?")
    role = (info.get("auth", {}) or {}).get("accessToken", {}).get("role", "?")
    if role in ("write", "admin", "fineGrained"):
        return "PASS", f"{name} (token role {role})"
    return "WARN", f"{name} (token role {role} — uploads may be refused)"


CHECKS = [
    ("task profile", check_task_profile),
    ("required env vars", check_required_env),
    ("config sanity", check_config_sanity),
    ("work dir writable", check_work_dir),
    ("assets", check_assets),
    ("source demo filter", check_source_filter),
    ("visual randomization pkg", check_vrand_package),
    ("stage entry points", check_stage_scripts),
    ("sart augmentation", check_sart),
    ("isaac lab install", check_isaaclab),
    ("gpu", check_gpu),
    ("disk space", check_disk),
    ("memory", check_memory),
    ("lerobot >= 0.4", check_lerobot),
    ("h5py", check_h5py),
    ("hugging face token", check_hf_token),
]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="hf80k preflight (env vars only)")
    ap.add_argument("--json", action="store_true", help="also print the rows as JSON")
    args = ap.parse_args(argv)

    try:
        cfg = orch.load_config()
    except SystemExit as exc:
        print(f"[preflight] configuration is unusable: {exc}")
        return 2

    rows = []
    for name, fn in CHECKS:
        try:
            status, detail = fn(cfg)
        except Exception as exc:                  # noqa: BLE001 - a broken check is a failure
            status, detail = "FAIL", f"check raised {exc!r}"
        rows.append((name, status, detail))

    width = max(len(n) for n, _, _ in rows)
    print("=" * (width + 60))
    print(f"{'CHECK'.ljust(width)}  STATUS  DETAIL")
    print("-" * (width + 60))
    for name, status, detail in rows:
        print(f"{name.ljust(width)}  {status:<6}  {detail}")
    print("=" * (width + 60))

    failed = [n for n, s, _ in rows if s == "FAIL"]
    warned = [n for n, s, _ in rows if s == "WARN"]
    print(f"[preflight] {len(rows) - len(failed) - len(warned)} pass, "
          f"{len(warned)} warn, {len(failed)} fail")
    if args.json:
        print(json.dumps([{"check": n, "status": s, "detail": d} for n, s, d in rows], indent=2))
    if failed:
        print("[preflight] blocking: " + ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
