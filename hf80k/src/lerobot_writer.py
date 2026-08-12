#!/usr/bin/env python3
"""Write one chunk's contract HDF5 + rendered RGB out as a LeRobot dataset v3.

WHY this file exists: the RL team trains from LeRobot v3 (they collect with
render/fr3_visual_randomization_v1/source/collect_demos_lerobot.py), while this
pipeline produces two separate HDF5 files -- contract actions/states at 10 Hz
(src/convert/schema_io.py) and camera frames also at 10 Hz (src/render/
render_viewpoints.py run with --every 2). This script is the join, and it is
the only place that decides which image belongs to which action.

The pairing is done BY TIME, not by frame number. Contract step k happens at
`timestamps[k]` seconds, rendered frame j at `j / fps`. Every step takes the
nearest frame, and a step with no frame within half a period is dropped, so a
one-frame length difference costs one frame instead of the whole episode. The
integral-ratio test of contract/join_rgb_contract.py is deliberately NOT used:
it throws a 176-step demo away when the length ratio comes out 2.006 instead
of 2.

A broken demo is skipped and reported, never fatal -- one bad episode out of
500 must not waste a multi-GB render. The exit code only turns nonzero when
fewer than --min-write-rate of the contract episodes reached the dataset.

`LeRobotDataset.finalize()` is what writes the parquet footers; without it the
dataset cannot be read back at all, so it runs from a `finally` block.

Memory: one episode of one camera is ~10 MB, so frames are read one at a time
straight out of h5py (the render step chunks the image datasets per frame) and
handed to the writer -- no episode-sized, let alone file-sized, buffers here.

Progress goes to stderr; the LAST line of stdout is the JSON summary.

    python3 lerobot_writer.py --contract contract.hdf5 --rgb rgb.hdf5 \
        --vrand-log vrand_log.json --out lerobot --repo-id myorg/fr3-cube-80k
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np

# INTERFACE.md section 4
PROFILE_IDS = {"nominal_lab": 0, "lab_variation": 1, "stress_tail": 2}
CONTRACT_CAMERAS = ("third_person_0", "third_person_1", "wrist")
# 아래 네 값은 재익님 수집기 collect_demos_lerobot.py와 글자까지 같아야 한다. 다르면
# 두 데이터셋을 이어 붙이거나 같은 학습 코드로 읽을 때 이름이 어긋난다.
#   task 문자열 :47, robot_type :272, 상태 이름 :100의 STATE_KEYS, 이미지 names :193
DEFAULT_TASK_STRING = "Stack three cubes into a three-level tower"
ROBOT_TYPE = "franka_fr3_osc"
DEFAULT_FPS = 10
ACTION_DIM = 7
JOINT_DIM = 9
POSE_DIM = 7
STATE_NAMES = ([f"joint_pos.{i}" for i in range(JOINT_DIM)]
               + [f"end_effector_pose.{i}" for i in range(POSE_DIM)]
               + [f"prev_actions.{i}" for i in range(ACTION_DIM)])
SUMMARY_SCHEMA = "fr3_cube.hf80k.lerobot_writer.v1"


def log(message: str) -> None:
    """Progress on stderr so stdout stays a parseable JSON document."""
    print(message, file=sys.stderr, flush=True)


def natural_key(name: str):
    """demo_2 before demo_10 (same ordering as render_viewpoints.py)."""
    match = re.search(r"(\d+)$", name)
    return (int(match.group(1)) if match else 1 << 30, name)


def import_lerobot():
    """Import lerobot >= 0.4 (dataset v3), honouring an out-of-tree site dir.

    The Isaac Lab image ships its own interpreter and the RL team installs
    lerobot beside it rather than into it; their collector reads the path from
    UWLAB_LEROBOT_SITE, so we accept that name as well as LEROBOT_SITE.
    """
    import importlib.metadata

    site = os.environ.get("LEROBOT_SITE") or os.environ.get("UWLAB_LEROBOT_SITE", "")
    if site and os.path.isdir(site) and site not in sys.path:
        sys.path.insert(0, site)
    try:
        version = importlib.metadata.version("lerobot")
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except (ImportError, importlib.metadata.PackageNotFoundError) as exc:
        raise SystemExit(
            "lerobot (>=0.4, dataset v3) is not importable. Install it or point "
            "LEROBOT_SITE at the directory it lives in.") from exc
    numbers = re.findall(r"\d+", version)
    if len(numbers) >= 2 and (int(numbers[0]), int(numbers[1])) < (0, 4):
        raise SystemExit(f"LeRobot dataset v3 needs lerobot>=0.4, found {version}")
    return version, LeRobotDataset


def build_features(roles, height: int, width: int) -> dict:
    """Feature spec of INTERFACE.md section 4, cameras taken from --cameras."""
    features = {
        f"observation.images.{role}": {
            "dtype": "video",
            "shape": (height, width, 3),
            "names": ["height", "width", "channels"],
        }
        for role in roles
    }
    features["observation.state"] = {
        "dtype": "float32", "shape": (len(STATE_NAMES),), "names": list(STATE_NAMES)}
    features["action"] = {
        "dtype": "float32", "shape": (ACTION_DIM,),
        "names": [f"osc_action.{i}" for i in range(ACTION_DIM)]}
    features["visual.profile_id"] = {
        "dtype": "int64", "shape": (1,), "names": ["profile_id"]}
    return features


def load_vrand_profiles(path: str):
    """demo name -> visual profile name, from the render stage's vrand_log.json.

    Two shapes are accepted, because the log either holds the per-episode
    records at the top level or under "episodes":

        {"demo_0": {"profile": "lab_variation", ...}, ...}
        {"seed": 0, "process_profile": "...", "episodes": {"demo_0": {...}}}

    A bare string value ({"demo_0": "stress_tail"}) works too, and a top-level
    "episode_profiles" mapping (what render_viewpoints.py stores in its HDF5
    attrs) is used as a second source. Entries that are neither are ignored,
    which is how the log's own metadata keys (seed, mixture, process, ...) get
    filtered out. A missing file means the render ran with a single profile for
    the whole chunk, so everything falls back to --profile.

    Returns (per-episode mapping, process-wide fallback or None).
    """
    if not path or not os.path.isfile(path):
        return {}, None
    with open(path) as handle:
        doc = json.load(handle)
    if not isinstance(doc, dict):
        return {}, None
    records = doc.get("episodes")
    if not isinstance(records, dict):
        records = doc
    profiles: dict[str, str] = {}
    for name, record in records.items():
        if isinstance(record, str):
            profiles[name] = record
        elif isinstance(record, dict) and isinstance(record.get("profile"), str):
            profiles[name] = record["profile"]
    plan = doc.get("episode_profiles")
    if isinstance(plan, dict):
        for name, profile in plan.items():
            if isinstance(profile, str):
                profiles.setdefault(name, profile)
    fallback = doc.get("process_profile")
    return profiles, fallback if isinstance(fallback, str) else None


def align_frames(timestamps, n_rgb: int, fps: float) -> np.ndarray:
    """Nearest rendered frame for every contract step, matched on TIME.

    Frame j was shot at j / fps seconds and contract step k belongs to
    timestamps[k]. A step whose nearest frame sits more than half a period away
    has simply run out of video, so the pairing stops there -- that truncates
    the pair to the shorter of the two streams, one frame at a time, without
    the all-or-nothing integer ratio test.
    """
    if n_rgb <= 0:
        return np.zeros(0, dtype=np.int64)
    times = np.asarray(timestamps, dtype=np.float64)
    index = np.clip(np.rint(times * float(fps)).astype(np.int64), 0, n_rgb - 1)
    error = np.abs(index.astype(np.float64) / float(fps) - times)
    close = error <= (0.5 / float(fps)) + 1e-9
    if not bool(np.all(close)):
        return index[:int(np.argmin(close))]
    return index


def build_state(joints, poses, actions) -> np.ndarray:
    """observation.state = joint_position(9) + actual_ee_pose(7) + prev action(7).

    The first step has no previous action, so it is zero-filled (section 4).
    """
    previous = np.zeros_like(actions)
    previous[1:] = actions[:-1]
    return np.concatenate([joints, poses, previous], axis=1).astype(np.float32)


def probe_image_shape(rgb_data, names, roles):
    """(height, width) from the first demo that carries every requested camera."""
    for name in names:
        group = rgb_data.get(name)
        obs = group.get("obs") if isinstance(group, h5py.Group) else None
        if obs is None or any(f"{role}_image" not in obs for role in roles):
            continue
        shape = obs[f"{roles[0]}_image"].shape
        if len(shape) == 4 and shape[3] == 3:
            return int(shape[1]), int(shape[2])
    return None


def prepare_episode(contract_group, rgb_group, roles, fps, geometry, min_steps,
                    max_frame_gap):
    """Validate one demo. Returns (payload, None) or (None, skip reason)."""
    required = {"actions": ACTION_DIM, "joint_position": JOINT_DIM,
                "actual_ee_pose": POSE_DIM}
    tracks = {}
    for key, width in required.items():
        if key not in contract_group:
            return None, f"contract dataset {key} missing"
        array = np.asarray(contract_group[key][()], dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != width:
            return None, f"contract {key} shape {tuple(array.shape)}"
        tracks[key] = array
    if "timestamps" not in contract_group:
        return None, "contract dataset timestamps missing"
    times = np.asarray(contract_group["timestamps"][()], dtype=np.float64)
    if times.ndim != 1:
        return None, f"contract timestamps shape {tuple(times.shape)}"
    lengths = {len(times)} | {len(array) for array in tracks.values()}
    if len(lengths) != 1:
        return None, f"contract tracks disagree on T {sorted(lengths)}"
    if any(not np.all(np.isfinite(array)) for array in tracks.values()):
        return None, "non-finite value in contract tracks"
    if len(times) > 1 and not np.all(np.diff(times) > 0):
        return None, "contract timestamps not strictly increasing"

    obs = rgb_group.get("obs")
    if obs is None:
        return None, "rendered demo has no obs group"
    height, width = geometry
    n_rgb = None
    for role in roles:
        key = f"{role}_image"
        dataset = obs.get(key)
        if dataset is None:
            return None, f"missing rendered camera {key}"
        if dataset.dtype != np.uint8:
            return None, f"{key} dtype {dataset.dtype}, expected uint8"
        if dataset.ndim != 4 or tuple(dataset.shape[1:]) != (height, width, 3):
            return None, (f"{key} shape {tuple(dataset.shape)}, expected "
                          f"(T,{height},{width},3)")
        if n_rgb is None:
            n_rgb = int(dataset.shape[0])
        elif int(dataset.shape[0]) != n_rgb:
            return None, f"{key} has {dataset.shape[0]} frames, expected {n_rgb}"

    # section 6 allows the two streams to differ by about one frame. A gap far
    # bigger than that is a rate bug (a render without --every 2 gives twice the
    # frames), and the nearest-frame rule would happily pair the actions with
    # the first half of the video instead of failing, so it is caught here.
    if abs(n_rgb - len(times)) > max_frame_gap:
        return None, (f"render T={n_rgb} vs contract T={len(times)} differ by more "
                      f"than {max_frame_gap} frames (rate mismatch, not rounding)")

    index = align_frames(times, n_rgb, fps)
    if len(index) < min_steps:
        return None, (f"only {len(index)} time-aligned steps "
                      f"(contract T={len(times)}, render T={n_rgb})")
    kept = len(index)
    state = build_state(tracks["joint_position"], tracks["actual_ee_pose"],
                        tracks["actions"])[:kept]
    payload = {"index": index, "state": state,
               "actions": tracks["actions"][:kept], "obs": obs,
               "contract_steps": int(len(times)), "render_frames": int(n_rgb)}
    return payload, None


def write_episode(dataset, payload, roles, profile_id: int, task: str) -> int:
    """Stream one episode into the writer, one frame at a time (never one file)."""
    images = {role: payload["obs"][f"{role}_image"] for role in roles}
    profile = np.asarray([profile_id], dtype=np.int64)
    state, actions, index = payload["state"], payload["actions"], payload["index"]
    for step, frame_index in enumerate(index):
        frame = {f"observation.images.{role}":
                 np.ascontiguousarray(handle[int(frame_index)])
                 for role, handle in images.items()}
        frame["observation.state"] = np.ascontiguousarray(state[step])
        frame["action"] = np.ascontiguousarray(actions[step])
        frame["visual.profile_id"] = profile.copy()
        frame["task"] = task
        dataset.add_frame(frame)
    dataset.save_episode()
    return int(len(index))


def drop_pending_episode(dataset) -> None:
    """Throw away a half-written episode so the next one starts clean.

    lerobot 0.4 only deletes the temporary PNGs of `image` features in
    clear_episode_buffer, so the directories backing `video` features are
    removed by hand -- otherwise a failure mid-chunk leaks a few hundred MB.
    """
    buffer = getattr(dataset, "episode_buffer", None)
    if buffer is None or int(buffer.get("size", 0)) == 0:
        return
    episode_index = buffer.get("episode_index", 0)
    if isinstance(episode_index, np.ndarray):
        episode_index = int(episode_index.reshape(-1)[0])
    if getattr(dataset, "image_writer", None) is not None:
        try:
            dataset._wait_image_writer()
        except Exception:  # best effort cleanup, the skip reason is what matters
            pass
    directories = []
    if hasattr(dataset, "_get_image_file_dir"):
        for key in getattr(dataset.meta, "video_keys", ()):
            try:
                directories.append(dataset._get_image_file_dir(int(episode_index), key))
            except Exception:
                pass
    try:
        dataset.clear_episode_buffer(delete_images=True)
    except TypeError:
        dataset.clear_episode_buffer()
    for directory in directories:
        if Path(directory).is_dir():
            shutil.rmtree(directory, ignore_errors=True)


def directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in Path(path).rglob("*") if item.is_file())


def prepare_out_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise SystemExit(f"{path} exists and is not empty; pass --overwrite "
                             f"to replace it")
        shutil.rmtree(path)


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # defaults are the file names of INTERFACE.md section 2, so running the
    # script inside a chunk directory needs no arguments at all. The underscore
    # spellings are accepted because src/orchestrate.py calls them that way.
    parser.add_argument("--contract", default="contract.hdf5")
    parser.add_argument("--rgb", default="rgb.hdf5")
    parser.add_argument("--vrand-log", "--vrand_log", dest="vrand_log",
                        default="vrand_log.json",
                        help="per-episode visual randomization log; missing is fine")
    parser.add_argument("--out", "--output", dest="out", default="lerobot",
                        help="LeRobot dataset root")
    parser.add_argument("--repo-id", "--repo_id", dest="repo_id",
                        default=os.environ.get("HF_REPO_ID", "")
                        or "local/fr3-cube-stack")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--profile", choices=sorted(PROFILE_IDS),
                        default="nominal_lab",
                        help="fallback profile for episodes the vrand log does not name")
    parser.add_argument("--cameras", default=",".join(CONTRACT_CAMERAS))
    parser.add_argument("--task-string", "--task_string", dest="task_string",
                        default=DEFAULT_TASK_STRING)
    parser.add_argument("--overwrite", action="store_true",
                        help="delete a non-empty --out before writing")
    parser.add_argument("--batch-encoding-size", type=int, default=10,
                        help="episodes buffered before one ffmpeg pass; chosen "
                             "here (the spec is silent), finalize() flushes the rest")
    parser.add_argument("--min-steps", type=int, default=2,
                        help="drop an episode with fewer aligned steps than this")
    parser.add_argument("--max-frame-gap", type=int, default=4,
                        help="max |render T - contract T| before the episode is "
                             "treated as a rate mismatch instead of rounding")
    parser.add_argument("--min-write-rate", type=float, default=0.9,
                        help="exit nonzero below this written/total ratio")
    # 기본이 켜짐이다. 재익님 규격이 성공 에피소드만 담으라고 정하고 있고
    # (uwlab_collection_manifest의 success_only), 생성 단계 필터가 놓친 것을 여기서
    # 한 번 더 거른다. 끄려면 --no-require-success를 준다.
    parser.add_argument("--require-success", dest="require_success",
                        action="store_true", default=True,
                        help="성공 표시가 없는 에피소드를 건너뛴다 (기본값)")
    parser.add_argument("--no-require-success", dest="require_success",
                        action="store_false",
                        help="성공 표시와 무관하게 전부 기록한다. 디버그용")
    return parser.parse_args()


def main():
    args = parse_args()
    roles = [role.strip() for role in args.cameras.split(",") if role.strip()]
    if not roles:
        raise SystemExit("--cameras is empty")
    out_dir = Path(args.out).resolve()
    prepare_out_dir(out_dir, args.overwrite)

    profiles, profile_fallback = load_vrand_profiles(args.vrand_log)
    contract = h5py.File(args.contract, "r", locking=False)
    rgb = h5py.File(args.rgb, "r", locking=False)
    if "data" not in contract or "data" not in rgb:
        raise SystemExit("both --contract and --rgb need a root 'data' group")
    contract_data, rgb_data = contract["data"], rgb["data"]
    names = sorted(contract_data.keys(), key=natural_key)

    skipped: list[dict] = []
    profile_counts: dict[str, int] = {}
    written, frames, finalize_error = 0, 0, None
    geometry = probe_image_shape(rgb_data, names, roles)
    version, LeRobotDataset = (None, None)
    dataset = None

    if geometry is None:
        for name in names:
            skipped.append({"episode": name,
                            "reason": f"no rendered frames for cameras {roles}"})
    else:
        height, width = geometry
        version, LeRobotDataset = import_lerobot()
        log(f"[lerobot] lerobot {version}, {len(names)} contract episodes, "
            f"{width}x{height}, cameras={roles}")
        dataset = LeRobotDataset.create(
            repo_id=args.repo_id,
            fps=args.fps,
            features=build_features(roles, height, width),
            root=out_dir,
            robot_type=ROBOT_TYPE,
            use_videos=True,
            batch_encoding_size=max(1, args.batch_encoding_size),
            video_backend=None,
        )
        try:
            for name in names:
                group = contract_data[name]
                # 표시가 없으면 실패로 본다. 기본값을 True로 두면 표시를 쓰지 않는 경로로
                # 만들어진 파일이 통째로 성공 취급된다.
                if args.require_success and not bool(group.attrs.get("success", False)):
                    skipped.append({"episode": name, "reason": "success attr is false"})
                    continue
                if name not in rgb_data:
                    skipped.append({"episode": name, "reason": "not in the render file"})
                    continue
                # 두 번째 관문. 렌더 단계가 기록된 마지막 상태를 다시 보고 매긴 판정이다.
                # 생성 단계 판정과 기준이 조금 다르다. 이쪽은 색 순서를 따지지 않는 대신
                # 그리퍼를 놓았는지를 본다. 둘 다 통과한 것만 남긴다.
                rgb_group = rgb_data[name]
                # 판정 표시가 아예 없으면 통과가 아니라 거부다. 씬 구성이나 Isaac 판본이
                # 바뀌어 판정이 돌지 않으면 표시가 없는데, 예전처럼 "있을 때만 본다"로
                # 두면 그 순간 이 관문이 통째로 사라진다.
                if args.require_success:
                    if not bool(rgb_group.attrs.get("replay_success_any_order", False)):
                        reason = ("render replay judged the tower not stacked"
                                  if "replay_success_any_order" in rgb_group.attrs
                                  else "render replay wrote no success verdict")
                        skipped.append({"episode": name, "reason": reason})
                        continue
                profile = profiles.get(name) or profile_fallback or args.profile
                if profile not in PROFILE_IDS:
                    skipped.append({"episode": name,
                                    "reason": f"unknown visual profile {profile!r}"})
                    continue
                payload, reason = prepare_episode(
                    group, rgb_data[name], roles, args.fps, geometry, args.min_steps,
                    args.max_frame_gap)
                if reason is not None:
                    skipped.append({"episode": name, "reason": reason})
                    continue
                try:
                    count = write_episode(
                        dataset, payload, roles, PROFILE_IDS[profile], args.task_string)
                except Exception as exc:  # one bad demo must not kill the chunk
                    drop_pending_episode(dataset)
                    skipped.append({"episode": name,
                                    "reason": f"{type(exc).__name__}: {exc}"})
                    log(f"[lerobot] {name}: FAILED {type(exc).__name__}: {exc}")
                    continue
                written += 1
                frames += count
                profile_counts[profile] = profile_counts.get(profile, 0) + 1
                if written <= 5 or written % 50 == 0:
                    log(f"[lerobot] {name}: {count} frames "
                        f"(profile={profile}) -> {written}/{len(names)}")
        finally:
            # finalize() writes the parquet footers; skipping it leaves the
            # dataset unreadable, so it must run even on the way out of a crash
            drop_pending_episode(dataset)
            try:
                dataset.finalize()
            except Exception as exc:  # still report, the chunk is unusable
                finalize_error = f"{type(exc).__name__}: {exc}"
                log(f"[lerobot] finalize failed: {finalize_error}")
            contract.close()
            rgb.close()

    if dataset is None:
        contract.close()
        rgb.close()

    total = len(names)
    rate = (written / total) if total else 0.0
    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "output_path": str(out_dir),
        "repo_id": args.repo_id,
        "lerobot_version": version,
        "fps": args.fps,
        "robot_type": ROBOT_TYPE,
        "task": args.task_string,
        "cameras": roles,
        "image_size": [geometry[1], geometry[0]] if geometry else None,
        "episodes_total": total,
        "episodes_written": written,
        "episodes_skipped_count": len(skipped),
        "episodes_skipped": skipped,
        "frames_written": frames,
        "write_rate": round(rate, 6),
        "profile_counts": profile_counts,
        "bytes_on_disk": directory_bytes(out_dir) if out_dir.exists() else 0,
        "finalize_error": finalize_error,
        # written이 0이면 실패다. min_write_rate를 0으로 낮춰도 이 조건은 남는다.
        "ok": bool(total) and written > 0 and rate >= args.min_write_rate
              and finalize_error is None,
    }
    print(json.dumps(summary))
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
