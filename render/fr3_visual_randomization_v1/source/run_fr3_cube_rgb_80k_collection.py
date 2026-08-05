#!/usr/bin/env python3
"""Collect the exact 50/40/10 FR3 RGB distillation mixture on four GPUs.

Completed LeRobot roots are discovered and reused on restart. Collection may
use fixed-size recovery chunks or one remaining root per GPU/profile to avoid
restarting Isaac Sim between small chunks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
ISAAC_PYTHON = Path("/opt/IsaacSim/python.sh")
AUDIT_SCRIPT = REPO_ROOT / "scripts_v2/tools/audit_fr3_cube_rgb_isolation_runtime.py"
COLLECT_SCRIPT = REPO_ROOT / "scripts_v2/tools/collect_demos_lerobot.py"
PROFILES = {"nominal_lab": 0, "lab_variation": 1, "stress_tail": 2}
DEFAULT_RESET_POOL_DIR = Path(
    "/home/ubuntu/jake/UWLab/datasets_final/cube_stack_one_two_20260710/ResetPoolsFinal"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_chunk(root: Path, *, episodes: int, profile: str, expert_sha: str) -> bool:
    manifest_path = root / "uwlab_collection_manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(
        manifest.get("lerobot_dataset_format") == "v3"
        and manifest.get("success_only") is True
        and int(manifest.get("successful_episodes", -1)) == episodes
        and manifest.get("visual_profile") == profile
        and manifest.get("expert_policy_sha256") == expert_sha
    )


def _completed_roots(output_root: Path, *, expert_sha: str) -> dict[int, list[Path]]:
    """Discover valid completed roots and associate them with their physical GPU."""
    by_gpu = {gpu: [] for gpu in range(4)}
    datasets_root = output_root / "datasets"
    if not datasets_root.is_dir():
        return by_gpu
    for root in sorted(datasets_root.iterdir()):
        if not root.is_dir() or root.name.startswith("."):
            continue
        match = re.match(r"^gpu([0-3])_", root.name)
        manifest_path = root / "uwlab_collection_manifest.json"
        if match is None or not manifest_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not (
            manifest.get("lerobot_dataset_format") == "v3"
            and manifest.get("success_only") is True
            and int(manifest.get("successful_episodes", 0)) > 0
            and manifest.get("visual_profile") in PROFILES
            and manifest.get("expert_policy_sha256") == expert_sha
        ):
            continue
        by_gpu[int(match.group(1))].append(root)
    return by_gpu


def _valid_cached_audit(path: Path, *, profile: str, num_envs: int, audit_steps: int) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(
        payload.get("pass") is True
        and payload.get("visual_profile") == profile
        and int(payload.get("num_envs", -1)) == num_envs
        and int(payload.get("sampled_steps", -1)) >= audit_steps
    )


def _run_logged(command: list[str], *, env: dict[str, str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n$ " + " ".join(command) + "\n")
        log.flush()
        subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )


def _worker(
    gpu: int,
    schedule: list[tuple[str, int]],
    *,
    output_root: Path,
    expert: Path,
    expert_sha: str,
    num_envs: int,
    audit_steps: int,
    chunk_size: int,
    seed_base: int,
    reset_pool_dir: Path,
    image_width: int,
    image_height: int,
    one_chunk_per_profile: bool,
    reuse_valid_gates: bool,
    seed_ordinal_start: int,
    hydra_args: list[str],
) -> list[Path]:
    # Isaac Sim 5.1's usdrt scenegraph currently requires the process-local
    # device to be cuda:0. Map each worker's physical GPU into that local slot;
    # direct cuda:1..3 selection fails before camera annotators can start.
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["UWLAB_PHYSICAL_RENDER_GPU"] = str(gpu)
    env["UWLAB_FR3_CUBE_RESET_POOL_DIR"] = str(reset_pool_dir)
    env["UWLAB_FR3_RGB_WIDTH"] = str(image_width)
    env["UWLAB_FR3_RGB_HEIGHT"] = str(image_height)
    device = "cuda:0"
    completed: list[Path] = []

    # Audit every visual profile used by this worker at the production env count.
    for profile, _count in schedule:
        audit = output_root / "gates" / f"gpu{gpu}_{profile}_{num_envs}env.json"
        if reuse_valid_gates and _valid_cached_audit(
            audit, profile=profile, num_envs=num_envs, audit_steps=audit_steps
        ):
            continue
        command = [
            str(ISAAC_PYTHON), str(AUDIT_SCRIPT),
            "--num-envs", str(num_envs),
            "--steps", str(audit_steps),
            "--output", str(audit),
            "--visual-profile", profile,
            "--device", device,
            "--headless", "--enable_cameras",
            *hydra_args,
        ]
        _run_logged(command, env=env, log_path=output_root / "logs" / f"gpu{gpu}_audit.log")
        payload = json.loads(audit.read_text(encoding="utf-8"))
        if not payload.get("pass", False):
            raise RuntimeError(f"GPU {gpu} {profile} isolation audit failed: {audit}")

    chunk_ordinal = seed_ordinal_start
    for profile, profile_count in schedule:
        if one_chunk_per_profile:
            profile_chunk_sizes = [profile_count]
        else:
            if profile_count % chunk_size:
                raise ValueError(f"{profile_count=} must be divisible by {chunk_size=}")
            profile_chunk_sizes = [chunk_size] * (profile_count // chunk_size)
        for profile_chunk, current_chunk_size in enumerate(profile_chunk_sizes):
            chunk_ordinal += 1
            seed = seed_base + gpu * 1000 + chunk_ordinal
            name = f"gpu{gpu}_{profile}_chunk{profile_chunk:02d}_{current_chunk_size:05d}eps"
            root = output_root / "datasets" / name
            if _valid_chunk(
                root, episodes=current_chunk_size, profile=profile, expert_sha=expert_sha
            ):
                completed.append(root)
                continue
            if root.exists() or (root.parent / f".{root.name}.shards").exists():
                raise RuntimeError(
                    f"Incomplete existing chunk requires manual inspection before retry: {root}"
                )
            command = [
                str(ISAAC_PYTHON), str(COLLECT_SCRIPT),
                "--expert-policy", str(expert),
                "--dataset-root", str(root),
                "--repo-id", f"local/fr3_cube_9k_80k_{name}",
                "--num-envs", str(num_envs),
                "--num-demos", str(current_chunk_size),
                "--visual-profile", profile,
                "--collection-seed", str(seed),
                "--device", device,
                "--headless", "--enable_cameras",
                *hydra_args,
            ]
            env["UWLAB_FR3_VISUAL_PROFILE"] = profile
            _run_logged(command, env=env, log_path=output_root / "logs" / f"gpu{gpu}_collection.log")
            if not _valid_chunk(
                root, episodes=current_chunk_size, profile=profile, expert_sha=expert_sha
            ):
                raise RuntimeError(f"Completed chunk failed validation: {root}")
            completed.append(root)
    return completed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expert-policy", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--audit-steps", type=int, default=20)
    parser.add_argument("--chunk-size", type=int, default=2000)
    parser.add_argument("--seed-base", type=int, default=42000)
    parser.add_argument("--reset-pool-dir", type=Path, default=DEFAULT_RESET_POOL_DIR)
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--image-height", type=int, default=360)
    parser.add_argument(
        "--one-chunk-per-profile",
        action="store_true",
        help="Collect each GPU/profile's remaining quota without intermediate Isaac Sim restarts.",
    )
    parser.add_argument(
        "--reuse-valid-gates",
        action="store_true",
        help="Reuse passing runtime isolation gates with the same profile/env-count/step contract.",
    )
    parser.add_argument("hydra_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.image_width <= 0 or args.image_height <= 0:
        raise ValueError("Image dimensions must be positive")
    if args.image_width * 9 != args.image_height * 16:
        raise ValueError(
            f"FR3 RGB capture must remain 16:9, got {args.image_width}x{args.image_height}"
        )

    expert = args.expert_policy.resolve()
    reset_pool_dir = args.reset_pool_dir.resolve()
    output_root = args.output_root.resolve()
    if not expert.is_file():
        raise FileNotFoundError(expert)
    required_pools = [
        reset_pool_dir / f"resets_{stage}_{reset_type}.pt"
        for stage in ("one_stacked", "two_stacked")
        for reset_type in ("reaching", "nearobject", "grasped", "neargoal")
    ]
    missing_pools = [str(path) for path in required_pools if not path.is_file()]
    if missing_pools:
        raise FileNotFoundError(f"Missing FR3 cube reset pools: {missing_pools}")
    output_root.mkdir(parents=True, exist_ok=True)
    expert_sha = _sha256(expert)

    # Per-GPU targets aggregate to the exact 40K/32K/8K mixture.
    target_schedules = {
        0: [("nominal_lab", 20000)],
        1: [("nominal_lab", 20000)],
        2: [("lab_variation", 16000), ("stress_tail", 4000)],
        3: [("lab_variation", 16000), ("stress_tail", 4000)],
    }
    completed_by_gpu = _completed_roots(output_root, expert_sha=expert_sha)
    results: list[Path] = [root for roots in completed_by_gpu.values() for root in roots]
    schedules: dict[int, list[tuple[str, int]]] = {}
    for gpu, targets in target_schedules.items():
        observed = {profile: 0 for profile in PROFILES}
        for root in completed_by_gpu[gpu]:
            manifest = json.loads((root / "uwlab_collection_manifest.json").read_text())
            observed[manifest["visual_profile"]] += int(manifest["successful_episodes"])
        remaining = []
        for profile, target_count in targets:
            if observed[profile] > target_count:
                raise RuntimeError(
                    f"GPU {gpu} {profile} already exceeds target: {observed[profile]} > {target_count}"
                )
            if observed[profile] < target_count:
                remaining.append((profile, target_count - observed[profile]))
        schedules[gpu] = remaining

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(
                _worker,
                gpu,
                schedule,
                output_root=output_root,
                expert=expert,
                expert_sha=expert_sha,
                num_envs=args.num_envs,
                audit_steps=args.audit_steps,
                chunk_size=args.chunk_size,
                seed_base=args.seed_base,
                reset_pool_dir=reset_pool_dir,
                image_width=args.image_width,
                image_height=args.image_height,
                one_chunk_per_profile=args.one_chunk_per_profile,
                reuse_valid_gates=args.reuse_valid_gates,
                seed_ordinal_start=len(completed_by_gpu[gpu]),
                hydra_args=args.hydra_args,
            ): gpu
            for gpu, schedule in schedules.items()
            if schedule
        }
        for future in as_completed(futures):
            gpu = futures[future]
            try:
                results.extend(future.result())
            except Exception as exc:
                print(f"[fatal] GPU worker {gpu}: {exc}", file=sys.stderr, flush=True)
                raise

    profile_counts = {profile: 0 for profile in PROFILES}
    parallelism_episode_counts: dict[str, int] = {}
    manifests = []
    for root in sorted(results):
        manifest_path = root / "uwlab_collection_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        episode_count = int(manifest["successful_episodes"])
        profile_counts[manifest["visual_profile"]] += episode_count
        collection_num_envs = int(
            manifest.get("num_envs", len(manifest.get("per_env_successful_episodes", [])))
        )
        parallelism_episode_counts[str(collection_num_envs)] = (
            parallelism_episode_counts.get(str(collection_num_envs), 0) + episode_count
        )
        manifests.append(str(manifest_path))
    expected = {"nominal_lab": 40000, "lab_variation": 32000, "stress_tail": 8000}
    if profile_counts != expected:
        raise RuntimeError(f"Mixture mismatch: observed={profile_counts}, expected={expected}")

    master = {
        "schema_version": "fr3_cube.rgb_distillation_80k.v1",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "expert_policy": str(expert),
        "expert_policy_sha256": expert_sha,
        "success_only": True,
        "lerobot_dataset_format": "v3",
        "capture_resolution": [args.image_width, args.image_height],
        "cameras": ["third_person_0", "third_person_1", "wrist"],
        "total_successful_episodes": sum(profile_counts.values()),
        "profile_counts": profile_counts,
        "profile_ratios": {key: value / 80000 for key, value in profile_counts.items()},
        "collection_num_envs_requested_per_gpu": args.num_envs,
        "collection_parallelism_episode_counts": parallelism_episode_counts,
        "reset_pool_dir": str(reset_pool_dir),
        "manifests": manifests,
        "dataset_roots": [str(Path(path).parent) for path in manifests],
    }
    master_path = output_root / "fr3_cube_rgb_80k_master_manifest.json"
    master_path.write_text(json.dumps(master, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"master_manifest": str(master_path), **profile_counts}, indent=2))


if __name__ == "__main__":
    main()
