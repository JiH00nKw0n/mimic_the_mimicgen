#!/usr/bin/env python3
"""Collect successful FR3 RGB expert rollouts directly as LeRobot Dataset v3."""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import hashlib
import importlib.metadata
import json
import os
import shutil
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
from isaaclab.app import AppLauncher


class _SplitRenderGpuAppLauncher(AppLauncher):
    """Keep process-local CUDA at zero while selecting the physical RTX GPU."""

    def _resolve_device_settings(self, launcher_args: dict):
        super()._resolve_device_settings(launcher_args)
        physical_gpu = os.environ.get("UWLAB_PHYSICAL_RENDER_GPU")
        if physical_gpu is not None:
            launcher_args["active_gpu"] = int(physical_gpu)
            print(
                "[INFO][UWLab]: process-local physics GPU "
                f"{launcher_args['physics_gpu']}, physical render GPU {physical_gpu}"
            )


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--task",
    default="OmniReset-Fr3PandaCube-FullStack-RelCartesianOSC-RGB-DataCollection-v0",
)
parser.add_argument("--expert-policy", type=Path, required=True, help="Exported JIT policy.pt")
parser.add_argument("--dataset-root", type=Path, required=True)
parser.add_argument("--repo-id", required=True, help="LeRobot dataset id, e.g. local/fr3_cube_omnireset")
parser.add_argument("--num-envs", type=int, default=32)
parser.add_argument("--num-demos", type=int, required=True)
parser.add_argument("--task-description", default="Stack three cubes into a three-level tower")
parser.add_argument("--deterministic", action="store_true")
parser.add_argument("--overwrite", action="store_true")
parser.add_argument("--keep-shards", action="store_true")
parser.add_argument("--image-writer-threads", type=int, default=1)
parser.add_argument(
    "--parallel-video-encoding",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Encode the three camera streams concurrently at episode finalization.",
)
parser.add_argument(
    "--encoder-threads",
    type=int,
    default=1,
    help="FFmpeg threads per camera. Keep at one when async episode finalization is enabled.",
)
parser.add_argument(
    "--finalize-workers",
    type=int,
    default=8,
    help=(
        "Background episode-finalization workers per collector. Writers are sharded per env, "
        "so different envs can encode concurrently while simulation continues. Set 0 for the "
        "legacy synchronous path."
    ),
)
parser.add_argument(
    "--visual-profile",
    choices=("nominal_lab", "lab_variation", "stress_tail"),
    required=True,
    help="Process-level visual profile; stored in every frame and the manifest.",
)
parser.add_argument("--collection-seed", type=int, default=42)
_SplitRenderGpuAppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
args.headless = True
args.enable_cameras = True
os.environ["UWLAB_FR3_VISUAL_PROFILE"] = args.visual_profile

app_launcher = _SplitRenderGpuAppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import uwlab_tasks  # noqa: F401,E402
from uwlab_tasks.utils.hydra import hydra_task_compose  # noqa: E402


CAMERA_KEYS = {
    "third_person_0_rgb": "observation.images.third_person_0",
    "third_person_1_rgb": "observation.images.third_person_1",
    "wrist_rgb": "observation.images.wrist",
}
STATE_KEYS = ("joint_pos", "end_effector_pose", "prev_actions")
PRIVILEGED_KEYS = ("cube_1_pose", "cube_2_pose", "cube_3_pose")
PROFILE_IDS = {"nominal_lab": 0, "lab_variation": 1, "stress_tail": 2}
DEFAULT_LEROBOT_SITE = Path("/home/ubuntu/jake/aidas/deps/lerobot_0_4_4_py311")


def _require_lerobot():
    site = Path(os.environ.get("UWLAB_LEROBOT_SITE", str(DEFAULT_LEROBOT_SITE)))
    if site.is_dir() and str(site) not in sys.path:
        sys.path.insert(0, str(site))
    try:
        version = importlib.metadata.version("lerobot")
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.datasets.aggregate import aggregate_datasets
    except (ImportError, importlib.metadata.PackageNotFoundError) as exc:
        raise RuntimeError(
            "LeRobot is not installed in the Isaac Python environment. "
            "Install stable lerobot>=0.4 (Dataset v3) before collection."
        ) from exc
    major_minor = tuple(int(part) for part in version.split("+")[0].split(".")[:2])
    if major_minor < (0, 4):
        raise RuntimeError(f"LeRobot Dataset v3 requires lerobot>=0.4, found {version}")
    return version, LeRobotDataset, aggregate_datasets


def _has_pending_frames(writer) -> bool:
    if hasattr(writer, "has_pending_frames"):
        return bool(writer.has_pending_frames())
    buffer = getattr(writer, "episode_buffer", None)
    return bool(buffer is not None and int(buffer.get("size", 0)) > 0)


def _clear_failed_episode(writer) -> None:
    """Clear an unsaved episode, including temporary PNGs for video features.

    LeRobot 0.4.4 only removes ``image_keys`` in ``clear_episode_buffer``;
    temporary frames backing ``video_keys`` otherwise accumulate during long
    success-only collection runs.
    """
    buffer = getattr(writer, "episode_buffer", None)
    if buffer is None:
        return
    episode_index = buffer.get("episode_index", 0)
    if isinstance(episode_index, np.ndarray):
        episode_index = episode_index.item() if episode_index.size == 1 else episode_index[0]
    if getattr(writer, "image_writer", None) is not None:
        writer._wait_image_writer()
    video_dirs = [
        writer._get_image_file_dir(int(episode_index), key)
        for key in getattr(writer.meta, "video_keys", ())
    ]
    writer.clear_episode_buffer(delete_images=True)
    for image_dir in video_dirs:
        if image_dir.is_dir():
            shutil.rmtree(image_dir)


def _add_expert_obs(env_cfg, agent_cfg):
    # The FR3 RGB config pre-installs this group so its one/two-stack terms are
    # rewritten consistently with the teacher during config construction.
    if getattr(env_cfg.observations, "expert_obs", None) is not None:
        return
    bc_cfg = agent_cfg.algorithm.offline_algorithm_cfg.behavior_cloning_cfg
    module_name, attribute_path = bc_cfg.experts_observation_group_cfg.split(":")
    module = __import__(module_name, fromlist=[attribute_path.split(".")[0]])
    cfg_cls = module
    for attribute in attribute_path.split("."):
        cfg_cls = getattr(cfg_cls, attribute)
    setattr(env_cfg.observations, "expert_obs", cfg_cls())


def _state_array(group: dict[str, torch.Tensor]) -> np.ndarray:
    return np.concatenate(
        [group[name].detach().cpu().numpy().reshape(group[name].shape[0], -1) for name in STATE_KEYS],
        axis=1,
    ).astype(np.float32, copy=False)


def _state_names(group: dict[str, torch.Tensor]) -> list[str]:
    names = []
    for key in STATE_KEYS:
        width = int(group[key][0].numel())
        names.extend(f"{key}.{index}" for index in range(width))
    return names


def _features(group: dict[str, torch.Tensor], action_dim: int) -> dict:
    first_image = group[next(iter(CAMERA_KEYS))]
    height, width, channels = (int(value) for value in first_image.shape[1:])
    features = {
        output_key: {
            "dtype": "video",
            "shape": (height, width, channels),
            "names": ["height", "width", "channels"],
        }
        for output_key in CAMERA_KEYS.values()
    }
    state_names = _state_names(group)
    action_names = [f"osc_action.{index}" for index in range(action_dim)]
    features.update(
        {
            "observation.state": {"dtype": "float32", "shape": (len(state_names),), "names": state_names},
            "action": {"dtype": "float32", "shape": (action_dim,), "names": action_names},
            "teacher.action_mean": {"dtype": "float32", "shape": (action_dim,), "names": action_names},
            "teacher.action_std": {"dtype": "float32", "shape": (action_dim,), "names": action_names},
            "next.reward": {"dtype": "float32", "shape": (1,), "names": ["reward"]},
            "next.done": {"dtype": "bool", "shape": (1,), "names": ["done"]},
            "next.success": {"dtype": "bool", "shape": (1,), "names": ["success"]},
            "visual.profile_id": {"dtype": "int64", "shape": (1,), "names": ["profile_id"]},
            "collection.seed": {"dtype": "int64", "shape": (1,), "names": ["seed"]},
        }
    )
    for key in PRIVILEGED_KEYS:
        width = int(group[key][0].numel())
        features[f"privileged.{key}"] = {
            "dtype": "float32",
            "shape": (width,),
            "names": [f"{key}.{index}" for index in range(width)],
        }
    return features


def _snapshot(group: dict[str, torch.Tensor]) -> tuple[dict[str, np.ndarray], np.ndarray]:
    images = {
        output_key: group[input_key].detach().cpu().numpy()
        for input_key, output_key in CAMERA_KEYS.items()
    }
    return images, _state_array(group)


@hydra_task_compose(args.task, "rsl_rl_cfg_entry_point", hydra_args=hydra_args)
def main(env_cfg, agent_cfg):
    lerobot_version, LeRobotDataset, aggregate_datasets = _require_lerobot()
    expert_path = args.expert_policy.resolve()
    if not expert_path.is_file():
        raise FileNotFoundError(f"Exported expert policy not found: {expert_path}")

    root = args.dataset_root.resolve()
    shard_root = root.parent / f".{root.name}.shards"
    for candidate in (root, shard_root):
        if candidate.exists():
            if not args.overwrite:
                raise FileExistsError(f"Refusing to overwrite {candidate}; pass --overwrite")
            shutil.rmtree(candidate)
    shard_root.mkdir(parents=True)

    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = args.device or env_cfg.sim.device
    env_cfg.seed = args.collection_seed
    _add_expert_obs(env_cfg, agent_cfg)
    env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array")
    base = env.unwrapped
    obs, _info = env.reset()

    expert = torch.jit.load(str(expert_path), map_location=base.device).to(base.device)
    expert.eval()
    with torch.inference_mode():
        mean0, _std0 = expert.compute_distribution(base.obs_buf["expert_obs"])
    action_dim = int(mean0.shape[1])
    data_group = base.obs_buf["data_collection"]
    features = _features(data_group, action_dim)
    fps = int(round(1.0 / float(base.step_dt)))

    writers = []
    shard_counts = [0] * args.num_envs
    for env_index in range(args.num_envs):
        shard = shard_root / f"env_{env_index:03d}"
        writers.append(
            LeRobotDataset.create(
                repo_id=f"{args.repo_id}-env-{env_index:03d}",
                root=shard,
                fps=fps,
                robot_type="franka_fr3_osc",
                features=features,
                use_videos=True,
                image_writer_processes=0,
                image_writer_threads=args.image_writer_threads,
                streaming_encoding=False,
                vcodec="h264",
                encoder_threads=args.encoder_threads,
            )
        )

    successful = 0
    attempted = 0
    finalize_pool = (
        concurrent.futures.ThreadPoolExecutor(
            max_workers=args.finalize_workers,
            thread_name_prefix="lerobot-finalize",
        )
        if args.finalize_workers > 0
        else None
    )
    # At most one save may mutate a given writer's metadata at once. The next
    # episode buffer is detached immediately, letting physics and PNG writes
    # proceed while the previous episode is encoded in the background.
    inflight: list[concurrent.futures.Future | None] = [None] * args.num_envs

    def finish_inflight(env_index: int) -> None:
        future = inflight[env_index]
        if future is not None:
            try:
                future.result()
            finally:
                inflight[env_index] = None

    def poll_completed_finalizations() -> None:
        for env_index, future in enumerate(inflight):
            if future is not None and future.done():
                finish_inflight(env_index)

    try:
        with contextlib.suppress(KeyboardInterrupt), torch.inference_mode():
            while successful < args.num_demos:
                poll_completed_finalizations()
                data_group = base.obs_buf["data_collection"]
                images, state = _snapshot(data_group)
                privileged = {
                    key: data_group[key].detach().cpu().numpy().astype(np.float32, copy=False)
                    for key in PRIVILEGED_KEYS
                }
                mean, std = expert.compute_distribution(base.obs_buf["expert_obs"])
                actions = mean if args.deterministic else torch.normal(mean, std)
                configured_clip = getattr(agent_cfg, "clip_actions", None)
                clip = 100.0 if configured_clip is None else float(configured_clip)
                actions = actions.clamp(-clip, clip)
                first_step = base.episode_length_buf == 0
                if bool(first_step.any()):
                    actions[first_step, :-1] = 0.0
                    actions[first_step, -1] = -1.0

                actions_cpu = actions.detach().cpu().numpy().astype(np.float32, copy=False)
                mean_cpu = mean.detach().cpu().numpy().astype(np.float32, copy=False)
                std_cpu = std.detach().cpu().numpy().astype(np.float32, copy=False)
                _next_obs, rewards, terminated, truncated, _extras = env.step(actions)
                done = terminated | truncated
                success = base.termination_manager.get_term("success").clone()
                reward_cpu = rewards.detach().cpu().numpy()
                done_cpu = done.detach().cpu().numpy()
                success_cpu = success.detach().cpu().numpy()

                for env_index, writer in enumerate(writers):
                    frame = {
                        key: np.ascontiguousarray(value[env_index]) for key, value in images.items()
                    }
                    frame.update(
                        {
                            "observation.state": np.ascontiguousarray(state[env_index]),
                            "action": np.ascontiguousarray(actions_cpu[env_index]),
                            "teacher.action_mean": np.ascontiguousarray(mean_cpu[env_index]),
                            "teacher.action_std": np.ascontiguousarray(std_cpu[env_index]),
                            "next.reward": np.asarray([reward_cpu[env_index]], dtype=np.float32),
                            "next.done": np.asarray([done_cpu[env_index]], dtype=np.bool_),
                            "next.success": np.asarray([success_cpu[env_index]], dtype=np.bool_),
                            "visual.profile_id": np.asarray(
                                [PROFILE_IDS[args.visual_profile]], dtype=np.int64
                            ),
                            "collection.seed": np.asarray([args.collection_seed], dtype=np.int64),
                            "task": args.task_description,
                        }
                    )
                    frame.update(
                        {
                            f"privileged.{key}": np.ascontiguousarray(value[env_index])
                            for key, value in privileged.items()
                        }
                    )
                    writer.add_frame(frame)
                    if not done_cpu[env_index]:
                        continue
                    # A second episode from this env cannot be committed until
                    # its preceding metadata update is complete. Other envs and
                    # the simulator remain asynchronous.
                    finish_inflight(env_index)
                    attempted += 1
                    if success_cpu[env_index] and successful < args.num_demos:
                        if finalize_pool is None:
                            writer.save_episode(parallel_encoding=args.parallel_video_encoding)
                        else:
                            episode_buffer = writer.episode_buffer
                            episode_index = int(episode_buffer["episode_index"])
                            writer.episode_buffer = writer.create_episode_buffer(
                                episode_index=episode_index + 1
                            )
                            inflight[env_index] = finalize_pool.submit(
                                writer.save_episode,
                                episode_buffer,
                                False,
                            )
                        shard_counts[env_index] += 1
                        successful += 1
                        if successful <= 10 or successful % 100 == 0:
                            print(
                                f"[LeRobot] success={successful}/{args.num_demos} attempts={attempted}",
                                flush=True,
                            )
                    else:
                        _clear_failed_episode(writer)
    finally:
        finalization_errors = []
        for env_index in range(args.num_envs):
            try:
                finish_inflight(env_index)
            except Exception as exc:  # finish every other shard before surfacing the first failure
                finalization_errors.append(exc)
        if finalize_pool is not None:
            finalize_pool.shutdown(wait=True, cancel_futures=False)
        for writer in writers:
            if _has_pending_frames(writer):
                _clear_failed_episode(writer)
            writer.finalize()
        env.close()
        if finalization_errors:
            raise RuntimeError(
                f"{len(finalization_errors)} asynchronous episode finalization(s) failed"
            ) from finalization_errors[0]

    source_repo_ids = [
        f"{args.repo_id}-env-{index:03d}"
        for index, count in enumerate(shard_counts)
        if count > 0
    ]
    source_roots = [
        shard_root / f"env_{index:03d}"
        for index, count in enumerate(shard_counts)
        if count > 0
    ]
    if not source_roots:
        raise RuntimeError("No successful LeRobot episodes were recorded")
    # Keep each source H.264 chunk as a separate final video file.  The stock
    # merge helper repeatedly concatenates the growing destination MP4, which
    # is quadratic in the number of tiled-env shards and needlessly expensive
    # for large success-only collection.  A zero rotation threshold makes the
    # official aggregator copy/reindex every source file without recompression.
    aggregate_datasets(
        repo_ids=source_repo_ids,
        aggr_repo_id=args.repo_id,
        roots=source_roots,
        aggr_root=root,
        video_files_size_in_mb=0.0,
    )
    merged = LeRobotDataset(repo_id=args.repo_id, root=root)
    if int(merged.meta.total_episodes) != successful:
        raise RuntimeError(
            f"Merged episode count mismatch: expected {successful}, got {merged.meta.total_episodes}"
        )

    manifest = {
        "schema_version": "fr3_cube.lerobot_collection.v2",
        "lerobot_version": lerobot_version,
        "lerobot_dataset_format": "v3",
        "repo_id": args.repo_id,
        "root": str(root),
        "expert_policy": str(expert_path),
        "expert_policy_sha256": hashlib.sha256(expert_path.read_bytes()).hexdigest(),
        "visual_profile": args.visual_profile,
        "visual_profile_id": PROFILE_IDS[args.visual_profile],
        "collection_seed": args.collection_seed,
        "successful_episodes": successful,
        "attempted_episodes": attempted,
        "num_envs": args.num_envs,
        "fps": fps,
        "features": features,
        "per_env_successful_episodes": shard_counts,
        "success_only": True,
        "parallel_video_encoding": args.parallel_video_encoding,
        "async_episode_finalization": finalize_pool is not None,
        "episode_finalize_workers": args.finalize_workers,
        "encoder_threads_per_camera": args.encoder_threads,
        "video_aggregation": "copy_and_reindex_without_recompression",
    }
    manifest_path = root / "uwlab_collection_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if not args.keep_shards:
        shutil.rmtree(shard_root)
    print(json.dumps({"dataset_root": str(root), "episodes": successful, "manifest": str(manifest_path)}))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        os._exit(1)
    else:
        simulation_app.close()
