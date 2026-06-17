#!/usr/bin/env python3
"""
(Runs INSIDE the Isaac Lab container, launched by 04_record_video.py.)

Replay recorded demonstrations and save an offscreen MP4 video.
==============================================================
Isaac Lab's Mimic scripts (record/replay/generate) do not record video by
themselves. But Isaac Lab CAN render offscreen on a headless server when you
launch the app with `--enable_cameras`. So this script:

  1. launches Isaac Sim headless with cameras enabled (offscreen rendering),
  2. opens a generated/annotated HDF5 dataset and reads back the recorded
     actions for a few episodes,
  3. resets the environment to each episode's recorded initial state and steps
     it forward applying the recorded actions (this is exactly what Isaac Lab's
     own `replay_demos.py` does),
  4. after every step calls `env.render()` to grab an RGB frame, and
  5. writes the collected frames to an MP4 with imageio.

We grab frames manually (instead of the gymnasium RecordVideo wrapper) because
replay needs the *unwrapped* env to call `reset_to(...)`, and mixing that with
the wrapper is fiddly. Manual `render()` is simpler and robust.

This file is intentionally defensive and chatty: if the env/API differs slightly
on your Isaac Lab version, the error messages should point at the exact call
that needs adjusting. The companion `04b_inspect_dataset.py` is a no-simulator
fallback if video rendering gives trouble.
"""

from __future__ import annotations

import argparse

# ---------------------------------------------------------------------------
# 1) Parse args and launch the simulator FIRST.
#    Isaac Lab requires the AppLauncher to start before importing isaaclab.*
#    modules, so the order here matters.
# ---------------------------------------------------------------------------
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Replay demos and record an MP4 (headless).")
parser.add_argument("--dataset_file", required=True, help="HDF5 dataset to replay.")
parser.add_argument("--task", default=None, help="Task name (default: read from dataset).")
parser.add_argument("--num_episodes", type=int, default=5, help="How many episodes to record.")
parser.add_argument("--video_dir", default="/workspace/isaaclab/outputs/videos", help="Where to write MP4s.")
parser.add_argument("--fps", type=int, default=30, help="Output video frames per second.")
parser.add_argument(
    "--cam_eye", default=None,
    help="Camera position 'x,y,z' (world). Closer to --cam_lookat = more zoomed in.",
)
parser.add_argument(
    "--cam_lookat", default=None,
    help="Camera target 'x,y,z' (world), e.g. the table where the action happens.",
)
parser.add_argument(
    "--enable_pinocchio", action="store_true", default=False,
    help="Needed for GR1T2: load Pinocchio + register the pick-place task modules.",
)
# This adds Isaac Lab's standard flags (--headless, --enable_cameras, --device, ...).
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# GR1T2 needs Pinocchio. Isaac Lab's own scripts import it BEFORE the simulator
# starts, to force the version Isaac Lab installed (not Isaac Sim's). We mirror
# that ordering exactly.
if args_cli.enable_pinocchio:
    import pinocchio  # noqa: F401

# Force the two flags we need for offscreen video on a headless box.
args_cli.headless = True
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# 2) Now it is safe to import the heavy modules.
# ---------------------------------------------------------------------------
import os

import gymnasium as gym
import imageio.v2 as imageio
import torch

from isaaclab.utils.datasets import HDF5DatasetFileHandler
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

# GR1T2's gym environments are not registered by default; importing these task
# packages (after the app has started) registers them so gym.make() can find
# the pick-place task. Harmless to skip for franka.
if args_cli.enable_pinocchio:
    import isaaclab_tasks.manager_based.locomanipulation.pick_place  # noqa: F401
    import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401


def main() -> None:
    os.makedirs(args_cli.video_dir, exist_ok=True)

    # --- open the dataset and figure out which environment it belongs to ------
    handler = HDF5DatasetFileHandler()
    handler.open(args_cli.dataset_file)
    env_name = args_cli.task or handler.get_env_name()
    episode_names = list(handler.get_episode_names())
    n_to_record = min(args_cli.num_episodes, len(episode_names))
    print(f"[info] dataset env = {env_name}")
    print(f"[info] dataset has {len(episode_names)} episodes; recording {n_to_record}.")

    # --- build a single-environment config for replay -------------------------
    # We disable recorders/terminations so the env does not reset or stop on its
    # own while we are manually driving it from recorded actions.
    env_cfg = parse_env_cfg(env_name, device=args_cli.device, num_envs=1)
    env_cfg.recorders = {}
    env_cfg.terminations = {}

    # Camera: env.render() uses the "viewer" camera. The default sits far back to
    # show the whole scene; pass --cam_eye/--cam_lookat to zoom in on the table.
    if args_cli.cam_eye and args_cli.cam_lookat:
        env_cfg.viewer.eye = tuple(float(v) for v in args_cli.cam_eye.split(","))
        env_cfg.viewer.lookat = tuple(float(v) for v in args_cli.cam_lookat.split(","))
        env_cfg.viewer.origin_type = "world"  # eye/lookat are absolute world coords
    env_cfg.viewer.resolution = (1280, 720)

    # render_mode="rgb_array" makes env.render() return an HxWx3 image array.
    env = gym.make(env_name, cfg=env_cfg, render_mode="rgb_array").unwrapped

    with torch.inference_mode():
        for i in range(n_to_record):
            ep_name = episode_names[i]
            episode = handler.load_episode(ep_name, env.device)

            # Reset, then snap the simulation to the episode's recorded initial
            # state so the replay starts from exactly the same configuration.
            env.reset()
            initial_state = episode.get_initial_state()
            env.reset_to(initial_state, torch.tensor([0], device=env.device), is_relative=True)

            frames = []
            # Pull recorded actions one at a time until the episode runs out.
            while True:
                action = episode.get_next_action()
                if action is None:
                    break
                # actions tensor must be shaped [num_envs, action_dim] = [1, D].
                env.step(action.unsqueeze(0))
                frame = env.render()  # HxWx3 uint8 RGB (offscreen)
                if frame is not None:
                    frames.append(frame)

            # Write this episode to its own MP4.
            out_path = os.path.join(args_cli.video_dir, f"{os.path.splitext(os.path.basename(args_cli.dataset_file))[0]}_{ep_name}.mp4")
            if frames:
                imageio.mimwrite(out_path, frames, fps=args_cli.fps, macro_block_size=None)
                print(f"[ok] wrote {out_path}  ({len(frames)} frames)")
            else:
                print(f"[warn] episode {ep_name} produced no frames (render returned None).")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
