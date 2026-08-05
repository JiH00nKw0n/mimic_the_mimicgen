#!/usr/bin/env python3
"""Render a diverse gallery from the production FR3 Cube RGB randomization task."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--task",
    default="OmniReset-Fr3PandaCube-FullStack-RelCartesianOSC-RGB-Play-v0",
)
parser.add_argument("--num-envs", type=int, default=8)
parser.add_argument("--warmup-steps", type=int, default=5)
parser.add_argument("--video-steps", type=int, default=100)
parser.add_argument("--video-envs", type=int, default=2)
parser.add_argument(
    "--appearance-profile",
    choices=("task-default", "lab-nominal"),
    default="task-default",
)
parser.add_argument(
    "--skip-reset",
    action="store_true",
    help="Render the composed robot scene without reset events (asset/debug preview).",
)
parser.add_argument("--output-dir", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
args.headless = True
args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import cv2  # noqa: E402
import gymnasium as gym  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

import uwlab_tasks  # noqa: F401,E402
from uwlab_tasks.utils.hydra import hydra_task_compose  # noqa: E402


CAMERAS = ("third_person_0", "third_person_1", "wrist")
LAB_INDOOR_HDRI = Path(
    "/home/ubuntu/.cache/uwlab/assets/Assets/PolyHaven/HDRIs/indoor/studio_small_04_1k.hdr"
)


def _solid_lab_material(term, *, colors, roughness, metallic, specular) -> None:
    """Constrain one existing appearance event to a lab-plausible solid material."""

    term.params.update(
        {
            "texture_prob": 0.0,
            "texture_config_path": None,
            "colors": colors,
            "diffuse_tint_range": None,
            "texture_scale_range": (0.8, 1.5),
            "roughness_range": roughness,
            "metallic_range": metallic,
            "specular_range": specular,
        }
    )


def _apply_lab_nominal_profile(env_cfg) -> None:
    """Apply the 70% nominal-lab visual bucket used by the draft contract."""

    _solid_lab_material(
        env_cfg.events.randomize_gripper,
        colors=[(0.90, 0.90, 0.88), (0.93, 0.93, 0.91), (0.96, 0.96, 0.94)],
        roughness=(0.20, 0.45),
        metallic=(0.0, 0.03),
        specular=(0.25, 0.50),
    )
    _solid_lab_material(
        env_cfg.events.randomize_cube_1_appearance,
        colors=[(0.66, 0.04, 0.03), (0.72, 0.06, 0.04), (0.77, 0.07, 0.05)],
        roughness=(0.35, 0.75),
        metallic=(0.0, 0.02),
        specular=(0.20, 0.45),
    )
    _solid_lab_material(
        env_cfg.events.randomize_cube_2_appearance,
        colors=[(0.03, 0.14, 0.50), (0.03, 0.17, 0.55), (0.04, 0.20, 0.61)],
        roughness=(0.35, 0.75),
        metallic=(0.0, 0.02),
        specular=(0.20, 0.45),
    )
    _solid_lab_material(
        env_cfg.events.randomize_cube_3_appearance,
        colors=[(0.025, 0.025, 0.025), (0.04, 0.04, 0.04), (0.055, 0.055, 0.055)],
        roughness=(0.35, 0.75),
        metallic=(0.0, 0.02),
        specular=(0.20, 0.45),
    )
    _solid_lab_material(
        env_cfg.events.randomize_table_appearance,
        colors=[(0.80, 0.79, 0.73), (0.84, 0.83, 0.77), (0.88, 0.87, 0.82)],
        roughness=(0.45, 0.85),
        metallic=(0.0, 0.05),
        specular=(0.20, 0.50),
    )
    for name in (
        "randomize_curtain_left_appearance",
        "randomize_curtain_right_appearance",
        "randomize_curtain_back_appearance",
    ):
        _solid_lab_material(
            getattr(env_cfg.events, name),
            colors=[(0.44, 0.45, 0.44), (0.48, 0.49, 0.48), (0.53, 0.53, 0.52)],
            roughness=(0.75, 1.0),
            metallic=(0.0, 0.02),
            specular=(0.10, 0.25),
        )
    _solid_lab_material(
        env_cfg.events.randomize_curtain_front_appearance,
        colors=[(0.84, 0.83, 0.78), (0.88, 0.87, 0.83), (0.92, 0.91, 0.87)],
        roughness=(0.65, 0.95),
        metallic=(0.0, 0.01),
        specular=(0.10, 0.30),
    )

    # The nominal bucket uses one neutral indoor environment map.  Per-episode
    # light variation belongs to the 25% and 5% buckets, not this preview.
    if not LAB_INDOOR_HDRI.is_file():
        raise FileNotFoundError(LAB_INDOOR_HDRI)
    env_cfg.events.randomize_sky_light = None
    env_cfg.scene.sky_light.spawn.texture_file = str(LAB_INDOOR_HDRI)
    env_cfg.scene.sky_light.spawn.intensity = 1100.0


def _rgb(base, camera_name: str, env_id: int) -> np.ndarray:
    frame = base.scene[camera_name].data.output["rgb"][env_id].detach().cpu().numpy()
    if frame.shape[-1] == 4:
        frame = frame[..., :3]
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame


def _label(frame: np.ndarray, label: str) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 245, 28), fill=(0, 0, 0))
    draw.text((8, 7), label, fill=(255, 255, 255))
    return np.asarray(image)


def _row(base, env_id: int) -> np.ndarray:
    return np.concatenate(
        [_label(_rgb(base, name, env_id), f"env {env_id} | {name}") for name in CAMERAS],
        axis=1,
    )


@hydra_task_compose(args.task, "rsl_rl_cfg_entry_point", hydra_args=hydra_args)
def main(env_cfg, _agent_cfg):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    env_cfg.scene.num_envs = args.num_envs
    if args.appearance_profile == "lab-nominal":
        _apply_lab_nominal_profile(env_cfg)
    env_cfg.sim.device = args.device or env_cfg.sim.device
    env_cfg.sim.render_interval = env_cfg.decimation
    print("[gallery] creating environment", flush=True)
    env = gym.make(args.task, cfg=env_cfg)
    print("[gallery] environment created", flush=True)
    base = env.unwrapped
    actions = torch.zeros(
        (base.num_envs, base.action_manager.total_action_dim),
        dtype=torch.float32,
        device=base.device,
    )
    try:
        if args.skip_reset:
            print("[gallery] warming up composed scene without reset events", flush=True)
            for _ in range(args.warmup_steps):
                base.sim.step(render=True)
                base.scene.update(dt=base.physics_dt)
        else:
            print("[gallery] resetting environment", flush=True)
            env.reset()
            print("[gallery] reset complete", flush=True)
            for _ in range(args.warmup_steps):
                env.step(actions)
        print("[gallery] warmup complete", flush=True)

        snapshot_paths = []
        for env_id in range(args.num_envs):
            path = args.output_dir / f"randomized_env_{env_id:02d}_three_camera.png"
            Image.fromarray(_row(base, env_id)).save(path)
            snapshot_paths.append(str(path.resolve()))

        # Two environment rows expose both cross-env diversity and all three
        # calibrated camera streams. Visual events fire every four seconds, so
        # a ten-second clip also shows temporal appearance/HDRI changes.
        sample = np.concatenate([_row(base, env_id) for env_id in range(args.video_envs)], axis=0)
        video_path = args.output_dir / "randomized_two_env_three_camera.mp4"
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            10.0,
            (sample.shape[1], sample.shape[0]),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not open video writer: {video_path}")
        try:
            for _ in range(args.video_steps):
                if args.skip_reset:
                    base.sim.step(render=True)
                    base.scene.update(dt=base.physics_dt)
                else:
                    env.step(actions)
                mosaic = np.concatenate([_row(base, env_id) for env_id in range(args.video_envs)], axis=0)
                writer.write(cv2.cvtColor(mosaic, cv2.COLOR_RGB2BGR))
        finally:
            writer.release()

        manifest = {
            "schema_version": "fr3_cube.visual_randomization_gallery.v1",
            "task": args.task,
            "num_envs": args.num_envs,
            "camera_roles": list(CAMERAS),
            "resolution_per_camera": [640, 360],
            "snapshot_paths": snapshot_paths,
            "video": str(video_path.resolve()),
            "video_steps": args.video_steps,
            "video_fps": 10,
            "visual_randomization": {
                "appearance_profile": args.appearance_profile,
                "appearance_interval_s": 4.0,
                "textures": 1 if args.appearance_profile == "lab-nominal" else 957,
                "hdris": 1 if args.appearance_profile == "lab-nominal" else 920,
                "camera_pose_randomization": "measured-v2 per-env reset jitter",
            },
        }
        manifest_path = args.output_dir / "gallery_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(manifest, indent=2), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
