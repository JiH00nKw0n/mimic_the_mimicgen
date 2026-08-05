#!/usr/bin/env python3
"""RTX runtime gate for FR3 RGB camera integrity and cross-env isolation."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

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
    default="OmniReset-Fr3PandaCube-FullStack-RelCartesianOSC-RGB-IsolationAudit-v0",
)
parser.add_argument("--num-envs", type=int, default=32)
parser.add_argument("--steps", type=int, default=20)
parser.add_argument("--warmup-steps", type=int, default=3)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument(
    "--visual-profile",
    choices=("nominal_lab", "lab_variation", "stress_tail"),
    required=True,
)
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


@hydra_task_compose(args.task, "rsl_rl_cfg_entry_point", hydra_args=hydra_args)
def main(env_cfg, _agent_cfg):
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = args.device or env_cfg.sim.device
    env_cfg.sim.render_interval = env_cfg.decimation
    env = gym.make(args.task, cfg=env_cfg)
    base = env.unwrapped
    actions = torch.zeros(
        (base.num_envs, base.action_manager.total_action_dim),
        dtype=torch.float32,
        device=base.device,
    )

    max_counts = {name: 0 for name in ("third_person_0", "third_person_1", "wrist")}
    min_rgb_std = {name: float("inf") for name in max_counts}
    mapping_all_steps = {name: True for name in max_counts}
    samples = 0
    try:
        env.reset()
        for step in range(args.warmup_steps + args.steps):
            env.step(actions)
            if step < args.warmup_steps:
                continue
            samples += 1
            counts = getattr(base, "_foreign_env_pixel_counts", {})
            mapping = getattr(base, "_foreign_env_pixel_mapping_ok", {})
            for camera_name in max_counts:
                camera = base.scene[camera_name]
                rgb = camera.data.output["rgb"].reshape(base.num_envs, -1).float()
                min_rgb_std[camera_name] = min(
                    min_rgb_std[camera_name], float(torch.std(rgb, dim=1).min().item())
                )
                camera_counts = counts.get(camera_name)
                if camera_counts is None:
                    max_counts[camera_name] = -1
                    mapping_all_steps[camera_name] = False
                else:
                    max_counts[camera_name] = max(max_counts[camera_name], int(camera_counts.max().item()))
                    mapping_all_steps[camera_name] &= bool(mapping.get(camera_name, False))
    finally:
        env.close()

    passed = bool(
        samples == args.steps
        and all(value == 0 for value in max_counts.values())
        and all(mapping_all_steps.values())
        and all(value >= 10.0 for value in min_rgb_std.values())
    )
    payload = {
        "schema_version": "fr3_cube.rgb_runtime_isolation_audit.v1",
        "pass": passed,
        "task": args.task,
        "visual_profile": args.visual_profile,
        "num_envs": args.num_envs,
        "sampled_steps": samples,
        "acceptance": {
            "max_foreign_pixels_per_camera_env": 0,
            "min_rgb_std": 10.0,
            "instance_id_mapping_complete": True,
        },
        "observed": {
            "max_foreign_pixel_count": max_counts,
            "min_rgb_std": min_rgb_std,
            "mapping_complete_every_step": mapping_all_steps,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    if not passed:
        raise RuntimeError("FR3 RGB isolation runtime gate failed")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        os._exit(1)
    else:
        simulation_app.close()
