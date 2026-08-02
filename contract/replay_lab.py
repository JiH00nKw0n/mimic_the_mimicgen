"""Replay lab-format episodes in the lab FR3 scene and judge tower success.

Thin sibling of lab_stack_mimic/replay_count.py that builds the env via
render/lab_env.build_env_cfg (procedural desk-slab fallback — no lab table USD
needed on aidas). Used to validate rl_to_lab conversions and to measure the
replay success of any lab-format dataset.

  ./run_isaac docker ... -p /repo/contract/replay_lab.py --device cpu \
      --dataset_file /out/rl_as_lab_3cube_smoke.hdf5 --report /out/r.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_file", required=True)
parser.add_argument("--report", default=None)
parser.add_argument("--table_usd", default="/nonexistent.usdc")
parser.add_argument("--select_episodes", type=int, nargs="+", default=[])
parser.add_argument("--settle_steps", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = False
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch  # noqa: E402
import gymnasium as gym  # noqa: E402
from isaaclab.utils.datasets import HDF5DatasetFileHandler  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
for rel in ("../render", "../lab_stack_mimic"):
    sys.path.insert(0, os.path.normpath(os.path.join(HERE, rel)))

from lab_env import build_env_cfg  # noqa: E402
from success_criteria import tower_status  # noqa: E402


def main() -> int:
    handler = HDF5DatasetFileHandler()
    handler.open(args.dataset_file)
    episode_names = list(handler.get_episode_names())
    indices = args.select_episodes or list(range(len(episode_names)))
    print(f"[replay] dataset={args.dataset_file} episodes={len(episode_names)} "
          f"replaying={len(indices)}", flush=True)

    cfg = build_env_cfg(args.device, args.table_usd, cameras=None, num_envs=1)
    env = gym.make("Isaac-Stack-Cube-Franka-IK-Rel-v0", cfg=cfg,
                   render_mode=None).unwrapped
    finger_idx = [i for i, n in enumerate(env.scene["robot"].joint_names)
                  if "finger" in n]
    env.reset()

    results = []
    any_ok = 0
    with torch.inference_mode():
        for n in indices:
            name = episode_names[n]
            episode = handler.load_episode(name, env.device)
            env.reset_to(episode.get_initial_state(),
                         torch.tensor([0], device=env.device), is_relative=True)
            steps = 0
            while steps < 3000:
                action = episode.get_next_action()
                if action is None:
                    break
                env.step(action.unsqueeze(0) if action.ndim == 1 else action)
                steps += 1
            for _ in range(args.settle_steps):
                env.sim.step(render=False)
                env.scene.update(env.sim.get_physics_dt())
            cubes = [env.scene[f"cube_{i}"].data.root_pos_w[0].tolist()
                     for i in (1, 2, 3)]
            fingers = env.scene["robot"].data.joint_pos[0, finger_idx].tolist()
            status = tower_status(cubes, fingers, canonical=False)
            any_ok += int(status["ok"])
            print(f"  {name}: steps={steps} ANY="
                  f"{'PASS' if status['ok'] else 'fail'} gaps={status['gaps']} "
                  f"xy_ok={status['xy_ok']} released={status['released']}",
                  flush=True)
            results.append({"demo": name, "steps": steps, "status": status})

    print(f"[replay] any-order success: {any_ok}/{len(indices)}", flush=True)
    if args.report:
        with open(args.report, "w") as handle:
            json.dump({"dataset": args.dataset_file, "n": len(indices),
                       "any_order_success": any_ok, "results": results},
                      handle, indent=2)
    env.close()
    simulation_app.close()
    return 0


if __name__ == "__main__":
    main()
