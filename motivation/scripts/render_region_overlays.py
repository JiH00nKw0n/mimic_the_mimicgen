"""Render per-task top-down scene images with reset-region boxes overlaid.

For each task: build the PUBLIC D0 env (no custom registration needed), reset,
capture the birdview camera, then project every ladder variant's placement box
(from the genaudit bounds registry — the single source of truth, E-variants
included) into pixel space and draw it. Matches the region figures of the
previous motivation report.

Server usage (robosuite_mimicgen venv, MUJOCO_GL=egl implied):
  PYTHONPATH=<repo>/motivation python render_region_overlays.py --out <dir>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from genaudit.config import load_task_spec
from genaudit.envs.bounds import BOUNDS, REFERENCE_OFFSETS

TASK_CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs" / "tasks"

# (task, D0 env class name, ladder variants to draw)
SCENES = (
    ("square", "Square_D0", ("D0", "D1", "D2")),
    ("threading", "Threading_D0", ("D0", "D1", "D2E")),
    ("coffee", "Coffee_D0", ("D0", "D1E", "D2E")),
    ("three_piece_assembly", "ThreePieceAssembly_D0", ("D0", "D1", "D2")),
    ("stack", "Stack_D0", ("D0", "D1", "D2E")),
    ("stack_three", "StackThree_D0", ("D0", "D1", "D2E")),
    ("mug_cleanup", "MugCleanup_D0", ("D0", "D1E", "D2E")),
    ("hammer_cleanup", "HammerCleanup_D0", ("D0", "D1")),
    ("coffee_preparation", "CoffeePreparation_D0", ("D0", "D1")),
)

VARIANT_COLORS = ["#2a78d6", "#008300", "#e34948"]  # narrow -> wide
OBJECT_STYLES = ["-", "--", ":", "-."]
IMG = 1024


def render_birdview(env_name: str):
    import mimicgen  # noqa: F401  (registers the env classes)
    import robosuite
    from robosuite.utils.camera_utils import get_camera_transform_matrix

    env = robosuite.make(
        env_name,
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=False,
        camera_names=["birdview"],
    )
    env.reset()
    frame = env.sim.render(height=IMG, width=IMG, camera_name="birdview")[::-1]
    world_to_pixel = get_camera_transform_matrix(env.sim, "birdview", IMG, IMG)
    env.close()
    return frame, world_to_pixel


def project(world_to_pixel: np.ndarray, points_xyz: np.ndarray) -> np.ndarray:
    homogeneous = np.hstack([points_xyz, np.ones((len(points_xyz), 1))])
    pixels = (world_to_pixel @ homogeneous.T).T
    pixels = pixels[:, :2] / pixels[:, 2:3]
    # sim.render output was vertically flipped to image convention above
    pixels[:, 1] = IMG - pixels[:, 1]
    return pixels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    for task, env_name, variants in SCENES:
        load_task_spec(TASK_CONFIG_DIR / f"{task}.yaml")  # loud validation
        frame, world_to_pixel = render_birdview(env_name)
        ref_x, ref_y, ref_z = REFERENCE_OFFSETS[task]

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(frame)
        ax.set_axis_off()

        objects = sorted(BOUNDS[task][variants[-1]].keys())
        for vi, variant in enumerate(variants):
            color = VARIANT_COLORS[vi if len(variants) == 3 else vi * 2]
            for oi, obj in enumerate(objects):
                bounds = BOUNDS[task][variant].get(obj)
                if bounds is None:
                    continue
                (x0, x1), (y0, y1) = bounds.x, bounds.y
                corners = np.array(
                    [[x0, y0], [x0, y1], [x1, y1], [x1, y0], [x0, y0]], dtype=float
                )
                world = np.column_stack(
                    [corners[:, 0] + ref_x, corners[:, 1] + ref_y,
                     np.full(len(corners), ref_z)]
                )
                pixel = project(world_to_pixel, world)
                if bounds.is_position_fixed:
                    ax.plot(pixel[0, 0], pixel[0, 1], marker="*", markersize=14,
                            color=color, markeredgecolor="white", zorder=5)
                else:
                    ax.plot(pixel[:, 0], pixel[:, 1], OBJECT_STYLES[oi % 4],
                            color=color, linewidth=2.2, zorder=4)

        variant_handles = [
            plt.Line2D([], [], color=VARIANT_COLORS[vi if len(variants) == 3 else vi * 2],
                       linewidth=3, label=variant)
            for vi, variant in enumerate(variants)
        ]
        object_handles = [
            plt.Line2D([], [], color="#0b0b0b", linestyle=OBJECT_STYLES[oi % 4],
                       linewidth=2, label=obj)
            for oi, obj in enumerate(objects)
        ]
        legend1 = ax.legend(handles=variant_handles, loc="upper left",
                            title="reset range", fontsize=10, framealpha=0.9)
        ax.add_artist(legend1)
        ax.legend(handles=object_handles, loc="upper right", fontsize=9, framealpha=0.9)
        ax.set_title(task, fontsize=14, fontweight="bold", loc="left")
        fig.tight_layout()
        path = out_dir / f"regions_{task}.png"
        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {path}")


if __name__ == "__main__":
    main()
