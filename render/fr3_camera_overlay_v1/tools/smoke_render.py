#!/usr/bin/env python3
"""Render RGB smoke checks from an applied portable FR3 camera overlay."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import yaml


ARM_JOINTS = [f"fr3v2_joint{index}" for index in range(1, 8)]


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"expected YAML mapping: {path}")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--overlay", type=Path, default=Path(__file__).resolve().parents[1] / "overlay.yaml"
    )
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-prim", required=True)
    parser.add_argument("--hand-tcp-prim", required=True)
    parser.add_argument("--camera-root-name", default="calibrated_cameras")
    parser.add_argument("--articulation-prim")
    parser.add_argument("--second-q", type=float, nargs=7)
    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument("--renderer", default="RayTracedLighting")
    args = parser.parse_args()
    if args.second_q is not None and args.articulation_prim is None:
        parser.error("--second-q requires --articulation-prim")
    return args


def camera_paths(args: argparse.Namespace, cameras: dict[str, Any]) -> dict[str, str]:
    return {
        role: (
            f"{args.hand_tcp_prim.rstrip('/')}/{camera['isaac_prim_name']}"
            if camera["parent_semantic"] == "hand_tcp"
            else f"{args.base_prim.rstrip('/')}/{args.camera_root_name}/{camera['isaac_prim_name']}"
        )
        for role, camera in cameras.items()
    }


def save_rgb(camera: Any, output: Path) -> dict[str, Any]:
    import numpy as np
    from PIL import Image

    rgba = camera.get_rgba()
    if rgba is None:
        raise RuntimeError(f"camera returned no RGBA frame: {camera.prim_path}")
    rgb = np.asarray(rgba)[:, :, :3]
    if rgb.dtype != np.uint8:
        scale = 255.0 if float(np.nanmax(rgb)) <= 1.5 else 1.0
        rgb = np.clip(rgb * scale, 0.0, 255.0).astype(np.uint8)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(output)
    return {
        "path": output.name,
        "shape": list(rgb.shape),
        "mean": float(np.mean(rgb)),
        "std": float(np.std(rgb)),
        "nonblank": bool(float(np.std(rgb)) > 1.0),
    }


def set_arm_q(robot: Any, q: list[float]) -> None:
    import numpy as np

    names = list(robot.dof_names)
    missing = [name for name in ARM_JOINTS if name not in names]
    if missing:
        raise RuntimeError(f"articulation is missing expected FR3 joints: {missing}; got {names}")
    indices = np.asarray([names.index(name) for name in ARM_JOINTS], dtype=np.int32)
    values = np.asarray(q, dtype=np.float32)
    robot.set_joint_positions(values, joint_indices=indices)
    robot.set_joint_velocities(np.zeros(7, dtype=np.float32), joint_indices=indices)


def add_smoke_lighting(stage: Any) -> None:
    from pxr import Gf, Sdf, UsdGeom, UsdLux

    root = Sdf.Path("/PortableOverlaySmokeLighting")
    dome = UsdLux.DomeLight.Define(stage, root.AppendChild("Dome"))
    dome.CreateIntensityAttr(450.0)
    key = UsdLux.DistantLight.Define(stage, root.AppendChild("Key"))
    key.CreateIntensityAttr(1600.0)
    UsdGeom.Xformable(key.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-50.0, 15.0, 20.0))


def main() -> int:
    args = parse_args()
    overlay_path = args.overlay.resolve()
    stage_path = args.stage.resolve()
    output = args.output_dir.resolve()
    if not overlay_path.is_file():
        raise SystemExit(f"overlay does not exist: {overlay_path}")
    if not stage_path.is_file():
        raise SystemExit(f"stage does not exist: {stage_path}")
    overlay = load_yaml(overlay_path)
    cameras_config = overlay["runtime"]["cameras"]
    paths = camera_paths(args, cameras_config)
    width = max(int(camera["intrinsics"]["width"]) for camera in cameras_config.values())
    height = max(int(camera["intrinsics"]["height"]) for camera in cameras_config.values())

    from isaacsim import SimulationApp

    app = SimulationApp(
        {"headless": True, "renderer": args.renderer, "width": width, "height": height}
    )
    try:
        import omni.usd
        from isaacsim.core.api import World
        from isaacsim.core.prims import Articulation
        from isaacsim.sensors.camera import Camera

        context = omni.usd.get_context()
        if not context.open_stage(str(stage_path)):
            raise RuntimeError(f"failed to open stage in Isaac context: {stage_path}")
        stage = context.get_stage()
        missing_prims = [path for path in paths.values() if not stage.GetPrimAtPath(path).IsValid()]
        if missing_prims:
            raise RuntimeError(f"applied Camera prims are missing: {missing_prims}")
        add_smoke_lighting(stage)

        world = World(stage_units_in_meters=1.0)
        sensors = {}
        for role, path in paths.items():
            intrinsics = cameras_config[role]["intrinsics"]
            sensors[role] = Camera(
                prim_path=path,
                frequency=15,
                resolution=(int(intrinsics["width"]), int(intrinsics["height"])),
            )
        robot = None
        if args.articulation_prim is not None:
            robot = world.scene.add(
                Articulation(
                    prim_paths_expr=args.articulation_prim,
                    name="portable_overlay_smoke_fr3",
                )
            )
        world.reset()
        for sensor in sensors.values():
            sensor.initialize()

        home_q = [float(value) for value in overlay["runtime"]["reference_robot_pose"]["arm_joint_pos_rad"]]
        if robot is not None:
            set_arm_q(robot, home_q)
        for _ in range(max(2, args.warmup_frames)):
            if robot is not None:
                set_arm_q(robot, home_q)
            world.step(render=True)
        world.render()

        output.mkdir(parents=True, exist_ok=True)
        images = {
            role: save_rgb(sensor, output / f"{role}_home.png")
            for role, sensor in sensors.items()
        }
        second_wrist = None
        if args.second_q is not None and robot is not None:
            second_q = [float(value) for value in args.second_q]
            for _ in range(max(2, args.warmup_frames)):
                set_arm_q(robot, second_q)
                world.step(render=True)
            world.render()
            second_wrist = save_rgb(sensors["wrist"], output / "wrist_second_q.png")
        else:
            second_q = None

        report = {
            "schema_version": "stage2.portable_camera_overlay_smoke_render.v1",
            "status": "pass" if all(item["nonblank"] for item in images.values()) else "fail",
            "camera_paths": paths,
            "home_q_rad": home_q if robot is not None else None,
            "second_q_rad": second_q,
            "images": images,
            "wrist_second_q": second_wrist,
            "sensor_pipeline": "isaacsim.sensors.camera.Camera_RGBA",
        }
        if second_wrist is not None and not second_wrist["nonblank"]:
            report["status"] = "fail"
        (output / "report.yaml").write_text(
            yaml.safe_dump(report, sort_keys=False), encoding="utf-8"
        )
        ok = report["status"] == "pass"
        print("portable_overlay_smoke_render_ok", str(ok).lower())
        print("output_dir", output)
        for role, image in images.items():
            print(role, image)
        if second_wrist is not None:
            print("wrist_second_q", second_wrist)
        sys.stdout.flush()
        if not ok:
            os._exit(2)
        return 0
    finally:
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
