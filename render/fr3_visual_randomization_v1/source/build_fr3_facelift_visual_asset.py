#!/usr/bin/env python3
"""Build an RGB-only FR3 facelift visual layer without changing robot physics.

The generated USDA references UWLab's existing NVIDIA FR3 articulation and
only hides/replaces its visual prims.  Joints, collision geometry, inertias,
actuators, cameras, and controller-facing prim names therefore remain exactly
the same as the state-teacher robot.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument(
    "--franka-description",
    type=Path,
    default=Path("/home/ubuntu/jake/ego_robot_video/ego4d_fr3/third_party/franka_description"),
)
parser.add_argument(
    "--base-usd",
    type=Path,
    default=Path(
        "/home/ubuntu/jake/UWLab/source/uwlab_assets/uwlab_assets/robots/fr3/asset/fr3_research3.usda"
    ),
)
parser.add_argument(
    "--output-dir",
    type=Path,
    default=Path(
        "/home/ubuntu/jake/UWLab/source/uwlab_assets/uwlab_assets/robots/fr3/asset/facelift"
    ),
)
parser.add_argument("--robot-model", choices=("fr3v2", "fr3v2_1"), default="fr3v2_1")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from pxr import Sdf, Usd, UsdGeom

from isaaclab.sim.converters import MeshConverter, MeshConverterCfg
from isaaclab.sim.schemas import schemas_cfg


def convert_visual(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    MeshConverter(
        MeshConverterCfg(
            asset_path=str(source),
            usd_dir=str(destination.parent),
            usd_file_name=destination.name,
            force_usd_conversion=True,
            # The same high-resolution facade is referenced by every tiled
            # environment.  Instanceable geometry prevents per-env mesh and
            # shader duplication while preserving the source triangles.
            make_instanceable=True,
            collision_props=schemas_cfg.CollisionPropertiesCfg(collision_enabled=False),
            mesh_collision_props=None,
        )
    )

    # MeshConverter authors disabled CollisionAPI schemas even for a
    # render-only conversion.  When this facade is parented below an existing
    # rigid body, PhysX still cooks those triangle meshes (and may fall back to
    # convex hulls).  Remove the schemas and attributes entirely: collision is
    # supplied only by the referenced, validated FR3 articulation.
    instance_layer = destination.parent / "Props" / "instanceable_meshes.usd"
    for usd_path in (destination, instance_layer):
        if not usd_path.is_file():
            continue
        stage = Usd.Stage.Open(str(usd_path))
        for prim in stage.TraverseAll():
            schemas = tuple(str(schema) for schema in prim.GetAppliedSchemas())
            if any("CollisionAPI" in schema for schema in schemas):
                prim.ClearMetadata("apiSchemas")
            for prop in tuple(prim.GetProperties()):
                name = prop.GetName()
                if name.startswith("physics:") or name.startswith("physxCollision:"):
                    prim.RemoveProperty(name)
        stage.GetRootLayer().Save()


def relative_asset_path(path: Path, layer_dir: Path) -> str:
    return Path(os.path.relpath(path.resolve(), layer_dir.resolve())).as_posix()


def main() -> None:
    source_root = args.franka_description.resolve()
    base_usd = args.base_usd.resolve()
    output_dir = args.output_dir.resolve()
    meshes_dir = output_dir / "meshes"
    output_dir.mkdir(parents=True, exist_ok=True)
    if meshes_dir.exists():
        shutil.rmtree(meshes_dir)
    meshes_dir.mkdir(parents=True, exist_ok=True)

    if not base_usd.is_file():
        raise FileNotFoundError(base_usd)

    # Task metadata is resolved relative to the selected robot USD.  Keep the
    # exact base-articulation metadata next to this visual-only wrapper.
    base_metadata = base_usd.parent / "metadata.yaml"
    if not base_metadata.is_file():
        raise FileNotFoundError(base_metadata)
    shutil.copy2(base_metadata, output_dir / "metadata.yaml")

    converted: dict[str, Path] = {}
    for index in range(8):
        role = f"link{index}"
        source = source_root / "meshes" / "robots" / args.robot_model / "visual" / f"{role}.dae"
        destination = meshes_dir / role / f"{role}.usd"
        destination.parent.mkdir(parents=True, exist_ok=True)
        convert_visual(source, destination)
        converted[role] = destination

    hand_sources = {
        "hand": source_root / "meshes/robot_ee/franka_hand_white/visual/hand.dae",
        "finger": source_root / "meshes/robot_ee/franka_hand_white/visual/finger.dae",
    }
    for role, source in hand_sources.items():
        destination = meshes_dir / role / f"{role}.usd"
        destination.parent.mkdir(parents=True, exist_ok=True)
        convert_visual(source, destination)
        converted[role] = destination

    layer_path = output_dir / "fr3_facelift_visual.usda"
    stage = Usd.Stage.CreateNew(str(layer_path))
    root = UsdGeom.Xform.Define(stage, "/fr3")
    stage.SetDefaultPrim(root.GetPrim())
    root.GetPrim().GetReferences().AddReference(relative_asset_path(base_usd, output_dir))

    bindings = {f"fr3_link{index}": f"link{index}" for index in range(8)}
    bindings.update(
        {
            "fr3_hand": "hand",
            "fr3_leftfinger": "finger",
            "fr3_rightfinger": "finger",
        }
    )
    for body_name, mesh_role in bindings.items():
        original = stage.OverridePrim(f"/fr3/{body_name}/visuals")
        UsdGeom.Imageable(original).CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
        replacement = UsdGeom.Xform.Define(stage, f"/fr3/{body_name}/facelift_visual")
        replacement.GetPrim().GetReferences().AddReference(
            relative_asset_path(converted[mesh_role], output_dir)
        )

    stage.GetRootLayer().Save()

    manifest = {
        "schema_version": "uwlab.fr3_facelift_visual.v1",
        "base_articulation_usd": str(base_usd),
        "visual_source": str(source_root),
        "robot_model": args.robot_model,
        "output_usd": str(layer_path),
        "physics_changed": False,
        "replaced_visuals": bindings,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
