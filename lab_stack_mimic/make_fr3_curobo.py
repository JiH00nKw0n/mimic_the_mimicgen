#!/usr/bin/env python3
"""Build fr3_curobo.yml from the INSTALLED cuRobo's own franka.yml (version-matched schema).

cuRobo only ships a Franka Panda config. FR3 shares Panda's link/joint naming (panda_* <-> fr3_*)
and near-identical kinematics, so we load the installed cuRobo's franka.yml + its franka collision
spheres, recursively rename panda_->fr3_ (matching the FR3 URDF link/joint names), point the
kinematics at our mesh-free FR3 URDF, and inline the (renamed) collision spheres. Using the
installed config as the base guarantees the schema matches this cuRobo version (no format_version /
grasp_contact_link_names mismatch).

    python make_fr3_curobo.py <bundled_franka.yml> <spheres.yml> <fr3_urdf> <asset_root> <out.yml>
"""

import sys
import yaml


def rename(o):
    if isinstance(o, dict):
        return {(k.replace("panda_", "fr3_") if isinstance(k, str) else k): rename(v) for k, v in o.items()}
    if isinstance(o, list):
        return [rename(x) for x in o]
    if isinstance(o, str):
        return o.replace("panda_", "fr3_")
    return o


def main():
    bundled, spheres_path, urdf, asset_root, out = sys.argv[1:6]
    cfg = rename(yaml.safe_load(open(bundled)))
    sph = rename(yaml.safe_load(open(spheres_path)))
    spheres = sph.get("collision_spheres", sph)

    k = cfg["robot_cfg"]["kinematics"]
    # drop USD-kinematics fields; we drive kinematics from the FR3 URDF
    for key in ("isaac_usd_path", "usd_path", "usd_robot_root", "usd_flip_joints", "usd_flip_joint_limits"):
        k.pop(key, None)
    k["use_usd_kinematics"] = False
    k["urdf_path"] = urdf
    k["asset_root_path"] = asset_root
    k["collision_spheres"] = spheres          # inline, FR3-named
    k.pop("mesh_link_names", None)            # we use spheres, not meshes (mesh-free URDF)

    yaml.safe_dump(cfg, open(out, "w"), default_flow_style=False, sort_keys=False)
    print(f"wrote {out}")
    print(f"  base_link={k['base_link']} ee_link={k.get('ee_link')} "
          f"n_collision_links={len(k['collision_link_names'])} "
          f"sphere_links={list(spheres.keys())[:4]}... "
          f"joints={k['cspace']['joint_names'] if 'cspace' in k else '?'}")


if __name__ == "__main__":
    main()
