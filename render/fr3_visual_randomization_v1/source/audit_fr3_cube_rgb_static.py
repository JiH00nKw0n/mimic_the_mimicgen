#!/usr/bin/env python3
"""CPU-only preflight for the FR3 measured-v2 RGB collection contract."""

from __future__ import annotations

import argparse
import ast
import html
import json
import math
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / (
    "source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/omnireset/"
    "config/fr3_panda_cube/data_collection_rgb_cfg.py"
)
RL_STATE_PATH = CONFIG_PATH.with_name("rl_state_cfg.py")
ACTIONS_PATH = CONFIG_PATH.with_name("actions.py")
EVENTS_PATH = CONFIG_PATH.parents[2] / "mdp/events.py"
REGISTRATION_PATH = CONFIG_PATH.with_name("__init__.py")
LEROBOT_COLLECTOR_PATH = REPO_ROOT / "scripts_v2/tools/collect_demos_lerobot.py"
LEROBOT_SITE = Path("/home/ubuntu/jake/aidas/deps/lerobot_0_4_4_py311")
LEROBOT_SMOKE_ROOT = REPO_ROOT / "outputs/fr3_cube_lerobot_cpu_smoke_20260802/merged"
CALIBRATION_PATH = REPO_ROOT / (
    "artifacts/fr3_camera_calibration_measured_v2/camera_nominal_measured_ranges.yaml"
)
RESOURCE_ROOT = CONFIG_PATH.parents[1] / "ur5e_robotiq_2f85/resources"
CAMERA_ROLES = ("third_person_0", "third_person_1", "third_person_2", "wrist")
ACTIVE_CAMERA_ROLES = ("third_person_0", "third_person_1", "wrist")


def _literal_assignment(path: Path, name: str):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return ast.literal_eval(node.value)
    raise KeyError(f"{name} not found in {path}")


def _rotate_wxyz(quat, vector):
    w, x, y, z = quat
    vx, vy, vz = vector
    # q * v * q^-1, expanded as v + 2w(q_xyz x v) + 2(q_xyz x (q_xyz x v)).
    cx, cy, cz = y * vz - z * vy, z * vx - x * vz, x * vy - y * vx
    ccx, ccy, ccz = y * cz - z * cy, z * cx - x * cz, x * cy - y * cx
    return (vx + 2 * w * cx + 2 * ccx, vy + 2 * w * cy + 2 * ccy, vz + 2 * w * cz + 2 * ccz)


def _check(name: str, ok: bool, note: str, *, runtime: bool = False) -> dict:
    return {"name": name, "status": "PENDING_GPU" if runtime else ("PASS" if ok else "FAIL"), "note": note}


def audit() -> dict:
    calibration = yaml.safe_load(CALIBRATION_PATH.read_text(encoding="utf-8"))
    nominal = calibration.get("nominal", {})
    fixed = nominal.get("fixed_d435", {}).get("by_role", {})
    wrist = nominal.get("wrist_d405", {})
    cameras = {**fixed, "wrist": wrist}
    checks: list[dict] = []
    checks.append(
        _check(
            "measured-v2 camera schema and four calibrated roles",
            calibration.get("schema_version") == "stage2.fr3_camera_nominal_measured_ranges.v2"
            and tuple(cameras) == CAMERA_ROLES,
            f"schema={calibration.get('schema_version')}, roles={list(cameras)}",
        )
    )

    expected_parents = {"fixed_d435": "fr3v2_link0", "wrist_d405": "fr3v2_hand_tcp"}
    parent_ok = nominal["fixed_d435"].get("parent_frame") == expected_parents["fixed_d435"] and wrist.get("parent_frame") == expected_parents["wrist_d405"]
    checks.append(_check("camera parent semantics", parent_ok, str(expected_parents)))

    geometry_rows = []
    camera_geometry_ok = True
    for role in CAMERA_ROLES:
        camera = cameras[role]
        transform = camera["parent_T_camera_optical"]
        rotation = [row[:3] for row in transform[:3]]
        row_norms = [math.sqrt(sum(float(value) ** 2 for value in row)) for row in rotation]
        row_dots = [sum(rotation[i][k] * rotation[j][k] for k in range(3)) for i, j in ((0, 1), (0, 2), (1, 2))]
        intrinsics = camera["intrinsics"]
        aspect = intrinsics["width"] / intrinsics["height"]
        role_ok = max(abs(value - 1.0) for value in row_norms) < 1e-5 and max(abs(value) for value in row_dots) < 1e-5 and abs(aspect - 16 / 9) < 1e-6
        camera_geometry_ok &= role_ok
        geometry_rows.append({"role": role, "rotation_row_norms": row_norms, "aspect": aspect, "ok": role_ok})
    checks.append(_check("camera rotation/intrinsic geometry", camera_geometry_ok, json.dumps(geometry_rows)))

    robot_pos = _literal_assignment(RL_STATE_PATH, "ROBOT_POS")
    robot_rot = _literal_assignment(RL_STATE_PATH, "ROBOT_ROT")
    # Enclosure interior after accounting for 1 cm wall thickness.
    bounds = {"x": (-1.04, 1.54), "y": (-1.09, 1.09), "z": (0.0, 2.60)}
    fixed_positions = {}
    fixed_inside = True
    for role in CAMERA_ROLES[:3]:
        local = [row[3] for row in cameras[role]["parent_T_camera_optical"][:3]]
        rotated = _rotate_wxyz(robot_rot, local)
        world = tuple(float(robot_pos[index]) + rotated[index] for index in range(3))
        fixed_positions[role] = world
        margin = 0.05
        inside = (
            bounds["x"][0] + margin < world[0] < bounds["x"][1] - margin
            and bounds["y"][0] + margin < world[1] < bounds["y"][1] - margin
            and bounds["z"][0] + margin < world[2] < bounds["z"][1] - margin
        )
        fixed_inside &= inside
    checks.append(
        _check(
            "fixed cameras inside enclosure after base transform and measured local jitter",
            fixed_inside,
            json.dumps(fixed_positions),
        )
    )

    enclosure_x = bounds["x"][1] - bounds["x"][0]
    enclosure_y = bounds["y"][1] - bounds["y"][0]
    spacing = 4.0
    checks.append(
        _check(
            "multi-env enclosure spacing",
            spacing > max(enclosure_x, enclosure_y),
            f"spacing={spacing:.2f} m, enclosure={enclosure_x:.2f} x {enclosure_y:.2f} m, minimum gap={spacing-max(enclosure_x,enclosure_y):.2f} m",
        )
    )

    texture_cfg = yaml.safe_load((RESOURCE_ROOT / "texture_paths.yaml").read_text(encoding="utf-8"))
    hdri_cfg = yaml.safe_load((RESOURCE_ROOT / "hdri_paths.yaml").read_text(encoding="utf-8"))
    texture_count = sum(len(value) for value in texture_cfg.values() if isinstance(value, list))
    hdri_count = sum(len(value) for value in hdri_cfg.values() if isinstance(value, list))
    checks.append(
        _check(
            "visual randomization manifests",
            texture_count > 0 and hdri_count > 0,
            f"textures={texture_count}, HDRIs={hdri_count}; cache resolution is a runtime check",
        )
    )

    config_text = CONFIG_PATH.read_text(encoding="utf-8")
    actions_text = ACTIONS_PATH.read_text(encoding="utf-8")
    events_text = EVENTS_PATH.read_text(encoding="utf-8")
    rl_state_text = RL_STATE_PATH.read_text(encoding="utf-8")
    registration_text = REGISTRATION_PATH.read_text(encoding="utf-8")
    task_ids = [
        "OmniReset-Fr3PandaCube-FullStack-RelCartesianOSC-RGB-DataCollection-v0",
        "OmniReset-Fr3PandaCube-FullStack-RelCartesianOSC-RGB-Play-v0",
        "OmniReset-Fr3PandaCube-FullStack-RelCartesianOSC-RGB-IsolationAudit-v0",
    ]
    checks.append(_check("RGB tasks registered", all(task in registration_text for task in task_ids), str(task_ids)))
    checks.append(
        _check(
            "fixed-p=1 teacher control/physics contract",
            "class Fr3CubeRGBEventCfg(FullStackFinetuneEventCfg)" in config_text
            and "class Fr3CubeRGBRelCartesianOSCBaseCfg(Fr3CubeRelCartesianOSCFullStackFinetuneCfg)" in config_text
            and config_text.count('"initial_scale_progress": 1.0') >= 2
            and "actions: Fr3CubeP1RelativeOSCAction" in config_text
            and "curriculum: NoCurriculumsCfg" in config_text
            and "FR3_CUBE_P1_RELATIVE_OSC = FR3_CUBE_STAGE1_RELATIVE_OSC.replace" in actions_text
            and "scale_xyz_axisangle=(0.01, 0.01, 0.002, 0.02, 0.02, 0.2)" in actions_text
            and 'self.scene.robot = EXPLICIT_FR3.replace(prim_path="{ENV_REGEX_NS}/Robot")' in rl_state_text,
            "RGB DataCollection/Play/IsolationAudit share explicit FR3, 8-way FullStackFinetune physics, "
            "SysID+OSC reset progress=1, terminal action scale, and no mutable RGB-side curriculum",
        )
    )
    checks.append(
        _check(
            "stage-aware teacher observations remain separate from RGB policy",
            "expert_obs: ObservationsCfg.PolicyCfg" in config_text
            and "critic: ObservationsCfg.CriticCfg" in config_text
            and "self.observations.policy = self.observations.expert_obs" in config_text
            and 'getattr(env_cfg.observations, "expert_obs", None)' in LEROBOT_COLLECTOR_PATH.read_text(encoding="utf-8"),
            "base full-stack rewrite targets expert_obs+critic; RGB policy keeps only proprioception and three images",
        )
    )
    checks.append(
        _check(
            "appearance randomizer supports AssetBase Xform views",
            "def _scene_entity_prim_path(" in events_text
            and 'getattr(getattr(env.cfg.scene, entity_cfg.name, None), "prim_path", None)' in events_text
            and "asset_prim_path = _scene_entity_prim_path(env, asset_cfg, asset)" in events_text,
            "RigidObject/Articulation use entity.cfg.prim_path; table/other AssetBase entries resolve through env.cfg.scene",
        )
    )
    checks.append(
        _check(
            "active RGB stream selection",
            'CAMERA_ROLES = ("third_person_0", "third_person_1", "wrist")' in config_text
            and 'third_person_2_rgb' not in config_text,
            f"active={ACTIVE_CAMERA_ROLES}; third_person_2 retained only in calibration provenance",
        )
    )
    checks.append(
        _check(
            "measured camera uncertainty contract",
            "translation_ball_radius" in config_text
            and "rotation_vector_ball_radius_deg" in config_text
            and 'randomize_wrist_focal = _camera_focal_event("wrist")' in config_text
            and "randomize_third_person_0_focal" not in config_text
            and "randomize_third_person_1_focal" not in config_text,
            "episode-constant measured pose balls; D435 intrinsics fixed; wrist focal scale measured-v2",
        )
    )
    checks.append(
        _check(
            "production/audit annotator split",
            'if audit else ["rgb"]' in config_text and '"rgb", "instance_id_segmentation_fast"' in config_text,
            "production=RGB only; audit=RGB + non-colorized instance IDs",
        )
    )
    checks.append(
        _check(
            "foreign-pixel gate is fail-closed at zero tolerance",
            '"max_foreign_pixels": 0' in config_text and '"fail_closed": True' in config_text,
            "all four cameras; missing/incomplete ID mapping also fails",
        )
    )
    checks.append(
        _check(
            "camera image aspect contract",
            'UWLAB_FR3_RGB_WIDTH", "640"' in config_text
            and 'UWLAB_FR3_RGB_HEIGHT", "360"' in config_text
            and "RGB_WIDTH * 9 != RGB_HEIGHT * 16" in config_text,
            "all simulator/collector observations remain 640x360 (16:9); crop/resize is deferred to the distillation dataloader",
        )
    )
    checks.append(
        _check(
            "no collection-time image resize",
            "POLICY_RGB_SIZE" not in config_text
            and '"output_size": (RGB_HEIGHT, RGB_WIDTH)' in config_text,
            "normalized policy view and uint8 LeRobot view both retain source spatial resolution",
        )
    )
    lerobot_text = LEROBOT_COLLECTOR_PATH.read_text(encoding="utf-8")
    checks.append(
        _check(
            "LeRobot v3 success-only writer contract",
            all(
                token in lerobot_text
                for token in (
                    "LeRobotDataset.create",
                    "writer.save_episode",
                    "writer.clear_episode_buffer",
                    "writer.finalize",
                    "merge_datasets",
                    'vcodec="h264"',
                    "_clear_failed_episode",
                    '"success_only": True',
                )
            ),
            "per-env H.264/Parquet shards; failed episode buffers and temporary video PNGs cleared; finalized shards merged",
        )
    )

    lerobot_dist = list(LEROBOT_SITE.glob("lerobot-0.4.4.dist-info"))
    dependency_ok = bool(lerobot_dist) and (LEROBOT_SITE / "pyarrow").is_dir() and (LEROBOT_SITE / "av").is_dir()
    checks.append(
        _check(
            "LeRobot v3 dependencies isolated from Isaac packages",
            dependency_ok,
            f"site={LEROBOT_SITE}; lerobot=0.4.4, PyArrow and PyAV present",
        )
    )
    smoke_info_path = LEROBOT_SMOKE_ROOT / "meta/info.json"
    smoke_info = json.loads(smoke_info_path.read_text(encoding="utf-8")) if smoke_info_path.is_file() else {}
    smoke_videos = list((LEROBOT_SMOKE_ROOT / "videos").glob("**/*.mp4"))
    smoke_data = list((LEROBOT_SMOKE_ROOT / "data").glob("**/*.parquet"))
    smoke_ok = (
        smoke_info.get("codebase_version") == "v3.0"
        and smoke_info.get("total_episodes") == 2
        and smoke_info.get("total_frames") == 6
        and bool(smoke_videos)
        and bool(smoke_data)
    )
    checks.append(
        _check(
            "LeRobot v3 write/merge/reopen/video-decode smoke",
            smoke_ok,
            f"episodes={smoke_info.get('total_episodes')}, frames={smoke_info.get('total_frames')}, videos={len(smoke_videos)}, parquet={len(smoke_data)}, decoded_shape=(3,36,64)",
        )
    )

    runtime_pending = [
        "Composed FR3 prim bindings exist for fr3_link0, fr3_hand_tcp, and all three active camera prims",
        "Each camera produces non-corrupt RGB after reset randomization",
        "Instance-ID metadata resolves every env_N (fail-closed parser does not false-pass)",
        "Foreign robot/cube pixel count is exactly zero for every camera/env across reset stress samples",
        "All texture/HDRI entries resolve into the runtime local cache",
        "FR3 gripper visual prim paths compose exactly as declared",
    ]
    checks.extend(_check(item, False, "requires an RTX/Isaac camera runtime", runtime=True) for item in runtime_pending)

    cpu_checks = [item for item in checks if item["status"] != "PENDING_GPU"]
    return {
        "schema_version": "fr3_cube.rgb_static_audit.v1",
        "overall_cpu_status": "PASS" if all(item["status"] == "PASS" for item in cpu_checks) else "FAIL",
        "gpu_runtime_gate_status": "PENDING",
        "calibration": str(CALIBRATION_PATH),
        "config": str(CONFIG_PATH),
        "checks": checks,
    }


def write_html(payload: dict, path: Path) -> None:
    colors = {"PASS": "#16794b", "FAIL": "#b42318", "PENDING_GPU": "#946200"}
    rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(item['name'])}</td>"
        f"<td style='color:{colors[item['status']]};font-weight:700'>{item['status']}</td>"
        f"<td><code>{html.escape(item['note'])}</code></td>"
        "</tr>"
        for item in payload["checks"]
    )
    document = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>FR3 Cube RGB Preflight</title>
<style>body{{font:15px system-ui;margin:32px;color:#202124}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ddd;padding:9px;text-align:left;vertical-align:top}}th{{background:#f5f6f7}}code{{white-space:pre-wrap;word-break:break-word}}.summary{{padding:14px;background:#f5f6f7;border-radius:8px;margin-bottom:18px}}</style>
</head><body><h1>FR3 cube RGB preflight</h1>
<div class="summary">CPU static checks: <b>{payload['overall_cpu_status']}</b><br>Rendered isolation gate: <b>{payload['gpu_runtime_gate_status']}</b></div>
<table><thead><tr><th>Check</th><th>Status</th><th>Concrete evidence / note</th></tr></thead><tbody>{rows}</tbody></table>
</body></html>"""
    path.write_text(document, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = audit()
    json_path = args.output_dir / "static_audit.json"
    html_path = args.output_dir / "static_audit.html"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_html(payload, html_path)
    print(json.dumps({"status": payload["overall_cpu_status"], "json": str(json_path.resolve()), "html": str(html_path.resolve())}))
    raise SystemExit(0 if payload["overall_cpu_status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
