#!/usr/bin/env python3
"""Re-render lab FR3 3-cube-stack demos from the calibrated REAL camera views.

Replays each demo's RECORDED states (same fidelity as ../record_video.py
--mode states — no physics divergence) while capturing RGB from the overlay's
cameras (third_person_0/1/2 fixed on the robot base + wrist D405 on the hand),
and writes a robomimic-style hdf5:

    data/demo_i/
        actions                     (T, 7)   copied from the source demo
        obs/third_person_0_image    (T, H, W, 3) uint8
        obs/third_person_1_image    (T, H, W, 3) uint8
        obs/third_person_2_image    (T, H, W, 3) uint8   (only if asked for)
        obs/wrist_image             (T, H, W, 3) uint8
        obs/<low-dim...>            copied through from the source demo's obs
    (root) attrs: env_args, fr3_camera_overlay, fr3_binding (full provenance)

--cameras picks which roles are rendered. The default is the three the LeRobot
contract keeps; third_person_2 is calibrated and still renderable, but nothing
downstream reads it and every extra camera is a full RTX pass (~25% of the
render time at four cameras), so it is off unless asked for.

ALIGNMENT: Isaac Lab's ActionStateRecorder logs states[t] AFTER actions[t]
(obs[t] is pre-step). So the image paired with actions[t] is rendered at the
PRE-action state: initial_state for t=0, states[t-1] for t>0 (the default
--state_offset pre). If a dataset turns out to store pre-step states instead,
pass --state_offset post. The first rendered demo prints an eef-position
diagnostic comparing the replayed sim TCP against the recorded obs/eef_pos for
both hypotheses — check it once per dataset.

Requires fr3_binding.yaml from probe_tcp_binding.py (same overlay revision).

Run (arpa, UWLab env) — one visual profile per process, which is how the hf80k
orchestrator drives it (one chunk = one profile):
    bash run_render.sh /home/ubuntu/jake/aidas/3cube_stack/datasets/random_generated_2000_FINAL.hdf5 \
        --count 25 --every 2 --vrand nominal_lab --preview_video 2
"""

from __future__ import annotations

import argparse
import sys
import json
import os
import re

from isaaclab.app import AppLauncher

# The roles the LeRobot feature spec keeps (INTERFACE.md section 4). Spelled out
# here as a string because argparse runs before AppLauncher, so overlay_cameras
# (and its ALL_ROLES) cannot be imported yet; main() validates against ALL_ROLES.
CONTRACT_CAMERAS = "third_person_0,third_person_1,wrist"

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True)
parser.add_argument("--output", default="", help="default: <dataset stem>_fr3cams.hdf5 beside the dataset")
parser.add_argument("--overlay", default=os.path.join(os.path.dirname(__file__), "fr3_camera_overlay_v1/overlay.yaml"))
parser.add_argument("--binding", default=os.path.join(os.path.dirname(__file__), "fr3_binding.yaml"))
parser.add_argument("--table_usd", default="/home/ubuntu/jake/aidas/3cube_stack/table_scene.usdc")
parser.add_argument("--task", default="", help="externally registered task id (e.g. the lab peg-insert env); "
                                               "default builds the lab 3-cube-stack scene")
parser.add_argument("--register", default="", help="comma-separated modules to import (task registration) "
                                                   "before gym.make, e.g. peg_register (needs PYTHONPATH)")
parser.add_argument("--demos", default="", help='explicit "demo_0,demo_7" list (overrides --start/--count)')
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--count", type=int, default=-1, help="-1 = all")
parser.add_argument("--width", type=int, default=640)
parser.add_argument("--height", type=int, default=360)
parser.add_argument("--cameras", default=CONTRACT_CAMERAS,
                    help="comma-separated camera roles to render (default: the three contract "
                         "cameras). Each extra role is a separate RTX pass — adding the unused "
                         "third_person_2 costs about 25%% of the render time")
parser.add_argument("--every", type=int, default=1, help="temporal subsample (VIEWING ONLY — breaks BC actions)")
parser.add_argument("--success-module", default="success_criteria",
                    help="재생 성공을 판정할 모듈 이름. 그 모듈의 replay_verdict(objects, "
                         "fingers)를 부른다. 큐브 쌓기는 success_criteria, 핀 삽입은 "
                         "peg_success_criteria다.")
parser.add_argument("--success-function", default="replay_verdict",
                    help="위 모듈에서 부를 함수 이름")
parser.add_argument("--success-verdict-attr", default="replay_success_any_order",
                    help="성공 여부를 적을 속성 이름. 기록 단계가 이 이름을 읽는다.")
parser.add_argument("--vrand-object-prims", default="",
                    help="시각 규격의 물체 이름을 이 장면의 프림 경로에 잇는 표를 JSON으로 "
                         "적는다. 예: {\"cube_1\": \"{ENV}/Peg\", \"cube_2\": \"\"}. "
                         "값이 빈 문자열이면 이 장면에 없는 물체라는 뜻이고, 매 에피소드 "
                         "건너뛴 항목으로 기록된다.")
parser.add_argument("--state_offset", choices=["pre", "post"], default="pre",
                    help="pre: image[t]=state before actions[t] (Isaac Lab recorder semantics); post: states[t] as-is")
parser.add_argument("--warmup", type=int, default=6, help="extra renders at each demo's first frame")
parser.add_argument("--no_write_root", action="store_true",
                    help="skip writing the recorded robot ROOT pose each frame (default writes it: the demos "
                         "were recorded with a wxyz-encoded 180-deg root yaw that Isaac Lab 3.0's xyzw spawn "
                         "reads as identity, so the spawn pose cannot be trusted across versions)")
parser.add_argument("--pose_mode", choices=["offset", "drive"], default="offset",
                    help="offset: cameras parented to robot links (fabric tracks them across the physics "
                         "sync step); drive: per-frame Camera.set_world_poses (debug fallback — broken "
                         "when physics steps recompose fabric)")
parser.add_argument("--double_render", action="store_true", help="render twice per step (annotator-lag paranoia)")
parser.add_argument("--preview_video", type=int, default=2,
                    help="write a tiled-grid mp4 (2 cameras per row) for the first N rendered demos")
parser.add_argument("--no_compress", action="store_true", help="skip gzip (bigger, faster)")
parser.add_argument("--append", action="store_true", help="resume: skip demos already fully rendered in the output")
parser.add_argument("--vrand", default="", choices=["", "nominal_lab", "lab_variation", "stress_tail", "mixture"],
                    help="apply the RL team's visual randomization contract. PREFERRED: one fixed "
                         "profile per process (the hf80k orchestrator renders one profile per "
                         "chunk, so the 50/40/10 split comes out across chunks). 'mixture' still "
                         "works and spreads episodes over the split within one process, but see "
                         "the note in main() on its process-scope lighting")
parser.add_argument("--vrand_config", default="/vrand/config", help="dir with the handoff's config yamls")
parser.add_argument("--vrand_root", default="/vrand", help="handoff package root (hdri/texture paths)")
parser.add_argument("--vrand_seed", type=int, default=0)
parser.add_argument("--vrand_log", default="",
                    help="per-episode randomization log path (default: vrand_log.json beside the "
                         "output hdf5, the name INTERFACE.md fixes). Override it when several "
                         "render processes share one output directory")
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--allow-legacy-quat", action="store_true",
                    help="Isaac Lab 2.x가 기록한 옛 WXYZ 사원수 파일을 일부러 읽을 때만 켠다. "
                         "이 파이프라인이 만든 파일에는 절대 쓰지 않는다.")
args = parser.parse_args()
args.headless = True
args.enable_cameras = True
app = AppLauncher(args).app

import h5py
import numpy as np
import torch
import gymnasium as gym
import imageio.v2 as imageio

from isaaclab.utils.datasets import HDF5DatasetFileHandler

import dataset_format

import lab_env
import visual_randomization as vrand_mod
from overlay_cameras import (
    ALL_ROLES, build_camera_cfgs, camera_link_transforms, camera_metadata,
    camera_quat_order, load_binding, load_overlay, pose_T_from_data, quat_wxyz_from_R,
)

# success_criteria.py는 이 폴더에 함께 복사해 뒀다. 원본은 lab_stack_mimic/에 있는데
# 그 경로를 sys.path에 끼워 넣으면 컨테이너 안에서는 그 폴더가 없어 import가 깨진다.
from success_criteria import tower_status  # noqa: F401  (기본 태스크의 판정 함수)


def _load_success_verdict(module_name: str, function_name: str):
    """재생 성공을 판정할 함수를 이름으로 찾아 온다.

    태스크마다 판정 기준이 다르다. 큐브는 세 개가 탑으로 쌓였는지 보고 핀은 구멍에
    꽂혔는지 본다. 두 모듈 모두 replay_verdict(objects, fingers)라는 같은 이름의 함수를
    두고, 어느 것을 부를지는 태스크 프로필이 정한다.

    찾지 못하면 여기서 멈춘다. 조용히 건너뛰면 성공 표시가 없는 데이터가 만들어지고,
    기록 단계가 그 에피소드를 전부 버려서 원인을 찾기 어려운 실패가 된다.
    """
    import importlib

    if not module_name:
        return None
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise SystemExit(
            f"성공 판정 모듈 {module_name!r}을 불러오지 못했다: {exc}. "
            f"프로필의 render.success.module을 확인해라.") from exc
    fn = getattr(module, function_name, None)
    if fn is None:
        raise SystemExit(
            f"{module_name}에 {function_name!r} 함수가 없다. "
            f"프로필의 render.success.function을 확인해라.")
    return fn


SUCCESS_VERDICT = _load_success_verdict(args.success_module, args.success_function)


def _parse_object_prims(text: str) -> dict:
    """--vrand-object-prims에 온 JSON을 읽는다. 잘못돼 있으면 여기서 멈춘다."""
    import json as _json

    if not text.strip():
        return {}
    try:
        table = _json.loads(text)
    except ValueError as exc:
        raise SystemExit(f"--vrand-object-prims를 읽지 못했다: {exc}") from exc
    if not isinstance(table, dict):
        raise SystemExit("--vrand-object-prims는 이름과 프림 경로를 잇는 사전이어야 한다")
    return {str(k): str(v) for k, v in table.items()}


VRAND_OBJECT_PRIMS = _parse_object_prims(args.vrand_object_prims)

IMG_KEY = {r: f"{r}_image" for r in ALL_ROLES}


def natural_key(name):
    m = re.search(r"(\d+)$", name)
    return (int(m.group(1)) if m else 1 << 30, name)


def grab_rgb(cam) -> np.ndarray:
    img = cam.data.output["rgb"][0]
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    return np.ascontiguousarray(img[..., :3]).astype(np.uint8, copy=False)


def preview_tile(imgs, roles, k) -> np.ndarray:
    """One preview frame: the rendered roles tiled two per row, black-padded.

    The grid used to be hardcoded 2x2 for exactly four cameras; --cameras makes
    the count variable, so an odd count gets a black filler tile.
    """
    tiles = [imgs[r][k] for r in roles]
    if len(tiles) % 2:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.concatenate(tiles[i:i + 2], axis=1) for i in range(0, len(tiles), 2)]
    return np.concatenate(rows, axis=0)


def select_roles(spec: str) -> tuple[str, ...]:
    """Parse --cameras into a validated role tuple (order kept as written)."""
    roles = tuple(r.strip() for r in spec.split(",") if r.strip())
    if not roles:
        raise SystemExit("--cameras is empty; give at least one role")
    unknown = [r for r in roles if r not in ALL_ROLES]
    if unknown:
        raise SystemExit(f"--cameras: unknown roles {unknown}; known roles are {list(ALL_ROLES)}")
    if len(set(roles)) != len(roles):
        raise SystemExit(f"--cameras: duplicate roles in {list(roles)}")
    return roles


def main():
    roles = select_roles(args.cameras)
    ov = load_overlay(args.overlay)
    hand_T_tcp, base_adapter, binding = load_binding(args.binding, ov)
    meta = camera_metadata(ov, args.width, args.height)

    out_path = args.output or os.path.join(
        os.path.dirname(args.dataset), os.path.splitext(os.path.basename(args.dataset))[0] + "_fr3cams.hdf5")

    # 파일 루트에 format_version이 없으면 Isaac Lab이 root_pose 사원수를 옛 WXYZ로
    # 보고 한 번 더 변환한다. 로봇 받침이 z축 180도에서 y축 180도로 돌아가고 카메라가
    # 책상 밑을 보게 되는데, 관절과 물체는 멀쩡해서 성공 판정으로는 잡히지 않는다.
    dataset_format.assert_modern_quaternion_format(
        args.dataset, "render", allow_legacy=args.allow_legacy_quat)
    handler = HDF5DatasetFileHandler()
    handler.open(args.dataset)
    all_names = sorted(handler.get_episode_names(), key=natural_key)
    if args.demos:
        names = [n.strip() for n in args.demos.split(",") if n.strip()]
        missing = [n for n in names if n not in all_names]
        if missing:
            raise SystemExit(f"demos not in dataset: {missing}")
    else:
        end = len(all_names) if args.count < 0 else min(len(all_names), args.start + args.count)
        names = all_names[args.start:end]
    print(f"[render] {args.dataset}: {len(all_names)} demos total, rendering {len(names)} "
          f"at {args.width}x{args.height} every={args.every} offset={args.state_offset} "
          f"cameras={list(roles)} -> {out_path}")
    if args.every > 1:
        print(f"[render] WARNING: --every {args.every} subsamples the IK-rel action stream too — "
              f"fine for VIEWING, NOT a valid BC dataset (per-step deltas skip steps)")

    cams = build_camera_cfgs(ov, hand_T_tcp, base_adapter, args.width, args.height,
                             roles=roles, standalone=(args.pose_mode == "drive"))
    link_T = camera_link_transforms(ov, hand_T_tcp, base_adapter, roles=roles)
    if args.task:
        for mod in filter(None, (m.strip() for m in args.register.split(","))):
            __import__(mod)
        from isaaclab_tasks.utils import parse_env_cfg
        cfg = lab_env.strip_env_cfg(parse_env_cfg(args.task, device=args.device, num_envs=1), cameras=cams)
        env = gym.make(args.task, cfg=cfg).unwrapped
    else:
        env = gym.make(lab_env.TASK, cfg=lab_env.build_env_cfg(args.device, args.table_usd, cameras=cams)).unwrapped
    robot = env.scene["robot"]
    cam_objs = {r: env.scene[r] for r in roles}
    randomizer = None
    vrand_plan: dict[str, str] = {}
    vrand_log: dict[str, dict] = {}
    origin = env.scene.env_origins
    finger_idx = [i for i, n in enumerate(robot.joint_names) if "finger" in n]
    i_hand = robot.body_names.index("fr3_hand")
    body_idx = {n: robot.body_names.index(n) for n in {ln for ln, _ in link_T.values()}}
    env.reset()

    quat_order = camera_quat_order()  # Isaac Lab 3.0 switched quats to (x,y,z,w) API-wide
    print(f"[render] Isaac Lab quaternion order: {quat_order}")

    def link_pose_T(i):
        d = robot.data
        if hasattr(d, "body_link_pos_w"):
            p, q = d.body_link_pos_w[0, i].cpu().numpy(), d.body_link_quat_w[0, i].cpu().numpy()
        else:
            p, q = d.body_pos_w[0, i].cpu().numpy(), d.body_quat_w[0, i].cpu().numpy()
        return pose_T_from_data(p, q, quat_order)

    # NOTE on Isaac Lab 3.0's mixed quaternion API (all verified empirically here):
    #   cfg/InitialStateCfg quats ......... (w,x,y,z)  — unchanged from 2.x
    #   classic write_*_to_sim pose quats .. (w,x,y,z)  — unchanged from 2.x, so the
    #       recorded dataset poses pass through RAW (converting them flips the robot)
    #   data reads (body_link_quat_w etc.) . (x,y,z,w)  — handled by pose_T_from_data
    #   camera OffsetCfg / set_world_poses . (x,y,z,w)  — handled by camera_quat_order

    def drive_cameras():
        # world pose from physx FK — camera prims do NOT track articulation links
        # through the USD hierarchy under fabric (see build_camera_cfgs docstring)
        for r in roles:
            ln, T_loc = link_T[r]
            W = link_pose_T(body_idx[ln]) @ T_loc
            q = quat_wxyz_from_R(W[:3, :3])
            if quat_order == "xyzw":
                q = np.array([q[1], q[2], q[3], q[0]])
            pos = torch.tensor(W[:3, 3], dtype=torch.float32, device=env.device).unsqueeze(0)
            quat = torch.tensor(q, dtype=torch.float32, device=env.device).unsqueeze(0)
            cam_objs[r].set_world_poses(positions=pos, orientations=quat, convention="opengl")

    src = h5py.File(args.dataset, "r")
    out = h5py.File(out_path, "a" if args.append else "w")
    data_grp = out.require_group("data")
    # provenance + robomimic-compatible env_args; refuse to append across settings
    env_args = json.dumps({
        "env_name": lab_env.TASK, "type": 5,
        "env_kwargs": {"cameras": list(roles), "camera_width": args.width, "camera_height": args.height,
                       "every": args.every, "state_offset": args.state_offset},
    })
    if args.append and "env_args" in data_grp.attrs and str(data_grp.attrs["env_args"]) != env_args:
        raise SystemExit(f"--append refused: existing file rendered with different settings\n"
                         f"  existing: {data_grp.attrs['env_args']}\n  current : {env_args}")
    data_grp.attrs["env_args"] = env_args
    data_grp.attrs["fr3_camera_overlay"] = json.dumps(meta)
    data_grp.attrs["fr3_binding"] = json.dumps(binding)
    data_grp.attrs["source_dataset"] = os.path.abspath(args.dataset)
    for k, v in src["data"].attrs.items():
        data_grp.attrs[f"source_{k}"] = v

    if args.vrand:
        # process scope: profile assignment, HDRI + dome light, floor.
        #
        # WHY ONE PROFILE PER PROCESS IS THE DOCUMENTED MODE: the contract puts
        # the HDRI, the dome light and the floor at PROCESS scope, so a process
        # can only ever have one of them. In 'mixture' mode we therefore apply
        # the MAJORITY profile's lighting and let the minority episodes run
        # under it — their per-episode colors and camera jitter are right, but
        # their illumination is the majority's, so a 'stress_tail' episode
        # inside a nominal-majority process is not really a stress_tail sample.
        # Running one profile per process (one chunk = one profile, the
        # orchestrator's layout) removes that compromise entirely; the 50/40/10
        # split then comes from how many chunks each profile gets.
        mixture = vrand_mod.load_config(args.vrand_config)["profiles"]["mixture"]
        if args.vrand == "mixture":
            plan = vrand_mod.episode_profile_plan(len(names), mixture, args.vrand_seed)
            process_profile = max(mixture, key=mixture.get)
        else:
            plan = [args.vrand] * len(names)
            process_profile = args.vrand
        vrand_plan = dict(zip(names, plan))
        randomizer = vrand_mod.VisualRandomizer(
            args.vrand_config, args.vrand_root, process_profile, args.vrand_seed,
            object_prims=VRAND_OBJECT_PRIMS)
        proc = randomizer.apply_process_scope()
        print(f"[vrand] process profile={process_profile} {proc}", flush=True)
        data_grp.attrs["visual_randomization"] = json.dumps({
            "contract": "fr3_visual_randomization_handoff_v1_320x180",
            "mode": args.vrand, "seed": args.vrand_seed,
            "process_profile": process_profile, "process": proc,
            "mixture": mixture, "episode_profiles": vrand_plan,
        })

    comp = {} if args.no_compress else {"compression": "gzip", "compression_opts": 4, "shuffle": True}
    total, previews_left = 0, args.preview_video
    warned_obs_keys: set[str] = set()
    diag_done = False

    with torch.inference_mode():
        for name in names:
            # a demo counts as complete only once its num_samples attr (written last) exists
            if args.append and name in data_grp and "num_samples" in data_grp[name].attrs:
                print(f"  [skip] {name} already rendered")
                previews_left = max(0, previews_left - 1)
                continue
            if randomizer is not None:
                randomizer.profile = vrand_plan[name]
                randomizer.spec = randomizer.profiles_doc["profiles"][vrand_plan[name]]
                randomizer.apply_episode_scope(env.scene)
                vrand_log[name] = randomizer.episode_report()
                print(f"  [vrand] {name}: profile={vrand_plan[name]} "
                      f"skipped={vrand_log[name]['skipped'][:3]}", flush=True)
            ep = handler.load_episode(name, env.device)
            if "states" not in ep.data:
                raise SystemExit(f"{name} has no per-step states — this renderer needs state replay "
                                 f"(actions-only datasets would need an open-loop replay mode)")
            S = ep.data["states"]
            jp = S["articulation"]["robot"]["joint_position"]
            jv = S["articulation"]["robot"]["joint_velocity"]
            rp = S["articulation"]["robot"].get("root_pose")
            rv = S["articulation"]["robot"].get("root_velocity")
            # rigid objects discovered from the recording (stack: cube_1..3; peg: peg)
            rig_names = list(S.get("rigid_object", {}).keys())
            rigs = {n: env.scene[n] for n in rig_names}
            cp = {n: S["rigid_object"][n]["root_pose"] for n in rig_names}
            cv = {n: S["rigid_object"][n]["root_velocity"] for n in rig_names}
            acts = src["data"][name]["actions"][()]
            T_s, T_a = jp.shape[0], acts.shape[0]
            T_use = min(T_s, T_a)
            if abs(T_s - T_a) > 1:
                print(f"  [warn] {name}: states T={T_s} vs actions T={T_a}; using first {T_use}")
            steps = list(range(0, T_use, args.every))

            def sync_step():
                # push physx state to fabric so the RENDERER sees it: without a physics
                # step, joint/pose writes render stale on Isaac Lab 3.0. PD holds the
                # written pose and every state is rewritten next frame, so the single
                # 10 ms step cannot drift the replay.
                env.scene.write_data_to_sim()
                env.sim.step(render=False)

            def write_state(s):
                if not args.no_write_root and rp is not None:
                    p = rp[s:s + 1].clone(); p[:, :3] += origin
                    robot.write_root_pose_to_sim(p)
                    robot.write_root_velocity_to_sim(rv[s:s + 1])
                robot.write_joint_state_to_sim(jp[s:s + 1], jv[s:s + 1])
                robot.set_joint_position_target(jp[s:s + 1])
                for n in rig_names:
                    p = cp[n][s:s + 1].clone(); p[:, :3] += origin
                    rigs[n].write_root_pose_to_sim(p)
                    rigs[n].write_root_velocity_to_sim(cv[n][s:s + 1])
                sync_step()

            def write_state_dict(sd):
                # manual initial-state write: env.reset_to() re-initializes sensors and
                # silently detaches the camera FrameViews (set_world_poses becomes a
                # no-op afterwards on Isaac Lab 3.0) — so never use it here
                art = sd["articulation"]["robot"]
                if not args.no_write_root and "root_pose" in art:
                    p = art["root_pose"].reshape(1, -1).clone(); p[:, :3] += origin
                    robot.write_root_pose_to_sim(p)
                    robot.write_root_velocity_to_sim(art["root_velocity"].reshape(1, -1))
                jq = art["joint_position"].reshape(1, -1)
                robot.write_joint_state_to_sim(jq, art["joint_velocity"].reshape(1, -1))
                robot.set_joint_position_target(jq)
                for n in rig_names:
                    if n not in sd.get("rigid_object", {}):
                        continue
                    ro = sd["rigid_object"][n]
                    p = ro["root_pose"].reshape(1, -1).clone(); p[:, :3] += origin
                    rigs[n].write_root_pose_to_sim(p)
                    rigs[n].write_root_velocity_to_sim(ro["root_velocity"].reshape(1, -1))
                sync_step()

            env.reset()
            pre = args.state_offset == "pre"
            init_state = None
            if pre:
                try:
                    init_state = ep.get_initial_state()
                except Exception as e:
                    print(f"  [warn] {name}: no initial_state ({e}); t=0 uses states[0]")

            # one-time alignment diagnostic against the recorded eef_pos, if present
            obs_eef = None
            if not diag_done and "obs" in src["data"][name] and "eef_pos" in src["data"][name]["obs"]:
                obs_eef = src["data"][name]["obs"]["eef_pos"][()]
            err_t, err_next, err_hand, n_diag = 0.0, 0.0, 0.0, 0

            imgs = {r: [] for r in roles}
            for k, t in enumerate(steps):
                s = (t - 1) if pre else t
                if s >= 0:
                    write_state(s)
                elif init_state is not None:
                    write_state_dict(init_state)
                else:
                    write_state(0)
                env.scene.update(env.physics_dt)  # refresh robot.data (FK diagnostics / drive mode)
                if args.pose_mode == "drive":
                    drive_cameras()
                n_render = (1 + args.warmup) if k == 0 else (2 if args.double_render else 1)
                for _ in range(n_render):
                    env.sim.render()
                for r in roles:
                    imgs[r].append(grab_rgb(cam_objs[r]))
                if obs_eef is not None and t + 1 < obs_eef.shape[0]:
                    hand_T = link_pose_T(i_hand)
                    hand = hand_T[:3, 3] - origin[0].cpu().numpy()
                    tcp = hand + hand_T[:3, :3] @ np.array([0.0, 0.0, 0.1034])
                    if k < 3:
                        print(f"    [align-dbg] t={t} hand={np.round(hand, 3).tolist()} "
                              f"tcp={np.round(tcp, 3).tolist()} obs[t]={np.round(obs_eef[t], 3).tolist()}")
                    err_t += float(np.linalg.norm(tcp - obs_eef[t]))
                    err_next += float(np.linalg.norm(tcp - obs_eef[t + 1]))
                    err_hand += float(np.linalg.norm(hand - obs_eef[t]))
                    n_diag += 1

            if obs_eef is not None and n_diag:
                diag_done = True
                a, b, c = err_t / n_diag, err_next / n_diag, err_hand / n_diag
                print(f"  [align] replayed frame vs recorded obs/eef_pos: mean |err| "
                      f"tcp@t={a * 1000:.1f} mm  tcp@t+1={b * 1000:.1f} mm  hand@t={c * 1000:.1f} mm")
                print(f"  [align] the dataset's eef convention is {'TCP (+0.1034)' if a <= c else 'the raw fr3_hand frame'}; "
                      f"the matching column should be mm-level (state_offset={args.state_offset} — "
                      f"if only @t+1 is small, rerun with the other offset)")

            # 성공 표시. 이 데모의 마지막 기록 상태만 보고 판정한다. 판정 함수는 태스크
            # 프로필이 고른다(--success-module). 장면에 판정할 물체가 없으면 None이 온다.
            st = None
            if SUCCESS_VERDICT is not None:
                final_objects = {n: cp[n][T_s - 1].tolist() for n in rig_names}
                st = SUCCESS_VERDICT(final_objects, jp[T_s - 1, finger_idx].tolist())

            if name in data_grp:  # overwrite a partial demo from an interrupted run
                del data_grp[name]
            g = data_grp.create_group(name)
            g.create_dataset("actions", data=acts[steps])
            og = g.create_group("obs")
            for r in roles:
                arr = np.stack(imgs[r])
                og.create_dataset(IMG_KEY[r], data=arr, chunks=(1,) + arr.shape[1:], **comp)
            if "obs" in src["data"][name]:  # carry the original low-dim obs through
                for key, ds in src["data"][name]["obs"].items():
                    if not isinstance(ds, h5py.Dataset) or ds.ndim == 0 or ds.shape[0] < T_use:
                        if key not in warned_obs_keys:
                            warned_obs_keys.add(key)
                            print(f"  [warn] skipping obs/{key} (subgroup, scalar, or shorter than {T_use})")
                        continue
                    og.create_dataset(key, data=ds[()][steps])
            if st is not None:
                g.attrs[args.success_verdict_attr] = bool(st["ok"])
                for _k, _v in (st.get("attrs") or {}).items():
                    g.attrs[_k] = _v
            g.attrs["num_samples"] = len(steps)  # LAST: doubles as the completeness marker for --append
            total += len(steps)
            out.flush()
            tag = (f"  {st.get('label', '성공')}={'YES' if st['ok'] else 'no'}"
                   if st is not None else "")
            print(f"  [done] {name}: {len(steps)} frames x {len(roles)} cams{tag}")

            if previews_left > 0:
                previews_left -= 1
                vid = os.path.splitext(out_path)[0] + f"_{name}_preview.mp4"
                w = imageio.get_writer(vid, fps=max(1, 30 // args.every), codec="libx264",
                                       quality=7, macro_block_size=8)
                for k in range(len(steps)):
                    w.append_data(preview_tile(imgs, roles, k))
                w.close()
                print(f"  [preview] {vid}")

    # recompute over ALL complete demos in the file so --append runs keep the attr truthful
    data_grp.attrs["total"] = int(sum(
        data_grp[n].attrs["num_samples"] for n in data_grp if "num_samples" in data_grp[n].attrs))
    out.close()
    src.close()

    if randomizer is not None:
        # demo name -> what that episode actually got. The LeRobot writer reads
        # it for visual.profile_id, and it is the only record of the per-episode
        # colors and camera jitter once the images are encoded to video.
        log_path = args.vrand_log or os.path.join(
            os.path.dirname(os.path.abspath(out_path)), "vrand_log.json")
        merged = {}
        if args.append and os.path.isfile(log_path):
            # a resumed run only re-randomizes the demos it re-renders, so keep
            # the entries written by the interrupted run
            with open(log_path) as fh:
                merged = json.load(fh)
        merged.update(vrand_log)
        with open(log_path, "w") as fh:
            json.dump(merged, fh, indent=1, sort_keys=True)
        print(f"[vrand] wrote {log_path}  ({len(merged)} episodes)")

    env.close()
    print(f"[render] wrote {out_path}  ({total} new samples this run)")


if __name__ == "__main__":
    # print failures BEFORE app.close(): SimulationApp.close() can os._exit(0),
    # which would swallow the traceback and fake a success exit code
    import traceback
    code = 0
    try:
        main()
    except BaseException:
        traceback.print_exc()
        sys.stderr.flush()
        code = 1
    finally:
        sys.stdout.flush()
        app.close()
    sys.exit(code)
