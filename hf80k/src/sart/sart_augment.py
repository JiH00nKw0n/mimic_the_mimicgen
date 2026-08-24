#!/usr/bin/env python3
"""SART 증강 실행기. 생성된 에피소드를 소스로 삼아 접근 구간만 다양화한 편을 더 만든다.

무엇을 하는가. MimicGen이 이미 만들어 둔 에피소드 하나를 고르고, 그 에피소드가 시작한
장면 그대로 로봇을 되돌린 뒤 다시 굴린다. 이때 물체를 옮기는 앞 구간과 구멍에 밀어 넣는
뒤 구간은 원본이 명령한 손 자세를 그대로 재생하고, 그 사이의 접근 구간만 새로 만든다.
새로 만드는 방법은 이렇다. 정밀 구간이 시작되는 손 자세를 중심으로 반지름 몇 센티미터의
공 안에서 자세 하나를 뽑고, 거기서 출발해 원래의 그 자세로 한 방향으로 되돌아온다.
물리를 실제로 돌려서 성공한 것만 파일에 남는다.

왜 이렇게 하는가. 벗어났다가 되돌아오는 움직임을 통째로 기록하면, 수렴을 방해하는
행동이 학습 자료에 섞인다. 그래서 원본의 마지막 접근 스텝 몇 개를 버리고 그 자리에
벗어남과 되돌아옴을 끼워 넣는다. 기록에 남는 것은 한 방향으로 모여드는 움직임뿐이다.

무엇을 쓰지 않는가. Isaac Lab의 데이터 생성기(DataGenerator)를 부르지 않는다. 웨이포인트
묶음과 실행 함수만 직접 쓴다. 기록은 환경에 붙은 기록기가 알아서 하므로, 이 파일은 HDF5
항목을 한 개도 직접 쓰지 않는다. 그래서 여기서 나온 편은 생성 단계가 만든 편과 같은
모양이 된다.

    isaaclab.sh -p src/sart/sart_augment.py \
        --task Isaac-PegInsert-LabFR3-IK-Rel-Mimic-v0 --headless --device cpu \
        --register peg_register,peg_success_hook,provenance_hooks \
        --dataset /work/chunks/chunk_00000/gen.hdf5 \
        --output  /work/chunks/chunk_00000/sart_00.hdf5 \
        --report  /work/chunks/chunk_00000/sart_00.json \
        --converge-object peg --converge-target socket

마지막 표준출력 한 줄은 ``SART_DONE {json}``이다. 부르는 쪽이 그 줄만 읽으면 된다.
"""
from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import math
import os
import sys
import time
import traceback

from isaaclab.app import AppLauncher

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

parser = argparse.ArgumentParser(description="SART 증강 실행기")
parser.add_argument("--task", required=True, help="Isaac에 등록된 태스크 이름")
parser.add_argument("--register", default="",
                    help="gym.make 전에 불러올 모듈들. 쉼표로 잇는다. 태스크 등록과 "
                         "성공 판정 훅이 여기서 들어온다")
parser.add_argument("--dataset", required=True, help="소스로 쓸 생성 결과 파일")
parser.add_argument("--output", required=True, help="증강 편을 쓸 조각 파일")
parser.add_argument("--report", default="", help="보고서를 쓸 JSON 경로")
parser.add_argument("--source-start", dest="source_start", type=int, default=0,
                    help="소스 목록에서 몇 번째부터 쓸지")
parser.add_argument("--source-count", dest="source_count", type=int, default=-1,
                    help="소스를 몇 편 쓸지. -1이면 끝까지")
parser.add_argument("--samples-per-source", dest="samples_per_source", type=int, default=4,
                    help="소스 한 편당 시도 횟수")
parser.add_argument("--radius-m", dest="radius_m", type=float, default=0.05,
                    help="접근 자세를 뽑는 공의 반지름, 미터")
parser.add_argument("--rotation-deg", dest="rotation_deg", type=float, default=10.0,
                    help="손 방향을 흔드는 최대 각도, 도")
parser.add_argument("--fix-position", dest="fix_position", action="store_true",
                    help="위치를 수렴 자세에 고정하고 방향만 바꾼다")
parser.add_argument("--divert-steps", dest="divert_steps", type=int, default=10)
parser.add_argument("--converge-steps", dest="converge_steps", type=int, default=20)
parser.add_argument("--settle-steps", dest="settle_steps", type=int, default=5)
parser.add_argument("--tail-steps", dest="tail_steps", type=int, default=25)
parser.add_argument("--floor-margin-m", dest="floor_margin_m", type=float, default=0.005,
                    help="고정 목표 위로 이만큼 띄운 높이가 공을 자르는 바닥면이 된다")
parser.add_argument("--converge-rule", dest="converge_rule", default="radial_gate",
                    help="radial_gate, descent_onset, tail_offset 중 하나")
parser.add_argument("--converge-object", dest="converge_object", default="",
                    help="옮기는 물체 이름. states/rigid_object 아래의 이름과 같다")
parser.add_argument("--converge-target", dest="converge_target", default="",
                    help="고정 목표 좌표계 이름. 환경의 get_object_poses가 돌려주는 이름이다")
parser.add_argument("--converge-radius-m", dest="converge_radius_m", type=float, default=0.016)
parser.add_argument("--grip-closed-m", dest="grip_closed_m", type=float, default=0.035)
parser.add_argument("--max-total-demos", dest="max_total_demos", type=int, default=-1,
                    help="이 프로세스가 쓸 증강 편수 상한. -1이면 상한 없음")
parser.add_argument("--max-consecutive-failures", dest="max_consecutive_failures",
                    type=int, default=3,
                    help="연달아 이만큼 예외가 나면 증강을 멈추고 지금까지 것을 지킨다")
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
simulation_app = AppLauncher(args).app

# ------------------------------------------------------- 시뮬레이터가 뜬 뒤의 임포트
import gymnasium as gym                                   # noqa: E402
import h5py                                               # noqa: E402
import numpy as np                                        # noqa: E402
import torch                                              # noqa: E402

import isaaclab.utils.math as PoseUtils                   # noqa: E402
from isaaclab_mimic.datagen.waypoint import (MultiWaypoint, WaypointSequence,   # noqa: E402
                                             WaypointTrajectory)
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

import sart_core                                          # noqa: E402
import sart_metrics                                       # noqa: E402


def _import_first(candidates, name):
    """같은 이름이 판마다 다른 자리에 있어서, 후보 경로를 차례로 시도한다."""
    errors = []
    for module_path in candidates:
        try:
            module = importlib.import_module(module_path)
        except Exception as exc:                          # noqa: BLE001
            errors.append(f"{module_path}: {exc!r}")
            continue
        if hasattr(module, name):
            return getattr(module, name)
        errors.append(f"{module_path}에 {name}이 없다")
    raise ImportError(f"{name}을 찾지 못했다. 시도한 곳: " + " | ".join(errors))


ActionStateRecorderManagerCfg = _import_first(
    ["isaaclab.envs.mdp.recorders.recorders_cfg", "isaaclab.envs.mdp.recorders"],
    "ActionStateRecorderManagerCfg")
DatasetExportMode = _import_first(
    ["isaaclab.managers.recorder_manager", "isaaclab.managers"], "DatasetExportMode")
HDF5DatasetFileHandler = _import_first(
    ["isaaclab.utils.datasets", "isaaclab.utils.datasets.hdf5_dataset_file_handler"],
    "HDF5DatasetFileHandler")


# ------------------------------------------------------------------------ 도구
def atomic_write_json(path: str, doc: dict) -> None:
    """임시 파일에 쓰고 이름을 바꾼다.

    보고서는 매 소스마다 다시 쓴다. 프로세스가 끝날 때 한 번만 쓰면 안 된다. Isaac Sim은
    파이썬의 종료 처리를 거치지 않고 바로 죽는 경우가 있어서, 그때 보고서가 통째로
    사라진다. src/env/provenance_hooks.py에 같은 이유가 적혀 있다.
    """
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def natural_key(name: str):
    """demo_2가 demo_10보다 앞에 오게 한다. 파이프라인의 다른 곳과 같은 규칙이다."""
    import re
    m = re.search(r"(\d+)$", name)
    return (int(m.group(1)) if m else 1 << 30, name)


def waypoints_of(sequence):
    """웨이포인트 묶음에서 낱개를 꺼낸다.

    판마다 꺼내는 이름이 달라서 세 가지를 차례로 시도한다. 셋 다 안 되면 무엇이 있는지
    적어서 오류를 낸다. 그래야 다음 사람이 서명을 다시 찾아보지 않아도 된다.
    """
    try:
        return list(sequence)
    except TypeError:
        pass
    for attr in ("sequence", "waypoints"):
        if hasattr(sequence, attr):
            return list(getattr(sequence, attr))
    raise RuntimeError("웨이포인트 낱개를 꺼내는 방법을 찾지 못했다. "
                       f"가진 이름은 {dir(sequence)}이다")


def commanded_pose_track(eef_pos, eef_quat, actions, root_quat, device):
    """기록된 관측과 행동에서 **명령한** 손 자세 궤적 (T, 4, 4)를 되살린다.

    Isaac Lab이 파일에 남기는 것은 실제로 도달한 자세와 그때 넣은 행동이다. 명령한
    자세는 남지 않는다. 그런데 증강 궤적의 앞뒤 구간은 원본이 명령한 대로 다시 명령해야
    하므로 그 값이 필요하다.

    되살리는 식은 환경의 action_to_target_eef_pose와 같다. 행동의 앞 세 개는 로봇 받침
    좌표계에서 본 위치 변화량이고 그다음 세 개는 방향 변화량이라, 받침의 방향으로 돌려
    월드 좌표계로 옮긴 뒤 그때의 손 자세에 얹는다. 이 되살림은 근사가 아니라 정확하다.
    기록기가 행동과 관측을 같은 시점, 즉 스텝을 밟기 직전에 같은 버퍼에서 가져다 쓰기
    때문이다.

    도달한 자세를 그대로 명령으로 쓰면 안 된다. 둘은 제어기가 뒤따라오는 만큼 차이가
    나고, 그 차이가 앞뒤 구간 전체에 치우침으로 남는다.

    회전은 전부 isaaclab.utils.math를 거친다. 이 컨테이너는 네 숫자 회전 표현을 XYZW
    순서로 쓰고 회전이 없는 상태가 (0, 0, 0, 1)이다. 한때 이것을 반대로 읽어 생성 697회가
    모두 실패했다. 순서를 손으로 바꾸는 코드를 여기에 쓰지 않는다.
    """
    pos = torch.as_tensor(np.asarray(eef_pos), dtype=torch.float32, device=device)
    quat = torch.as_tensor(np.asarray(eef_quat), dtype=torch.float32, device=device)
    act = torch.as_tensor(np.asarray(actions), dtype=torch.float32, device=device)
    n = act.shape[0]
    root = torch.as_tensor(np.asarray(root_quat), dtype=torch.float32,
                           device=device).reshape(1, 4).expand(n, 4)

    curr_rot = PoseUtils.matrix_from_quat(quat[:n])
    delta_position = PoseUtils.quat_apply(root, act[:, 0:3])
    delta_rotation = PoseUtils.quat_apply(root, act[:, 3:6])

    angle = torch.linalg.norm(delta_rotation, dim=-1)
    axis = delta_rotation / angle.unsqueeze(-1).clamp_min(1e-9)
    axis = torch.where((angle < 1e-9).unsqueeze(-1), torch.zeros_like(axis), axis)
    delta_rot_mat = PoseUtils.matrix_from_quat(PoseUtils.quat_from_angle_axis(angle, axis))

    target_rot = torch.matmul(delta_rot_mat, curr_rot)
    target_pos = pos[:n] + delta_position
    return PoseUtils.make_pose(target_pos, target_rot)


# --------------------------------------------------------------------- 소스 읽기
def read_source(path: str, name: str, object_name: str) -> dict:
    """소스 한 편에서 필요한 배열만 읽는다. 전부 numpy로 돌려준다."""
    with h5py.File(path, "r") as fh:
        grp = fh["data"][name]
        obs = grp["obs"]
        states = grp["states"]
        out = {
            "eef_pos": np.asarray(obs["eef_pos"][()], dtype=np.float32),
            "eef_quat": np.asarray(obs["eef_quat"][()], dtype=np.float32),
            "actions": np.asarray(grp["actions"][()], dtype=np.float32),
            "joints": np.asarray(states["articulation"]["robot"]["joint_position"][()],
                                 dtype=np.float32),
            "root_pose": np.asarray(states["articulation"]["robot"]["root_pose"][()],
                                    dtype=np.float32),
        }
        rigid = states["rigid_object"]
        if object_name not in rigid:
            raise KeyError(f"{name}의 states/rigid_object에 {object_name!r}이 없다. "
                           f"있는 이름은 {list(rigid.keys())}이다")
        out["obj_pose"] = np.asarray(rigid[object_name]["root_pose"][()], dtype=np.float32)
        # 관측 쪽 물체 위치가 있으면 좌표계가 같은지 한 번 비교해 본다. 진단용일 뿐이라
        # 값이 달라도 실행에는 영향이 없다.
        obs_key = f"{object_name}_pos"
        out["obs_obj_pos0"] = (np.asarray(obs[obs_key][0], dtype=np.float32)
                               if obs_key in obs else None)
    return out


def prepare_source(path: str, name: str, cfg: dict, device) -> dict:
    """소스 한 편을 증강에 쓸 수 있는 모양으로 미리 계산해 둔다."""
    raw = read_source(path, name, cfg["converge_object"])
    n = int(min(raw["eef_pos"].shape[0], raw["actions"].shape[0],
                raw["joints"].shape[0], raw["obj_pose"].shape[0]))
    if n < 4:
        raise ValueError(f"{name}이 {n}스텝뿐이라 증강할 수 없다")

    target_pose = commanded_pose_track(raw["eef_pos"][:n], raw["eef_quat"][:n],
                                       raw["actions"][:n], raw["root_pose"][0, 3:7], device)
    gripper = torch.as_tensor(raw["actions"][:n, 6:7], dtype=torch.float32, device=device)

    t_grasp = sart_core.grasp_step(raw["joints"][:n, 7:9], cfg["grip_closed_m"])
    rules = sart_core.converge_step_all(
        raw["eef_pos"][:n, 2], raw["obj_pose"][:n, :2], cfg["target_xy"],
        t_grasp, cfg["tail_steps"], cfg["converge_radius_m"])
    t_conv, used_fallback = rules[cfg["converge_rule"]]

    plan = sart_core.plan_segments(n, t_conv, cfg["divert_steps"],
                                   cfg["converge_steps"], cfg["settle_steps"])

    # 명령한 자세와 다음 스텝에 실제로 도달한 자세의 차이. 제어기가 뒤따라오는 크기다.
    commanded = target_pose[:, :3, 3].detach().cpu().numpy()
    achieved = raw["eef_pos"][:n]
    lag = float(np.median(np.linalg.norm(commanded[:-1] - achieved[1:], axis=1))) if n > 1 else 0.0

    return {
        "name": name,
        "n_steps": n,
        "target_pose": target_pose,
        "gripper": gripper,
        "t_grasp": int(t_grasp),
        "t_conv": int(plan["t_conv"]),
        "t_conv_by_rule": {k: int(v[0]) for k, v in rules.items()},
        "t_conv_fallback": {k: bool(v[1]) for k, v in rules.items()},
        "used_fallback": bool(used_fallback),
        "plan": plan,
        "recon_lag_m": lag,
        "obs_obj_pos0": raw["obs_obj_pos0"],
        "obj_pose0": raw["obj_pose"][0, :3],
        "eef_pos0": raw["eef_pos"][0],
    }


# ------------------------------------------------------------------- 궤적 만들기
def build_trajectory(source: dict, offset_pose, grip_conv):
    """계획의 다섯 구간을 순서대로 Isaac Lab 웨이포인트로 만든다.

    구간 순서와 첨자 계산은 sart_core.plan_segments가 정한다. 여기서 다시 세지 않는다.
    """
    plan = source["plan"]
    target_pose = source["target_pose"]
    gripper = source["gripper"]
    conv_pose = target_pose[plan["t_conv"]]

    traj = WaypointTrajectory()
    for seg in plan["segments"]:
        if seg["kind"] == "verbatim":
            start, stop = int(seg["start"]), int(seg["stop"])
            if stop <= start:
                continue
            traj.add_waypoint_sequence(WaypointSequence.from_poses(
                target_pose[start:stop], gripper[start:stop], 0.0))
        elif seg["kind"] == "target":
            goal = offset_pose if seg["pose"] == "offset" else conv_pose
            steps = int(seg["steps"])
            if steps < 1:
                continue
            traj.add_waypoint_sequence_for_target_pose(
                pose=goal, gripper_action=grip_conv, num_steps=steps, action_noise=0.0)
        elif seg["kind"] == "hold":
            steps = int(seg["steps"])
            if steps < 1:
                continue
            traj.add_waypoint_sequence(WaypointSequence.from_poses(
                conv_pose.unsqueeze(0).repeat(steps, 1, 1),
                grip_conv.unsqueeze(0).repeat(steps, 1), 0.0))
        else:
            raise ValueError(f"모르는 구간 종류 {seg['kind']!r}이다")
    return traj


async def execute_waypoints(env, eef, waypoints, success_term) -> tuple:
    """웨이포인트를 하나씩 실행한다. 성공은 스텝마다의 논리합이다.

    큐를 넘기지 않으면 실행 함수가 스스로 env.step을 부른다. 그래서 환경 하나만 띄운 이
    실행기가 데이터 생성기 없이도 돌아간다. 성공을 논리합으로 모으는 것은 Isaac Lab의
    데이터 생성기가 하는 방식과 같다.

    Returns:
        (성공 여부, 스텝마다 손끝이 가라고 명령받은 거리 목록[m]).
    """
    success = False
    # 한 스텝에 손끝이 얼마나 멀리 가라고 명령받는지를 잰다.
    #
    # 예전에는 명령이 [-1, 1]로 잘렸는지를 셌는데 그 값은 절대 0이 아닐 수 없다.
    # target_eef_pose_to_action이 돌려주는 위치 성분의 단위가 미터라서, 잘림 문턱 1.0은
    # 한 스텝에 1미터를 가라는 명령을 뜻한다. 이 태스크의 반지름은 0.06미터라 그 근처도
    # 가지 못한다. 그래서 잘림 대신 실제 명령 거리를 재고, 그것이 제어기가 한 스텝에 갈 수
    # 있는 거리를 넘는지는 사람이 값을 보고 판단한다.
    step_dists = []
    for wp in waypoints:
        probe = env.target_eef_pose_to_action({eef: wp.pose}, {eef: wp.gripper_action}, None, 0)
        step_dists.append(float(torch.linalg.vector_norm(probe[:3]).item()))
        step_ok = await MultiWaypoint({eef: wp}).execute(
            env, success_term, env_id=0, env_action_queue=None)
        success = success or bool(step_ok)
    return success, step_dists


# ------------------------------------------------------------------------ 본문
def main() -> int:
    started = time.time()
    rng = np.random.default_rng(int(args.seed))

    for name in [m.strip() for m in args.register.split(",") if m.strip()]:
        importlib.import_module(name)

    if not args.converge_object or not args.converge_target:
        raise SystemExit("--converge-object와 --converge-target을 모두 적어야 한다")
    if args.converge_rule not in sart_core.CONVERGE_RULES:
        raise SystemExit(f"--converge-rule은 {', '.join(sart_core.CONVERGE_RULES)} "
                         f"중 하나여야 한다")

    report = {
        "ok": False,
        "reason": "아직 끝나지 않았다",
        "task": args.task,
        "dataset": args.dataset,
        "output": args.output,
        "sources_used": 0,
        "attempts": 0,
        "successes": 0,
        "written": 0,
        "dgr_pct": 0.0,
        "degenerate_offsets": 0,
        "reset_pose_mismatch": 0,
        "errors": 0,
        "disabled_early": False,
        "stamp_skipped": False,
        "converge_rule_used": args.converge_rule,
        "t_conv_fallback_frac": 0.0,
        "t_conv_by_rule": {},
        "step_cmd_dist_max_m": 0.0,
        "step_cmd_dist_median_m": 0.0,
        "approach_std_peak_m": None,
        "approach_std_tail_m": None,
        "approach_std_peak_over_tail": None,
        "offset_pos_std_m": 0.0,
        "recon_lag_m": 0.0,
        "approach_std_peak_m": 0.0,
        "approach_std_profile": {},
        "params": {
            "samples_per_source": int(args.samples_per_source),
            "radius_m": float(args.radius_m),
            "rotation_deg": float(args.rotation_deg),
            "fix_position": bool(args.fix_position),
            "divert_steps": int(args.divert_steps),
            "converge_steps": int(args.converge_steps),
            "settle_steps": int(args.settle_steps),
            "tail_steps": int(args.tail_steps),
            "floor_margin_m": float(args.floor_margin_m),
            "converge_rule": args.converge_rule,
            "converge_object": args.converge_object,
            "converge_target": args.converge_target,
            "converge_radius_m": float(args.converge_radius_m),
            "grip_closed_m": float(args.grip_closed_m),
            "seed": int(args.seed),
        },
        "seconds": 0.0,
    }
    atomic_write_json(args.report, report)

    # ---------------------------------------------------------------- 환경 준비
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    # 손끝 자세를 관측 사전에서 이름으로 꺼내야 한다. 붙여 놓은 한 덩어리면 꺼내지 못한다.
    env_cfg.observations.policy.concatenate_terms = False
    # 시간 초과로 에피소드가 끊기면 안 된다. 길이는 웨이포인트 개수가 정한다.
    if hasattr(env_cfg.terminations, "time_out"):
        env_cfg.terminations.time_out = None
    # 성공 판정 항을 들어내서 손으로 부른다. 그대로 두면 성공한 순간 환경이 에피소드를
    # 끝내고, 그때 기록기가 버퍼를 파일로 내보내 버린다.
    success_term = getattr(env_cfg.terminations, "success", None)
    if success_term is None:
        raise SystemExit("이 태스크에는 success 종료 항이 없어 성공을 판정할 수 없다")
    env_cfg.terminations.success = None

    env_cfg.recorders = ActionStateRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = os.path.dirname(os.path.abspath(args.output))
    env_cfg.recorders.dataset_filename = os.path.basename(args.output)
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY

    env = gym.make(args.task, cfg=env_cfg).unwrapped
    eef = list(env.cfg.subtask_configs.keys())[0]
    ids = torch.tensor([0], device=env.device)

    # 고정 목표 좌표계를 환경에서 직접 읽는다. 프로필에 좌표를 적어 두지 않는 이유는,
    # 적어 두면 환경의 기하가 바뀔 때 프로필만 옛 값으로 남기 때문이다.
    env.reset()
    object_poses = env.get_object_poses(env_ids=[0])
    if args.converge_target not in object_poses:
        raise SystemExit(f"환경이 {args.converge_target!r} 좌표계를 내놓지 않는다. "
                         f"있는 이름은 {list(object_poses.keys())}이다")
    target_xyz = object_poses[args.converge_target][0][:3, 3].detach().cpu().numpy()
    floor_z = float(target_xyz[2]) + float(args.floor_margin_m)
    print(f"[sart] 고정 목표 {args.converge_target} = {np.round(target_xyz, 4).tolist()}, "
          f"공을 자르는 바닥 높이 {floor_z:.4f} m", flush=True)

    prep_cfg = {
        "converge_object": args.converge_object,
        "converge_rule": args.converge_rule,
        "converge_radius_m": float(args.converge_radius_m),
        "grip_closed_m": float(args.grip_closed_m),
        "tail_steps": int(args.tail_steps),
        "divert_steps": int(args.divert_steps),
        "converge_steps": int(args.converge_steps),
        "settle_steps": int(args.settle_steps),
        "target_xy": (float(target_xyz[0]), float(target_xyz[1])),
    }

    # ---------------------------------------------------------------- 소스 목록
    with h5py.File(args.dataset, "r") as fh:
        all_names = sorted(fh["data"].keys(), key=natural_key)
    start = max(0, int(args.source_start))
    stop = len(all_names) if args.source_count < 0 else start + int(args.source_count)
    names = all_names[start:stop]
    if not names:
        report.update({"ok": False, "reason": f"소스가 없다. 전체 {len(all_names)}편 중 "
                                              f"{start}부터 {stop}까지를 요청했다"})
        atomic_write_json(args.report, report)
        print("SART_DONE " + json.dumps(report, ensure_ascii=False), flush=True)
        return 1
    print(f"[sart] 소스 {len(names)}편 ({names[0]} ~ {names[-1]}), "
          f"편당 {args.samples_per_source}회 시도", flush=True)

    handler = HDF5DatasetFileHandler()
    handler.open(args.dataset)

    # 소스 파일이 지금 장면과 같은 장면에서 만들어졌는지 먼저 본다.
    #
    # 증강은 기록된 초기 상태로 장면을 되돌려 놓고 시작한다. 그런데 장면 정의가 바뀐 뒤에
    # 만든 파일과 그 전에 만든 파일은 강체 목록이 다르다. 예를 들어 책상을 충돌만 있는
    # 프림에서 강체로 바꾼 뒤에는 상태 트리에 desk_surface가 들어간다. 옛 파일에는 그것이
    # 없어서 되돌리기가 KeyError로 죽는데, 그 오류만 보고는 원인을 알 수 없다.
    try:
        with h5py.File(args.dataset, "r") as fh:
            recorded = set(fh["data"][names[0]]["states"]["rigid_object"].keys())
    except Exception:                                     # noqa: BLE001
        recorded = set()
    scene_rigid = set()
    for key in ("rigid_objects", "rigid_object"):
        got = getattr(env.scene, key, None)
        if isinstance(got, dict):
            scene_rigid = set(got.keys())
            break
    if recorded and scene_rigid and recorded != scene_rigid:
        missing = sorted(scene_rigid - recorded)
        extra = sorted(recorded - scene_rigid)
        report.update({
            "ok": False,
            "reason": ("소스 파일의 장면과 지금 장면의 강체 목록이 다르다. "
                       f"파일에 없는데 장면이 요구하는 것: {missing or '없음'}. "
                       f"파일에는 있는데 장면에 없는 것: {extra or '없음'}. "
                       "장면 정의를 바꾸기 전에 만든 파일이라는 뜻이므로, 지금 코드로 "
                       "생성을 다시 돌린 gen.hdf5를 써야 한다"),
            "scene_rigid_objects": sorted(scene_rigid),
            "recorded_rigid_objects": sorted(recorded),
        })
        print("[sart] " + report["reason"], flush=True)
        atomic_write_json(args.report, report)
        print("SART_DONE " + json.dumps(report, ensure_ascii=False), flush=True)
        return 0

    sources, initial_states = [], {}
    for name in names:
        try:
            source = prepare_source(args.dataset, name, prep_cfg, env.device)
            initial_states[name] = handler.load_episode(name, env.device).get_initial_state()
        except Exception as exc:                          # noqa: BLE001
            print(f"[sart] {name} 준비 실패, 건너뛴다: {type(exc).__name__}: {exc}", flush=True)
            continue
        sources.append(source)
    if not sources:
        report.update({"ok": False, "reason": "쓸 수 있는 소스가 한 편도 없다"})
        atomic_write_json(args.report, report)
        print("SART_DONE " + json.dumps(report, ensure_ascii=False), flush=True)
        return 1

    first = sources[0]
    if first["obs_obj_pos0"] is not None:
        gap = float(np.linalg.norm(first["obs_obj_pos0"] - first["obj_pose0"]))
        print(f"[sart] 진단: 관측의 물체 위치와 상태 트리의 물체 위치가 {gap:.4f} m 다르다 "
              f"(0에 가까워야 좌표계가 같다)", flush=True)
    report["t_conv_by_rule"] = {
        rule: [int(s["t_conv_by_rule"][rule]) for s in sources[:10]]
        for rule in sart_core.CONVERGE_RULES
    }
    report["recon_lag_m"] = round(float(np.median([s["recon_lag_m"] for s in sources])), 5)
    report["t_conv_fallback_frac"] = round(
        float(np.mean([1.0 if s["used_fallback"] else 0.0 for s in sources])), 4)
    report["sources_used"] = len(sources)
    atomic_write_json(args.report, report)

    # ---------------------------------------------------------------- 증강 반복
    cap = None if args.max_total_demos is None or args.max_total_demos < 0 else int(args.max_total_demos)
    max_angle = math.radians(float(args.rotation_deg))
    written_sources, offsets = [], []
    attempts = successes = 0
    degenerate = mismatch = errors = 0
    step_dist_max, step_dist_all = 0.0, []
    consecutive = 0
    disabled_early = False

    for round_index in range(int(args.samples_per_source)):
        if disabled_early or (cap is not None and successes >= cap):
            break
        for source in sources:
            if disabled_early or (cap is not None and successes >= cap):
                break
            attempts += 1
            try:
                plan = source["plan"]
                conv_pose = source["target_pose"][plan["t_conv"]]
                grip_conv = source["gripper"][plan["t_conv"]]
                center = conv_pose.detach().cpu().numpy().astype(np.float64)
                try:
                    offset_np = sart_core.sample_offset(
                        center, floor_z, float(args.radius_m), max_angle, rng,
                        fix_position=bool(args.fix_position))
                except sart_core.DegenerateOffset as exc:
                    degenerate += 1
                    consecutive = 0
                    print(f"[sart] {source['name']} 회차 {round_index}: 뽑을 자세가 없어 "
                          f"건너뛴다 ({exc})", flush=True)
                    continue
                offset_pose = torch.as_tensor(offset_np, dtype=torch.float32, device=env.device)

                # 기록기를 먼저 비우고 그다음에 장면을 되돌린다. 순서를 바꾸면 안 된다.
                # reset_to는 되돌리기 직전에 기록기의 내보내기를 부르는데, 버퍼를 비우지
                # 않았으면 앞 시도의 내용이 그대로 파일에 나간다.
                env.recorder_manager.reset(env_ids=ids)
                env.reset_to(initial_states[source["name"]], ids, is_relative=True)

                landed = env.get_robot_eef_pose(eef, env_ids=[0])[0][:3, 3]
                gap = float(torch.linalg.norm(
                    landed - torch.as_tensor(source["eef_pos0"], dtype=landed.dtype,
                                             device=landed.device)).item())
                if gap > 0.002:
                    mismatch += 1
                    consecutive = 0
                    print(f"[sart] {source['name']} 회차 {round_index}: 되돌린 손 자세가 "
                          f"{gap * 1000:.1f} mm 어긋나 건너뛴다", flush=True)
                    continue

                traj = build_trajectory(source, offset_pose, grip_conv)
                waypoints = waypoints_of(traj.get_full_sequence())
                ok, step_dists = asyncio.run(
                    execute_waypoints(env, eef, waypoints, success_term))
                if step_dists:
                    step_dist_max = max(step_dist_max, max(step_dists))
                    step_dist_all.extend(step_dists)

                verdict = torch.tensor([[ok]], dtype=torch.bool, device=env.device)
                env.recorder_manager.set_success_to_episodes(ids, verdict)
                # 성공 판정 훅이 이 자리에서 값을 거짓으로 바꿔 넣는다. 그래서 넣은 값이
                # 아니라 다시 읽은 값이 실제로 파일에 나간 판정이다.
                final = bool(verdict[0, 0].item())
                env.recorder_manager.export_episodes(ids)

                if final:
                    successes += 1
                    written_sources.append(source["name"])
                    offsets.append(offset_np[:3, 3].copy())
                consecutive = 0
                print(f"[sart] {source['name']} 회차 {round_index}: 웨이포인트 "
                      f"{len(waypoints)}개, 성공 {final}, 누적 성공률 "
                      f"{100.0 * successes / max(attempts, 1):.1f}%", flush=True)
            except Exception as exc:                      # noqa: BLE001
                errors += 1
                consecutive += 1
                print(f"[sart] {source['name']} 회차 {round_index} 시도가 죽었다: "
                      f"{type(exc).__name__}: {exc}", flush=True)
                traceback.print_exc()
                try:
                    env.recorder_manager.reset(env_ids=ids)
                except Exception:                         # noqa: BLE001
                    pass
                if consecutive >= int(args.max_consecutive_failures):
                    disabled_early = True
                    print(f"[sart] 연달아 {consecutive}번 죽었다. 증강을 여기서 멈추고 "
                          f"지금까지 만든 {successes}편을 지킨다.", flush=True)
                    break

        report.update({
            "attempts": attempts, "successes": successes,
            "dgr_pct": round(100.0 * successes / max(attempts, 1), 1),
            "degenerate_offsets": degenerate, "reset_pose_mismatch": mismatch,
            "errors": errors, "disabled_early": disabled_early,
            # 한 스텝에 손끝이 가라고 명령받은 거리[m]. 최대와 중앙값을 함께 적는다.
            # 최대가 제어기가 한 스텝에 실제로 갈 수 있는 거리보다 크면, 옆으로 비키는
            # 구간을 divert_steps로 더 길게 잡아야 명령을 따라갈 수 있다.
            "step_cmd_dist_max_m": round(step_dist_max, 5),
            "step_cmd_dist_median_m": round(
                float(np.median(step_dist_all)) if step_dist_all else 0.0, 5),
            "seconds": round(time.time() - started, 1),
        })
        atomic_write_json(args.report, report)

    # ---------------------------------------------------------------- 마무리
    for closer in (getattr(env, "recorder_manager", None), env):
        try:
            closer.close()
        except Exception:                                 # noqa: BLE001
            pass
    try:
        handler.close()
    except Exception:                                     # noqa: BLE001
        pass

    if len(offsets) > 1:
        report["offset_pos_std_m"] = round(float(np.std(np.array(offsets), axis=0).mean()), 5)

    # 증강 편마다 어느 소스에서 나왔는지 적는다. 다양성 측정이 이 값으로 편을 묶는다.
    if os.path.isfile(args.output):
        try:
            with h5py.File(args.output, "a") as fh:
                demo_names = sorted(fh["data"].keys(), key=natural_key)
                if len(demo_names) != len(written_sources):
                    report["stamp_skipped"] = True
                    print(f"[sart] 파일에 {len(demo_names)}편이 있는데 성공한 시도는 "
                          f"{len(written_sources)}번이라 소스 표시를 붙이지 않는다.", flush=True)
                else:
                    for demo_name, source_name, offset in zip(demo_names, written_sources, offsets):
                        fh["data"][demo_name].attrs[sart_metrics.SOURCE_ATTR] = source_name
                        fh["data"][demo_name].attrs["sart_offset_m"] = np.asarray(
                            offset, dtype=np.float32)
        except Exception as exc:                          # noqa: BLE001
            report["stamp_skipped"] = True
            print(f"[sart] 소스 표시를 붙이지 못했다: {type(exc).__name__}: {exc}", flush=True)

        try:
            measured = sart_metrics.report(args.output)
            # peak_m이 None이면 "재지 못했다"는 뜻이고 0.0이면 "재 봤더니 0"이라는 뜻이다.
            # 0.0은 증강이 원본 복사로 무너졌다는 신호라 둘을 섞으면 안 된다.
            report["approach_std_peak_m"] = measured.get("peak_m")
            report["approach_std_tail_m"] = measured.get("tail_m")
            report["approach_std_peak_over_tail"] = measured.get("peak_over_tail")
            report["approach_std_profile"] = measured.get("profile", {})
            report["approach_std_groups"] = measured.get("n_groups", 0)
            if measured.get("note"):
                report["approach_std_note"] = measured["note"]
        except Exception as exc:                          # noqa: BLE001
            print(f"[sart] 다양성 측정에 실패했다: {type(exc).__name__}: {exc}", flush=True)

    report.update({
        "ok": successes > 0,
        "reason": "" if successes > 0 else "성공한 증강 편이 하나도 없다",
        "written": successes,
        "seconds": round(time.time() - started, 1),
    })
    atomic_write_json(args.report, report)
    print("SART_DONE " + json.dumps(report, ensure_ascii=False), flush=True)
    # 종료 코드는 "내가 죽었는가"만 말한다. 얼마나 만들었는지는 보고서의 written이 말한다.
    #
    # 예전에는 성공이 0이면 1을 돌려줬다. 그러면 오케스트레이터의 run_parallel이 그것을
    # 프로세스 실패로 보고 StageError를 던지고, 같은 청크에서 다른 프로세스가 이미 만들어
    # 놓은 증강 편까지 통째로 버렸다. 프로세스를 넷으로 나눠 돌릴 때 하나만 수율 0이어도
    # 나머지 셋의 결과가 사라진다. 증강 수율 0은 정상적인 결과이지 오류가 아니다.
    return 0


if __name__ == "__main__":
    code = 1
    try:
        code = main()
    except SystemExit as exc:
        # 여기서 직접 찍고 나간다. 아래 simulation_app.close()가 프로세스를 강제로
        # 끝내기 때문에, 파이썬이 평소처럼 종료할 때 찍어 주는 오류 메시지가 화면에
        # 나오지 못한다. 그러면 아무 말 없이 종료 코드 0으로 끝나서 원인을 알 수 없다.
        message = str(exc.code) if not isinstance(exc.code, int) else ""
        if message:
            print(message, file=sys.stderr, flush=True)
            print(message, flush=True)
        code = exc.code if isinstance(exc.code, int) else 1
    except Exception:                                     # noqa: BLE001
        import traceback

        traceback.print_exc()
        sys.stderr.flush()
        sys.stdout.flush()
        code = 1
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        simulation_app.close()
    sys.exit(code)
