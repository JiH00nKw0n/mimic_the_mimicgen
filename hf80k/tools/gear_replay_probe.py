"""기어 시연을 재생하면서 어디서 어긋나는지 찍는다.

왜 필요한가. 재생이 20편 전부 "과제 미완료"로 끝나는데, 그 한 줄로는 원인을 알 수 없다.
손끝이 궤적을 못 따라가는 것인지, 기어를 못 쥐는 것인지, 쥐었지만 축에 못 넣는 것인지,
아니면 다 했는데 성공 판정이 잘못된 것인지 구분해야 한다.

무엇을 찍는가. 스텝마다 네 가지를 기록된 값과 비교한다.

  1. 손끝 위치가 기록된 자리에서 얼마나 벗어났는지
  2. 기어 셋이 기록된 자리에서 얼마나 벗어났는지
  3. 손가락이 얼마나 열려 있는지와 기록된 값
  4. 기어마다 성공 판정의 네 성분 (축 오차, 높이 차, 기울기, 속도)

쓰는 법. Isaac Lab 실행기로 돌린다. 등록 모듈을 끼워 넣을 필요가 없게 직접 import 한다.

    isaaclab.sh -p tools/gear_replay_probe.py --task Isaac-MultiGear-LabFR3-IK-Rel-Mimic-v0 \
        --input_file /opt/hf80k/assets/multigear_annotated.hdf5 --device cpu --demos 1
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", required=True)
parser.add_argument("--input_file", required=True)
parser.add_argument("--demos", type=int, default=1, help="앞에서 몇 편을 볼지")
parser.add_argument("--every", type=int, default=40, help="몇 스텝마다 찍을지")
parser.add_argument("--check-init", action="store_true",
                    help="리셋 직후의 받침 자세와 손끝 자세만 찍고 끝낸다. 사원수 순서를 "
                         "확인하는 용도다")
parser.add_argument("--seat-trace", default="",
                    help="기어마다 스텝별 축오차와 높이차를 전부 모아 이 경로에 JSON으로 "
                         "쓴다. 기어가 축을 끝까지 내려갔다가 튕겨 나온 것인지, 애초에 "
                         "내려가지 못한 것인지 가리는 용도다")
parser.add_argument("--mode", choices=["open", "closed"], default="open",
                    help="open은 파일에 적힌 명령을 그대로 넣는다. closed는 매 스텝 지금 "
                         "손끝 자세와 기록된 목표 자세의 차이로 명령을 다시 만든다")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = False

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import json  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import gear_register  # noqa: E402,F401  태스크를 gym에 등록한다
from isaaclab.utils.datasets import HDF5DatasetFileHandler  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

from gear_geom import (AXIS_XY_TOL_M, GEARS, SPEED_TOL_M_S, TILT_TOL_RAD,  # noqa: E402
                       Z_TOL_M, axis_xy_error, tilt_rad_from_quat)


def eef_pos(env):
    """손끝 위치를 ee_frame에서 직접 읽는다. 환경 기준이다.

    get_robot_eef_pose는 관측 버퍼를 읽으므로 리셋 전에는 쓸 수 없다. 사원수 순서를
    확인하는 단계에서는 리셋 전 상태도 봐야 하므로 프레임에서 바로 읽는다.
    """
    f = env.scene["ee_frame"]
    return f.data.target_pos_w[0, 0] - env.scene.env_origins[0]


def world(env, name):
    """환경 기준 위치와 XYZW 자세."""
    obj = env.scene[name]
    pos = obj.data.root_pos_w - env.scene.env_origins
    quat = obj.data.root_quat_w
    if quat.ndim == 1 or quat.shape[-1] != 4:
        quat = quat.reshape(-1, 4)
    return pos, quat


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    handler = HDF5DatasetFileHandler()
    handler.open(args_cli.input_file)
    names = list(handler.get_episode_names())
    names.sort(key=lambda s: int(s.split("_")[1]))

    report = []
    for ep_name in names[:args_cli.demos]:
        episode = handler.load_episode(ep_name, env.device)
        initial_state = episode.data["initial_state"]
        actions = episode.data["actions"]
        rec_eef = episode.data["obs"]["datagen_info"]["eef_pose"]["franka"]
        rec_target = episode.data["obs"]["datagen_info"]["target_eef_pose"]["franka"]
        rec_obj = {k: episode.data["obs"]["datagen_info"]["object_pose"][k]
                   for k in ("base",) + GEARS}
        rec_grip = episode.data["obs"]["gripper_pos"]

        env.sim.reset()
        env.recorder_manager.reset()

        if args_cli.check_init:
            # 리셋 전에 설정값 그대로의 상태를 본다
            rp = env.scene["robot"].data.root_pos_w[0] - env.scene.env_origins[0]
            rq = env.scene["robot"].data.root_quat_w[0]
            ee3 = eef_pos(env)
            print(f"\n=== {ep_name} 리셋 전 (설정값 그대로) ===", flush=True)
            print(f"  받침 위치 {[round(float(v), 4) for v in rp]}", flush=True)
            print(f"  받침 사원수(파일에 적힌 순서 그대로) "
                  f"{[round(float(v), 4) for v in rq]}", flush=True)
            print(f"  손끝 위치 {[round(float(v), 4) for v in ee3]}", flush=True)
            print(f"  기록된 손끝 위치 "
                  f"{[round(float(v), 4) for v in rec_eef[0][:3, 3]]}", flush=True)
            print(f"  차이 "
                  f"{float(torch.linalg.vector_norm(ee3 - rec_eef[0][:3, 3])):.4f} m",
                  flush=True)
            print(f"  파일의 initial_state 받침 자세 "
                  f"{[round(float(v), 4) for v in initial_state['articulation']['robot']['root_pose'].reshape(-1)]}",
                  flush=True)
            print(f"  파일의 initial_state 관절 "
                  f"{[round(float(v), 4) for v in initial_state['articulation']['robot']['joint_position'].reshape(-1)]}",
                  flush=True)

        env.reset_to(initial_state, None, is_relative=True)

        if args_cli.check_init:
            rp = env.scene["robot"].data.root_pos_w[0] - env.scene.env_origins[0]
            rq = env.scene["robot"].data.root_quat_w[0]
            ee3 = eef_pos(env)
            jp = env.scene["robot"].data.joint_pos[0]
            print(f"  --- reset_to 뒤 ---", flush=True)
            print(f"  받침 위치 {[round(float(v), 4) for v in rp]}", flush=True)
            print(f"  받침 사원수 {[round(float(v), 4) for v in rq]}", flush=True)
            print(f"  관절 {[round(float(v), 4) for v in jp]}", flush=True)
            print(f"  손끝 위치 {[round(float(v), 4) for v in ee3]}", flush=True)
            print(f"  기록된 손끝 위치 "
                  f"{[round(float(v), 4) for v in rec_eef[0][:3, 3]]}", flush=True)
            print(f"  차이 "
                  f"{float(torch.linalg.vector_norm(ee3 - rec_eef[0][:3, 3])):.4f} m",
                  flush=True)
            for g in ("base",) + GEARS:
                pos, quat = world(env, g)
                # 기울기를 함께 찍는다. 파일에 적힌 사원수의 성분 순서가 reset_to가
                # 기대하는 순서와 다르면 z축 회전이 x축 회전으로 들어가 물체가 눕는다.
                # 위치만 보면 그 사고가 보이지 않는다. 눕혀도 원점 높이는 그대로다.
                print(f"  {g:7s} 실제 {[round(float(v), 4) for v in pos[0]]} "
                      f"기록 {[round(float(v), 4) for v in rec_obj[g][0][:3, 3]]} "
                      f"사원수 {[round(float(v), 4) for v in quat[0]]} "
                      f"기울기 {float(tilt_rad_from_quat(quat)[0]):.4f} rad", flush=True)

            # 손끝을 어디서 읽어야 기록과 맞는지 찾는다. ee_frame의 목표 프레임 전부와
            # 로봇 몸통 링크 전부를 기록된 손끝과 비교해 가장 가까운 것을 보여 준다.
            print("  --- 어느 프레임이 기록된 손끝과 맞나 ---", flush=True)
            target = rec_eef[0][:3, 3]
            org = env.scene.env_origins[0]
            cands = []
            f = env.scene["ee_frame"]
            fnames = getattr(f.data, "target_frame_names", None) or [
                f"ee_frame[{i}]" for i in range(f.data.target_pos_w.shape[1])]
            for i, nm in enumerate(fnames):
                cands.append((f"ee_frame/{nm}", f.data.target_pos_w[0, i] - org))
            rb = env.scene["robot"]
            for i, nm in enumerate(rb.data.body_names):
                cands.append((f"body/{nm}", rb.data.body_pos_w[0, i] - org))
            cands.sort(key=lambda c: float(torch.linalg.vector_norm(c[1] - target)))
            for nm, pos in cands[:8]:
                d = float(torch.linalg.vector_norm(pos - target))
                print(f"    {d:7.4f} m  {nm:28s} {[round(float(v), 4) for v in pos]}",
                      flush=True)
            print(f"    기록된 손끝 {[round(float(v), 4) for v in target]}", flush=True)
            continue

        print(f"\n=== {ep_name}: {actions.shape[0]}스텝 재생 "
              f"({'열린 고리' if args_cli.mode == 'open' else '닫힌 고리'}) ===", flush=True)
        print("  스텝 | 손끝오차 | 큰기어오차 작은기어오차 중간기어오차 | 손가락(실제/기록)",
              flush=True)
        rows = []
        seat = {g: {"axis": [], "dz": [], "tilt": []} for g in GEARS}
        seat["eef_err"], seat["grip"] = [], []
        for i in range(actions.shape[0]):
            if args_cli.mode == "open":
                # 파일에 적힌 명령을 그대로 넣는다. 이웃한 기록 자세의 차이라서 제어기가
                # 뒤처지면 그 오차가 계속 쌓인다.
                act = torch.as_tensor(actions[i]).reshape(1, -1)
            else:
                # 매 스텝 다시 만든다. 기록된 목표 자세를 그대로 쓰고 지금 실제 손끝
                # 자세와의 차이를 명령으로 만든다. 뒤처진 만큼이 다음 명령에 자동으로
                # 더해지므로 오차가 쌓이지 않는다. MimicGen의 생성도 같은 방식으로
                # 환경을 몬다. env의 target_eef_pose_to_action이 그 변환을 한다.
                grip = torch.as_tensor(actions[i][6:7]).reshape(1).to(env.device)
                act = env.target_eef_pose_to_action(
                    {"franka": torch.as_tensor(rec_target[i]).to(env.device)},
                    {"franka": grip}, None, 0).reshape(1, -1)
            env.step(act)
            if args_cli.seat_trace:
                # 매 스텝 걷는다. 찍는 주기(--every)와 무관하게 전부 모아야 기어가
                # 가장 깊이 내려간 순간을 놓치지 않는다.
                bp, bq = world(env, "base")
                for g in GEARS:
                    gp, gq = world(env, g)
                    seat[g]["axis"].append(
                        round(float(axis_xy_error(gp, gq, bp, bq, g)[0]), 6))
                    seat[g]["dz"].append(
                        round(abs(float(gp[0, 2]) - float(bp[0, 2])), 6))
                    # 기울기도 매 스텝 걷는다. 중간 기어가 축에 8 mm 걸려 멈출 때
                    # 기울기가 0.05 rad로 나오는데, 그 기울기가 축에 닿기 전부터
                    # 있던 것인지 닿은 뒤에 생긴 것인지로 원인이 갈린다. 앞쪽이면
                    # 쥐는 자세가 틀어진 것이고 뒤쪽이면 접촉 풀이가 문제다.
                    seat[g]["tilt"].append(
                        round(float(tilt_rad_from_quat(gq)[0]), 6))
                eef_now = env.get_robot_eef_pose("franka", env_ids=[0])[0]
                seat["eef_err"].append(round(float(torch.linalg.vector_norm(
                    eef_now[:3, 3] - rec_eef[i][:3, 3])), 6))
                seat["grip"].append(round(float(
                    env.scene["robot"].data.joint_pos[0, -2:].mean()), 5))
            if i % args_cli.every and i != actions.shape[0] - 1:
                continue
            eef = env.get_robot_eef_pose("franka", env_ids=[0])[0]
            eef_err = float(torch.linalg.vector_norm(eef[:3, 3] - rec_eef[i][:3, 3]))
            errs = {}
            for g in GEARS:
                p, _ = world(env, g)
                errs[g] = float(torch.linalg.vector_norm(p[0] - rec_obj[g][i][:3, 3]))
            fingers = env.scene["robot"].data.joint_pos[0, -2:]
            print(f"  {i:5d} | {eef_err:8.4f} | "
                  f"{errs['large']:10.4f} {errs['small']:12.4f} {errs['medium']:12.4f} | "
                  f"{float(fingers.mean()):.4f}/{float(rec_grip[i].mean()):.4f}", flush=True)
            rows.append({"step": i, "eef_err_m": round(eef_err, 5),
                         **{f"{g}_err_m": round(errs[g], 5) for g in GEARS}})

        # 마지막 상태에서 성공 판정의 네 성분을 기어마다 찍는다
        print("  === 마지막 상태의 성공 판정 ===", flush=True)
        base_pos, base_quat = world(env, "base")
        verdict = {}
        for g in GEARS:
            pos, quat = world(env, g)
            ax = float(axis_xy_error(pos, quat, base_pos, base_quat, g)[0])
            dz = abs(float(pos[0, 2]) - float(base_pos[0, 2]))
            tilt = float(tilt_rad_from_quat(quat)[0])
            vel = env.scene[g].data.root_vel_w
            if vel.ndim == 1 or vel.shape[-1] < 3:
                vel = vel.reshape(-1, 6)
            sp = float(torch.linalg.vector_norm(vel[0, :3]))
            ok = (ax < AXIS_XY_TOL_M and dz < Z_TOL_M
                  and tilt < TILT_TOL_RAD and sp < SPEED_TOL_M_S)
            print(f"    {g:7s} 축오차 {ax:.5f}(<{AXIS_XY_TOL_M}) "
                  f"높이차 {dz:.5f}(<{Z_TOL_M}) "
                  f"기울기 {tilt:.4f}(<{TILT_TOL_RAD}) "
                  f"속도 {sp:.4f}(<{SPEED_TOL_M_S}) -> {'통과' if ok else '실패'}",
                  flush=True)
            verdict[g] = {"axis_err_m": round(ax, 6), "z_err_m": round(dz, 6),
                          "tilt_rad": round(tilt, 5), "speed_m_s": round(sp, 5), "ok": ok}
        # 기록된 마지막 상태로 같은 판정을 해 본다. 판정 자체가 틀렸는지 가린다.
        print("  === 기록된 마지막 상태로 같은 판정 ===", flush=True)
        rec_verdict = {}
        last = actions.shape[0] - 1
        rb_pos = rec_obj["base"][last][:3, 3].reshape(1, 3)
        rb_rot = rec_obj["base"][last][:3, :3].reshape(1, 3, 3)
        for g in GEARS:
            gp = rec_obj[g][last][:3, 3].reshape(1, 3)
            gr = rec_obj[g][last][:3, :3].reshape(1, 3, 3)
            # 회전 행렬에서 기울기를 직접 구한다. 행렬의 (2,2)가 몸통 z축의 월드 z 성분이다
            tilt = float(torch.acos(torch.clamp(gr[0, 2, 2], -1.0, 1.0)))
            # 축 오차는 행렬로 직접 계산한다
            off = torch.tensor([[0.0, 0.0, 0.0]], device=gp.device)
            from gear_geom import AXIS_X, AXIS_Z
            off[0, 0], off[0, 2] = AXIS_X[g], AXIS_Z
            ga = gp + (gr @ off.unsqueeze(-1)).squeeze(-1)
            ba = rb_pos + (rb_rot @ off.unsqueeze(-1)).squeeze(-1)
            ax = float(torch.linalg.vector_norm((ga - ba)[0, :2]))
            dz = abs(float(gp[0, 2]) - float(rb_pos[0, 2]))
            ok = ax < AXIS_XY_TOL_M and dz < Z_TOL_M and tilt < TILT_TOL_RAD
            print(f"    {g:7s} 축오차 {ax:.5f} 높이차 {dz:.5f} 기울기 {tilt:.4f} "
                  f"-> {'통과' if ok else '실패'}", flush=True)
            rec_verdict[g] = {"axis_err_m": round(ax, 6), "z_err_m": round(dz, 6),
                              "tilt_rad": round(tilt, 5), "ok": ok}

        entry = {"episode": ep_name, "mode": args_cli.mode,
                 "steps": int(actions.shape[0]),
                 "trace": rows, "replay_verdict": verdict,
                 "recorded_verdict": rec_verdict}
        if args_cli.seat_trace:
            entry["seat_trace"] = seat
        report.append(entry)

    if args_cli.seat_trace:
        with open(args_cli.seat_trace, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False)
        print(f"\n[probe] 스텝별 자취를 {args_cli.seat_trace}에 썼다", flush=True)
        # 자취까지 들어간 JSON은 한 편에 수만 줄이라 터미널로 내보내지 않는다.
        for item in report:
            item.pop("seat_trace", None)
    print("\n=== JSON ===", flush=True)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
