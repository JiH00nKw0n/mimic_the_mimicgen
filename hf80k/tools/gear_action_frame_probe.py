"""상대 명령 하나를 넣으면 손끝이 어느 방향으로 얼마나 움직이는지 재 본다.

왜 필요한가. 2026-10-22 생성에서 손끝 목표는 기어와 받침 자리에 제대로 찍히는데
(x 0.14-0.23, z 0.74-0.90) 정작 팔은 x 1.3-1.6 m 자리를 헤매다 끝났다. 시연이
손끝을 x 0.089-0.413, z 0.743-1.209 안에서만 움직이므로 x 1.5 m는 팔이 닿을 수
있는 자리가 아니다. 즉 팔이 명령을 따라가지 못하는 것이 아니라 명령이 엉뚱하게
작용하고 있다. 후보가 둘이다.

  (1) 좌표계가 틀렸다. gear_mimic_env.target_eef_pose_to_action은 월드 기준 차이를
      로봇 받침 좌표계로 돌려서 내보낸다. 받침이 z축으로 180도 돌아가 있으므로 이
      회전은 (dx, dy, dz)를 (-dx, -dy, dz)로 바꾼다. 제어기가 이 명령을 월드 기준
      으로 해석한다면 팔은 x와 y에서 정반대로 간다.
  (2) 크기가 틀렸다. 제어기가 명령을 미터가 아닌 다른 단위로 읽으면 한 스텝에
      명령보다 몇 배 많이 움직인다. 생성 로그의 첫 다섯 스텝에서 x 명령은 -0.009 m
      였는데 손끝은 -0.075 m 움직였다.

그래서 축 하나씩 알려진 크기의 명령만 넣고 손끝이 실제로 간 거리를 잰다. 세 축의
결과를 받침 사원수와 맞춰 보면 (1)과 (2)가 각각 맞는지 바로 갈린다.

    isaaclab.sh -p tools/gear_action_frame_probe.py \
        --task Isaac-MultiGear-LabFR3-IK-Rel-Mimic-v0 --device cpu \
        --cmd 0.005 --steps 10
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", required=True)
parser.add_argument("--cmd", type=float, default=0.005,
                    help="한 스텝에 넣을 상대 위치 명령의 크기, 미터")
parser.add_argument("--steps", type=int, default=10,
                    help="한 축마다 같은 명령을 몇 스텝 넣을지")
parser.add_argument("--settle", type=int, default=30,
                    help="리셋 뒤 아무 명령 없이 몇 스텝 기다릴지")
parser.add_argument("--rot", type=float, default=0.02,
                    help="한 스텝에 넣을 상대 회전 명령의 크기, 라디안. 축 하나씩 "
                         "넣어 손끝이 어느 축으로 도는지 본다")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = False

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab.utils.math as PoseUtils  # noqa: E402

import gear_register  # noqa: E402,F401  태스크를 gym에 등록한다
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402


def eef_pos(env):
    """MimicGen이 쓰는 그 함수로 읽은 손끝 위치. 명령을 만드는 쪽과 같은 값이다."""
    name = list(env.cfg.subtask_configs.keys())[0]
    pose = env.get_robot_eef_pose(name, env_ids=[0])[0]
    pos, _ = PoseUtils.unmake_pose(pose)
    return pos.clone()


def eef_rot(env):
    """MimicGen이 쓰는 그 함수로 읽은 손끝 회전행렬. 명령을 만드는 쪽과 같은 값이다."""
    name = list(env.cfg.subtask_configs.keys())[0]
    pose = env.get_robot_eef_pose(name, env_ids=[0])[0]
    _, rot = PoseUtils.unmake_pose(pose)
    return rot.clone()


def hand_pos(env):
    """장면에서 직접 읽은 손 몸체 위치. 환경 기준이다. 위 값과 같은 자리를 가리키는지
    맞춰 보려고 따로 읽는다."""
    robot = env.scene["robot"]
    names = robot.data.body_names
    cand = [i for i, n in enumerate(names) if "hand" in n.lower()]
    if not cand:
        cand = [len(names) - 1]
    p = robot.data.body_pos_w[0, cand[0]] - env.scene.env_origins[0]
    return p.clone(), names[cand[0]]


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    dim = env.action_space.shape[1]
    zero = torch.zeros((1, dim), device=env.device)
    zero[0, -1] = 1.0                     # 손가락은 열어 둔다

    env.reset()
    # 리셋 직후부터 아무 명령 없이 몇 스텝을 세어 가며 본다. 팔이 집 자세에 가만히
    # 있어야 한다. 시연 20편의 첫 관절값이 모두 집 자세
    # [0, -0.7854, 0, -2.3562, 0, 1.5708, 0.7854]이고 그 자세의 손끝이 z 1.2089다.
    # 생성 로그의 첫 명령을 만들 때 손끝이 z 1.573에 있었으므로 그 사이에 팔이
    # 움직였다는 뜻이다. 어느 스텝에서 움직이는지 여기서 센다.
    robot = env.scene["robot"]
    jn = robot.data.joint_names
    # 설정에 적어 둔 초기 관절값이 실제로 관절에 꽂혔는지 본다. 세 자리를 구분해야
    # 한다. (1) 환경 설정이 들고 있는 값, (2) 물체가 들고 있는 기본값, (3) 리셋 직후
    # 관절에 들어가 있는 값. 셋 중 어디서 끊기는지에 따라 고칠 자리가 다르다.
    print("\n=== 초기 관절값이 어디까지 전해졌나 ===", flush=True)
    print("  관절 이름 순서", list(jn), flush=True)
    try:
        print("  환경 설정의 init_state.joint_pos",
              dict(robot.cfg.init_state.joint_pos), flush=True)
    except Exception as exc:                       # noqa: BLE001
        print("  환경 설정을 읽지 못했다", type(exc).__name__, exc, flush=True)
    print("  물체가 들고 있는 default_joint_pos",
          [round(float(v), 4) for v in robot.data.default_joint_pos[0]], flush=True)
    print("  시연 20편의 첫 관절값 "
          "[0.0, -0.7854, 0.0, -2.3562, 0.0, 1.5708, 0.7854, 0.04, 0.04]", flush=True)
    arm = [i for i, n in enumerate(jn) if "finger" not in n.lower()]
    print("\n=== 리셋 직후부터 영 명령으로 지켜본다 ===", flush=True)
    for k in range(args_cli.settle + 1):
        if k in (0, 1, 2, 3, 5, 10, 20, args_cli.settle):
            jp = [round(float(robot.data.joint_pos[0, i]), 4) for i in arm]
            hp0, _ = hand_pos(env)
            print(f"  {k:3d}스텝  손끝 "
                  f"{[round(float(v), 4) for v in eef_pos(env)]}  손 몸체 "
                  f"{[round(float(v), 4) for v in hp0]}  관절 {jp}", flush=True)
        if k < args_cli.settle:
            env.step(zero)

    root_quat = robot.data.root_quat_w[0]
    root_pos = robot.data.root_pos_w[0] - env.scene.env_origins[0]
    hp, hname = hand_pos(env)
    print("\n=== 리셋하고 %d스텝 기다린 뒤 ===" % args_cli.settle, flush=True)
    print("  로봇 받침 위치(환경 기준) "
          f"{[round(float(v), 4) for v in root_pos]}", flush=True)
    print("  로봇 받침 사원수(읽은 순서 그대로, XYZW) "
          f"{[round(float(v), 4) for v in root_quat]}", flush=True)
    print(f"  get_robot_eef_pose가 준 손끝 위치 "
          f"{[round(float(v), 4) for v in eef_pos(env)]}", flush=True)
    print(f"  장면에서 읽은 '{hname}' 몸체 위치 "
          f"{[round(float(v), 4) for v in hp]}", flush=True)
    print("  시연 demo_0의 손끝은 x 0.089-0.413, y -0.218-0.138, z 0.743-1.209 "
          "안에서만 움직였다.", flush=True)

    axes = (("x", 0), ("y", 1), ("z", 2))
    for label, k in axes:
        env.reset()
        for _ in range(args_cli.settle):
            env.step(zero)
        before = eef_pos(env)
        act = zero.clone()
        act[0, k] = args_cli.cmd
        for _ in range(args_cli.steps):
            env.step(act)
        after = eef_pos(env)
        moved = after - before
        cmd_vec = torch.zeros(3, device=env.device)
        cmd_vec[k] = args_cli.cmd * args_cli.steps
        # 두 가설이 예측하는 변위
        as_world = cmd_vec
        as_base = PoseUtils.quat_apply(root_quat.reshape(1, 4),
                                       cmd_vec.reshape(1, 3))[0]
        print(f"\n--- {label}축에만 {args_cli.cmd:+.4f}를 {args_cli.steps}스텝 "
              f"넣었다 (명령 합계 {float(cmd_vec[k]):+.4f} m) ---", flush=True)
        print(f"  손끝이 실제로 간 거리 "
              f"{[round(float(v), 4) for v in moved]}  크기 "
              f"{float(torch.linalg.vector_norm(moved)):.4f} m", flush=True)
        print("  제어기가 명령을 월드 기준으로 읽는다면 "
              f"{[round(float(v), 4) for v in as_world]}", flush=True)
        print("  제어기가 명령을 받침 기준으로 읽는다면 "
              f"{[round(float(v), 4) for v in as_base]}", flush=True)
        gain = (float(torch.linalg.vector_norm(moved))
                / max(float(torch.linalg.vector_norm(cmd_vec)), 1e-9))
        print(f"  간 거리 / 명령 크기 = {gain:.3f} 배", flush=True)

    # ----------------------------------------------------------------- 회전 명령
    # 위치는 받침 좌표계로 들어가는 것을 이미 확인했다. 회전도 같은지는 따로 봐야 한다.
    # 재생에서 중간 기어가 축에 가로세로로는 0.26 mm까지 맞으면서 6도에서 12도 기울어
    # 걸리는 것을 보았고, 손가락은 그 동안 계속 닫혀 있었다. 쥔 물체가 기울어진다는
    # 것은 손끝 자체가 기울어졌다는 뜻이므로 회전 명령이 어떻게 작용하는지 재야 한다.
    print("\n\n=== 회전 명령은 어느 축으로 도나 ===", flush=True)
    for label, k in axes:
        env.reset()
        for _ in range(args_cli.settle):
            env.step(zero)
        before = eef_rot(env)
        act = zero.clone()
        act[0, 3 + k] = args_cli.rot
        for _ in range(args_cli.steps):
            env.step(act)
        after = eef_rot(env)
        # 실제로 일어난 회전을 월드 기준 축각으로 바꾼다
        done_mat = after.matmul(before.transpose(-1, -2))
        done = PoseUtils.axis_angle_from_quat(PoseUtils.quat_from_matrix(done_mat))
        cmd_vec = torch.zeros(3, device=env.device)
        cmd_vec[k] = args_cli.rot * args_cli.steps
        as_world = cmd_vec
        as_base = PoseUtils.quat_apply(root_quat.reshape(1, 4),
                                       cmd_vec.reshape(1, 3))[0]
        print(f"\n--- 회전 {label}축에만 {args_cli.rot:+.4f} rad을 "
              f"{args_cli.steps}스텝 넣었다 (합계 {float(cmd_vec[k]):+.4f} rad) ---",
              flush=True)
        print(f"  손끝이 실제로 돈 축각(월드 기준) "
              f"{[round(float(v), 4) for v in done]}  크기 "
              f"{float(torch.linalg.vector_norm(done)):.4f} rad", flush=True)
        print("  제어기가 회전 명령을 월드 기준으로 읽는다면 "
              f"{[round(float(v), 4) for v in as_world]}", flush=True)
        print("  제어기가 회전 명령을 받침 기준으로 읽는다면 "
              f"{[round(float(v), 4) for v in as_base]}", flush=True)
        gain = (float(torch.linalg.vector_norm(done))
                / max(float(torch.linalg.vector_norm(cmd_vec)), 1e-9))
        print(f"  돈 각 / 명령 각 = {gain:.3f} 배", flush=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
