"""무작위 배치로 리셋한 직후 장면이 멀쩡한지 본다. 아무 명령도 넣지 않고 지켜본다.

왜 필요한가. 생성 시도의 70%가 기어를 받침 높이에서 1.74 m 떨어진 자리에 남긴다.
그 자리는 책상도 바닥도 아니다. 그런데 기록된 배치를 그대로 되돌리는 재생에서는
그런 일이 없다. 둘의 차이는 리셋 때 물체를 무작위로 옮기는 이벤트 하나뿐이다.

그래서 리셋만 걸고 로봇에 아무 명령도 주지 않은 채 몇 스텝을 지켜본다. 기어가
가만히 있으면 리셋은 멀쩡하고 시도 중에 망가지는 것이다. 리셋 직후부터 떨어지거나
솟으면 무작위 배치가 기어를 받침이나 책상 안에 박아 넣은 것이다.

    isaaclab.sh -p tools/gear_reset_probe.py \
        --task Isaac-MultiGear-LabFR3-IK-Rel-Mimic-v0 --device cpu --resets 5
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", required=True)
parser.add_argument("--resets", type=int, default=5, help="리셋을 몇 번 볼지")
parser.add_argument("--steps", type=int, default=60, help="리셋 뒤 몇 스텝을 지켜볼지")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = False

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import gear_register  # noqa: E402,F401  태스크를 gym에 등록한다
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

from gear_geom import GEARS, tilt_rad_from_quat  # noqa: E402


def read(env, name):
    """환경 기준 위치와 속도 크기."""
    obj = env.scene[name]
    pos = (obj.data.root_pos_w - env.scene.env_origins)[0]
    vel = obj.data.root_vel_w
    if vel.ndim == 1 or vel.shape[-1] < 3:
        vel = vel.reshape(-1, 6)
    return pos, float(torch.linalg.vector_norm(vel[0, :3]))


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    # 아무것도 하지 않는 명령. 손끝 상대 자세 6개와 손가락 1개다.
    zero = torch.zeros((1, env.action_space.shape[1]), device=env.device)

    for trial in range(args_cli.resets):
        env.reset()
        print(f"\n=== 리셋 {trial} ===", flush=True)
        bp, _ = read(env, "base")
        print(f"  받침 z {float(bp[2]):.4f}  xy "
              f"({float(bp[0]):.4f}, {float(bp[1]):.4f})", flush=True)
        # 뽑힌 자리와 서로의 거리. 기어가 튀어 나가는 것이 겹쳐 놓인 탓인지 가린다.
        # 무작위 배치는 기어끼리 0.126 m, 기어와 받침 0.091 m를 지키게 되어 있다.
        xy = {g: read(env, g)[0] for g in GEARS}
        for g in GEARS:
            print(f"    {g:7s} xy ({float(xy[g][0]):.4f}, {float(xy[g][1]):.4f})  "
                  f"받침까지 {float(torch.linalg.vector_norm(xy[g][:2] - bp[:2])):.4f}",
                  flush=True)
        # 리셋 직후의 기울기. 기어가 눕혀져 있으면 가만히 두어도 쓰러진다. 무작위
        # 배치가 기어에 써넣는 사원수의 성분 순서가 틀리면 z축 회전이 x축 회전으로
        # 읽혀 기어가 90도 눕는다. 그 경우를 여기서 바로 가린다.
        for g in GEARS:
            q = env.scene[g].data.root_quat_w[0]
            print(f"    {g:7s} 사원수(읽은 순서 그대로) "
                  f"{[round(float(v), 4) for v in q]}  기울기 "
                  f"{float(tilt_rad_from_quat(q.reshape(1, 4))[0]):.4f} rad", flush=True)
        pairs = [(a, b) for i, a in enumerate(GEARS) for b in GEARS[i + 1:]]
        print("    기어끼리 " + ", ".join(
            f"{a}-{b} {float(torch.linalg.vector_norm(xy[a][:2] - xy[b][:2])):.4f}"
            for a, b in pairs), flush=True)
        watch = sorted({0, 1, 5, 20, args_cli.steps})
        done = 0
        for target in watch:
            while done < target:
                env.step(zero)
                done += 1
            line = [f"  {done:4d}스텝"]
            for g in GEARS:
                p, sp = read(env, g)
                line.append(f"{g} z={float(p[2]):.4f} 속도={sp:.3f}")
            print("  ".join(line), flush=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
