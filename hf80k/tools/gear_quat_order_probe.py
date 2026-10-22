"""기어 자세를 써넣을 때 사원수 성분을 어느 순서로 넣어야 하는지 실험으로 정한다.

왜 필요한가. 읽는 쪽(`.data.root_quat_w`)과 쓰는 쪽(`write_root_pose_to_sim`)의 성분
순서가 같다는 보장이 없고, 문서와 실제가 어긋난 적이 있다. 순서를 틀리면 z축 20도
회전이 x축 160도 회전으로 들어가 기어가 뒤집히고, 뒤집힌 기어는 작업면 안에 박혀
튕겨 나간다. 무작위 배치에서 기어 셋의 69%가 바닥 아래로 떨어진 일이 그것이다.

무엇을 하는가. 같은 z축 회전 하나를 두 순서로 써넣고 각각 결과를 본다.

  wxyz 순서로 넣기   (cos, 0, 0, sin)
  xyzw 순서로 넣기   (0, 0, sin, cos)

그리고 세 가지를 찍는다. 써넣은 직후 읽힌 사원수, 기울기, 그리고 30스텝 뒤의 높이와
속도다. 옳은 순서는 기울기가 0에 가깝고 높이가 그대로이며 속도가 0인 쪽이다.

    isaaclab.sh -p tools/gear_quat_order_probe.py \
        --task Isaac-MultiGear-LabFR3-IK-Rel-Mimic-v0 --device cpu --yaw-deg 20
"""

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", required=True)
parser.add_argument("--yaw-deg", type=float, default=20.0, help="시험에 쓸 z축 회전 각도")
parser.add_argument("--steps", type=int, default=30, help="써넣은 뒤 몇 스텝을 지켜볼지")
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


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    zero = torch.zeros((1, env.action_space.shape[1]), device=env.device)
    dev = env.device
    half = math.radians(args_cli.yaw_deg) * 0.5

    orders = {
        "wxyz": [math.cos(half), 0.0, 0.0, math.sin(half)],
        "xyzw": [0.0, 0.0, math.sin(half), math.cos(half)],
    }
    print(f"\n시험 각도 {args_cli.yaw_deg}도를 z축 회전으로 써넣는다", flush=True)

    for label, comps in orders.items():
        env.reset()
        quat = torch.tensor([comps], device=dev, dtype=torch.float32)
        print(f"\n=== {label} 순서로 써넣기: {[round(c, 4) for c in comps]} ===", flush=True)
        for name in GEARS:
            obj = env.scene[name]
            pos = obj.data.root_pos_w[:1].clone()
            obj.write_root_pose_to_sim(torch.cat([pos, quat], dim=1), env_ids=[0])
            obj.write_root_velocity_to_sim(torch.zeros((1, 6), device=dev), env_ids=[0])
        for name in GEARS:
            q = env.scene[name].data.root_quat_w[0]
            print(f"  {name:7s} 읽힌 사원수 {[round(float(v), 4) for v in q]}  "
                  f"기울기 {float(tilt_rad_from_quat(q.reshape(1, 4))[0]):.4f} rad",
                  flush=True)
        for _ in range(args_cli.steps):
            env.step(zero)
        for name in GEARS:
            obj = env.scene[name]
            z = float((obj.data.root_pos_w - env.scene.env_origins)[0, 2])
            vel = obj.data.root_vel_w.reshape(-1, 6)
            sp = float(torch.linalg.vector_norm(vel[0, :3]))
            q = env.scene[name].data.root_quat_w[0]
            print(f"  {name:7s} {args_cli.steps}스텝 뒤 z={z:.4f} 속도={sp:.3f} "
                  f"기울기 {float(tilt_rad_from_quat(q.reshape(1, 4))[0]):.4f} rad",
                  flush=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
