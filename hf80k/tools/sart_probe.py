#!/usr/bin/env python3
"""SART 실행기를 쓰기 전에 컨테이너 안의 함수 서명을 눈으로 확인하는 탐색기.

왜 이것부터 돌리는가. SART 실행기는 Isaac Lab이 이미 갖고 있는 웨이포인트 도구를 그대로
쓴다. 그런데 이 컨테이너의 판이 어떤 이름과 어떤 인자를 갖고 있는지는 원문을 읽어서만
알 수 있고, 원문을 잘못 읽으면 시뮬레이터를 띄운 뒤에야 오류가 난다. 시뮬레이터를 한 번
띄우는 데 60초에서 90초가 든다. 이 파일은 시뮬레이터를 띄우지 않고 모듈만 불러서 필요한
정보를 한 번에 찍는다. 5분이면 끝난다.

세 가지를 여기서 정한다. 웨이포인트 묶음에서 낱개를 꺼내는 방법, 환경 설정을 도와주는
함수를 그대로 부를 수 있는지, 그리고 보간 함수가 스텝 수를 어떻게 세는지다.

컨테이너 안에서 이렇게 돌린다.

    ssh arpa-l40s
    docker run --rm -v /host/tools/sart_probe.py:/tmp/f.py \
      --entrypoint /workspace/isaaclab/_isaac_sim/python.sh fr3-hf80k:12 /tmp/f.py
"""
from __future__ import annotations

import inspect
import sys


def line(text: str = "") -> None:
    print(text, flush=True)


def show(label: str, obj) -> None:
    try:
        line(f"  {label}{inspect.signature(obj)}")
    except (TypeError, ValueError) as exc:
        line(f"  {label} 서명을 읽지 못했다: {exc}")


def main() -> int:
    line("=" * 72)
    line("SART 탐색기. 시뮬레이터를 띄우지 않고 모듈만 불러서 확인한다.")
    line("=" * 72)

    try:
        from isaaclab_mimic.datagen.waypoint import (MultiWaypoint, Waypoint,
                                                     WaypointSequence, WaypointTrajectory)
    except Exception as exc:                       # noqa: BLE001
        line(f"실패: isaaclab_mimic.datagen.waypoint를 불러오지 못했다: {exc!r}")
        return 1
    line("불러오기 성공 (시뮬레이터 없이 들어왔다)")

    line("\n[1] 웨이포인트 묶음의 속성. 낱개를 꺼내는 이름을 여기서 고른다.")
    line("  Waypoint 멤버:           " + str([m for m in dir(Waypoint) if not m.startswith("_")]))
    line("  WaypointSequence 멤버:   " + str([m for m in dir(WaypointSequence) if not m.startswith("_")]))
    line("  WaypointTrajectory 멤버: " + str([m for m in dir(WaypointTrajectory) if not m.startswith("_")]))
    line("  WaypointTrajectory에 execute가 있는가: " + str(hasattr(WaypointTrajectory, "execute")))

    line("\n[2] 실행기가 부를 함수들의 서명")
    show("WaypointSequence.from_poses", WaypointSequence.from_poses)
    show("WaypointTrajectory.add_waypoint_sequence", WaypointTrajectory.add_waypoint_sequence)
    show("WaypointTrajectory.add_waypoint_sequence_for_target_pose",
         WaypointTrajectory.add_waypoint_sequence_for_target_pose)
    show("WaypointTrajectory.get_full_sequence", WaypointTrajectory.get_full_sequence)
    show("MultiWaypoint.execute", MultiWaypoint.execute)
    line("  MultiWaypoint.execute가 코루틴인가: "
         + str(inspect.iscoroutinefunction(MultiWaypoint.execute)))

    line("\n[3] 환경과 기록기 쪽 서명")
    try:
        from isaaclab.envs.manager_based_env import ManagerBasedEnv
        show("ManagerBasedEnv.reset_to", ManagerBasedEnv.reset_to)
    except Exception as exc:                       # noqa: BLE001
        line(f"  ManagerBasedEnv를 불러오지 못했다: {exc!r}")
    try:
        from isaaclab.utils.datasets import HDF5DatasetFileHandler
        show("HDF5DatasetFileHandler.open", HDF5DatasetFileHandler.open)
        show("HDF5DatasetFileHandler.load_episode", HDF5DatasetFileHandler.load_episode)
    except Exception as exc:                       # noqa: BLE001
        line(f"  HDF5DatasetFileHandler를 불러오지 못했다: {exc!r}")
    try:
        from isaaclab.managers.recorder_manager import RecorderManager
        show("RecorderManager.reset", RecorderManager.reset)
        show("RecorderManager.set_success_to_episodes", RecorderManager.set_success_to_episodes)
        show("RecorderManager.export_episodes", RecorderManager.export_episodes)
    except Exception as exc:                       # noqa: BLE001
        line(f"  RecorderManager를 불러오지 못했다: {exc!r}")
    try:
        from isaaclab_mimic.datagen import generation as mimic_generation
        show("generation.setup_env_config", mimic_generation.setup_env_config)
    except Exception as exc:                       # noqa: BLE001
        line(f"  generation.setup_env_config를 불러오지 못했다: {exc!r}")

    line("\n[4] 보간이 만드는 웨이포인트 개수. num_steps=10으로 재 본다.")
    try:
        import numpy as np
        import torch

        start = torch.eye(4, dtype=torch.float32)
        goal = torch.eye(4, dtype=torch.float32)
        goal[0, 3] = 0.1
        grip = torch.zeros(1, dtype=torch.float32)
        traj = WaypointTrajectory()
        traj.add_waypoint_sequence(WaypointSequence.from_poses(
            start.unsqueeze(0), grip.unsqueeze(0), 0.0))
        traj.add_waypoint_sequence_for_target_pose(
            pose=goal, gripper_action=grip, num_steps=10, action_noise=0.0)
        seq = traj.get_full_sequence()
        for attempt in ("list(seq)", "seq.sequence", "seq.waypoints"):
            try:
                if attempt == "list(seq)":
                    items = list(seq)
                else:
                    items = list(getattr(seq, attempt.split(".")[1]))
                line(f"  {attempt} 로 {len(items)}개를 꺼냈다 "
                     f"(첫 구간 1개 + num_steps=10이므로 11이면 시작 자세를 뺀 것이다)")
                break
            except Exception as exc:               # noqa: BLE001
                line(f"  {attempt} 실패: {exc!r}")
        else:
            line("  낱개를 꺼내는 방법을 찾지 못했다. dir(seq) = " + str(dir(seq)))
        line("  numpy %s / torch %s" % (np.__version__, torch.__version__))
    except Exception as exc:                       # noqa: BLE001
        line(f"  개수 확인에 실패했다: {exc!r}")

    line("\n끝났다. 위 네 항목을 읽고 src/sart/sart_augment.py의 전제와 맞는지 확인한다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
