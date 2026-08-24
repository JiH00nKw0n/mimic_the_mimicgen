#!/usr/bin/env python3
"""사람이 조종해 녹화한 시연을 시뮬레이터에서 다시 재생해, 작업에 성공한 편만 남긴다.

왜 이 단계가 필요한가.

녹화 파일에는 매 시점의 관절 각도와 로봇에게 준 명령이 들어 있다. 사람이 실제로 작업을
해냈으니 그 파일은 성공한 시연이다. 그런데 같은 명령을 시뮬레이터에서 다시 실행한다고 해서
같은 결과가 나오지는 않는다. 녹화할 때와 재생할 때의 물리 계산이 조금씩 달라 손끝 위치가
서서히 어긋나기 때문이다. 어긋남이 쌓이면 큐브를 놓치거나 탑이 무너진다.

MimicGen은 시연의 명령을 잘라 새 장면에 옮겨 붙이는 방식으로 새 에피소드를 만든다. 그래서
재생조차 되지 않는 시연을 넣으면 옮겨 붙인 것도 당연히 실패한다. 이 단계가 그런 시연을
미리 걸러 낸다. 실제로 큐브 쌓기에서 사람 시연 50편 중 13편만 재생에 성공했다.

무엇을 하는가.

시연 한 편마다 이렇게 한다. 녹화된 첫 상태로 장면을 되돌리고, 녹화된 명령을 한 스텝씩
그대로 실행하고, 마지막 상태를 보고 작업에 성공했는지 판정한다. 성공한 편만 출력 파일에
쓴다. 판정 방법은 작업마다 다르므로 작업 설정 파일의 render.success 항목이 정한다.
큐브 쌓기는 세 개가 탑으로 섰는지 보고, 핀 꽂기는 핀이 구멍에 꽂혔는지 본다.

컨테이너 안에서 이렇게 돌린다.

    replay_filter.py --task <태스크 이름> --register <끼워 넣을 모듈들> \
        --input /work/prepare/raw.hdf5 --output /work/prepare/replayed.hdf5 \
        --report /work/prepare/replay_report.json --headless --device cpu

보통은 직접 부르지 않는다. `make prepare`가 이 파일과 어노테이션을 순서대로 부른다.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="시연을 다시 재생해 성공한 편만 남긴다")
parser.add_argument("--input", required=True,
                    help="재생할 시연 파일. 우리 장면 형식이어야 한다")
parser.add_argument("--output", required=True,
                    help="성공한 편만 담을 파일")
parser.add_argument("--report", default="",
                    help="편마다의 판정 결과를 적을 JSON 경로. 비우면 출력 파일 옆에 쓴다")
parser.add_argument("--task", required=True, help="Isaac에 등록된 환경 이름")
parser.add_argument("--register", default="",
                    help="gym.make 전에 불러올 모듈들. 쉼표로 잇는다")
parser.add_argument("--success-module", dest="success_module", default="success_criteria",
                    help="성공을 판정할 모듈 이름")
parser.add_argument("--success-function", dest="success_function", default="replay_verdict",
                    help="위 모듈에서 부를 함수 이름")
parser.add_argument("--objects", default="",
                    help="판정에 넘길 강체 이름들. 쉼표로 잇는다. 비우면 장면에 있는 것을 다 넘긴다")
parser.add_argument("--count", type=int, default=-1,
                    help="앞에서 몇 편만 재생할지. -1이면 전부")
parser.add_argument("--max-steps", dest="max_steps", type=int, default=3000,
                    help="한 편에 허용할 최대 스텝 수. 무한 재생을 막는다")
parser.add_argument("--settle-steps", dest="settle_steps", type=int, default=0,
                    help="판정 전에 물리만 더 돌릴 스텝 수. 물체가 멈춘 뒤 판정하고 싶을 때 쓴다")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = False
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym                                   # noqa: E402
import torch                                              # noqa: E402

from isaaclab.utils.datasets import HDF5DatasetFileHandler  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

# 태스크 등록 모듈과 성공 판정 모듈을 이름으로 불러올 수 있게 경로를 붙인다.
# 등록 모듈은 태스크마다 다른 폴더에 있다. 큐브는 src/env, 핀은 src/env_peg다.
# 어느 쪽인지 모르므로 있는 폴더를 전부 붙인다. 판정 모듈은 src/render에 있다.
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
for rel in ("render", "env", "env_peg", "sart", "convert", "."):
    path = os.path.normpath(os.path.join(SRC, rel))
    if os.path.isdir(path) and path not in sys.path:
        sys.path.insert(0, path)


def _import_first(candidates, name):
    """같은 이름이 판마다 다른 자리에 있어서, 후보 경로를 차례로 시도한다."""
    errors = []
    for module_path in candidates:
        try:
            return getattr(importlib.import_module(module_path), name)
        except (ImportError, AttributeError) as exc:
            errors.append(f"{module_path}: {exc}")
    raise SystemExit(f"{name}을 찾지 못했다. 시도한 경로: {'; '.join(errors)}")


ActionStateRecorderManagerCfg = _import_first(
    ["isaaclab.envs.mdp.recorders.recorders_cfg", "isaaclab.envs.mdp.recorders"],
    "ActionStateRecorderManagerCfg")
DatasetExportMode = _import_first(
    ["isaaclab.managers.recorder_manager", "isaaclab.managers"], "DatasetExportMode")


def load_verdict(module_name: str, function_name: str):
    """성공을 판정할 함수를 이름으로 찾아 온다.

    찾지 못하면 여기서 멈춘다. 조용히 건너뛰면 판정 없이 전부 통과시키거나 전부 버리게 되고,
    둘 다 나중에 원인을 찾기 어려운 실패가 된다.
    """
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise SystemExit(f"성공 판정 모듈 {module_name!r}을 불러오지 못했다: {exc}") from exc
    fn = getattr(module, function_name, None)
    if fn is None:
        raise SystemExit(f"{module_name}에 {function_name!r} 함수가 없다")
    return fn


def finger_indices(env) -> list:
    """그리퍼 손가락 관절의 자리 번호를 찾는다. 판정 함수가 이 값을 받는다."""
    names = env.scene["robot"].data.joint_names
    idx = [i for i, n in enumerate(names) if "finger" in n.lower()]
    if not idx:
        raise SystemExit(f"그리퍼 손가락 관절을 찾지 못했다. 있는 이름: {names}")
    return idx


def main() -> int:
    for name in [m.strip() for m in args.register.split(",") if m.strip()]:
        importlib.import_module(name)

    verdict_fn = load_verdict(args.success_module, args.success_function)

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    # 시간 초과로 에피소드가 끊기면 안 된다. 길이는 녹화된 명령의 개수가 정한다.
    if hasattr(env_cfg.terminations, "time_out"):
        env_cfg.terminations.time_out = None
    # 성공 종료 항을 들어낸다. 그대로 두면 성공한 순간 환경이 에피소드를 끝내고,
    # 그때 기록기가 버퍼를 파일로 내보내 버려 우리가 판정할 기회가 사라진다.
    if hasattr(env_cfg.terminations, "success"):
        env_cfg.terminations.success = None

    env_cfg.recorders = ActionStateRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = os.path.dirname(os.path.abspath(args.output))
    env_cfg.recorders.dataset_filename = os.path.basename(args.output)
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY

    env = gym.make(args.task, cfg=env_cfg).unwrapped
    ids = torch.tensor([0], device=env.device)
    env.reset()
    fingers = finger_indices(env)

    wanted = [n.strip() for n in args.objects.split(",") if n.strip()]

    # 입력 파일이 지금 장면과 같은 장면에서 녹화됐는지 먼저 본다.
    #
    # 재생은 녹화된 첫 상태로 장면을 되돌려 놓고 시작한다. 되돌리려면 지금 장면에 있는
    # 강체마다 기록된 자세가 파일에 있어야 한다. 수집 환경이 다르면 물체 이름부터 달라서
    # 되돌리기가 실패하는데, 그 오류만 보고는 원인을 알 수 없다. 예를 들어 핀 꽂기 원본은
    # insertive_object와 receptive_object라는 이름을 쓰고 우리 장면은 peg와 desk_surface를
    # 쓴다. 이런 파일은 재생 앞에 환경 변환을 거쳐야 한다.
    import h5py

    with h5py.File(args.input, "r") as fh:
        data = fh.get("data")
        if data is None or not len(data.keys()):
            raise SystemExit(f"{args.input}에 data 그룹이 없거나 비어 있다")
        first = sorted(data.keys())[0]
        states = data[first].get("states")
        recorded = set(states["rigid_object"].keys()) if (
            states is not None and "rigid_object" in states) else set()
        try:
            recorded_env = json.loads(data.attrs.get("env_args", "{}")).get("env_name", "")
        except (ValueError, TypeError):
            recorded_env = ""
    scene_rigid = set(getattr(env.scene, "rigid_objects", {}) or {})
    if recorded and scene_rigid and recorded != scene_rigid:
        raise SystemExit(
            f"입력 파일의 장면과 지금 장면이 다르다.\n"
            f"  파일이 녹화된 환경: {recorded_env or '적혀 있지 않다'}\n"
            f"  파일에 기록된 물체: {sorted(recorded)}\n"
            f"  지금 장면의 물체:   {sorted(scene_rigid)}\n"
            f"  지금 장면의 환경:   {args.task}\n"
            f"다른 환경에서 녹화된 파일은 재생하기 전에 우리 장면 형식으로 옮겨야 한다. "
            f"src/prepare/ 아래의 환경 변환 단계를 먼저 거쳐라.")

    handler = HDF5DatasetFileHandler()
    handler.open(args.input)
    names = list(handler.get_episode_names())
    if args.count >= 0:
        names = names[: args.count]
    if not names:
        raise SystemExit(f"{args.input}에 재생할 시연이 없다")
    print(f"[replay] {len(names)}편을 재생한다: {names[0]} 부터 {names[-1]} 까지", flush=True)

    report = {"input": args.input, "output": args.output, "task": args.task,
              "success_rule": f"{args.success_module}.{args.success_function}",
              "demos": {}}
    passed = 0
    with torch.inference_mode():
        for name in names:
            episode = handler.load_episode(name, env.device)
            env.recorder_manager.reset(env_ids=ids)
            env.reset_to(episode.get_initial_state(), ids, is_relative=True)

            steps = 0
            while steps < args.max_steps:
                action = episode.get_next_action()
                if action is None:
                    break
                env.step(action.unsqueeze(0) if action.ndim == 1 else action)
                steps += 1
            for _ in range(args.settle_steps):
                env.sim.step(render=False)
                env.scene.update(env.sim.get_physics_dt())

            # 판정에 넘길 물체 자세를 장면에서 읽는다. 이름을 지정하지 않으면 장면에 있는
            # 강체를 전부 넘긴다. 판정 함수가 자기가 쓸 것만 골라 쓴다.
            scene_rigid = getattr(env.scene, "rigid_objects", {}) or {}
            picked = wanted or list(scene_rigid.keys())
            objects = {}
            for obj_name in picked:
                if obj_name not in scene_rigid:
                    continue
                data = env.scene[obj_name].data
                pos = data.root_pos_w[0] - env.scene.env_origins[0]
                quat = data.root_quat_w[0]
                objects[obj_name] = [float(v) for v in pos] + [float(v) for v in quat]
            finger_pos = [float(v) for v in env.scene["robot"].data.joint_pos[0, fingers]]

            status = verdict_fn(objects, finger_pos)
            ok = bool(status["ok"]) if status else False
            passed += int(ok)

            env.recorder_manager.set_success_to_episodes(
                ids, torch.tensor([[ok]], dtype=torch.bool, device=env.device))
            env.recorder_manager.export_episodes(ids)

            report["demos"][name] = {"steps": steps, "ok": ok,
                                     "detail": (status or {}).get("attrs", {})}
            print(f"  {name}: {steps}스텝 재생, {'성공' if ok else '실패'}"
                  f"{'' if not status else ' ' + json.dumps((status or {}).get('attrs', {}), ensure_ascii=False)}",
                  flush=True)

    report["replayed"] = len(names)
    report["passed"] = passed
    report["pass_rate"] = round(passed / max(1, len(names)), 4)
    print(f"[replay] {len(names)}편 중 {passed}편이 재생에 성공했다 "
          f"({100.0 * passed / max(1, len(names)):.1f}%)", flush=True)

    report_path = args.report or (args.output + ".report.json")
    tmp = report_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    os.replace(tmp, report_path)

    env.close()
    # 한 편도 통과하지 못하면 다음 단계가 빈 파일을 들고 돈다. 여기서 알린다.
    return 0 if passed > 0 else 1


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
