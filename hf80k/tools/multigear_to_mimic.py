#!/usr/bin/env python3
"""기어 조립 시연을 MimicGen이 읽는 형식으로 바꾼다.

왜 필요한가. 기어 조립 시연 20편은 잘 만들어져 있지만 MimicGen에 그대로 넣을 수 없다.
세 가지가 어긋난다.

첫째, 구간 경계 표시가 없다. MimicGen은 시연마다 "여기까지가 집는 구간, 여기부터가
꽂는 구간"이라는 표시가 obs/datagen_info/subtask_term_signals 자리에 있어야 구간별로
궤적을 옮겨 붙인다. 기어 시연에는 그 자리가 아예 없다.

둘째, 동작 명령의 형식이 다르다. 기어 시연은 관절 일곱 개의 절대 목표 각도에 손가락
열림을 더한 8차원이다. MimicGen은 손끝 자세를 물체 기준으로 옮겨 다시 쓰므로 손끝
좌표계의 상대 명령 7차원이 필요하다.

셋째, 기록 주기가 다르다. 시연은 초당 120스텝이고 생성은 초당 20스텝이다.

무엇을 하는가. 위 셋을 한 번에 바꿔 핀 꽂기의 peg_annotated.hdf5와 같은 구조의 파일을
만든다. 그 구조는 실제 파일을 열어 확인한 것이다.

구간을 어떻게 자르는가. 시연에는 스텝마다 phase_id가 붙어 있고 단계 이름이 따로
남아 있다. 큰 기어를 집고 꽂고, 작은 기어를 집고 꽂고, 중간 기어를 집고 꽂는 순서라서
구간이 여섯 개 나온다. 경계는 단계 이름으로 찾는다. 번호로 찾으면 안 된다. 데모마다
단계 개수가 32개에서 40개까지 다르기 때문이다. 중간 기어를 눌러 앉히는 두드림이
시연마다 두 번에서 네 번까지 다르고, 작은 기어가 톱니를 못 맞추면 탐색 단계가 더 붙는다.

마진을 왜 두는가. 경계를 단계가 바뀌는 바로 그 스텝에 두면 움직임 중간에서 잘린다.
손가락이 막 닫힌 순간은 아직 쥔 것이 안정되지 않았고, 꽂은 직후는 기어가 아직 자리를
잡는 중이다. 그래서 경계를 조금 뒤로 미룬다. 신호는 걸쇠형이라 한 번 켜지면 끝까지
켜져 있고, 그래서 한 구간의 끝 마진이 다음 구간의 시작 마진과 같은 값이 된다.

쓰는 법.

    python3 tools/multigear_to_mimic.py \
        --input  /work/multigear/source/multigear_20_success_only.hdf5 \
        --episodes /work/multigear/source/episodes \
        --output /work/multigear/multigear_annotated.hdf5

마진을 바꾸려면 --grasp-margin-s와 --seat-margin-s를 준다. 둘 다 초 단위다.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import h5py
import numpy as np

# 생성 환경의 기록 주기. 물리 주기 0.01초에 데시메이션 5이므로 초당 20스텝이다.
TARGET_HZ = 20.0
# 시연이 기록된 주기. 파일의 timestamps 차이가 1/120이다.
SOURCE_HZ = 120.0

# 조립 순서. 시연이 이 순서로만 만들어져 있다.
GEARS = ("large", "small", "medium")

# 로봇 받침의 자리와 방향. legacy_multi_gear/config/scene_layout.json의 값이고
# 우리 핀 꽂기 환경의 ROBOT_POS/ROBOT_ROT와 한 자리까지 같다. 사원수는 xyzw 순서이고
# z축 180도 회전이다. 이 순서는 peg_annotated.hdf5의 root_pose를 열어 명령과 맞춰 보며
# 확인했다. wxyz로 해석하면 명령이 재현되지 않는다.
ROBOT_POS = (0.72, 0.138, 0.722)
ROBOT_QUAT_XYZW = (0.0, 0.0, 1.0, 0.0)

# 손가락 목표가 이 값 이상이면 열라는 뜻으로 본다, 미터. 시연은 열 때 0.04를 주고
# 닫을 때 기어마다 0.0078에서 0.0178 사이를 준다. 그 사이를 가르는 값이다.
GRIP_OPEN_THRESHOLD_M = 0.035

# 이진 그리퍼 명령의 부호. Isaac Lab의 BinaryJointPositionAction이 음수를 닫기로,
# 0 이상을 열기로 읽는다. 컨테이너의 binary_joint_actions.py에서 확인했다.
GRIP_OPEN = 1.0
GRIP_CLOSE = -1.0

# IK 상대 제어에서 한 스텝에 허용하는 이동량의 배율. 핀 꽂기가 쓰는 1.0과 같게 둔다.
# 기어 시연의 한 스텝 이동량은 평균 2.9 mm, 최대 15.3 mm로 핀 꽂기의 13.6 mm보다
# 작아서 배율을 올릴 이유가 없다.
ARM_SCALE = 1.0


def quat_xyzw_to_matrix(q: np.ndarray) -> np.ndarray:
    """xyzw 순서의 사원수를 3x3 회전 행렬로 바꾼다. 입력 (..., 4), 출력 (..., 3, 3)."""
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    m = np.empty(q.shape[:-1] + (3, 3), dtype=np.float64)
    m[..., 0, 0] = 1 - 2 * (y * y + z * z)
    m[..., 0, 1] = 2 * (x * y - z * w)
    m[..., 0, 2] = 2 * (x * z + y * w)
    m[..., 1, 0] = 2 * (x * y + z * w)
    m[..., 1, 1] = 1 - 2 * (x * x + z * z)
    m[..., 1, 2] = 2 * (y * z - x * w)
    m[..., 2, 0] = 2 * (x * z - y * w)
    m[..., 2, 1] = 2 * (y * z + x * w)
    m[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return m


def make_pose(pos: np.ndarray, rot: np.ndarray) -> np.ndarray:
    """위치 (..., 3)과 회전 행렬 (..., 3, 3)을 4x4 동차 자세 행렬로 묶는다."""
    n = pos.shape[0]
    out = np.zeros((n, 4, 4), dtype=np.float32)
    out[:, :3, :3] = rot
    out[:, :3, 3] = pos
    out[:, 3, 3] = 1.0
    return out


def quat_wxyz_from_matrix(m: np.ndarray) -> np.ndarray:
    """회전 행렬 (..., 3, 3)을 wxyz 순서의 사원수 (..., 4)로 바꾼다.

    네 경우로 나누는 이유는 정밀도다. 대각 성분 중 가장 큰 쪽을 분모로 쓰면 0으로
    나누는 일이 없다. 한 식만 쓰면 특정 회전에서 분모가 0에 가까워져 값이 튄다.
    """
    m = np.asarray(m, dtype=np.float64)
    shape = m.shape[:-2]
    m = m.reshape(-1, 3, 3)
    n = m.shape[0]
    q = np.empty((n, 4), dtype=np.float64)
    trace = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]

    big_w = trace > 0.0
    if big_w.any():
        mm = m[big_w]
        s = np.sqrt(trace[big_w] + 1.0) * 2.0
        q[big_w, 0] = 0.25 * s
        q[big_w, 1] = (mm[:, 2, 1] - mm[:, 1, 2]) / s
        q[big_w, 2] = (mm[:, 0, 2] - mm[:, 2, 0]) / s
        q[big_w, 3] = (mm[:, 1, 0] - mm[:, 0, 1]) / s

    rest = ~big_w
    if rest.any():
        idx = np.flatnonzero(rest)
        for i in idx:
            d = np.array([m[i, 0, 0], m[i, 1, 1], m[i, 2, 2]])
            k = int(np.argmax(d))
            if k == 0:
                s = np.sqrt(1.0 + m[i, 0, 0] - m[i, 1, 1] - m[i, 2, 2]) * 2.0
                q[i] = [(m[i, 2, 1] - m[i, 1, 2]) / s, 0.25 * s,
                        (m[i, 0, 1] + m[i, 1, 0]) / s, (m[i, 0, 2] + m[i, 2, 0]) / s]
            elif k == 1:
                s = np.sqrt(1.0 + m[i, 1, 1] - m[i, 0, 0] - m[i, 2, 2]) * 2.0
                q[i] = [(m[i, 0, 2] - m[i, 2, 0]) / s, (m[i, 0, 1] + m[i, 1, 0]) / s,
                        0.25 * s, (m[i, 1, 2] + m[i, 2, 1]) / s]
            else:
                s = np.sqrt(1.0 + m[i, 2, 2] - m[i, 0, 0] - m[i, 1, 1]) * 2.0
                q[i] = [(m[i, 1, 0] - m[i, 0, 1]) / s, (m[i, 0, 2] + m[i, 2, 0]) / s,
                        (m[i, 1, 2] + m[i, 2, 1]) / s, 0.25 * s]

    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    return q.reshape(shape + (4,))


def axis_angle_from_matrix(m: np.ndarray) -> np.ndarray:
    """회전 행렬 (..., 3, 3)에서 축각 벡터 (..., 3)을 뽑는다.

    축각 벡터란 회전축 방향의 단위벡터에 회전각을 곱한 값이다. 크기가 각이고 방향이
    축이라서 세 숫자로 회전 하나를 적을 수 있다. IK 상대 명령의 회전 성분이 이 형식이다.

    행렬에서 바로 arccos로 각을 구하면 안 된다. 각이 0에 가까울 때 arccos의 기울기가
    무한으로 가서 입력의 작은 오차가 크게 번진다. 실제로 그렇게 짜서 핀 꽂기의 기록된
    명령과 5.7e-4 rad 어긋났다. 사원수를 거쳐 atan2로 구하면 작은 각에서도 안정하다.
    기준이 되는 Isaac Lab의 코드도 사원수를 거친다.
    """
    q = quat_wxyz_from_matrix(m)
    # 사원수는 q와 -q가 같은 회전이다. w가 음수면 뒤집어 각이 180도를 넘지 않게 한다.
    q = np.where(q[..., :1] < 0.0, -q, q)
    w = np.clip(q[..., 0], -1.0, 1.0)
    vec = q[..., 1:]
    norm = np.linalg.norm(vec, axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(norm[..., 0], w)
    out = np.zeros(q.shape[:-1] + (3,), dtype=np.float64)
    safe = norm[..., 0] > 1e-12
    out[safe] = vec[safe] / norm[safe] * angle[safe, None]
    return out


def quat_apply_inverse_xyzw(q_xyzw, v: np.ndarray) -> np.ndarray:
    """사원수 q의 역회전을 벡터 v에 적용한다. q는 xyzw 하나, v는 (..., 3)."""
    rot = quat_xyzw_to_matrix(np.asarray(q_xyzw, dtype=np.float64))
    return v @ rot          # 전치 행렬을 곱하는 것과 같다


def phase_bounds(phase_id: np.ndarray, names: list[str], target: str):
    """단계 이름 target이 차지하는 구간의 첫 스텝과 마지막 스텝을 돌려준다.

    없으면 (None, None)을 돌려준다. 같은 이름이 떨어져 두 번 나오면 마지막 덩어리를
    쓴다. 두드림처럼 되풀이되는 단계가 있어서다.
    """
    if target not in names:
        return None, None
    idx = names.index(target)
    hit = np.flatnonzero(phase_id == idx)
    if hit.size == 0:
        return None, None
    return int(hit[0]), int(hit[-1])


def subtask_boundaries(phase_id: np.ndarray, names: list[str],
                       grasp_margin: int, seat_margin: int, n: int) -> dict:
    """구간 여섯 개의 끝 스텝을 단계 이름에서 찾는다. 120 Hz 기준 스텝 번호다.

    집기 구간은 손가락이 닫히는 단계가 끝난 뒤 grasp_margin만큼 더 간 자리에서 끝낸다.
    쥔 것이 안정될 시간을 준다.

    꽂기 구간은 손을 빼는 단계가 끝난 뒤 seat_margin만큼 더 간 자리에서 끝낸다.
    손을 빼는 동작을 꽂기 구간에 넣는 이유는, 그 동작이 이미 꽂아 둔 기어를 건드리지
    않도록 받침 모양에 맞춰 나와야 하기 때문이다. 다음 구간은 손이 어디 있든 거기서
    다음 기어로 간다.
    """
    out = {}
    for gear in GEARS:
        _, close_end = phase_bounds(phase_id, names, f"{gear}: close fingers")
        if close_end is None:
            raise ValueError(f"'{gear}: close fingers' 단계가 없다")
        out[f"grasp_{gear}"] = min(close_end + grasp_margin, n - 1)

        _, retreat_end = phase_bounds(phase_id, names, f"{gear}: retreat")
        if retreat_end is None:
            # 손을 빼는 단계가 없으면 놓는 단계의 끝을 쓴다.
            _, retreat_end = phase_bounds(phase_id, names, f"{gear}: release")
        if retreat_end is None:
            raise ValueError(f"'{gear}: retreat'도 '{gear}: release'도 없다")
        out[f"seat_{gear}"] = min(retreat_end + seat_margin, n - 1)
    return out


def convert_demo(src, npz_path: str, grasp_margin: int, seat_margin: int, stride: int):
    """시연 한 편을 바꾼다. 돌려주는 값은 그룹에 쓸 사전이다."""
    names = np.load(npz_path, allow_pickle=True)["phases"].tolist()
    phase_id = src["phase_id"][:].astype(np.int64)
    n_src = phase_id.shape[0]

    bounds = subtask_boundaries(phase_id, names, grasp_margin, seat_margin, n_src)

    # 120 Hz를 20 Hz로 줄인다. 마지막 스텝은 성공이 확인된 자리라서 반드시 남긴다.
    keep = np.arange(0, n_src, stride)
    if keep[-1] != n_src - 1:
        keep = np.append(keep, n_src - 1)
    n = keep.shape[0]

    eef_pos = src["obs/robot_eef_pos"][:][keep].astype(np.float64)
    eef_quat = src["obs/robot_eef_quat"][:][keep].astype(np.float64)   # xyzw
    eef_rot = quat_xyzw_to_matrix(eef_quat)
    eef_pose = make_pose(eef_pos.astype(np.float32), eef_rot.astype(np.float32))

    # 목표 자세는 다음 스텝에서 실제로 도달한 자세로 둔다. 시연에 명령된 손끝 목표가
    # 스텝마다 저장돼 있지 않기 때문이다. 이렇게 두면 아래에서 만드는 상대 명령과
    # 서로 어긋나지 않는다. 마지막 스텝은 더 갈 곳이 없으므로 제자리로 둔다.
    tgt_pos = np.vstack([eef_pos[1:], eef_pos[-1:]])
    tgt_rot = np.concatenate([eef_rot[1:], eef_rot[-1:]], axis=0)
    target_pose = make_pose(tgt_pos.astype(np.float32), tgt_rot.astype(np.float32))

    # 상대 명령. 월드 기준 차이를 구한 뒤 로봇 받침 좌표계로 돌린다. 핀 꽂기 환경의
    # target_eef_pose_to_action과 같은 식이다. 그 함수의 주석이 이 회전을 "받침이
    # z축으로 180도 돌아간 FR3에서 생성이 수렴하게 만든 수정"이라고 적고 있다.
    dpos_world = tgt_pos - eef_pos
    drot_mat = np.einsum("nij,nkj->nik", tgt_rot, eef_rot)       # tgt @ eef^T
    drot_world = axis_angle_from_matrix(drot_mat)
    dpos = quat_apply_inverse_xyzw(ROBOT_QUAT_XYZW, dpos_world)
    drot = quat_apply_inverse_xyzw(ROBOT_QUAT_XYZW, drot_world)

    finger_target = src["actions"][:, 7][keep]
    grip = np.where(finger_target >= GRIP_OPEN_THRESHOLD_M, GRIP_OPEN, GRIP_CLOSE)

    actions = np.concatenate([dpos, drot, grip[:, None]], axis=1).astype(np.float32)

    # 걸쇠형 구간 신호. 경계 스텝부터 끝까지 참이다. 마지막 구간은 신호를 만들지 않는다.
    # 구간이 여섯 개면 신호는 다섯 개다. MimicGen이 그렇게 센다.
    order = [f"grasp_{GEARS[0]}", f"seat_{GEARS[0]}",
             f"grasp_{GEARS[1]}", f"seat_{GEARS[1]}",
             f"grasp_{GEARS[2]}"]
    signals = {}
    for name in order:
        cut_src = bounds[name]
        # 줄인 뒤의 번호로 옮긴다. 경계 이상인 첫 자리를 쓴다.
        cut = int(np.searchsorted(keep, cut_src, side="left"))
        cut = min(cut, n - 1)
        sig = np.zeros(n, dtype=bool)
        sig[cut:] = True
        signals[name] = sig

    objects, object_raw = {}, {}
    for key in ("base",) + GEARS:
        p = src[f"obs/{key}_pose"][:][keep].astype(np.float64)
        object_raw[key] = p.astype(np.float32)
        objects[key] = make_pose(p[:, :3].astype(np.float32),
                                 quat_xyzw_to_matrix(p[:, 3:]).astype(np.float32))

    joint_pos = src["obs/robot_joint_pos"][:][keep].astype(np.float32)
    joint_vel = src["obs/robot_joint_vel"][:][keep].astype(np.float32)

    return dict(n=n, keep=keep, actions=actions, eef_pose=eef_pose,
                target_pose=target_pose, signals=signals, objects=objects,
                object_raw=object_raw,
                joint_pos=joint_pos, joint_vel=joint_vel,
                eef_pos=eef_pos.astype(np.float32), eef_quat=eef_quat.astype(np.float32),
                bounds=bounds, phase_names=names, n_src=n_src)


def write_demo(group, conv: dict, src):
    """바꾼 값들을 peg_annotated.hdf5와 같은 구조로 적는다."""
    n = conv["n"]
    group.create_dataset("actions", data=conv["actions"])
    group.create_dataset("obs/actions", data=conv["actions"])
    group.create_dataset("obs/eef_pos", data=conv["eef_pos"])
    group.create_dataset("obs/eef_quat", data=conv["eef_quat"])
    group.create_dataset("obs/joint_pos", data=conv["joint_pos"])
    group.create_dataset("obs/joint_vel", data=conv["joint_vel"])
    group.create_dataset("obs/gripper_pos", data=conv["joint_pos"][:, 7:9])

    # 물체마다 위치와 방향을 따로 적는다. MimicGen의 생성은 datagen_info만 읽지만
    # 우리 파이프라인의 convert와 render 단계가 obs 아래의 이 이름들을 찾는다.
    for key, raw in conv["object_raw"].items():
        group.create_dataset(f"obs/{key}_pos", data=raw[:, :3])
        group.create_dataset(f"obs/{key}_quat", data=raw[:, 3:])

    # 동작 관리자가 실제로 관절에 보낸 값. 앞 여섯 개가 배율을 적용한 손끝 상대 명령이고
    # 뒤 두 개가 손가락 두 개에 주는 같은 목표다. 열 때 0.04, 닫을 때 0.0이다.
    grip_cmd = np.where(conv["actions"][:, 6:7] >= 0.0, 0.04, 0.0).astype(np.float32)
    group.create_dataset("processed_actions", data=np.concatenate(
        [conv["actions"][:, :6] * ARM_SCALE, grip_cmd, grip_cmd], axis=1).astype(np.float32))

    group.create_dataset("obs/datagen_info/eef_pose/franka", data=conv["eef_pose"])
    group.create_dataset("obs/datagen_info/target_eef_pose/franka", data=conv["target_pose"])
    for key, pose in conv["objects"].items():
        group.create_dataset(f"obs/datagen_info/object_pose/{key}", data=pose)
    for key, sig in conv["signals"].items():
        group.create_dataset(f"obs/datagen_info/subtask_term_signals/{key}", data=sig)

    # 초기 상태. 생성이 장면을 이 자리로 되돌린 다음 궤적을 옮겨 붙인다.
    root_pose = np.asarray([[*ROBOT_POS, *ROBOT_QUAT_XYZW]], dtype=np.float32)
    group.create_dataset("initial_state/articulation/robot/root_pose", data=root_pose)
    group.create_dataset("initial_state/articulation/robot/root_velocity",
                         data=np.zeros((1, 6), dtype=np.float32))
    group.create_dataset("initial_state/articulation/robot/joint_position",
                         data=conv["joint_pos"][:1])
    group.create_dataset("initial_state/articulation/robot/joint_velocity",
                         data=conv["joint_vel"][:1])
    for key in ("base",) + GEARS:
        p = src[f"obs/{key}_pose"][:1].astype(np.float32)
        group.create_dataset(f"initial_state/rigid_object/{key}/root_pose", data=p)
        group.create_dataset(f"initial_state/rigid_object/{key}/root_velocity",
                             data=np.zeros((1, 6), dtype=np.float32))

    # 스텝마다의 상태.
    group.create_dataset("states/articulation/robot/joint_position", data=conv["joint_pos"])
    group.create_dataset("states/articulation/robot/joint_velocity", data=conv["joint_vel"])
    group.create_dataset("states/articulation/robot/root_pose",
                         data=np.repeat(root_pose, n, axis=0))
    group.create_dataset("states/articulation/robot/root_velocity",
                         data=np.zeros((n, 6), dtype=np.float32))
    for key in ("base",) + GEARS:
        p = src[f"obs/{key}_pose"][:][conv["keep"]].astype(np.float32)
        group.create_dataset(f"states/rigid_object/{key}/root_pose", data=p)
        group.create_dataset(f"states/rigid_object/{key}/root_velocity",
                             data=np.zeros((n, 6), dtype=np.float32))

    group.attrs["num_samples"] = n
    group.attrs["success"] = True
    group.attrs["source_demo_number"] = int(src.attrs["source_demo_number"])
    group.attrs["layout_json"] = src.attrs["layout_json"]
    group.attrs["subtask_bounds_source_steps"] = json.dumps(conv["bounds"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="기어 시연 HDF5")
    ap.add_argument("--episodes", required=True,
                    help="episodes 폴더. 단계 이름을 여기의 trajectory.npz에서 읽는다")
    ap.add_argument("--output", required=True, help="만들 파일")
    ap.add_argument("--grasp-margin-s", type=float, default=0.10,
                    help="손가락이 닫힌 뒤 집기 구간을 더 끌 시간, 초. 쥔 것이 "
                         "안정될 틈을 준다 (기본 0.10)")
    ap.add_argument("--seat-margin-s", type=float, default=0.10,
                    help="손을 뺀 뒤 꽂기 구간을 더 끌 시간, 초 (기본 0.10)")
    ap.add_argument("--task-id", default="Isaac-MultiGear-LabFR3-IK-Rel-Mimic-v0")
    args = ap.parse_args(argv)

    stride = int(round(SOURCE_HZ / TARGET_HZ))
    grasp_margin = int(round(args.grasp_margin_s * SOURCE_HZ))
    seat_margin = int(round(args.seat_margin_s * SOURCE_HZ))

    print(f"[변환] 초당 {SOURCE_HZ:.0f}스텝을 {TARGET_HZ:.0f}스텝으로 줄인다 "
          f"({stride}스텝에 하나)")
    print(f"[변환] 집기 마진 {args.grasp_margin_s}초({grasp_margin}스텝), "
          f"꽂기 마진 {args.seat_margin_s}초({seat_margin}스텝)")

    src = h5py.File(args.input, "r")
    keys = sorted(src["data"], key=lambda x: int(x.split("_")[1]))

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    total = 0
    with h5py.File(args.output, "w") as out:
        data = out.create_group("data")
        for i, key in enumerate(keys):
            s = src["data"][key]
            sdn = int(s.attrs["source_demo_number"])
            npz = os.path.join(args.episodes, f"demo_{sdn:02d}", "trajectory.npz")
            if not os.path.isfile(npz):
                print(f"  건너뜀  {key}: {npz}가 없다")
                continue
            conv = convert_demo(s, npz, grasp_margin, seat_margin, stride)
            write_demo(data.create_group(f"demo_{i}"), conv, s)
            total += conv["n"]
            cuts = {k: int(np.searchsorted(conv["keep"], v)) for k, v in conv["bounds"].items()}
            print(f"  {key} -> demo_{i}  {conv['n_src']}스텝을 {conv['n']}스텝으로  "
                  f"단계 {len(conv['phase_names'])}개  경계 {cuts}")
        data.attrs["total"] = total
        data.attrs["env_args"] = json.dumps({
            "env_name": args.task_id,
            "type": 2,
            "sim_args": {"dt": 1.0 / 100.0, "decimation": 5,
                         "render_interval": 2, "num_envs": 1},
        })
        data.attrs["grasp_margin_s"] = args.grasp_margin_s
        data.attrs["seat_margin_s"] = args.seat_margin_s

    print(f"[변환] 시연 {len(keys)}편, 전체 {total}스텝을 {args.output}에 적었다")
    return 0


if __name__ == "__main__":
    sys.exit(main())
