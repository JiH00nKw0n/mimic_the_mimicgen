#!/usr/bin/env python3
"""Dependency-free pose/action adapter for Cube Stage-1 model 4500."""

from __future__ import annotations

import argparse
import math
from typing import Iterable, Sequence

ACTION_SCALE = (0.02, 0.02, 0.02, 0.02, 0.02, 0.2)


def _vec(values: Iterable[float], size: int, name: str) -> tuple[float, ...]:
    out = tuple(float(v) for v in values)
    if len(out) != size or not all(math.isfinite(v) for v in out):
        raise ValueError(f"{name} must have {size} finite values")
    return out


def quat_normalize(q: Sequence[float]) -> tuple[float, float, float, float]:
    w, x, y, z = _vec(q, 4, "quaternion")
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-12:
        raise ValueError("zero quaternion")
    return (w / n, x / n, y / n, z / n)


def quat_conjugate(q: Sequence[float]) -> tuple[float, float, float, float]:
    w, x, y, z = quat_normalize(q)
    return (w, -x, -y, -z)


def quat_multiply(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float, float]:
    aw, ax, ay, az = quat_normalize(a)
    bw, bx, by, bz = quat_normalize(b)
    return quat_normalize((
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ))


def axis_angle_to_quat(v: Sequence[float]) -> tuple[float, float, float, float]:
    x, y, z = _vec(v, 3, "axis_angle")
    angle = math.sqrt(x * x + y * y + z * z)
    if angle < 1e-12:
        return (1.0, 0.0, 0.0, 0.0)
    s = math.sin(angle / 2.0) / angle
    return quat_normalize((math.cos(angle / 2.0), x * s, y * s, z * s))


def quat_to_axis_angle(q: Sequence[float]) -> tuple[float, float, float]:
    w, x, y, z = quat_normalize(q)
    if w < 0.0:
        w, x, y, z = -w, -x, -y, -z
    s = math.sqrt(x * x + y * y + z * z)
    if s < 1e-12:
        return (0.0, 0.0, 0.0)
    angle = 2.0 * math.atan2(s, max(-1.0, min(1.0, w)))
    return (x * angle / s, y * angle / s, z * angle / s)


def action_to_target_pose(current_pos, current_quat_wxyz, action):
    pos = _vec(current_pos, 3, "current_pos")
    quat = quat_normalize(current_quat_wxyz)
    raw = _vec(action, 7, "action")
    delta = tuple(raw[i] * ACTION_SCALE[i] for i in range(6))
    return {
        "position": tuple(pos[i] + delta[i] for i in range(3)),
        "quaternion_wxyz": quat_multiply(axis_angle_to_quat(delta[3:]), quat),
        "processed_cartesian_delta": delta,
        "gripper": "open" if raw[6] > 0.0 else "close",
    }


def target_pose_to_action(current_pos, current_quat_wxyz, target_pos, target_quat_wxyz, gripper):
    pos = _vec(current_pos, 3, "current_pos")
    quat = quat_normalize(current_quat_wxyz)
    target_pos = _vec(target_pos, 3, "target_pos")
    target_quat = quat_normalize(target_quat_wxyz)
    dp = tuple(target_pos[i] - pos[i] for i in range(3))
    dr = quat_to_axis_angle(quat_multiply(target_quat, quat_conjugate(quat)))
    arm = tuple(dp[i] / ACTION_SCALE[i] for i in range(3)) + tuple(
        dr[i] / ACTION_SCALE[i + 3] for i in range(3)
    )
    if isinstance(gripper, str):
        if gripper.lower() not in {"open", "close"}:
            raise ValueError("gripper must be open or close")
        g = 1.0 if gripper.lower() == "open" else -1.0
    else:
        g = 1.0 if float(gripper) > 0.0 else -1.0
    return arm + (g,)


def self_test():
    p = (0.4, -0.2, 0.5)
    q = quat_normalize((0.9, 0.1, -0.2, 0.3))
    for action in (
        (0, 0, 0, 0, 0, 0, -1),
        (0.5, -0.25, 0.75, 0.2, -0.4, 0.1, 1),
        (-1.2, 0.8, -0.1, -0.3, 0.15, -0.25, -0.2),
    ):
        target = action_to_target_pose(p, q, action)
        rebuilt = target_pose_to_action(p, q, target["position"], target["quaternion_wxyz"], target["gripper"])
        for i in range(6):
            if abs(float(action[i]) - rebuilt[i]) > 1e-9:
                raise AssertionError((action, rebuilt, i))
    print("PASS: Cube Stage-1 pose/action round-trip")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if not args.self_test:
        parser.error("use --self-test")
    self_test()

