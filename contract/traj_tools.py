"""Pure-python trajectory tools for the FR3 cube control-contract conversion.

Everything here is stdlib-only (like the vendored adapter) so it runs in any
python — laptop, robosuite venv, or Isaac Sim's bundled interpreter.

Conventions match the contract: positions in metres, quaternions wxyz,
robot-base frame, 10 Hz policy rate.
"""
from __future__ import annotations

import math
from typing import Sequence

from adapter import (  # vendored controller_adapter (package-relative import
    ACTION_SCALE,      # is avoided so Isaac scripts can sys.path this dir)
    action_to_target_pose,
    quat_conjugate,
    quat_multiply,
    quat_normalize,
    quat_to_axis_angle,
    target_pose_to_action,
)

POLICY_DT = 0.1  # contract: 10 Hz


def slerp(qa: Sequence[float], qb: Sequence[float], t: float):
    """Shortest-arc slerp between two wxyz quaternions."""
    a = quat_normalize(qa)
    b = quat_normalize(qb)
    dot = sum(x * y for x, y in zip(a, b))
    if dot < 0.0:
        b = tuple(-v for v in b)
        dot = -dot
    dot = min(1.0, dot)
    if dot > 0.9995:
        mixed = tuple((1 - t) * x + t * y for x, y in zip(a, b))
        return quat_normalize(mixed)
    theta = math.acos(dot)
    sa = math.sin((1 - t) * theta) / math.sin(theta)
    sb = math.sin(t * theta) / math.sin(theta)
    return quat_normalize(tuple(sa * x + sb * y for x, y in zip(a, b)))


def resample_pose_track(
    times: Sequence[float],
    positions: Sequence[Sequence[float]],
    quats_wxyz: Sequence[Sequence[float]],
    grippers: Sequence[float],
    target_dt: float = POLICY_DT,
):
    """Time-resample a pose track to the contract policy rate.

    positions: linear interpolation; quaternions: slerp; gripper: zero-order
    hold (value at or before the sample time). Returns (times, pos, quat, grip)
    tuples of equal length, covering [times[0], times[-1]].
    """
    if not (len(times) == len(positions) == len(quats_wxyz) == len(grippers)):
        raise ValueError("track arrays must have equal length")
    if len(times) < 2:
        raise ValueError("need at least two samples to resample")
    out_t, out_p, out_q, out_g = [], [], [], []
    duration = times[-1] - times[0]
    steps = max(1, int(round(duration / target_dt)))
    j = 0
    for k in range(steps + 1):
        t = min(times[0] + k * target_dt, times[-1])
        while j + 1 < len(times) - 1 and times[j + 1] <= t:
            j += 1
        t0, t1 = times[j], times[j + 1]
        w = 0.0 if t1 <= t0 else max(0.0, min(1.0, (t - t0) / (t1 - t0)))
        pos = tuple((1 - w) * a + w * b for a, b in zip(positions[j], positions[j + 1]))
        quat = slerp(quats_wxyz[j], quats_wxyz[j + 1], w)
        out_t.append(t - times[0])
        out_p.append(pos)
        out_q.append(quat)
        out_g.append(grippers[j if w < 1.0 else j + 1])
    return out_t, out_p, out_q, out_g


def track_to_actions(positions, quats_wxyz, grippers):
    """Contract actions from a 10 Hz actual-EE track.

    Step-t action commands the step-t+1 pose relative to the step-t ACTUAL
    pose (contract: reference is the current actual EE pose). Output length is
    len(track) - 1. Returns (actions, targets, deltas) where targets[k] is the
    commanded [x,y,z,qw,qx,qy,qz] and deltas[k] the scaled cartesian delta.
    """
    actions, targets, deltas = [], [], []
    for k in range(len(positions) - 1):
        action = target_pose_to_action(
            positions[k], quats_wxyz[k],
            positions[k + 1], quats_wxyz[k + 1],
            grippers[k + 1],
        )
        actions.append(action)
        targets.append(tuple(positions[k + 1]) + quat_normalize(quats_wxyz[k + 1]))
        deltas.append(tuple(action[i] * ACTION_SCALE[i] for i in range(6)))
    return actions, targets, deltas


def round_trip_errors(positions, quats_wxyz, actions):
    """Max |pose - reconstruct(action)| over the track (contract smoke item)."""
    max_pos = 0.0
    max_rot = 0.0
    for k, action in enumerate(actions):
        rec = action_to_target_pose(positions[k], quats_wxyz[k], action)
        target_pos = positions[k + 1]
        max_pos = max(max_pos, max(
            abs(a - b) for a, b in zip(rec["position"], target_pos)))
        q_err = quat_multiply(rec["quaternion_wxyz"], quat_conjugate(quats_wxyz[k + 1]))
        angle = 2.0 * math.atan2(
            math.sqrt(sum(v * v for v in q_err[1:])), abs(q_err[0]))
        max_rot = max(max_rot, angle)
    return {"max_position_error_m": max_pos, "max_rotation_error_rad": max_rot}


def action_percentiles(actions, percentiles=(1, 5, 50, 95, 99)):
    """Per-dimension percentiles of raw actions (report requirement)."""
    out = {}
    dims = ["dx", "dy", "dz", "drx", "dry", "drz", "gripper"]
    for i, name in enumerate(dims):
        values = sorted(a[i] for a in actions)
        out[name] = {
            f"p{p}": values[min(len(values) - 1, int(round(p / 100 * (len(values) - 1))))]
            for p in percentiles
        }
    return out


def envelope_fraction(actions, reference_low, reference_high):
    """Fraction of raw arm actions outside the RL reference envelope per dim."""
    out = {}
    dims = ["dx", "dy", "dz", "drx", "dry", "drz"]
    for i, name in enumerate(dims):
        n_out = sum(1 for a in actions
                    if not (reference_low[i] <= a[i] <= reference_high[i]))
        out[name] = n_out / len(actions) if actions else 0.0
    return out
