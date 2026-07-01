"""Shared trajectory execution: IK -> collision-check -> env.step -> record -> success.

Both SartAugmentor and SkillGenPipeline funnel through execute_trajectory so the
IK/collision/replay/filter behavior is identical regardless of which augmentor produced
the EEF trajectory.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from .interfaces import CollisionChecker, DataWriter, IKSolver, SimEnv
from .math_utils import Pose


def execute_trajectory(
    env: SimEnv,
    ik: IKSolver,
    collision: Optional[CollisionChecker],
    writer: DataWriter,
    poses: Sequence[Pose],
    grippers: Sequence[float],
    meta: dict,
    world_idx: int = 0,
    reject_on_ik_fail: bool = True,
    reject_on_collision: bool = True,
) -> bool:
    """Replay an EEF-pose trajectory open-loop; return task success (reward>=1).

    Returns False early (episode marked failed) if IK/collision reject a step — this is
    the by-construction validity gate; the reward>=1 check is the task success filter.
    """
    writer.begin_episode(meta)
    env.set_world(world_idx)
    env.reset(seed=meta.get("seed"))

    seed_q: Optional[np.ndarray] = None
    last_reward = 0.0
    for pose, grip in zip(poses, grippers):
        q = ik.solve(pose, seed_q)
        if q is None:
            if reject_on_ik_fail:
                writer.end_episode(False)
                return False
            continue
        if reject_on_collision and collision is not None and not collision.is_collision_free(q):
            writer.end_episode(False)
            return False
        seed_q = q
        action = np.concatenate([np.asarray(pose, dtype=float), [float(grip)]])
        obs, reward, terminated, truncated, info = env.step(action)
        last_reward = float(reward)
        writer.record({"action": action, "obs": obs, "reward": reward, "q": q})
        if terminated or truncated:
            break

    success = last_reward >= 1.0
    writer.end_episode(success)
    return success


def waypoints_to_poses(waypoints) -> tuple:
    """Split a Waypoint list into (poses, grippers)."""
    poses: List[Pose] = [w.eef_pose for w in waypoints]
    grippers: List[float] = [w.gripper for w in waypoints]
    return poses, grippers
