"""In-memory mock backends so the pure-python core runs & tests WITHOUT Isaac Lab/cuRobo.

These stand in for the real SimEnv / IKSolver / CollisionChecker / MotionPlanner /
DataWriter. They let us validate the SART + CP-Gen algorithms end-to-end today;
swap them for curobo_backend / isaaclab_env implementations when running for real.

Action convention used across the skeleton:
    action = np.ndarray shape (8,) = [x, y, z, qx, qy, qz, qw, gripper]  (EEF command)
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from .math_utils import Pose, pos_error


class MockEnv:
    """Toy env: 'success' iff the final commanded EEF is within tol of a goal pose.

    Tighter augmentation (smaller sphere) -> more successes, so the success filter
    actually does something in tests.
    """

    def __init__(self, goal_pose: Optional[Pose] = None, success_pos_tol: float = 0.02,
                 num_joints: int = 7, dt: float = 0.05):
        self.goal_pose = goal_pose
        self.success_pos_tol = success_pos_tol
        self._num_joints = num_joints
        self.dt = dt
        self._t = 0.0
        self._last_eef: Optional[Pose] = None

    def reset(self, seed: Optional[int] = None) -> dict:
        self._t = 0.0
        self._last_eef = None
        return {"eef_pose": np.zeros(7)}

    def set_world(self, world_idx: int) -> None:
        self._world_idx = world_idx

    def step(self, action: np.ndarray):
        self._last_eef = np.asarray(action[:7], dtype=float)
        self._t += self.dt
        if self.goal_pose is not None and self._last_eef is not None:
            reward = 1.0 if pos_error(self._last_eef, self.goal_pose) < self.success_pos_tol else 0.0
        else:
            reward = 1.0
        obs = {"eef_pose": self._last_eef}
        return obs, reward, False, False, {}

    def get_time(self) -> float:
        return self._t

    @property
    def num_joints(self) -> int:
        return self._num_joints


class MockIK:
    """Always-solvable deterministic IK (pose -> pseudo joints)."""

    def __init__(self, num_joints: int = 7, workspace_radius: float = 5.0):
        self.num_joints = num_joints
        self.workspace_radius = workspace_radius

    def solve(self, eef_pose: Pose, seed_q: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        if np.linalg.norm(eef_pose[:3]) > self.workspace_radius:
            return None  # out of reach -> no IK
        base = np.concatenate([eef_pose[:3], eef_pose[3:7]])
        q = np.resize(base, self.num_joints).astype(float)
        return q


class MockCollision:
    """No obstacles by default; optionally reject configs outside a joint box."""

    def __init__(self, q_abs_limit: float = 10.0):
        self.q_abs_limit = q_abs_limit

    def is_collision_free(self, q: np.ndarray) -> bool:
        return bool(np.all(np.abs(q) <= self.q_abs_limit))

    def update_world(self, obstacles: object) -> None:
        self._obstacles = obstacles


class MockPlanner:
    """Straight-line joint interpolation (stand-in for cuRobo transit planning)."""

    def __init__(self, steps: int = 20):
        self.steps = steps

    def plan(self, q_start: np.ndarray, q_goal: np.ndarray) -> Optional[np.ndarray]:
        ts = np.linspace(0.0, 1.0, self.steps)[:, None]
        return (1 - ts) * q_start[None, :] + ts * q_goal[None, :]


class InMemoryDataWriter:
    """Collects episodes in RAM; mirrors the DataWriter Protocol."""

    def __init__(self):
        self.episodes: List[dict] = []
        self._cur: Optional[dict] = None

    def begin_episode(self, meta: dict) -> None:
        self._cur = {"meta": dict(meta), "steps": []}

    def record(self, step: dict) -> None:
        assert self._cur is not None, "begin_episode() first"
        self._cur["steps"].append(step)

    def end_episode(self, success: bool) -> None:
        assert self._cur is not None
        self._cur["success"] = success
        self.episodes.append(self._cur)
        self._cur = None

    def save(self, path: str) -> None:
        # in-memory: nothing to flush. Kept for Protocol parity.
        pass

    @property
    def num_success(self) -> int:
        return sum(1 for e in self.episodes if e.get("success"))


# --------------------------------------------------------------------------- #
# Toy fixtures (used by examples/ and tests/) — a peg-in-hole-ish demo.
# --------------------------------------------------------------------------- #
def make_toy_demo():
    """3-segment demo: reach(transit) -> grasp(skill) -> insert_peg(insert@socket)."""
    from .skills import Demo, SkillSegment, SkillType, Waypoint

    def pose(x, y, z):
        return np.array([x, y, z, 0.0, 0.0, 0.0, 1.0])

    socket = pose(0.5, 0.0, 0.10)
    wps = [
        Waypoint(t=0, eef_pose=pose(0.30, 0.0, 0.40), gripper=0.0),   # reach
        Waypoint(t=1, eef_pose=pose(0.40, 0.0, 0.30), gripper=0.0),
        Waypoint(t=2, eef_pose=pose(0.50, 0.0, 0.25), gripper=1.0),   # grasp
        Waypoint(t=3, eef_pose=pose(0.50, 0.0, 0.18), gripper=1.0),
        Waypoint(t=4, eef_pose=pose(0.50, 0.0, 0.12), gripper=1.0),   # align
        Waypoint(t=5, eef_pose=pose(0.50, 0.0, 0.10), gripper=1.0),   # insert convergence
    ]
    segs = [
        SkillSegment("reach", SkillType.TRANSIT, 0, 1, ref_object=None),
        SkillSegment("grasp", SkillType.SKILL, 2, 3, ref_object="socket"),
        SkillSegment("insert_peg", SkillType.INSERT, 4, 5, ref_object="socket"),
    ]
    return Demo(
        waypoints=wps, segments=segs, world_idx=0,
        object_poses={"socket": socket},
        meta={"radius::insert_peg": 0.01},
    )


def toy_scene_sampler(demo, rng):
    """Perturb the socket pose to make a new scene."""
    new = dict(demo.object_poses)
    src = demo.object_poses["socket"].copy()
    src[:2] += rng.uniform(-0.03, 0.03, size=2)  # jitter x,y
    new["socket"] = src
    return new
