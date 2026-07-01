"""Backend interfaces (Protocols) that decouple the generation algorithms from
the concrete simulator / motion stack.

Our pipeline talks ONLY to these. Swapping MuJoCo/pinocchio (RoboManipBaselines)
for Isaac Lab + cuRobo means implementing these Protocols — the SART / CP-Gen
algorithms above them do not change.
"""
from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

import numpy as np

from .math_utils import Pose


@runtime_checkable
class SimEnv(Protocol):
    """Gymnasium-style manipulation env (Isaac Lab task / MuJoCo / mock)."""

    dt: float

    def reset(self, seed: Optional[int] = None) -> dict:
        """Reset to the demo's world; return observation dict."""
        ...

    def set_world(self, world_idx: int) -> None:
        """Select the object-layout variant (a.k.a. reset config)."""
        ...

    def step(self, action: np.ndarray) -> tuple:
        """Return (obs, reward, terminated, truncated, info). reward>=1 == success."""
        ...

    def get_time(self) -> float:
        ...

    @property
    def num_joints(self) -> int:
        ...


@runtime_checkable
class IKSolver(Protocol):
    """EEF pose -> joint config. cuRobo (Isaac Lab) or pinocchio (RMB)."""

    def solve(self, eef_pose: Pose, seed_q: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """Return joint positions, or None if no IK solution."""


@runtime_checkable
class MotionPlanner(Protocol):
    """Collision-free joint-space plan between two configs (cuRobo).

    Used for free-space *transit* stitching (the SkillGen/CP-Gen axis).
    """

    def plan(self, q_start: np.ndarray, q_goal: np.ndarray) -> Optional[np.ndarray]:
        """Return (T, num_joints) joint trajectory, or None if planning failed."""


@runtime_checkable
class CollisionChecker(Protocol):
    """World-collision query against the current scene / point cloud (cuRobo world).

    Replaces SART's hand-annotated 'acceptable sphere' with an online check.
    """

    def is_collision_free(self, q: np.ndarray) -> bool:
        ...

    def update_world(self, obstacles: object) -> None:
        """Load scene geometry (mesh / point cloud from hand camera)."""


@runtime_checkable
class DataWriter(Protocol):
    """Episode writer to our synthetic-data schema (HDF5 / Isaac Lab Mimic)."""

    def begin_episode(self, meta: dict) -> None:
        ...

    def record(self, step: dict) -> None:
        ...

    def end_episode(self, success: bool) -> None:
        ...

    def save(self, path: str) -> None:
        ...
