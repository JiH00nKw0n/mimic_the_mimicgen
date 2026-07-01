"""Isaac Lab env adapter: fills the SimEnv Protocol on top of an Isaac Lab task.

Guarded imports so the pure-python core runs without Isaac Lab. This is the piece that
does NOT exist in RoboManipBaselines (RMB supports MuJoCo/IsaacGym/PyBullet, not Isaac
Lab) — so porting SART to our stack = implementing this adapter + using CuroboIK.

TODO(wire): back this with a ManagerBasedRLEnv (or DirectRLEnv) for the peg/gear task,
mapping our (8,) EEF action to the task's action term, and exposing reward>=1 as success.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def _require_isaaclab():
    try:
        import isaaclab  # noqa: F401
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "Isaac Lab not installed / not launched. Run inside the Isaac Lab python env "
            "(isaaclab.sh -p ...). Until then use synthgen.mocks.MockEnv for the core path."
        ) from e


class IsaacLabEnv:
    """Thin SimEnv wrapper around an Isaac Lab manipulation task."""

    def __init__(self, task_name: str, device: str = "cuda:0", dt: float = 1.0 / 60.0,
                 num_joints: int = 7):
        _require_isaaclab()
        self.task_name = task_name
        self.device = device
        self.dt = dt
        self._num_joints = num_joints
        # import gymnasium as gym
        # import isaaclab_tasks  # noqa: registers tasks
        # self._env = gym.make(task_name, ...)
        raise NotImplementedError(
            "IsaacLabEnv skeleton: create the ManagerBasedRLEnv for the peg/gear task."
        )

    def reset(self, seed: Optional[int] = None) -> dict:
        # obs, _ = self._env.reset(seed=seed); return self._to_obs_dict(obs)
        raise NotImplementedError

    def set_world(self, world_idx: int) -> None:
        # select the object-layout variant / randomization seed for this reset
        raise NotImplementedError

    def step(self, action: np.ndarray):
        # convert (8,) EEF+gripper action to the task action term, step, unpack
        # obs, reward, terminated, truncated, info = self._env.step(torch_action)
        raise NotImplementedError

    def get_time(self) -> float:
        raise NotImplementedError

    @property
    def num_joints(self) -> int:
        return self._num_joints
