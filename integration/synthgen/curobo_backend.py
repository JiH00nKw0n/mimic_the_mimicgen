"""cuRobo backend: IK + collision-free planning + world-collision checking.

This is where OUR existing cuRobo assets plug in. Fills the IKSolver / MotionPlanner
/ CollisionChecker Protocols. Imports are guarded so the pure-python core (mocks) runs
without cuRobo installed; instantiating these classes without cuRobo raises a clear error.

Reference: cuRobo docs https://curobo.org ; CP-Gen already wraps cuRobo in
robot_data workspace — augmentation_methods/cpgen/repo/demo_aug/envs/motion_planners/curobo_mp.py
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .math_utils import Pose


def _require_curobo():
    try:
        import curobo  # noqa: F401
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "cuRobo not installed. Install per https://curobo.org/get_started/1_install_instructions.html "
            "(GPU required). Until then use synthgen.mocks for the pure-python path."
        ) from e


class CuroboIK:
    """EEF pose -> joint config via cuRobo IKSolver.

    TODO(wire): build IKSolverConfig from robot_cfg (UR5e / our arm), reuse a warm
    seed for temporal consistency along a trajectory.
    """

    def __init__(self, robot_cfg: str, world_cfg: Optional[object] = None,
                 tensor_device: str = "cuda:0"):
        _require_curobo()
        self.robot_cfg = robot_cfg
        self.world_cfg = world_cfg
        self.device = tensor_device
        # from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
        # self._ik = IKSolver(IKSolverConfig.load_from_robot_config(robot_cfg, world_cfg, ...))
        raise NotImplementedError(
            "CuroboIK skeleton: wire IKSolverConfig.load_from_robot_config here."
        )

    def solve(self, eef_pose: Pose, seed_q: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        # from curobo.types.math import Pose as CuPose
        # result = self._ik.solve_single(CuPose(position, quaternion_wxyz), seed_config=seed_q)
        # return result.solution[0].cpu().numpy() if result.success else None
        raise NotImplementedError


class CuroboPlanner:
    """Collision-free joint-space transit planning via cuRobo MotionGen.

    Used for SkillGen/CP-Gen free-space stitching between skill segments.
    """

    def __init__(self, robot_cfg: str, world_cfg: Optional[object] = None,
                 tensor_device: str = "cuda:0"):
        _require_curobo()
        self.robot_cfg = robot_cfg
        self.world_cfg = world_cfg
        self.device = tensor_device
        # from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig
        # self._mg = MotionGen(MotionGenConfig.load_from_robot_config(robot_cfg, world_cfg, ...))
        raise NotImplementedError("CuroboPlanner skeleton: wire MotionGen here.")

    def plan(self, q_start: np.ndarray, q_goal: np.ndarray) -> Optional[np.ndarray]:
        # result = self._mg.plan_single_js(JointState(q_start), JointState(q_goal), ...)
        # return result.get_interpolated_plan().position.cpu().numpy() if result.success else None
        raise NotImplementedError


class CuroboCollision:
    """World-collision query. Load the hand-camera point cloud / scene mesh here so the
    SART sphere sampling is validated against real geometry (an upgrade over hand spheres).
    """

    def __init__(self, robot_cfg: str, world_cfg: Optional[object] = None,
                 tensor_device: str = "cuda:0"):
        _require_curobo()
        self.robot_cfg = robot_cfg
        self.world_cfg = world_cfg
        self.device = tensor_device
        # from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig
        # self._rw = RobotWorld(RobotWorldConfig.load_from_config(robot_cfg, world_cfg, ...))
        raise NotImplementedError("CuroboCollision skeleton: wire RobotWorld here.")

    def is_collision_free(self, q: np.ndarray) -> bool:
        # d = self._rw.get_collision_distance(JointState(q)); return bool(d.min() > 0)
        raise NotImplementedError

    def update_world(self, obstacles: object) -> None:
        # from curobo.geom.types import WorldConfig, PointCloud/Mesh
        # self._rw.update_world(WorldConfig(...))  # obstacles from GenerateMergedPointCloud
        raise NotImplementedError
