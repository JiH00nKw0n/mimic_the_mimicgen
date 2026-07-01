"""synthgen — integrate SART & CP-Gen into our Isaac Lab + cuRobo synthetic-data pipeline.

Public API:
    from synthgen import (
        Demo, Waypoint, SkillSegment, SkillType,
        SartAugmentor, SartConfig,
        KeypointTrajectoryTransform, CpGenConfig, GeometrySample,
        SkillGenPipeline, PipelineConfig,
        execute_trajectory,
    )
Backends (implement per your stack):
    from synthgen.mocks import MockEnv, MockIK, MockCollision, MockPlanner, InMemoryDataWriter
    from synthgen.curobo_backend import CuroboIK, CuroboPlanner, CuroboCollision
    from synthgen.isaaclab_env import IsaacLabEnv
    from synthgen.data_schema import HDF5DataWriter
"""
from .cpgen_transform import (
    CpGenConfig,
    GeometrySample,
    KeypointTrajectoryTransform,
    keypoint_object_transform,
    rigid_object_transform,
)
from .interfaces import CollisionChecker, DataWriter, IKSolver, MotionPlanner, SimEnv
from .pipeline import PipelineConfig, SceneSampler, SkillGenPipeline
from .runtime import execute_trajectory, waypoints_to_poses
from .sart_augmentor import SartAugmentor, SartConfig
from .skills import Demo, Keypoint, SkillSegment, SkillType, Waypoint, is_insert_skill

__version__ = "0.0.1"

__all__ = [
    "Demo", "Waypoint", "Keypoint", "SkillSegment", "SkillType", "is_insert_skill",
    "SartAugmentor", "SartConfig",
    "KeypointTrajectoryTransform", "CpGenConfig", "GeometrySample",
    "rigid_object_transform", "keypoint_object_transform",
    "SkillGenPipeline", "PipelineConfig", "SceneSampler",
    "execute_trajectory", "waypoints_to_poses",
    "SimEnv", "IKSolver", "MotionPlanner", "CollisionChecker", "DataWriter",
    "__version__",
]
