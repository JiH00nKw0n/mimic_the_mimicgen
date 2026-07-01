"""SkillGen-family generator with SART & CP-Gen as pluggable parts.

This is our synthetic-data loop (see robot_data workspace — augmentation_methods/MIMICGEN_FAMILY_DEFINITION.md):

    for each new scene:
        transform each skill segment          <- CP-Gen mode (pose+geometry) or rigid
        stitch transits                       <- cuRobo MotionPlanner
        replay open-loop + success-filter     <- runtime.execute_trajectory
        [optional] SART local boost on insert <- SartAugmentor around precision waypoints
    until N successes.

Transform mode:
    'rigid'    -> MimicGen/SkillGen object-centric SE(3) (geometry forced identity)
    'keypoint' -> CP-Gen: object-centric + geometry(scale) sampling (insert stays tight)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from .cpgen_transform import CpGenConfig, GeometrySample, KeypointTrajectoryTransform
from .interfaces import CollisionChecker, DataWriter, IKSolver, MotionPlanner, SimEnv
from .math_utils import Pose
from .runtime import execute_trajectory, waypoints_to_poses
from .sart_augmentor import SartAugmentor
from .skills import Demo, SkillSegment, SkillType, Waypoint

# scene_sampler(demo, rng) -> {object_name: new_world_pose}
SceneSampler = Callable[[Demo, np.random.Generator], Dict[str, Pose]]


@dataclass
class PipelineConfig:
    transform_mode: str = "keypoint"        # 'keypoint' (CP-Gen) | 'rigid' (SkillGen)
    num_success: int = 100
    max_attempts: int = 1000
    sart_boost: bool = True                 # add SART local variants on insert skills
    seed: int = 0


class SkillGenPipeline:
    def __init__(
        self,
        env: SimEnv,
        ik: IKSolver,
        writer: DataWriter,
        planner: Optional[MotionPlanner] = None,
        collision: Optional[CollisionChecker] = None,
        transform: Optional[KeypointTrajectoryTransform] = None,
        sart: Optional[SartAugmentor] = None,
        cfg: PipelineConfig = field(default_factory=PipelineConfig),
    ):
        self.env = env
        self.ik = ik
        self.writer = writer
        self.planner = planner
        self.collision = collision
        self.transform = transform or KeypointTrajectoryTransform(CpGenConfig(seed=cfg.seed))
        self.sart = sart
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

    def _transform_demo(self, demo: Demo, new_object_poses: Dict[str, Pose], world_idx: int) -> Demo:
        """Build a new-scene Demo by transforming each segment (indices recomputed)."""
        new_wps: List[Waypoint] = []
        new_segs: List[SkillSegment] = []
        idx = 0
        for seg in demo.segments:
            # rigid mode = keypoint transform with identity geometry (no scale)
            geom = GeometrySample.identity() if self.cfg.transform_mode == "rigid" else None
            poses: List[Pose] = self.transform.transform_segment(demo, seg, new_object_poses, geom)
            src_wps = demo.segment_waypoints(seg)
            seg_start = idx
            for k, p in enumerate(poses):
                grip = src_wps[k].gripper if k < len(src_wps) else 0.0
                new_wps.append(Waypoint(t=float(idx), eef_pose=p, gripper=grip))
                idx += 1
            new_segs.append(
                SkillSegment(
                    name=seg.name, skill_type=seg.skill_type,
                    t_start=seg_start, t_end=idx - 1,
                    ref_object=seg.ref_object, keypoints=seg.keypoints,
                )
            )
        return Demo(
            waypoints=new_wps, segments=new_segs, world_idx=world_idx,
            object_poses=new_object_poses, meta=demo.meta,
        )

    def generate(self, demo: Demo, scene_sampler: SceneSampler) -> int:
        """Run the generation loop until num_success successes. Returns total written."""
        successes = 0
        attempts = 0
        while successes < self.cfg.num_success and attempts < self.cfg.max_attempts:
            attempts += 1
            world_idx = attempts  # each attempt = a new scene / reset config
            new_poses = scene_sampler(demo, self.rng)
            gen_demo = self._transform_demo(demo, new_poses, world_idx)
            poses, grippers = waypoints_to_poses(gen_demo.waypoints)
            meta = {
                "source": f"skillgen:{self.cfg.transform_mode}",
                "attempt": attempts,
                "world_idx": world_idx,
                "seed": self.cfg.seed,
            }
            ok = execute_trajectory(
                self.env, self.ik, self.collision, self.writer,
                poses, grippers, meta, world_idx=world_idx,
            )
            if ok:
                successes += 1
                # SART local boost around this scene's insert precision waypoints
                if self.cfg.sart_boost and self.sart is not None:
                    successes += self.sart.augment(gen_demo, world_idx=world_idx)
        return successes

    def has_transits(self, demo: Demo) -> bool:
        """Transit segments would be stitched by self.planner (cuRobo) in the real run."""
        return any(s.skill_type == SkillType.TRANSIT for s in demo.segments)
