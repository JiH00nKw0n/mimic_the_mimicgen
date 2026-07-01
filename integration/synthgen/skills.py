"""Skill-segment data structures shared by SART and CP-Gen.

A demo is a waypoint sequence + an object-centric segmentation (SkillGen output).
Both augmentors consume the SAME structures, which is what lets them plug into one
pipeline. See ../INTEGRATION context and robot_data workspace — augmentation_methods/INTEGRATION_PLAN.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import numpy as np

from .math_utils import Pose

# 삽입/정렬 '동작' 키워드 (물체명 아님). cpgen/adapters 와 동일 규약.
INSERT_KEYWORDS = (
    "insert", "hole", "thread", "mesh", "assemble", "assembly",
    "align", "mate", "seat", "fit", "engage",
)


def is_insert_skill(name: str) -> bool:
    n = name.lower()
    return any(k in n for k in INSERT_KEYWORDS)


class SkillType(Enum):
    TRANSIT = "transit"   # free-space; stitched by cuRobo motion planning
    SKILL = "skill"       # contact-rich core; replayed (object-centric transform)
    INSERT = "insert"     # tight-tolerance sub-skill; SART local augmentation target


@dataclass
class Waypoint:
    t: float
    eef_pose: Pose                       # [x,y,z, qx,qy,qz,qw] in world
    gripper: float = 0.0                 # normalized gripper command
    joint_pos: Optional[np.ndarray] = None


@dataclass
class Keypoint:
    """CP-Gen: a point tracked relative to a task object (in that object's frame)."""
    name: str
    obj_frame: str                       # which object this keypoint is anchored to
    local_pos: np.ndarray                # position in the object frame


@dataclass
class SkillSegment:
    name: str
    skill_type: SkillType
    t_start: int
    t_end: int                           # inclusive
    ref_object: Optional[str] = None     # object frame for object-centric transform
    keypoints: List[Keypoint] = field(default_factory=list)   # CP-Gen keypoint-traj

    @property
    def is_insert(self) -> bool:
        return self.skill_type == SkillType.INSERT or is_insert_skill(self.name)


@dataclass
class Demo:
    """One source demonstration + its segmentation + scene info."""
    waypoints: List[Waypoint]
    segments: List[SkillSegment]
    world_idx: int = 0
    object_poses: Dict[str, Pose] = field(default_factory=dict)   # src scene object poses
    meta: Dict[str, object] = field(default_factory=dict)

    def segment_waypoints(self, seg: SkillSegment) -> List[Waypoint]:
        return self.waypoints[seg.t_start : seg.t_end + 1]

    @classmethod
    def from_skillgen(
        cls,
        waypoints: List[Waypoint],
        skill_segments: List[tuple],
        **kw,
    ) -> "Demo":
        """Build from SkillGen output: [(name, t0, t1), ...].

        Segment type is inferred: insert-keyword -> INSERT, else SKILL; callers can
        mark explicit TRANSIT segments by naming them 'transit'/'reach'/'move'.
        """
        segs: List[SkillSegment] = []
        for name, t0, t1 in skill_segments:
            low = name.lower()
            if is_insert_skill(name):
                st = SkillType.INSERT
            elif any(k in low for k in ("transit", "reach", "move", "approach", "retract")):
                st = SkillType.TRANSIT
            else:
                st = SkillType.SKILL
            segs.append(SkillSegment(name=name, skill_type=st, t_start=t0, t_end=t1))
        return cls(waypoints=waypoints, segments=segs, **kw)
