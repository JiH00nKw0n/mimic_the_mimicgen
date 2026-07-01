"""CP-Gen transform mode: object-centric segment transform with geometry (scale) change.

Grafts CP-Gen's distinctive axis onto our SkillGen generator. Instead of only re-posing a
skill segment (rigid SE(3), MimicGen), we also sample object GEOMETRY (scale) so gear/peg
instances of different sizes are covered — the "keypoint-trajectory constraint" idea, in
its object-centric-transform form.

Two transforms provided:
  * rigid_object_transform  — the MimicGen/SkillGen primitive (baseline mode)
  * KeypointTrajectoryTransform — CP-Gen mode (pose + geometry), insert-aware tolerance

Both return a transformed EEF-pose trajectory the pipeline then IK's / plans / replays.
Insert/align segments get NARROW pose range + geometry OFF (tolerance preserved); transit
and non-insert skills get wider ranges. Mirrors robot_data workspace — augmentation_methods/cpgen/adapters.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .math_utils import (
    Pose,
    mat_from_pose,
    pose_from_mat,
)
from .skills import Demo, SkillSegment


@dataclass
class GeometrySample:
    """Scale applied in the reference-object frame (uniform or per-axis)."""
    scale: np.ndarray  # shape (3,)

    @staticmethod
    def identity() -> "GeometrySample":
        return GeometrySample(scale=np.ones(3))

    def matrix(self) -> np.ndarray:
        S = np.eye(4)
        S[0, 0], S[1, 1], S[2, 2] = self.scale
        return S


@dataclass
class CpGenConfig:
    # 삽입/정렬 구간 허용 오차(=pose 샘플 범위)
    insert_pos_tol_m: float = 0.004
    insert_rot_tol_deg: float = 3.0
    # 접근/비삽입 스킬 pose 범위
    transport_pos_bound_m: float = 0.05
    transport_rot_bound_deg: float = 30.0
    # geometry(scale) 샘플 범위 — 삽입 구간에서는 강제로 1.0 (형상 tolerance 보존)
    scale_range: tuple = (0.85, 1.15)
    apply_geometry: bool = True
    seed: int = 0


def rigid_object_transform(
    src_eef: Pose, src_obj: Pose, new_obj: Pose
) -> Pose:
    """MimicGen primitive: T_eef' = T_obj' * (T_obj)^-1 * T_eef  (object-centric)."""
    T = mat_from_pose(new_obj) @ np.linalg.inv(mat_from_pose(src_obj)) @ mat_from_pose(src_eef)
    return pose_from_mat(T)


def keypoint_object_transform(
    src_eef: Pose, src_obj: Pose, new_obj: Pose, geom: GeometrySample
) -> Pose:
    """CP-Gen primitive: object-centric transform with geometry (scale) in object frame.

        T_eef' = T_obj' * G * (T_obj)^-1 * T_eef
    G scales the relative offset in the object frame, so a larger/smaller instance moves
    the tracked keypoints (and thus the EEF) consistently.
    """
    T = (
        mat_from_pose(new_obj)
        @ geom.matrix()
        @ np.linalg.inv(mat_from_pose(src_obj))
        @ mat_from_pose(src_eef)
    )
    return pose_from_mat(T)


class KeypointTrajectoryTransform:
    """CP-Gen transform mode for the pipeline."""

    def __init__(self, cfg: CpGenConfig = CpGenConfig()):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

    def sample_geometry(self, seg: SkillSegment) -> GeometrySample:
        if seg.is_insert or not self.cfg.apply_geometry:
            return GeometrySample.identity()  # 삽입: 형상 고정
        lo, hi = self.cfg.scale_range
        s = self.rng.uniform(lo, hi)  # uniform scale (per-axis 확장은 여기서)
        return GeometrySample(scale=np.array([s, s, s]))

    def transform_segment(
        self,
        demo: Demo,
        seg: SkillSegment,
        new_object_poses: Dict[str, Pose],
        geom: Optional[GeometrySample] = None,
    ) -> List[Pose]:
        """Transform one skill segment's EEF trajectory to the new scene."""
        ref = seg.ref_object
        if ref is None or ref not in new_object_poses or ref not in demo.object_poses:
            # no object anchor -> pass through unchanged (e.g. free-space handled by planner)
            return [w.eef_pose for w in demo.segment_waypoints(seg)]
        src_obj = demo.object_poses[ref]
        new_obj = new_object_poses[ref]
        geom = geom if geom is not None else self.sample_geometry(seg)
        return [
            keypoint_object_transform(w.eef_pose, src_obj, new_obj, geom)
            for w in demo.segment_waypoints(seg)
        ]
